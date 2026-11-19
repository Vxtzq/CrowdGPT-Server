"""Sparse (`SPK1`) delta blocks: the server half of the transfer cut.

The wire contract lives in two places by necessity — the client encodes, the server
decodes — so these tests build a file byte-for-byte the way the client's
`_write_delta_file_sparse` does and assert the server reads it correctly. The
failure mode being guarded against is subtle: a sparse delta has a *variable-length*
base block, so every reader that assumes `model_size * 2` silently reads the engram
tail from the wrong offset and aggregates garbage.
"""
import json
import struct
import unittest

try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
    import torch  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")

import numpy as np

import krum_stream as K

MODEL_SIZE = 8
DIM = 2


def _sparse_block(entries, total=MODEL_SIZE):
    idx = np.array([e[0] for e in entries], dtype=np.uint32)
    vals = torch.tensor([e[1] for e in entries], dtype=torch.float32).to(torch.bfloat16)
    return (K.SPARSE_MAGIC + struct.pack("<II", len(entries), total)
            + idx.tobytes() + vals.view(torch.int16).numpy().astype(np.uint16).tobytes())


def _delta_file(tmpdir, entries, *, engram=(), codec="sparse"):
    """Write a delta exactly like the client does: header, base block, engram tail."""
    if codec == "sparse":
        block = _sparse_block(entries)
    else:                                        # dense: plain bf16 values, no indices
        t = torch.tensor([float(v) for v in entries], dtype=torch.float32).to(torch.bfloat16)
        block = t.view(torch.int16).numpy().astype(np.uint16).tobytes()
    meta = {"taskId": "t", "isDelta": True, "hasEngram": bool(len(engram)),
            "engramSparseCount": len(engram)}
    if codec == "sparse":
        meta.update({"baseCodec": "sparse", "baseBytes": len(block), "baseCount": len(entries)})
    head = json.dumps(meta).encode()
    path = f"{tmpdir}/delta.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(head))); f.write(head); f.write(block)
        f.write(struct.pack("<I", len(engram)))
        for row, values in engram:
            f.write(struct.pack("<I", row))
            f.write(torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
                    .view(torch.int16).numpy().astype(np.uint16).tobytes())
    return path, block


class SparseReadTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def test_header_declares_a_variable_length_block(self):
        path, block = _delta_file(self.tmp, [(1, 2.0), (5, -1.0)])
        df = K.open_delta(path, MODEL_SIZE)
        self.assertEqual(df.base_codec, "sparse")
        self.assertEqual(df.base_bytes, len(block))
        self.assertNotEqual(df.base_bytes, MODEL_SIZE * 2)   # the whole point

    def test_dense_deltas_keep_the_old_arithmetic(self):
        path, _ = _delta_file(self.tmp, [0.5] * MODEL_SIZE, codec="dense")
        df = K.open_delta(path, MODEL_SIZE)
        self.assertEqual(df.base_codec, "dense")
        self.assertEqual(df.base_bytes, MODEL_SIZE * 2)
        self.assertEqual([round(float(v), 2) for v in K._read_base(df)], [0.5] * MODEL_SIZE)

    def test_a_sparse_block_reconstructs_as_a_dense_delta(self):
        path, _ = _delta_file(self.tmp, [(1, 2.0), (5, -1.0)])
        df = K.open_delta(path, MODEL_SIZE)
        v = K._read_base(df)
        self.assertEqual(v.numel(), MODEL_SIZE)
        self.assertEqual(round(float(v[1]), 3), 2.0)
        self.assertEqual(round(float(v[5]), 3), -1.0)
        self.assertEqual(float(v.abs().sum()), 3.0)          # nothing invented elsewhere

    def test_the_merge_path_scatters_with_the_scale(self):
        path, _ = _delta_file(self.tmp, [(1, 2.0), (5, -1.0)])
        df = K.open_delta(path, MODEL_SIZE)
        w = torch.zeros(MODEL_SIZE)
        K.stream_add_base_delta(w, df, 0.5)
        self.assertEqual([round(float(x), 3) for x in w.tolist()], [0, 1, 0, 0, 0, -0.5, 0, 0])

    def test_the_outer_momentum_pair_scatter_uses_both_scales(self):
        path, _ = _delta_file(self.tmp, [(1, 2.0), (5, -1.0)])
        df = K.open_delta(path, MODEL_SIZE)
        w = torch.zeros(MODEL_SIZE); g = torch.zeros(MODEL_SIZE)
        K.stream_add_base_delta_pair(w, 0.8, g, -0.2, df)
        self.assertEqual(round(float(w[1]), 3), 1.6)
        self.assertEqual(round(float(g[1]), 3), -0.4)
        self.assertEqual(round(float(g[5]), 3), 0.2)

    def test_the_engram_tail_is_found_after_the_entries_not_after_a_dense_block(self):
        """A reader using `base_bytes` sees the tail; one using `model_size*2` would not."""
        path, _ = _delta_file(self.tmp, [(1, 2.0)], engram=[(3, [0.5, -0.5])])
        df = K.open_delta(path, MODEL_SIZE)
        with open(path, "rb") as f:
            f.seek(df.base_offset + df.base_bytes)
            count = struct.unpack("<I", f.read(4))[0]
            row = struct.unpack("<I", f.read(4))[0]
        self.assertEqual(count, 1)
        self.assertEqual(row, 3)
        # And the wrong offset really does disagree, so this test can fail.
        self.assertNotEqual(df.base_offset + MODEL_SIZE * 2, df.base_offset + df.base_bytes)

    def test_a_block_for_the_wrong_model_is_rejected(self):
        path, _ = _delta_file(self.tmp, [(1, 2.0)])
        df = K.open_delta(path, MODEL_SIZE)
        with self.assertRaises(ValueError):
            K._parse_sparse(K._read_block(df), MODEL_SIZE + 1)

    def test_a_truncated_block_is_rejected(self):
        with self.assertRaises(ValueError):
            K._parse_sparse(K.SPARSE_MAGIC + struct.pack("<II", 4, MODEL_SIZE) + b"\x00" * 6,
                            MODEL_SIZE)

    def test_a_dense_block_is_not_parsed_as_sparse(self):
        with self.assertRaises(ValueError):
            K._parse_sparse(b"\x00" * 20, MODEL_SIZE)


if __name__ == "__main__":
    unittest.main()
