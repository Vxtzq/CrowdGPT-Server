"""A volunteer that runs out of VRAM must not be flagged for saying so.

The client halves its batch size when a build OOMs. Before this, that silently
changed its hardware signature (the signature buckets bs to a power of two), so a
16 GB card that downscaled stopped being compatible with its own pool and its
deltas looked like a different peer's -- the false-positive path the author
reported. These tests pin the two halves of the fix: identity is bucketed on the
*declared* batch size, and a delta that declares a downscale is spared the
byzantine flag (it is still judged by the objective gate).
"""

import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # the repo's CI job is stdlib-only; skip there instead of erroring on import
    import torch  # noqa: F401
    import server  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"needs server dependencies: {exc}")


class HwSignatureTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(server.client_hardware)

    def tearDown(self):
        server.client_hardware.clear()
        server.client_hardware.update(self._saved)

    def _sig(self, bs, declared, cc="8.9"):
        server.client_hardware["card"] = {
            "backend": "CUDA", "bs": bs, "bs_declared": declared, "vram_gb": 16.0,
            "gpu": "NVIDIA GeForce RTX 4060 Ti", "autocast": "bf16", "optim": "any",
            "compute_cap": cc,
        }
        return server._hw_signature("card")

    def test_downscale_keeps_the_same_identity(self):
        self.assertEqual(self._sig(16, 16), self._sig(8, 16))

    def test_a_different_declared_size_is_a_different_machine(self):
        self.assertNotEqual(self._sig(16, 16)[1], self._sig(32, 32)[1])

    def test_missing_declared_value_falls_back_to_the_live_one(self):
        self.assertEqual(self._sig(16, 0)[1], self._sig(16, 16)[1])


class OomExemptionTest(unittest.TestCase):
    def _delta(self, meta):
        path = os.path.join(tempfile.mkdtemp(), "d.bin")
        blob = json.dumps(meta).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)
        return path

    def test_a_declared_downscale_is_not_a_byzantine_flag(self):
        calls = []
        saved = server._flag_byzantine
        server._flag_byzantine = lambda *a, **k: calls.append(a)
        try:
            before = server._OOM_EXEMPT["count"]
            server._flag_byzantine_from_delta(
                self._delta({"nodeKey": "k", "nodeIp": "1.2.3.4", "oomDownscales": 2}),
                "cross-check outlier")
            self.assertEqual(server._OOM_EXEMPT["count"], before + 1)
            self.assertEqual(calls, [])
        finally:
            server._flag_byzantine = saved

    def test_without_a_declared_downscale_the_flag_still_fires(self):
        calls = []
        saved = server._flag_byzantine
        server._flag_byzantine = lambda *a, **k: calls.append(a)
        try:
            server._flag_byzantine_from_delta(
                self._delta({"nodeKey": "k", "nodeIp": "1.2.3.4"}), "cross-check outlier")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "k")
        finally:
            server._flag_byzantine = saved


if __name__ == "__main__":
    unittest.main()
