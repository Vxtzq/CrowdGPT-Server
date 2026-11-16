"""Tests for the opt-in NGD1/NGD2 delta codec used by the upload path."""
import struct
import zlib

import numpy as np
import pytest

from orchestrator.compression import (
    BLOCK,
    FRAME_MAGIC,
    MAGIC,
    compress_delta,
    compression_ratio,
    decompress_delta,
    iter_frames,
    pack_frames,
    unpack_frames,
)


def test_round_trip_for_uniform_magnitude():
    values = np.full(2048, 0.25, dtype="float32")
    values[::2] *= -1
    back = decompress_delta(compress_delta(values))
    assert np.array_equal(back, values)


def test_exact_round_trip_for_zero():
    values = np.zeros(2048, dtype="float32")
    assert np.array_equal(decompress_delta(compress_delta(values)), values)


def test_empty_delta_round_trips():
    back = decompress_delta(compress_delta(np.zeros(0, dtype="float32")))
    assert back.size == 0


def test_partial_block_uses_its_real_length():
    # Regression for the zero-padded tail block: a single trailing value must
    # keep scale 1.0, not 1/1024.
    values = np.zeros(BLOCK + 1, dtype="float32")
    values[-1] = 1.0
    back = decompress_delta(compress_delta(values))
    assert back[-1] == pytest.approx(1.0, rel=1e-2)


def test_size_target_for_model_scale():
    values = np.random.default_rng(7).standard_normal(4_000_000).astype("float32") * 0.01
    blob = compress_delta(values)
    assert compression_ratio(values.nbytes, blob) > 10.0


def test_framed_container_round_trip_and_errors():
    blocks = [np.full(64, 0.25, dtype=np.float32), np.zeros(32, dtype=np.float32)]
    blob = pack_frames(blocks)
    assert blob.startswith(FRAME_MAGIC)
    back = unpack_frames(blob)
    assert back.size == 96
    assert np.allclose(back[:64], 0.25)
    assert np.all(back[64:] == 0)
    with pytest.raises(ValueError):
        list(iter_frames(blob[:-5]))
    with pytest.raises(ValueError):
        list(iter_frames(b"bad"))


def test_trailing_bytes_are_rejected():
    blob = pack_frames([np.zeros(8, dtype=np.float32)]) + b"x"
    with pytest.raises(ValueError, match="trailing"):
        list(iter_frames(blob))


def test_frame_total_mismatch_is_rejected():
    body = pack_frames([np.zeros(8, dtype=np.float32)])
    # Rewrite the declared total (first 8 bytes after the magic) to 9.
    forged = body[:4] + struct.pack("<Q", 9) + body[12:]
    with pytest.raises(ValueError, match="mismatch"):
        list(iter_frames(forged))


def test_absurd_frame_count_is_rejected_without_looping():
    blob = FRAME_MAGIC + struct.pack("<QI", 0, 2 ** 32 - 1)
    with pytest.raises(ValueError, match="truncated"):
        list(iter_frames(blob))


def test_total_limit_rejects_before_decoding():
    blob = pack_frames([np.zeros(2048, dtype=np.float32)])
    with pytest.raises(ValueError, match="exceeds limit"):
        list(iter_frames(blob, max_total_params=1024))


def test_declared_size_guard_stops_oversized_expansion():
    # A valid header that claims one value, followed by far more inflated data.
    raw = MAGIC + struct.pack("<QI", 1, BLOCK) + b"\x00" * 2_000_000
    with pytest.raises(ValueError):
        decompress_delta(zlib.compress(raw))


def test_malformed_compressed_stream_is_a_value_error():
    with pytest.raises(ValueError):
        decompress_delta(b"not zlib")
    with pytest.raises(ValueError):
        decompress_delta(zlib.compress(b"NOPE" + struct.pack("<QI", 0, BLOCK)))


def test_client_emitted_frame_layout_is_decoded():
    # Independent encoding of the layout the v3.3 client's _ngd1_compress
    # writes, so the server decoder and the wire contract cannot drift.
    values = np.random.default_rng(11).standard_normal(1500).astype("float32") * 0.02
    n = int(values.size)
    nb = max(1, (n + 1023) // 1024)
    padded = np.zeros(nb * 1024, dtype=np.float32)
    padded[:n] = values
    matrix = padded.reshape(nb, 1024)
    counts = np.full(nb, 1024, dtype=np.float64)
    counts[-1] = n - (nb - 1) * 1024
    scales = (np.abs(matrix).sum(axis=1) / counts).astype(np.float16)
    bits = (matrix >= 0).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    frame = zlib.compress(b"NGD1" + struct.pack("<QI", n, 1024) + scales.tobytes() + packed, 6)

    back = decompress_delta(frame)
    assert back.size == n
    assert np.allclose(np.sign(back), np.sign(values))
    assert np.mean(np.abs(back - values)) < 0.01
