"""Tests for the residual (two bit-plane) NGD1R/NGD3 delta codec.

The plain codec quantises a 1024-value block to one fp16 mean-absolute scale
plus sign bits, so the reconstruction error is the magnitude error. The
residual codec adds a second sign plane with its own scale, taken from
``x - x1``. These tests pin the wire format, the error reduction and the
guard rails that stop a malformed frame from expanding without bound.
"""

import struct
import zlib

import numpy as np
import pytest

from orchestrator.compression import (
    BLOCK,
    FRAME_MAGIC,
    FRAME_MAGIC_RESIDUAL,
    MAGIC,
    MAGIC_RESIDUAL,
    compress_delta,
    compression_ratio,
    container_is_residual,
    decompress_delta,
    iter_frames,
    pack_frames,
    unpack_frames,
)


def _skewed(count, seed=0):
    """Small signed values resembling a training delta (not symmetric noise)."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.01, size=count).astype("float32")
    values[rng.random(count) < 0.05] *= 6.0
    return values


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def test_residual_round_trip_for_uniform_magnitude():
    values = np.full(2048, 0.25, dtype="float32")
    values[::2] *= -1
    back = decompress_delta(compress_delta(values, residual=True))
    assert np.array_equal(back, values)


def test_residual_round_trip_for_zero():
    values = np.zeros(4096, dtype="float32")
    assert np.array_equal(decompress_delta(compress_delta(values, residual=True)), values)


def test_residual_empty_delta_round_trips():
    back = decompress_delta(compress_delta(np.zeros(0, dtype="float32"), residual=True))
    assert back.size == 0


def test_residual_cuts_reconstruction_error():
    # The second plane is a variance reduction, not an error bound: an element
    # whose residual is smaller than the residual scale overshoots. What it
    # guarantees is a lower *mean* absolute error, which is what a coordinator
    # averages over 658M parameters.
    values = _skewed(64 * BLOCK)
    plain = decompress_delta(compress_delta(values))
    residual = decompress_delta(compress_delta(values, residual=True))
    plain_mae = _mae(plain, values)
    residual_mae = _mae(residual, values)
    assert residual_mae < plain_mae / 1.4, (plain_mae, residual_mae)
    plain_rel = float(np.linalg.norm(plain - values) / np.linalg.norm(values))
    residual_rel = float(np.linalg.norm(residual - values) / np.linalg.norm(values))
    assert residual_rel < plain_rel


def test_residual_costs_about_twice_the_bytes():
    values = _skewed(64 * BLOCK)
    plain = len(compress_delta(values))
    residual = len(compress_delta(values, residual=True))
    assert 1.7 < residual / plain < 2.3, (plain, residual)


def test_residual_partial_tail_block_uses_its_real_length():
    values = np.array([0.5], dtype="float32")
    assert decompress_delta(compress_delta(values, residual=True))[0] == pytest.approx(0.5)


def test_frame_magics_select_the_layout():
    values = _skewed(BLOCK)
    assert decompress_delta(compress_delta(values)).size == BLOCK
    for residual, magic in ((False, MAGIC), (True, MAGIC_RESIDUAL)):
        raw = zlib.decompress(compress_delta(values, residual=residual))
        assert raw[: len(magic)] == magic
        # The stored count and block size are the same in both layouts.
        assert struct.unpack("<QI", raw[len(magic): len(magic) + 12]) == (BLOCK, BLOCK)


def test_container_magic_follows_the_codec():
    blocks = [_skewed(BLOCK, seed=1), _skewed(2 * BLOCK, seed=2)]
    assert pack_frames(blocks)[: len(FRAME_MAGIC)] == FRAME_MAGIC
    blob = pack_frames(blocks, residual=True)
    assert blob[: len(FRAME_MAGIC_RESIDUAL)] == FRAME_MAGIC_RESIDUAL
    total, count = struct.unpack("<QI", blob[len(FRAME_MAGIC_RESIDUAL): len(FRAME_MAGIC_RESIDUAL) + 12])
    assert (total, count) == (3 * BLOCK, 2)


def test_container_is_residual():
    blocks = [_skewed(BLOCK)]
    assert container_is_residual(pack_frames(blocks)) is False
    assert container_is_residual(pack_frames(blocks, residual=True)) is True


def test_container_is_residual_rejects_garbage():
    with pytest.raises(ValueError):
        container_is_residual(b"XXXX" + b"\x00" * 12)


def test_iter_frames_round_trips_the_residual_container():
    parts = [_skewed(BLOCK, seed=3), _skewed(BLOCK + 7, seed=4)]
    blob = pack_frames(parts, residual=True)
    decoded = list(iter_frames(blob, expect_residual=True))
    assert [int(p.size) for p in decoded] == [BLOCK, BLOCK + 7]
    joined = np.concatenate(parts)
    plain = np.concatenate(list(iter_frames(pack_frames(parts))))
    assert _mae(np.concatenate(decoded), joined) < _mae(plain, joined)


def test_iter_frames_rejects_a_container_that_contradicts_the_codec():
    blocks = [_skewed(BLOCK)]
    with pytest.raises(ValueError, match="NGD3"):
        list(iter_frames(pack_frames(blocks), expect_residual=True))
    with pytest.raises(ValueError, match="NGD2"):
        list(iter_frames(pack_frames(blocks, residual=True), expect_residual=False))


def test_unpack_frames_residual_matches_the_input():
    values = _skewed(5 * BLOCK)
    decoded = unpack_frames(pack_frames([values], residual=True))
    assert decoded.shape == values.shape
    plain = unpack_frames(pack_frames([values]))
    assert _mae(decoded, values) < _mae(plain, values)


def test_truncated_residual_container_is_rejected():
    blob = pack_frames([_skewed(BLOCK)], residual=True)
    with pytest.raises(ValueError):
        list(iter_frames(blob[:-8], expect_residual=True))


def test_trailing_bytes_after_a_residual_frame_are_rejected():
    blob = pack_frames([_skewed(BLOCK)], residual=True) + b"\x00\x00"
    with pytest.raises(ValueError):
        list(iter_frames(blob, expect_residual=True))


def test_residual_frame_total_mismatch_is_rejected():
    blob = pack_frames([_skewed(BLOCK)], residual=True)
    header = len(FRAME_MAGIC_RESIDUAL)
    total, count = struct.unpack("<QI", blob[header: header + 12])
    tampered = blob[:header] + struct.pack("<QI", total + BLOCK, count) + blob[header + 12:]
    with pytest.raises(ValueError):
        list(iter_frames(tampered, expect_residual=True))


def test_residual_respects_the_total_limit():
    blob = pack_frames([_skewed(BLOCK), _skewed(BLOCK)], residual=True)
    with pytest.raises(ValueError):
        list(iter_frames(blob, max_total_params=BLOCK, expect_residual=True))


def test_malformed_residual_stream_is_a_value_error():
    raw = bytearray(zlib.decompress(compress_delta(_skewed(BLOCK), residual=True)))
    raw[20:24] = b"\xff\xff\xff\xff"
    with pytest.raises(ValueError):
        decompress_delta(zlib.compress(bytes(raw), 6))


def test_residual_declared_size_guard_stops_oversized_expansion():
    values = _skewed(4 * BLOCK)
    blob = compress_delta(values, residual=True)
    plain_size = len(compress_delta(values))
    with pytest.raises(ValueError):
        decompress_delta(blob, max_raw_bytes=plain_size)


def test_residual_costs_less_than_dense_bf16():
    values = _skewed(64 * BLOCK)
    compressed = len(compress_delta(values, residual=True))
    assert compression_ratio(values.nbytes, b"\x00" * compressed) > 3.0


def test_plain_frames_inside_a_residual_container_still_decode():
    # The per-frame bound uses the larger residual size, so a peer that emits a
    # single-plane frame under the NGD3 magic is decodable rather than corrupt.
    frame = compress_delta(_skewed(BLOCK))
    blob = FRAME_MAGIC_RESIDUAL + struct.pack("<QI", BLOCK, 1) + struct.pack("<I", len(frame)) + frame
    decoded = list(iter_frames(blob, expect_residual=True))
    assert int(decoded[0].size) == BLOCK
