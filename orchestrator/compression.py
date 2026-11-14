"""1-bit delta compression with per-block scales plus zlib framing.

A block of 1024 params becomes one fp16 scale plus 1024 sign bits (128 bytes).
For 657,787,392 params that is ~80.3 MB before entropy coding, under the
200 MB target. The codec is deliberately simple so a client and a validator
can reproduce it byte-for-byte.

The ``NGD2`` container wraps independent ``NGD1`` frames. It is opt-in on the
wire (client metadata ``deltaCodec == "ngd1"``); the server uses
:func:`iter_frames` to rebuild the dense bf16 delta stream.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

BLOCK = 1024
MAGIC = b"NGD1"
FRAME_MAGIC = b"NGD2"

# ``MAGIC`` (4) + count/block header (12).
_DELTA_HEADER = len(MAGIC) + 12


def _expected_raw_size(n: int, block: int) -> int:
    """Exact uncompressed NGD1 payload size for ``n`` values and ``block``."""
    nb = (n + block - 1) // block
    return _DELTA_HEADER + 2 * nb + (nb * block + 7) // 8


def compress_delta(values, *, block: int = BLOCK, level: int = 6) -> bytes:
    if block <= 0:
        raise ValueError("block must be positive")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = arr.size
    nb = (n + block - 1) // block
    padded = np.zeros(nb * block, dtype=np.float32)
    padded[:n] = arr
    matrix = padded.reshape(nb, block)
    if nb:
        # The final block is zero-padded; divide by its real length so a
        # partial block keeps the right mean-absolute scale.
        counts = np.full(nb, block, dtype=np.float64)
        counts[-1] = n - (nb - 1) * block
        scales = (np.abs(matrix).sum(axis=1) / counts).astype(np.float16)
    else:
        scales = np.zeros(0, dtype=np.float16)
    bits = (matrix >= 0).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    raw = MAGIC + struct.pack("<QI", n, block) + scales.tobytes() + packed
    return zlib.compress(raw, level)


def decompress_delta(blob: bytes, *, max_raw_bytes: int | None = None) -> np.ndarray:
    """Decode one NGD1 frame.

    ``max_raw_bytes`` bounds the declared uncompressed size before the rest of
    the stream is inflated, so a small compressed frame cannot expand without
    limit. Malformed, truncated, oversized, or trailing data raises ValueError.
    """
    try:
        inflater = zlib.decompressobj()
        head = inflater.decompress(blob, _DELTA_HEADER)
        if len(head) < _DELTA_HEADER:
            raise ValueError("truncated delta header")
        if not head.startswith(MAGIC):
            raise ValueError("bad delta magic")
        n, block = struct.unpack_from("<QI", head, 4)
        if block <= 0:
            raise ValueError(f"invalid block size {block}")
        expected = _expected_raw_size(n, block)
        if max_raw_bytes is not None and expected > max_raw_bytes:
            raise ValueError(f"delta too large: {expected} > {max_raw_bytes} bytes")
        raw = head
        if len(raw) < expected:
            raw += inflater.decompress(inflater.unconsumed_tail, expected - len(raw) + 1)
        if len(raw) != expected:
            raise ValueError("truncated or oversized delta")
        if inflater.unconsumed_tail:
            raise ValueError("oversized delta")
        if inflater.unused_data:
            raise ValueError("trailing bytes in delta")
        if not inflater.eof:
            raise ValueError("truncated delta stream")
    except (zlib.error, struct.error) as exc:
        raise ValueError(f"bad delta stream: {exc}") from exc

    nb = (n + block - 1) // block
    pos = _DELTA_HEADER
    scales = np.frombuffer(raw, dtype=np.float16, count=nb, offset=pos); pos += nb * 2
    packed = np.frombuffer(raw, dtype=np.uint8, offset=pos)
    bits = np.unpackbits(packed, bitorder="little")[:nb * block]
    out = np.where(bits == 1, scales.repeat(block), -scales.repeat(block))[:n]
    return out.astype(np.float32)


def pack_frames(blocks) -> bytes:
    """Frame several independent 1-bit blocks under one container."""
    blocks = list(blocks)
    frames = [compress_delta(block) for block in blocks]
    total = sum(int(np.asarray(block).size) for block in blocks)
    parts = [FRAME_MAGIC, struct.pack("<QI", total, len(frames))]
    for frame in frames:
        parts.append(struct.pack("<I", len(frame)))
        parts.append(frame)
    return b"".join(parts)


def iter_frames(blob: bytes, *, max_total_params: int | None = None):
    """Yield decoded blocks; raises on a truncated or trailing container.

    ``max_total_params`` caps both the declared total and each frame's declared
    uncompressed size. The NGD2 wire format uses :data:`BLOCK`-sized blocks.
    """
    if not blob.startswith(FRAME_MAGIC):
        raise ValueError("bad frame magic")
    pos = len(FRAME_MAGIC)
    if len(blob) < pos + 12:
        raise ValueError("truncated frame header")
    total, count = struct.unpack_from("<QI", blob, pos); pos += 12
    if max_total_params is not None and total > max_total_params:
        raise ValueError(f"frame total {total} exceeds limit {max_total_params}")
    # Every frame needs at least its 4-byte length prefix.
    if count > (len(blob) - pos) // 4:
        raise ValueError("truncated frame header")
    seen = 0
    for _ in range(count):
        if len(blob) < pos + 4:
            raise ValueError("truncated frame header")
        (size,) = struct.unpack_from("<I", blob, pos); pos += 4
        if size <= 0:
            raise ValueError("empty frame")
        frame = blob[pos:pos + size]
        if len(frame) != size:
            raise ValueError("truncated frame")
        limit = None
        if max_total_params is not None:
            limit = _expected_raw_size(max(0, max_total_params - seen), BLOCK)
        block = decompress_delta(frame, max_raw_bytes=limit)
        seen += int(block.size)
        if max_total_params is not None and seen > max_total_params:
            raise ValueError(f"frame total exceeds limit {max_total_params}")
        yield block
        pos += size
    if pos != len(blob):
        raise ValueError("trailing bytes after frames")
    if seen != total:
        raise ValueError(f"frame total mismatch: {seen} != {total}")


def unpack_frames(blob: bytes, *, max_total_params: int | None = None) -> np.ndarray:
    parts = list(iter_frames(blob, max_total_params=max_total_params))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def compression_ratio(original_bytes: int, compressed: bytes) -> float:
    return float(original_bytes) / max(1, len(compressed))


__all__ = ["BLOCK", "FRAME_MAGIC", "MAGIC", "compress_delta", "compression_ratio",
           "decompress_delta", "iter_frames", "pack_frames", "unpack_frames"]
