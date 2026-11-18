"""1-bit delta compression with per-block scales plus zlib framing.

A block of 1024 params becomes one fp16 scale plus 1024 sign bits (128 bytes).
For 657,787,392 params that is ~80.3 MB before entropy coding, under the
200 MB target. The codec is deliberately simple so a client and a validator
can reproduce it byte-for-byte.

Two frame variants share the layout:

``NGD1``
    one bit-plane: ``magic | n | block | scale | bits``.
``NGD1R`` (residual)
    two bit-planes: ``magic | n | block | scale1 | scale2 | bits2 | bits1``.
    ``bits1``/``scale1`` are the plain 1-bit approximation and
    ``bits2``/``scale2`` encode the sign of the leftover error, so the
    reconstruction is ``x1 + x2``. It costs roughly twice the bytes of
    ``NGD1`` for about a third of the reconstruction error.

The ``NGD2``/``NGD3`` containers wrap independent ``NGD1``/``NGD1R`` frames
respectively (``NGD3`` may still carry a plain frame). Both are opt-in on the
wire (client metadata ``deltaCodec`` is ``"ngd1"`` or ``"ngd1r"``); the server
uses :func:`iter_frames` to rebuild the dense bf16 delta stream.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

BLOCK = 1024
MAGIC = b"NGD1"
MAGIC_RESIDUAL = b"NGD1R"
FRAME_MAGIC = b"NGD2"
FRAME_MAGIC_RESIDUAL = b"NGD3"

# ``MAGIC`` (4 or 5) + count/block header (12).
_DELTA_HEADER = len(MAGIC) + 12
_DELTA_HEADER_RESIDUAL = len(MAGIC_RESIDUAL) + 12


def _plane_bytes(nb: int, block: int) -> int:
    return (nb * block + 7) // 8


def _expected_raw_size(n: int, block: int, residual: bool = False) -> int:
    """Exact uncompressed frame size for ``n`` values and ``block``."""
    nb = (n + block - 1) // block
    if residual:
        return _DELTA_HEADER_RESIDUAL + 4 * nb + 2 * _plane_bytes(nb, block)
    return _DELTA_HEADER + 2 * nb + _plane_bytes(nb, block)


def _block_stats(matrix: np.ndarray, counts: np.ndarray):
    """Per-block mean-absolute scale plus the sign bits it quantises to."""
    scales = (np.abs(matrix).sum(axis=1) / counts).astype(np.float16)
    bits = (matrix >= 0).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    return scales, bits, packed


def compress_delta(values, *, block: int = BLOCK, level: int = 6,
                   residual: bool = False) -> bytes:
    """Encode one frame; ``residual=True`` adds the second bit-plane."""
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
    else:
        counts = np.zeros(0, dtype=np.float64)
    scales, bits, packed = _block_stats(matrix, counts) if nb else (
        np.zeros(0, dtype=np.float16), np.zeros((0, block), dtype=np.uint8), b"")
    if not residual:
        raw = MAGIC + struct.pack("<QI", n, block) + scales.tobytes() + packed
        return zlib.compress(raw, level)
    if nb:
        recon = np.repeat(scales.astype(np.float32)[:, None], block, axis=1)
        recon *= np.where(bits == 1, np.float32(1.0), np.float32(-1.0))
        leftover = matrix - recon
        del recon
        if n < nb * block:
            # The zero padding reconstructs to +/-scales, not to zero, so it
            # would otherwise swamp the residual scale of the final block.
            leftover = leftover.reshape(-1)
            leftover[n:] = 0.0
            leftover = leftover.reshape(nb, block)
        rscales, _, rpacked = _block_stats(leftover, counts)
        del leftover
    else:
        rscales, rpacked = np.zeros(0, dtype=np.float16), b""
    raw = (MAGIC_RESIDUAL + struct.pack("<QI", n, block)
           + scales.tobytes() + rscales.tobytes() + rpacked + packed)
    return zlib.compress(raw, level)


def decompress_delta(blob: bytes, *, max_raw_bytes: int | None = None) -> np.ndarray:
    """Decode one frame, plain or residual; the magic selects the layout.

    ``max_raw_bytes`` bounds the declared uncompressed size before the rest of
    the stream is inflated, so a small compressed frame cannot expand without
    limit. Malformed, truncated, oversized, or trailing data raises ValueError.
    """
    try:
        inflater = zlib.decompressobj()
        head = inflater.decompress(blob, _DELTA_HEADER_RESIDUAL)
        if len(head) >= len(MAGIC_RESIDUAL) and head.startswith(MAGIC_RESIDUAL):
            residual, magic_len = True, len(MAGIC_RESIDUAL)
        elif len(head) >= len(MAGIC) and head.startswith(MAGIC):
            residual, magic_len = False, len(MAGIC)
        elif len(head) < len(MAGIC):
            raise ValueError("truncated delta header")
        else:
            raise ValueError("bad delta magic")
        header = magic_len + 12
        if len(head) < header:
            raise ValueError("truncated delta header")
        n, block = struct.unpack_from("<QI", head, magic_len)
        if block <= 0:
            raise ValueError(f"invalid block size {block}")
        expected = _expected_raw_size(n, block, residual)
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
    pos = header
    scales = np.frombuffer(raw, dtype=np.float16, count=nb, offset=pos); pos += nb * 2
    if not np.all(np.isfinite(scales)):
        raise ValueError("non-finite delta scale")
    if not residual:
        packed = np.frombuffer(raw, dtype=np.uint8, offset=pos)
        bits = np.unpackbits(packed, bitorder="little")[:nb * block]
        out = np.where(bits == 1, scales.repeat(block), -scales.repeat(block))[:n]
        return out.astype(np.float32)

    plane = _plane_bytes(nb, block)
    rscales = np.frombuffer(raw, dtype=np.float16, count=nb, offset=pos); pos += nb * 2
    if not np.all(np.isfinite(rscales)):
        raise ValueError("non-finite residual scale")
    rbits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, count=plane, offset=pos),
                          bitorder="little")[:nb * block]; pos += plane
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, count=plane, offset=pos),
                         bitorder="little")[:nb * block]
    first = np.where(bits == 1, scales.repeat(block), -scales.repeat(block))
    second = np.where(rbits == 1, rscales.repeat(block), -rscales.repeat(block))
    return (first + second)[:n].astype(np.float32)


def pack_frames(blocks, *, residual: bool = False) -> bytes:
    """Frame several independent blocks under one container."""
    blocks = list(blocks)
    frames = [compress_delta(block, residual=residual) for block in blocks]
    total = sum(int(np.asarray(block).size) for block in blocks)
    parts = [FRAME_MAGIC_RESIDUAL if residual else FRAME_MAGIC,
             struct.pack("<QI", total, len(frames))]
    for frame in frames:
        parts.append(struct.pack("<I", len(frame)))
        parts.append(frame)
    return b"".join(parts)


def container_is_residual(blob: bytes) -> bool:
    """True for an ``NGD3`` container, False for ``NGD2``."""
    if blob.startswith(FRAME_MAGIC_RESIDUAL):
        return True
    if blob.startswith(FRAME_MAGIC):
        return False
    raise ValueError("bad frame magic")


def iter_frames(blob: bytes, *, max_total_params: int | None = None,
                expect_residual: bool | None = None):
    """Yield decoded blocks; raises on a truncated or trailing container.

    ``max_total_params`` caps both the declared total and each frame's declared
    uncompressed size. ``expect_residual`` rejects a container whose
    ``NGD2``/``NGD3`` magic disagrees with the negotiated ``deltaCodec``.
    """
    residual = container_is_residual(blob)
    if expect_residual is not None and residual != expect_residual:
        raise ValueError(
            f"container is {'NGD3' if residual else 'NGD2'}, "
            f"expected {'NGD3' if expect_residual else 'NGD2'}")
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
            # Use the larger (residual) bound so a plain frame inside an NGD3
            # container is still accepted; a residual frame inside NGD2 exceeds
            # the plain bound only when it actually grows past it.
            limit = _expected_raw_size(max(0, max_total_params - seen), BLOCK, True)
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


__all__ = ["BLOCK", "FRAME_MAGIC", "FRAME_MAGIC_RESIDUAL", "MAGIC", "MAGIC_RESIDUAL",
           "compress_delta", "compression_ratio", "container_is_residual",
           "decompress_delta", "iter_frames", "pack_frames", "unpack_frames"]
