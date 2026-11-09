"""Activation fingerprints for step receipts.

A 32-token block is 258 bytes: a 2-byte token count plus one 64-bit SimHash per
token (8 bytes each). The projection is a fixed random matrix, so a validator
with the same model can recompute the block from activations.
"""
from __future__ import annotations

import hashlib
import numpy as np

DIM = 1536
BITS = 64
BLOCK = 32
SEED = 0x0C0FFEE

_MATRIX = None


def projection(dim: int = DIM, bits: int = BITS, seed: int = SEED) -> np.ndarray:
    global _MATRIX
    if _MATRIX is None or _MATRIX.shape != (dim, bits):
        rng = np.random.default_rng(seed)
        _MATRIX = rng.standard_normal((dim, bits), dtype=np.float32)
    return _MATRIX


def simhash64(vector, matrix: np.ndarray | None = None) -> int:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    m = projection(v.shape[0]) if matrix is None else matrix
    bits = (v @ m) > 0
    out = 0
    for i, b in enumerate(bits):
        if b: out |= (1 << i)
    return out


def block_fingerprint(hidden) -> bytes:
    """``hidden`` is [tokens, dim]; returns 2 + 8*tokens bytes (258 for 32)."""
    arr = np.asarray(hidden, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("hidden must be [tokens, dim]")
    count = min(BLOCK, arr.shape[0])
    out = bytearray(int(count).to_bytes(2, "little"))
    for i in range(count):
        out.extend(simhash64(arr[i]).to_bytes(8, "little"))
    return bytes(out)


def fingerprint_root(blocks) -> str:
    h = hashlib.sha256()
    for block in blocks:
        h.update(block)
    return h.hexdigest()


def compare_blocks(expected: bytes, actual: bytes) -> int:
    """Number of differing 64-bit tokens (0 means identical)."""
    if len(expected) != len(actual) or len(expected) < 2:
        return max(len(expected), len(actual))
    n = int.from_bytes(expected[:2], "little")
    diff = 0
    for i in range(n):
        a = expected[2 + 8 * i:10 + 8 * i]
        b = actual[2 + 8 * i:10 + 8 * i]
        if a != b: diff += 1
    return diff


def verify_root(claimed_root: str, blocks) -> bool:
    return hmac_compare(claimed_root, fingerprint_root(blocks))


def hmac_compare(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(str(a).lower(), str(b).lower())


__all__ = ["BLOCK", "DIM", "block_fingerprint", "compare_blocks",
           "fingerprint_root", "projection", "simhash64", "verify_root"]
