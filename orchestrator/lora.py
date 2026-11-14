"""LoRA adapter wire format and train-mode identity for the coordinator.

The wire payload is deliberately independent of the client's model code: a
client sends one entry per adapter with the target weight's name, the offset of
that weight in the flat base-parameter stream, its shape, and the two fp16
factors.  The coordinator expands those adapters into the dense base delta the
existing cross-check and aggregation pipeline already consumes, so an adapter
upload (~17 MB at rank 16 on the 658M model) rides the same path as a 1.3 GB
full-parameter delta.

The training mode is part of a pool's identity rather than a hint: a low-rank
direction and a full-parameter update are not comparable in the cross-check, so
``signatures_compatible`` keeps them apart and ``check_submission`` rejects a
submission whose declared mode, rank, or trained tensors disagree with its task.

Nothing here imports torch or FastAPI; the module is pure numpy + struct so it
can be exercised without the HTTP layer.
"""
from __future__ import annotations

import struct

import numpy as np

ALPHA = 32.0
WIRE_MAGIC = b"NGL1"
WIRE_HEADER = struct.Struct("<HIIIH")  # name_len, offset, rows, cols, rank
MAX_ADAPTERS = 4096


# ============ TRAIN MODE ============

def train_mode_of(value) -> str:
    """Normalize a declared training mode.

    Anything unknown (including a missing header from a client that predates
    LoRA) means ``dense``, so old clients keep matching old pools."""
    return "lora" if str(value or "").strip().lower() in ("lora", "adapter", "adapters") else "dense"


def rank_of(value) -> int:
    """Non-negative integer rank; 0 when absent or malformed."""
    try:
        rank = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return rank if rank > 0 else 0


def trained_params(mode: str, *, has_engram: bool) -> list[str]:
    """The tensor groups a client in this mode is allowed to claim."""
    base = ["adapters"] if train_mode_of(mode) == "lora" else ["base"]
    return base + (["engram"] if has_engram else [])


def signature_mode(sig) -> tuple[str, int]:
    """``(mode, rank)`` from a client/pool hardware signature.

    Mode and rank were appended as positions 7 and 8, so a signature persisted
    before LoRA existed (7 elements) still reads as ``dense``/0 and keeps
    matching other legacy signatures."""
    mode = train_mode_of(sig[7] if len(sig) > 7 else "")
    rank = rank_of(sig[8]) if mode == "lora" and len(sig) > 8 else 0
    return mode, rank


def signatures_compatible(client_sig, pool_sig) -> bool:
    """Pool isolation: equal train mode and equal rank (dense is rank 0)."""
    return signature_mode(client_sig) == signature_mode(pool_sig)


def check_submission(task_meta: dict, body: dict) -> tuple[bool, int]:
    """Validate a submission's declared train mode against its task.

    Returns ``(is_lora, rank)``.  Raises ``ValueError`` with a machine-readable
    reason when the submission could not be aggregated with the task's pool:
    a dense delta must never enter a LoRA pool (or the reverse), the declared
    rank must equal the pool's, and the claimed trained tensors must be a
    subset of what the task allows.  A dense client that sends none of the new
    fields validates against a dense task, which is the pre-LoRA wire shape.
    """
    declared_rank = rank_of(body.get("loraRank"))
    is_lora = bool(body.get("isLoRA")) or declared_rank > 0
    expected_mode = train_mode_of(task_meta.get("trainMode"))
    if is_lora != (expected_mode == "lora"):
        raise ValueError("train_mode_mismatch")
    if not is_lora:
        return False, 0
    expected_rank = rank_of(task_meta.get("loraRank"))
    if expected_rank and declared_rank != expected_rank:
        raise ValueError("lora_rank_mismatch")
    allowed = set(task_meta.get("trainedParams") or [])
    claimed = set(body.get("trainedParams") or [])
    if claimed and not claimed <= allowed:
        raise ValueError("trained_params_mismatch")
    return True, declared_rank or expected_rank


# ============ WIRE FORMAT ============

def _adapter_patch(item: dict, model_size: int, alpha: float) -> tuple[int, np.ndarray]:
    """Flattened ``b @ a * (alpha / rank)`` for one adapter, plus its offset."""
    a = np.asarray(item["a"], dtype=np.float32)
    b = np.asarray(item["b"], dtype=np.float32)
    name = item.get("name", "?")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(f"{name}: incompatible adapter shapes")
    rank, cols = int(a.shape[0]), int(a.shape[1])
    rows = int(b.shape[0])
    if rank <= 0 or rows <= 0 or cols <= 0:
        raise ValueError(f"{name}: empty adapter factors")
    offset = int(item["offset"])
    size = rows * cols
    if offset < 0 or offset + size > model_size:
        raise ValueError(f"{name}: adapter span outside the model")
    return offset, (b @ a * (alpha / rank)).reshape(-1)


def encode_payload(adapters: list[dict], *, alpha: float = ALPHA) -> bytes:
    """Encode ``[{name, offset, a, b}, ...]`` into the compact upload payload.

    The coordinator only decodes this format; the encoder exists so clients and
    tests have a reference implementation to round-trip against.  Offsets and
    shapes are validated here so a malformed client cannot claim a bad span.
    """
    parts = [WIRE_MAGIC, struct.pack("<I", len(adapters))]
    for item in adapters:
        name = str(item["name"]).encode("utf-8")
        if not name or len(name) > 65535:
            raise ValueError("adapter name length out of range")
        a = np.asarray(item["a"], dtype=np.float16)
        b = np.asarray(item["b"], dtype=np.float16)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"{item['name']}: incompatible adapter shapes")
        rows, cols, rank = int(b.shape[0]), int(a.shape[1]), int(a.shape[0])
        offset = int(item["offset"])
        if offset < 0 or rows <= 0 or cols <= 0 or rank <= 0:
            raise ValueError(f"{item['name']}: invalid shape or offset")
        parts.append(WIRE_HEADER.pack(len(name), offset, rows, cols, rank))
        parts.append(name)
        parts.append(a.tobytes(order="C"))
        parts.append(b.tobytes(order="C"))
    return b"".join(parts)


def decode_payload(blob: bytes, *, model_size: int) -> list[dict]:
    """Parse and bounds-check a wire payload.  Does not allocate the dense delta."""
    if not blob.startswith(WIRE_MAGIC):
        raise ValueError("bad LoRA payload magic")
    pos = len(WIRE_MAGIC)
    if len(blob) < pos + 4:
        raise ValueError("truncated LoRA payload")
    count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    if count <= 0 or count > MAX_ADAPTERS:
        raise ValueError("LoRA adapter count out of range")
    adapters = []
    spans = []
    for _ in range(count):
        if len(blob) < pos + WIRE_HEADER.size:
            raise ValueError("truncated LoRA adapter header")
        nlen, offset, rows, cols, rank = WIRE_HEADER.unpack_from(blob, pos)
        pos += WIRE_HEADER.size
        name = blob[pos:pos + nlen].decode("utf-8", "strict")
        pos += nlen
        a_n, b_n = rank * cols, rows * rank
        need = (a_n + b_n) * 2
        if len(blob) < pos + need:
            raise ValueError(f"{name}: truncated adapter factors")
        a = np.frombuffer(blob, dtype=np.float16, count=a_n, offset=pos).reshape(rank, cols).astype(np.float32)
        pos += a_n * 2
        b = np.frombuffer(blob, dtype=np.float16, count=b_n, offset=pos).reshape(rows, rank).astype(np.float32)
        pos += b_n * 2
        if rank <= 0 or rows <= 0 or cols <= 0 or offset < 0:
            raise ValueError(f"{name}: invalid shape or offset")
        if offset + rows * cols > int(model_size):
            raise ValueError(f"{name}: adapter exceeds model size")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"{name}: NaN/Inf adapter factors")
        spans.append((offset, offset + rows * cols))
        adapters.append({"name": name, "offset": offset, "a": a, "b": b})
    if pos != len(blob):
        raise ValueError("trailing bytes after LoRA payload")
    spans.sort()
    for (_a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError("overlapping LoRA adapter spans")
    return adapters


def iter_dense_chunks(adapters: list[dict], model_size: int, *, alpha: float = ALPHA,
                      chunk: int = 10_000_000):
    """Yield the expanded delta as fp32 chunks of at most ``chunk`` parameters.

    The coordinator writes these straight to its temp file, so expanding a
    full-model adapter set never needs the multi-gigabyte dense array."""
    model_size = int(model_size)
    if model_size <= 0:
        raise ValueError("model_size must be positive")
    chunk = max(1, int(chunk))
    patches = []
    for item in adapters:
        offset, patch = _adapter_patch(item, model_size, alpha)
        patches.append((offset, offset + patch.size, patch))
    patches.sort(key=lambda p: p[0])
    for start in range(0, model_size, chunk):
        end = min(model_size, start + chunk)
        out = np.zeros(end - start, dtype=np.float32)
        for p_start, p_end, patch in patches:
            if p_end <= start or p_start >= end:
                continue
            lo, hi = max(p_start, start), min(p_end, end)
            out[lo - start:hi - start] += patch[lo - p_start:hi - p_start]
        yield out


def expand_to_dense(adapters: list[dict], model_size: int, *, alpha: float = ALPHA) -> np.ndarray:
    """Expand adapters into one dense fp32 delta (tests, tools, small models)."""
    out = np.zeros(int(model_size), dtype=np.float32)
    for item in adapters:
        offset, patch = _adapter_patch(item, int(model_size), alpha)
        out[offset:offset + patch.size] += patch
    return out


__all__ = ["ALPHA", "MAX_ADAPTERS", "WIRE_HEADER", "WIRE_MAGIC", "check_submission",
           "decode_payload", "encode_payload", "expand_to_dense", "iter_dense_chunks",
           "rank_of", "signature_mode", "signatures_compatible", "train_mode_of",
           "trained_params"]
