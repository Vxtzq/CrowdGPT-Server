"""Reference krum_stream implementation for the public server repository.

This is a correctness-first reference, not a tuned production build: it keeps
one delta row vector in memory at a time and computes pairwise squared L2 in
full. The private deployment's version may use a different streaming strategy,
but the delta format and the returned shapes are the contract the coordinator
depends on. Operators can replace this file with a faster implementation that
returns the same values.
"""
import json
import struct
import torch
import numpy as np

class DeltaFile:
    def __init__(self, path, model_size):
        self.path = path
        self.model_size = model_size
        with open(path, "rb") as f:
            raw = f.read(4)
            if len(raw) != 4:
                raise ValueError("short delta header")
            n = struct.unpack("<I", raw)[0]
            self.meta = json.loads(f.read(n).decode("utf-8"))
            self.base_offset = 4 + n
        self.tokens = int(self.meta.get("tokensProcessed") or self.meta.get("tokens") or 1)
        # The base block is variable length once a client uses a sparse or framed
        # codec. The header declares it, so the engram tail is found after the
        # block rather than after the dense `model_size * 2` bytes. A dense delta
        # leaves the field unset and keeps the old arithmetic exactly.
        self.base_codec = str(self.meta.get("baseCodec") or "dense")
        self.base_bytes = int(self.meta.get("baseBytes") or self.model_size * 2)

def open_delta(path, model_size):
    return DeltaFile(path, model_size)

SPARSE_MAGIC = b"SPK1"
SPARSE_HEADER_BYTES = 12   # magic(4) + uint32 entry count + uint32 flat length


def _read_block(df):
    """The raw base block, whose length the header declares."""
    with open(df.path, "rb") as f:
        f.seek(df.base_offset)
        raw = f.read(df.base_bytes)
    if len(raw) != df.base_bytes:
        raise ValueError("short base delta")
    return raw


def _parse_sparse(raw, model_size):
    """Decode a `SPK1` block to (int64 indices, float32 values).

    Shared by every reader so the wire format has one implementation: a sparse
    delta must be usable by the merge path *and* by the norm/distance helpers,
    otherwise a round would aggregate correctly and still be sanity-checked wrong.
    """
    if raw[:4] != SPARSE_MAGIC:
        raise ValueError(f"not a sparse block: {raw[:4]!r}")
    count, total = struct.unpack("<II", raw[4:SPARSE_HEADER_BYTES])
    need = SPARSE_HEADER_BYTES + count * 6
    if len(raw) < need:
        raise ValueError(f"short sparse block: {len(raw)} < {need}")
    if total != model_size:
        raise ValueError(f"sparse block covers {total} params, expected {model_size}")
    idx = np.frombuffer(raw[SPARSE_HEADER_BYTES:SPARSE_HEADER_BYTES + count * 4], dtype=np.uint32)
    vals = torch.from_numpy(
        np.frombuffer(raw[SPARSE_HEADER_BYTES + count * 4:need], dtype=np.int16).copy()
    ).view(torch.bfloat16).to(torch.float32)
    return torch.from_numpy(idx.astype(np.int64)), vals


def _read_base(df):
    raw = _read_block(df)
    if getattr(df, "base_codec", "dense") == "sparse":
        idx, vals = _parse_sparse(raw, df.model_size)
        v = torch.zeros(df.model_size, dtype=torch.float32)
        v.index_add_(0, idx, vals)
        return v
    return torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).view(torch.bfloat16).to(torch.float32)

def streaming_pairwise_sq_l2(dfs, max_ram_bytes=None):
    n = len(dfs)
    dist = torch.zeros((n, n), dtype=torch.float64)
    sq = torch.zeros(n, dtype=torch.float64)
    bad = [False] * n
    vecs = []
    for i, df in enumerate(dfs):
        try:
            v = _read_base(df)
            if not bool(torch.isfinite(v).all()):
                bad[i] = True
                v = torch.zeros(df.model_size, dtype=torch.float32)
            vecs.append(v)
            sq[i] = torch.sum(v * v).item()
        except Exception:
            bad[i] = True
            vecs.append(torch.zeros(df.model_size, dtype=torch.float32))
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((vecs[i] - vecs[j]) ** 2).item()
            dist[i, j] = dist[j, i] = d
    return dist, bad, sq

def stream_add_base_delta(weights, df, scale):
    if getattr(df, "base_codec", "dense") == "sparse":
        # Scatter straight into the weights: materialising a dense float32 copy of
        # a 657.8 M-param delta would cost 2.6 GB per peer per round, which is the
        # transfer win given back on the receiving side.
        idx, vals = _parse_sparse(_read_block(df), df.model_size)
        weights[: df.model_size].index_add_(0, idx, vals * scale)
        return
    v = _read_base(df)
    weights[: df.model_size] += v * scale

def stream_add_base_delta_pair(weights, w_scale, extra, e_scale, df):
    """One read of the delta, two scaled accumulations.

    The aggregator needs the delta twice once outer momentum is on: folded into
    the weights at the merged scale, and folded into the carried gradient buffer
    at the (unscaled) effective weight. Reading the file twice would double the
    round's I/O for a 1.3 GB delta, so both adds share one decode.
    """
    if getattr(df, "base_codec", "dense") == "sparse":
        idx, vals = _parse_sparse(_read_block(df), df.model_size)
        weights[: df.model_size].index_add_(0, idx, vals * w_scale)
        extra[: df.model_size].index_add_(0, idx, vals * e_scale)
        return
    v = _read_base(df)
    weights[: df.model_size] += v * w_scale
    extra[: df.model_size] += v * e_scale

def read_engram_tail(df, dim):
    with open(df.path, "rb") as f:
        f.seek(df.base_offset + df.base_bytes)
        raw = f.read(4)
        if len(raw) != 4:
            return None, None
        count = struct.unpack("<I", raw)[0]
        if count == 0:
            return None, None
        idx = np.frombuffer(f.read(count * 4), dtype=np.uint32)
        vals = torch.frombuffer(bytearray(f.read(count * dim * 2)), dtype=torch.uint16).view(torch.bfloat16).reshape(-1, dim)
    return idx, vals

def plan_round(*args, **kwargs):
    return None
