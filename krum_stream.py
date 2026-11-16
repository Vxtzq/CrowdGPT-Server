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

def open_delta(path, model_size):
    return DeltaFile(path, model_size)

def _read_base(df):
    with open(df.path, "rb") as f:
        f.seek(df.base_offset)
        raw = f.read(df.model_size * 2)
    if len(raw) != df.model_size * 2:
        raise ValueError("short base delta")
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
    v = _read_base(df)
    weights[: df.model_size] += v * scale

def read_engram_tail(df, dim):
    with open(df.path, "rb") as f:
        f.seek(df.base_offset + df.model_size * 2)
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
