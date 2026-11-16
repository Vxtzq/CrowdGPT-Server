"""Validator mode: sampled replay and fingerprint comparison."""
from __future__ import annotations

import hashlib
import random
from typing import Callable, Iterable, Optional

from .fingerprints import compare_blocks, verify_root


def sample_indices(seed: int, n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0: return []
    rng = random.Random(int(seed))
    k = min(int(k), int(n))
    return sorted(rng.sample(range(int(n)), k))


def replay_fraction(items: Iterable, fraction: float, seed: int) -> list:
    items = list(items)
    k = max(0, min(len(items), int(round(len(items) * max(0.0, min(1.0, fraction))))))
    idx = sample_indices(seed, len(items), k)
    return [items[i] for i in idx]


class ValidationService:
    def __init__(self, store=None):
        self.store = store

    def validate_claimed_root(self, claimed_root: Optional[str], blocks: list[bytes]) -> bool:
        if not claimed_root: return False
        return verify_root(claimed_root, blocks)

    def compare_sample(self, claimed: dict[int, bytes],
                       recompute: Callable[[int], bytes],
                       indices: Iterable[int]) -> dict:
        """Return divergent indices and counts for a sampled replay."""
        checked = 0; divergent = []
        for i in indices:
            if i not in claimed: continue
            checked += 1
            actual = recompute(i)
            if compare_blocks(claimed[i], actual) > 0:
                divergent.append(i)
        return {"checked": checked, "divergent": divergent,
                "ok": checked > 0 and not divergent}

    def submission_seed(self, submission_id: str) -> int:
        return int(hashlib.sha256(submission_id.encode()).hexdigest()[:16], 16)


__all__ = ["ValidationService", "replay_fraction", "sample_indices"]
