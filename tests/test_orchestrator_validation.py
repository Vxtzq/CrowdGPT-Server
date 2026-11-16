import numpy as np

from orchestrator.fingerprints import block_fingerprint, fingerprint_root
from orchestrator.validation import ValidationService, replay_fraction, sample_indices


def _blocks(seed=4, n=4):
    rng = np.random.default_rng(seed)
    return [block_fingerprint(rng.standard_normal((32, 1536)).astype("float32")) for _ in range(n)]


def test_honest_blocks_pass_and_mutation_is_flagged():
    blocks = _blocks()
    root = fingerprint_root(blocks)
    svc = ValidationService()
    assert svc.validate_claimed_root(root, blocks) is True
    mutated = list(blocks); mutated[1] = mutated[1][:-1] + bytes([mutated[1][-1] ^ 0xFF])
    assert svc.validate_claimed_root(root, mutated) is False
    rec = lambda i: mutated[i]
    out = svc.compare_sample({i: blocks[i] for i in range(len(blocks))}, rec, [0, 1, 2, 3])
    assert out["ok"] is False and out["divergent"] == [1]


def test_fabricated_root_rejected():
    blocks = _blocks(seed=9)
    svc = ValidationService()
    assert svc.validate_claimed_root("f" * 64, blocks) is False


def test_sampling_is_deterministic_and_bounded():
    assert sample_indices(7, 100, 5) == sample_indices(7, 100, 5)
    assert len(sample_indices(7, 3, 99)) == 3
    items = list(range(50))
    assert replay_fraction(items, 0.1, 5) == replay_fraction(items, 0.1, 5)
    assert len(replay_fraction(items, 0.1, 5)) == 5
