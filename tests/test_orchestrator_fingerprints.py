import numpy as np

from orchestrator.fingerprints import (BLOCK, block_fingerprint, compare_blocks,
                                       fingerprint_root, simhash64, verify_root)


def test_block_is_258_bytes_for_32_tokens():
    hidden = np.random.default_rng(1).standard_normal((BLOCK, 1536)).astype("float32")
    fp = block_fingerprint(hidden)
    assert len(fp) == 258
    assert int.from_bytes(fp[:2], "little") == 32


def test_same_activation_same_fingerprint_and_one_mutation_differs():
    rng = np.random.default_rng(2)
    hidden = rng.standard_normal((8, 1536)).astype("float32")
    a = block_fingerprint(hidden)
    b = block_fingerprint(hidden.copy())
    assert a == b
    mutated = hidden.copy(); mutated[3] += 5.0
    assert compare_blocks(a, block_fingerprint(mutated)) >= 1


def test_root_binds_every_block():
    rng = np.random.default_rng(3)
    blocks = [block_fingerprint(rng.standard_normal((32, 1536)).astype("float32")) for _ in range(3)]
    root = fingerprint_root(blocks)
    assert verify_root(root, blocks) is True
    assert verify_root(root, blocks[:-1] + [blocks[-1][:-1]]) is False
    assert len(root) == 64


def test_simhash_is_deterministic():
    v = np.arange(1536, dtype="float32")
    assert simhash64(v) == simhash64(v)
