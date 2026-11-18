"""DiLoCo's outer step: the streaming form must equal the vector definition.

The server cannot afford to materialise the client mean, so it applies the
update as two streaming accumulations plus one carried vector. That is only safe
if it is *algebraically* the same update, which is what these tests pin down --
including across several rounds, because the momentum term is what distinguishes
DiLoCo from plain weighted-mean SGD and an off-by-one in the carry would look
fine for the first round and drift afterwards.
"""

import unittest

# The repo's checks job is standard-library only (no torch, no pytest), and runs
# `python -m unittest discover`. Skip this module there instead of erroring on import.
try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
    import torch  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from outer_step import (  # noqa: E402
    DEFAULT_MOMENTUM,
    MAX_MOMENTUM,
    buffer_bytes,
    nesterov_update,
    resolve_momentum,
    streaming_scales,
)


def _rounds(n_rounds=3, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [[(torch.randn(size, generator=g), 0.3),
             (torch.randn(size, generator=g), 0.7)] for _ in range(n_rounds)]


def _vector_path(theta0, rounds, *, lr, momentum, nesterov=True):
    theta = theta0.clone()
    prev_grad = None
    for deltas in rounds:
        mean = theta + sum(eff * d for d, eff in deltas)
        grad = theta - mean
        theta = nesterov_update(theta, grad, prev_grad, lr=lr, momentum=momentum,
                                nesterov=nesterov)
        prev_grad = grad
    return theta


def _streaming_path(theta0, rounds, *, lr, momentum, nesterov=True):
    """Exactly what ``apply_round_aggregation`` does once momentum is enabled."""
    w = theta0.clone()
    carried = torch.zeros_like(w)
    for deltas in rounds:
        scales = streaming_scales(lr=lr, momentum=momentum, nesterov=nesterov)
        w = w + scales["carry"] * carried
        carried = torch.zeros_like(w)
        for d, eff in deltas:
            w = w + scales["weight"] * eff * d
            carried = carried + scales["buffer"] * eff * d
    return w


def test_streaming_matches_the_vector_update_across_rounds():
    theta0 = torch.randn(16, generator=torch.Generator().manual_seed(9))
    rounds = _rounds(3)
    ref = _vector_path(theta0, rounds, lr=0.05, momentum=0.9)
    got = _streaming_path(theta0, rounds, lr=0.05, momentum=0.9)
    assert torch.allclose(ref, got, atol=1e-7)


def test_streaming_matches_after_many_rounds():
    """Five rounds: a wrong carry would still pass a one-round check."""
    theta0 = torch.randn(16, generator=torch.Generator().manual_seed(10))
    rounds = _rounds(5, seed=11)
    ref = _vector_path(theta0, rounds, lr=0.03, momentum=0.95)
    got = _streaming_path(theta0, rounds, lr=0.03, momentum=0.95)
    assert torch.allclose(ref, got, atol=1e-7)


def test_heavy_ball_form_also_matches():
    theta0 = torch.randn(16, generator=torch.Generator().manual_seed(12))
    rounds = _rounds(4, seed=13)
    ref = _vector_path(theta0, rounds, lr=0.05, momentum=0.9, nesterov=False)
    got = _streaming_path(theta0, rounds, lr=0.05, momentum=0.9, nesterov=False)
    assert torch.allclose(ref, got, atol=1e-7)


def test_zero_momentum_is_exactly_the_pre_diloco_update():
    """Default deployments must be bit-identical to weighted-mean SGD."""
    rounds = _rounds(2, seed=14)
    scales = streaming_scales(lr=0.05, momentum=0.0)
    assert scales["carry"] == 0.0 and scales["weight"] == 0.05
    theta0 = torch.randn(16, generator=torch.Generator().manual_seed(15))
    plain = theta0.clone()
    for deltas in rounds:
        for d, eff in deltas:                     # same accumulation order as the server
            plain = plain + 0.05 * eff * d
    got = _streaming_path(theta0, rounds, lr=0.05, momentum=0.0)
    assert torch.equal(plain, got)


def test_nesterov_scales_the_update_by_one_plus_momentum():
    # 1.0 is clamped to MAX_MOMENTUM: the outer step runs once per round, so a
    # value that close to 1 compounds over very little real work.
    assert streaming_scales(lr=0.1, momentum=1.0)["weight"] == pytest.approx(0.1 * (1 + MAX_MOMENTUM))
    assert streaming_scales(lr=0.1, momentum=0.9)["weight"] == pytest.approx(0.19)
    assert streaming_scales(lr=0.1, momentum=0.9)["factor"] == pytest.approx(1.9)


def test_resolve_momentum_parses_and_clamps():
    assert resolve_momentum(None) == 0.0            # absent = off, the shipped default
    assert resolve_momentum("") == 0.0
    assert resolve_momentum("0.9") == pytest.approx(0.9)
    assert resolve_momentum(0.85) == pytest.approx(0.85)
    assert resolve_momentum("5") == MAX_MOMENTUM    # clamped, not accepted
    assert resolve_momentum("-2") == 0.0
    assert resolve_momentum("nan") == 0.0
    assert resolve_momentum("garbage") == 0.0
    assert resolve_momentum("garbage", default=DEFAULT_MOMENTUM) == DEFAULT_MOMENTUM


def test_buffer_bytes_reports_the_memory_guard_cost():
    assert buffer_bytes(657_800_000, 4) == pytest.approx(2.45, abs=0.01)
    assert buffer_bytes(1_000_000, 2) == pytest.approx(0.0018, abs=0.0005)
