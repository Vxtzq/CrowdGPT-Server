"""The objective gate's own visibility.

`select_by_objective` accepts every delta when no replay evaluator is registered,
which is correct -- a server that cannot measure must not reject -- but it means a
misconfigured deployment verifies nothing while looking healthy. `gate_state` is
the machine-readable answer to "are we actually measuring?", surfaced in the round
status and re-announced every round the gate is unmeasured.
"""

import unittest

try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest: {_missing}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solo_check import (  # noqa: E402
    MIN_PAIRED_WINDOWS,
    Verdict,
    accept_unmeasured,
    gate_state,
    summarise,
)


def _evaluate(_path):
    return [5.0, 5.1]


def test_no_evaluator_is_loudly_unmeasured():
    state = gate_state(evaluate=None)
    assert state["level"] == "unmeasured"
    assert state["measured"] is False
    assert state["paired"] is False
    assert "ACCEPTED without being measured" in state["warning"]


def test_a_scalar_evaluator_is_degraded_not_ok():
    """A single loss has no window noise: the margin is a fixed floor again."""
    state = gate_state(evaluate=_evaluate, base_windows=None)
    assert state["level"] == "degraded"
    assert state["measured"] is True
    assert state["paired"] is False
    assert "fixed floor" in state["warning"]


def test_too_few_windows_is_degraded():
    state = gate_state(evaluate=_evaluate, base_windows=[5.0, 5.1])
    assert state["level"] == "degraded"
    assert state["windows"] == 2
    assert str(MIN_PAIRED_WINDOWS) in state["warning"]


def test_enough_windows_is_ok():
    state = gate_state(evaluate=_evaluate, base_windows=[5.0] * MIN_PAIRED_WINDOWS,
                       bf16_band=3.4e-5)
    assert state["level"] == "ok"
    assert state["paired"] is True
    assert state["windows"] == MIN_PAIRED_WINDOWS
    assert state["bf16Band"] == pytest.approx(3.4e-5)
    assert state["warning"] is None


def test_min_windows_is_configurable():
    state = gate_state(evaluate=_evaluate, base_windows=[5.0, 5.1], min_windows=2)
    assert state["level"] == "ok"


def test_summarise_counts_the_unmeasured_acceptances():
    verdicts = [accept_unmeasured("a"), accept_unmeasured("b"),
                Verdict("c", True, "accepted", base_loss=5.0, candidate_loss=4.9,
                        regression=-0.1, margin=0.01)]
    summary = summarise(verdicts)
    assert summary["unmeasured"] == 2
    assert summary["accepted"] == 3


def test_summarise_reports_zero_when_every_delta_was_measured():
    summary = summarise([Verdict("a", True, "accepted", regression=-0.1, margin=0.01)])
    assert summary["unmeasured"] == 0
