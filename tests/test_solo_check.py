"""Acceptance-rule tests: the gate must be unable to reject an update it did not measure.

Each test states the false positive it prevents, because the old geometry gate
rejected honest clients rather than damaged ones.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solo_check import (  # noqa: E402
    check_round_modules,
    DEFAULT_FLOOR,
    SubmissionScopeError,
    accept_unmeasured,
    check_module_scope,
    estimate_margin,
    judge,
    summarise,
    verdict_by_replay,
)


def test_improvement_is_accepted():
    accepted, regression = judge(8.0, 7.5, 0.01)
    assert accepted is True
    assert regression == pytest.approx(-0.5)


def test_no_measurable_change_is_accepted():
    # An honest client whose update happens to be neutral must not be punished.
    accepted, regression = judge(8.0, 8.0005, 0.01)
    assert accepted is True
    assert regression == pytest.approx(0.0005)


def test_equality_is_accepted():
    accepted, regression = judge(8.0, 8.0, 0.0)
    assert accepted is True
    assert regression == 0.0


def test_measured_regression_is_rejected():
    accepted, regression = judge(8.0, 8.5, 0.01)
    assert accepted is False
    assert regression == pytest.approx(0.5)


def test_non_finite_candidate_is_rejected_without_a_comparison():
    for bad in (math.nan, math.inf, -math.inf):
        accepted, regression = judge(8.0, bad, 0.01)
        assert accepted is False
        assert regression == math.inf


def test_a_single_baseline_sample_yields_the_floor():
    assert estimate_margin([]) == DEFAULT_FLOOR
    assert estimate_margin([8.0]) == DEFAULT_FLOOR
    assert estimate_margin([None, math.nan, 8.0]) == DEFAULT_FLOOR


def test_margin_grows_with_evaluation_noise():
    tight = estimate_margin([8.0, 8.0001, 7.9999])
    loose = estimate_margin([8.0, 8.2, 7.8])
    assert tight < loose
    assert loose > DEFAULT_FLOOR


def test_margin_never_drops_below_the_floor():
    assert estimate_margin([8.0, 8.0, 8.0, 8.0]) == DEFAULT_FLOOR


def test_replay_verdict_accepts_a_measured_improvement():
    verdict = verdict_by_replay("sub-1", lambda: 7.9, base_loss=8.0, margin=0.01,
                                advisory={"distance": 12.5})
    assert verdict.accepted is True
    assert verdict.reason == "accepted"
    assert verdict.regression == pytest.approx(-0.1)
    assert verdict.advisory == {"distance": 12.5}
    assert verdict.as_record()["submission_id"] == "sub-1"


def test_replay_verdict_rejects_only_a_measured_regression():
    verdict = verdict_by_replay("sub-2", lambda: 8.4, base_loss=8.0, margin=0.01)
    assert verdict.accepted is False
    assert verdict.reason == "held_out_regression"
    assert verdict.regression == pytest.approx(0.4)


def test_replay_verdict_rejects_a_failing_evaluator_rather_than_crashing():
    verdict = verdict_by_replay("sub-3", lambda: math.nan, base_loss=8.0, margin=0.01)
    assert verdict.accepted is False


def test_without_an_evaluator_everything_is_accepted():
    # The core anti-false-positive property: no measurement, no rejection.
    for sid in ("a", "b", "c"):
        verdict = accept_unmeasured(sid)
        assert verdict.accepted is True
        assert verdict.reason == "accepted_without_replay_evaluator"
        assert verdict.regression is None


def test_module_scope_accepts_a_clean_module_delta():
    names = ["blocks.0.attn.qkv.weight", "blocks.5.mlp.w2.weight", "blocks.3.norm.weight"]
    assert check_module_scope(names, split=(0, 6)) == []


def test_module_scope_rejects_a_block_outside_the_split():
    problems = check_module_scope(["blocks.7.attn.qkv.weight"], split=(0, 6))
    assert len(problems) == 1
    assert "blocks.7" in problems[0]


def test_module_scope_rejects_client_local_heads():
    problems = check_module_scope(["solo_heads.0.phi.weight"], split=(0, 6))
    assert len(problems) == 1
    assert "client-local" in problems[0]


def test_module_scope_rejects_engram_rows():
    assert check_module_scope(["engram.table.weight"], split=(0, 6)) != []


def test_module_scope_allows_the_readout_only_when_expected():
    assert check_module_scope(["lm_head.weight"], split=(18, 24)) != []
    assert check_module_scope(["lm_head.weight"], split=(18, 24),
                              readout_names=("lm_head.weight",)) == []


def test_module_scope_rejects_an_unparseable_block_name():
    assert check_module_scope(["blocks.abc.attn.qkv.weight"], split=(0, 6)) != []


def test_scope_error_is_a_value_error():
    assert issubclass(SubmissionScopeError, ValueError)


def test_summarise_counts_and_reports_the_worst_regression():
    verdicts = [
        verdict_by_replay("a", lambda: 7.9, base_loss=8.0, margin=0.01),
        verdict_by_replay("b", lambda: 8.4, base_loss=8.0, margin=0.01),
        accept_unmeasured("c"),
    ]
    summary = summarise(verdicts)
    assert summary["submissions"] == 3
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    assert summary["rejected_ids"] == ["b"]
    assert summary["max_regression"] == pytest.approx(0.4)
    # A round normally shares one margin, so duplicates collapse to one value.
    assert summary["margins"] == [0.01]


# --- select_by_objective: the round-level decision, without a socket -----------

from types import SimpleNamespace

from solo_check import select_by_objective


def _stub_open(sizes):
    """Pretend to open a delta; paths absent from ``sizes`` are unreadable."""
    def open_delta(path, model_size):
        if path not in sizes:
            raise OSError("truncated delta")
        return SimpleNamespace(path=path, tokens=sizes[path])
    return open_delta


def test_no_evaluator_keeps_every_readable_delta():
    kept, weights, rejected, unreadable, verdicts = select_by_objective(
        ["a", "b"], open_delta=_stub_open({"a": 300, "b": 100}), model_size=8)
    assert kept == ["a", "b"]
    assert rejected == [] and unreadable == []
    assert weights[0] == pytest.approx(0.75) and weights[1] == pytest.approx(0.25)
    assert sum(weights) == pytest.approx(1.0)
    assert all(v.accepted for v in verdicts)
    assert all(v.reason == "accepted_without_replay_evaluator" for v in verdicts)


def test_only_a_measured_regression_is_rejected():
    losses = {"a": 5.00, "b": 5.50, "c": 4.90}   # b gets worse, c gets better
    kept, weights, rejected, unreadable, verdicts = select_by_objective(
        ["a", "b", "c"], open_delta=_stub_open({"a": 1, "b": 1, "c": 1}),
        model_size=8, evaluate=lambda p: losses[p], base_loss=5.00, margin=0.01)
    assert rejected == ["b"] and unreadable == []
    assert kept == ["a", "c"]
    assert sum(weights) == pytest.approx(1.0)
    reasons = {v.submission_id: v.reason for v in verdicts}
    assert reasons == {"a": "accepted", "b": "held_out_regression", "c": "accepted"}


def test_an_unreadable_delta_is_reported_not_fatal():
    kept, weights, rejected, unreadable, _ = select_by_objective(
        ["a", "gone", "c"], open_delta=_stub_open({"a": 1, "c": 1}), model_size=8)
    assert unreadable == ["gone"]
    assert kept == ["a", "c"] and rejected == []
    assert sum(weights) == pytest.approx(1.0)


def test_a_round_where_every_delta_regresses_keeps_nothing():
    kept, weights, rejected, unreadable, _ = select_by_objective(
        ["a", "b"], open_delta=_stub_open({"a": 1, "b": 1}), model_size=8,
        evaluate=lambda p: 6.0, base_loss=5.0, margin=0.01)
    assert kept == [] and weights == []
    assert rejected == ["a", "b"] and unreadable == []


# --- check_round_modules: one round's declarations must agree ----------------

_SPLIT4 = [[0, 6], [6, 12], [12, 18], [18, 24]]


def _decl(sid, module, split=_SPLIT4, count=4):
    return {"submissionId": sid, "soloModule": module, "soloModules": count,
            "soloSplit": split}


def test_a_consistent_round_has_no_problems():
    out = check_round_modules([_decl("a", 0), _decl("b", 1), _decl("c", 2), _decl("d", 3)])
    assert out["problems"] == []
    assert out["missing"] == []
    assert out["modules"] == {0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]}


def test_two_peers_on_the_same_module_is_a_problem():
    out = check_round_modules([_decl("a", 1), _decl("b", 1)])
    assert any("more than one peer" in p for p in out["problems"])
    assert out["missing"] == [0, 2, 3]


def test_peers_disagreeing_about_the_split_is_a_problem():
    other = [[0, 12], [12, 24]]
    out = check_round_modules([_decl("a", 0), _decl("b", 0, split=other, count=2)])
    assert any("disagree about the split" in p for p in out["problems"])


def test_a_round_missing_a_module_reports_it():
    out = check_round_modules([_decl("a", 0), _decl("b", 1), _decl("c", 2)])
    assert out["missing"] == [3]
    assert out["problems"] == []


def test_legacy_submissions_have_no_module_declarations():
    out = check_round_modules([{"submissionId": "a"}, {"submissionId": "b"}])
    assert out["problems"] == []
    assert out["missing"] == []
    assert out["modules"] == {}


def test_an_out_of_range_module_is_a_problem():
    out = check_round_modules([_decl("a", 9)])
    assert any("outside 0..3" in p for p in out["problems"])


def test_a_split_of_the_wrong_length_is_a_problem():
    out = check_round_modules([_decl("a", 0, split=[[0, 12]], count=4)])
    assert any("split has 1 ranges but declares 4 modules" in p for p in out["problems"])
