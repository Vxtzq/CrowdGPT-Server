"""The public status document must never present a training loss as held-out."""

import math

from orchestrator.status import eval_fields


def test_unevaluated_round_publishes_no_loss():
    fields = eval_fields({"round": 372, "eval": None})
    assert fields["evalRound"] == 372
    assert fields["evalStatus"] == "unevaluated"
    assert fields["heldoutLoss"] is None
    assert fields["heldoutCi"] is None
    assert fields["evalSource"] is None


def test_missing_state_is_unevaluated():
    for state in (None, {}):
        fields = eval_fields(state)
        assert fields["evalStatus"] == "unevaluated"
        assert fields["heldoutLoss"] is None
        assert fields["evalRound"] is None


def test_evaluated_round_projects_the_record():
    fields = eval_fields({
        "round": 372,
        "updatedAt": "2026-11-14T21:00:00+00:00",
        "eval": {
            "mode": "heldout", "loss": "2.913456", "ciLow": 2.88, "ciHigh": 2.95,
            "windows": 8, "manifestSha256": "ab" * 32, "modelSha256": "cd" * 32,
            "source": "eval_service", "evaluatedAt": "2026-11-14T20:59:00+00:00",
        },
    })
    assert fields["evalStatus"] == "evaluated"
    assert fields["heldoutLoss"] == 2.913456
    assert isinstance(fields["heldoutLoss"], float)
    assert fields["heldoutCi"] == {"low": 2.88, "high": 2.95}
    assert fields["evalSource"] == "eval_service"
    assert fields["evalMode"] == "heldout"
    assert fields["evalEvaluatedAt"] == "2026-11-14T20:59:00+00:00"
    assert fields["evalRound"] == 372


def test_one_sided_and_missing_confidence_intervals():
    assert eval_fields({"eval": {"loss": 3.0}})["heldoutCi"] is None
    assert eval_fields({"eval": {"loss": 3.0, "ciLow": 2.9}})["heldoutCi"] == {"low": 2.9, "high": None}
    assert eval_fields({"eval": {"loss": 3.0, "ciHigh": 3.1}})["heldoutCi"] == {"low": None, "high": 3.1}


def test_invalid_losses_are_never_coerced_into_a_number():
    for bad in (None, "", "abc", float("nan"), float("inf"), float("-inf"), True):
        fields = eval_fields({"round": 7, "eval": {"loss": bad, "source": "eval_service"}})
        assert fields["evalStatus"] == "invalid", bad
        assert fields["heldoutLoss"] is None, bad


def test_non_mapping_eval_is_unevaluated():
    for value in ("2.9", [2.9], 2.9):
        fields = eval_fields({"round": 1, "eval": value})
        assert fields["evalStatus"] == "unevaluated"
        assert fields["heldoutLoss"] is None


def test_round_coercion():
    assert eval_fields({"round": "12"})["evalRound"] == 12
    assert eval_fields({"round": "not-a-round"})["evalRound"] is None
    assert eval_fields({"round": None})["evalRound"] is None


def test_training_loss_can_never_be_read_as_heldout():
    fields = eval_fields({"round": 5, "trainLoss": 3.08, "lastLoss": 3.08})
    assert fields["heldoutLoss"] is None
    assert fields["evalStatus"] == "unevaluated"


def test_result_is_a_fresh_mapping():
    first = eval_fields({})
    first["heldoutLoss"] = 1.0
    assert eval_fields({})["heldoutLoss"] is None


def test_projection_is_finite_or_absent():
    fields = eval_fields({"eval": {"loss": 2.5, "ciLow": 2.4, "ciHigh": 2.6}})
    for name in ("heldoutLoss",):
        assert math.isfinite(fields[name])
