"""Public `status.json` projection for the held-out evaluation fields.

`status.json` used to publish only `lastLoss`, a *training* number. The round
record and the nutrition label already carry a held-out evaluation, but the
public status document did not, so a reader could not tell the two apart.

These helpers keep the two separate (``NG-6``: reported, checked, and held-out
numbers are never relabelled):

- ``trainLoss`` / ``lastLoss`` are always the reported training loss;
- ``heldoutLoss`` is only ever a value from an evaluated round;
- a round that was not evaluated publishes ``null`` rather than carrying the
  previous round's number forward, which would silently attribute a stale
  measurement to a new checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping

_UNEVALUATED: dict[str, Any] = {
    "heldoutLoss": None,
    "heldoutCi": None,
    "evalRound": None,
    "evalSource": None,
    "evalMode": None,
    "evalStatus": "unevaluated",
    "evalEvaluatedAt": None,
}


def eval_fields(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project a ``{"round", "eval", "updatedAt"}`` record into status fields.

    ``eval`` is the dict ``server.py::_round_eval`` produces: ``loss`` may be a
    float or a formatted string, and ``ciLow``/``ciHigh`` may be missing.
    Anything unparseable is reported as unevaluated or invalid — never coerced
    into a plausible-looking loss.
    """
    fields = dict(_UNEVALUATED)
    state = state or {}
    round_no = state.get("round")
    if round_no is not None:
        try:
            fields["evalRound"] = int(round_no)
        except (TypeError, ValueError):
            fields["evalRound"] = None
    evaluation = state.get("eval")
    if not isinstance(evaluation, Mapping):
        return fields
    raw_loss = evaluation.get("loss")
    if isinstance(raw_loss, bool):  # bool is an int subclass; a flag is not a loss
        fields["evalStatus"] = "invalid"
        return fields
    try:
        loss = float(raw_loss)
    except (TypeError, ValueError):
        fields["evalStatus"] = "invalid"
        return fields
    if loss != loss or loss in (float("inf"), float("-inf")):  # NaN / ±inf
        fields["evalStatus"] = "invalid"
        return fields
    fields["heldoutLoss"] = loss
    low, high = evaluation.get("ciLow"), evaluation.get("ciHigh")
    fields["heldoutCi"] = ({"low": low, "high": high}
                           if low is not None or high is not None else None)
    fields["evalSource"] = evaluation.get("source")
    fields["evalMode"] = evaluation.get("mode")
    fields["evalEvaluatedAt"] = evaluation.get("evaluatedAt") or state.get("updatedAt")
    fields["evalStatus"] = "evaluated"
    return fields
