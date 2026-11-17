"""Objective-based acceptance for client submissions.

Crowd-v1's cross-check judged *geometry*: it clustered the round's deltas by
pairwise L2 distance, kept the closest pair as a "core", and discarded anything
further from that core than a ratio of the core distance. Honest stochastic
updates are not close to each other -- two clients that trained on different
shards, or for different numbers of tokens, produce deltas whose distance is
dominated by sampling noise rather than by dishonesty. That rule therefore
discarded honest work whenever a round was heterogeneous, and at `n == 2` it
discarded *both* clients as soon as they disagreed by 10 % of the smaller norm.
Nothing in that path ever asked whether an update helped the model.

This module decides on the *objective*: apply the delta to the base weights and
measure held-out loss. A submission is rejected only when it measurably makes
the loss worse, by a margin the round's own measurement noise justifies. Every
other case is accepted, including "no measurable change" -- a client that
safely did nothing is not a liar. Distance survives as an advisory number in the
round record, never as a verdict.

With SOLO a submission carries one module's delta instead of the whole model, so
the replay evaluates a single module on a small held-out slice rather than 1 B
parameters. `check_module_scope` enforces that a SOLO submission cannot smuggle
in tensors from another module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Sequence

# A held-out loss regression smaller than this is not evidence of anything: the
# server evaluates on a finite slice, and two evaluations of identical weights
# differ by sampling noise alone.
DEFAULT_FLOOR = 1e-4
DEFAULT_SIGMAS = 3.0


class SubmissionScopeError(ValueError):
    """A SOLO submission carries parameters outside the module it claims."""


@dataclass(frozen=True)
class Verdict:
    """One submission's acceptance decision, with the numbers behind it."""

    submission_id: str
    accepted: bool
    reason: str
    base_loss: float | None = None
    candidate_loss: float | None = None
    regression: float | None = None
    margin: float | None = None
    advisory: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return asdict(self)


def estimate_margin(baseline_losses: Sequence[float], *, sigmas: float = DEFAULT_SIGMAS,
                    floor: float = DEFAULT_FLOOR) -> float:
    """Noise margin from repeated evaluations of the *same* base weights.

    The spread of identical-weight evaluations is the floor below which a loss
    change cannot be attributed to the submission. With fewer than two samples
    there is no estimate, so the fixed floor is used.
    """
    values = [float(v) for v in baseline_losses if v is not None and math.isfinite(float(v))]
    if len(values) < 2:
        return float(floor)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    stderr = math.sqrt(var / len(values))
    return max(float(floor), float(sigmas) * stderr)


def judge(base_loss: float, candidate_loss: float, margin: float) -> tuple[bool, float]:
    """Accept unless the candidate measurably regresses the held-out loss.

    Returns `(accepted, regression)` where `regression = candidate - base`.
    Equality and improvement are accepted: the gate exists to catch submissions
    that damage the model, not to reward submissions that happen to help.
    """
    if not math.isfinite(candidate_loss):
        return False, float("inf")
    regression = float(candidate_loss) - float(base_loss)
    return (regression <= float(margin)), regression


def verdict_by_replay(submission_id: str, evaluate: Callable[[], float], *,
                      base_loss: float, margin: float,
                      advisory: dict | None = None) -> Verdict:
    """Evaluate the candidate weights once and accept unless they regress.

    `evaluate` is supplied by the caller because only the server knows how to
    load a delta onto the base model; this function owns the decision rule so
    that the rule is testable without a model.
    """
    candidate = float(evaluate())
    accepted, regression = judge(base_loss, candidate, margin)
    reason = "accepted" if accepted else "held_out_regression"
    return Verdict(submission_id=submission_id, accepted=accepted, reason=reason,
                   base_loss=float(base_loss), candidate_loss=candidate,
                   regression=regression, margin=float(margin),
                   advisory=dict(advisory or {}))


def accept_unmeasured(submission_id: str, *, advisory: dict | None = None) -> Verdict:
    """The verdict when no replay evaluator is configured.

    The server cannot invent a rejection it has not measured, so it accepts and
    records that fact. This is what removes the false positives: the only way a
    submission is rejected is a measured regression.
    """
    return Verdict(submission_id=submission_id, accepted=True,
                   reason="accepted_without_replay_evaluator",
                   advisory=dict(advisory or {}))


def check_module_scope(names: Iterable[str], *, split: tuple[int, int],
                       readout_names: Iterable[str] = ()) -> list[str]:
    """Verify a SOLO delta only touches its own module (plus the shared readout).

    `split` is the module's `(lo, hi)` block range. Block tensors are matched as
    `blocks.<i>.…`; anything outside `[lo, hi)` is an error unless it is one of
    the readout tensors the terminal module legitimately owns. Head tensors are
    client-local and must never appear in an upload.
    """
    lo, hi = int(split[0]), int(split[1])
    readout = set(readout_names)
    problems: list[str] = []
    for name in names:
        if name.startswith("solo_heads."):
            problems.append(f"{name}: SOLO heads are client-local and must not be uploaded")
            continue
        if name.startswith("engram."):
            problems.append(f"{name}: engram rows are not part of a module delta")
            continue
        if name in readout:
            continue
        if name.startswith("blocks."):
            parts = name.split(".")
            try:
                index = int(parts[1])
            except (IndexError, ValueError):
                problems.append(f"{name}: cannot parse block index")
                continue
            if not (lo <= index < hi):
                problems.append(f"{name}: block {index} is outside module {lo}-{hi}")
            continue
        problems.append(f"{name}: not owned by module {lo}-{hi}")
    return problems


def summarise(verdicts: Sequence[Verdict]) -> dict:
    """Round-level counts for the log and the public status record."""
    accepted = [v for v in verdicts if v.accepted]
    rejected = [v for v in verdicts if not v.accepted]
    return {
        "submissions": len(verdicts),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_ids": [v.submission_id for v in rejected],
        "max_regression": max((v.regression for v in rejected if v.regression is not None),
                              default=None),
        "margins": sorted({round(v.margin, 8) for v in verdicts if v.margin is not None}),
    }


def select_by_objective(paths, *, open_delta, model_size, evaluate=None,
                       base_loss=None, margin=None):
    """Pick the deltas a round keeps, by the objective and nothing else.

    Pure so it can be tested without importing ``server`` (which binds a port):
    ``open_delta`` is the caller's delta parser and ``evaluate(path)`` replays a
    delta onto the base weights and returns held-out loss.

    Returns ``(kept_paths, weights, rejected_paths, unreadable_paths, verdicts)``.
    With no evaluator every readable delta is kept, because a server that cannot
    measure a delta has no business rejecting it.
    """
    unreadable = []
    readable = []
    for p in paths:
        try:
            readable.append(open_delta(p, model_size))
        except Exception:
            unreadable.append(p)
    verdicts = []
    for d in readable:
        sid = getattr(d, "path", str(d))
        advisory = {"tokens": getattr(d, "tokens", None)}
        if evaluate is None or base_loss is None:
            verdicts.append(accept_unmeasured(sid, advisory=advisory))
        else:
            verdicts.append(verdict_by_replay(sid, lambda p=sid: evaluate(p),
                                              base_loss=base_loss, margin=margin,
                                              advisory=advisory))
    kept = [(d, v) for d, v in zip(readable, verdicts) if v.accepted]
    rejected = [d for d, v in zip(readable, verdicts) if not v.accepted]
    if not kept:
        return ([], [], [getattr(d, "path", str(d)) for d in rejected], unreadable, verdicts)
    tokens = [max(1, int(getattr(d, "tokens", 0) or 1)) for d, _ in kept]
    total = sum(tokens)
    return ([getattr(d, "path", str(d)) for d, _ in kept], [t / total for t in tokens],
            [getattr(d, "path", str(d)) for d in rejected], unreadable, verdicts)
