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


def paired_margin(base_windows: Sequence[float], cand_windows: Sequence[float], *,
                  sigmas: float = DEFAULT_SIGMAS, floor: float = DEFAULT_FLOOR,
                  bf16_band: float = 0.0) -> tuple[float, float]:
    """Margin for a *paired* comparison: same windows, base vs candidate.

    Why this exists (review note: "way too much false positives with crowd v1,
    especially with bf16"): ``estimate_margin`` measures repeated evaluations of
    the *same* weights. On a fixed held-out slice in eval mode those repeat
    exactly, so its stderr is zero and the margin collapses to ``DEFAULT_FLOOR``
    (1e-4). Any rounding-level change then counts as a regression, and bf16
    deltas move the loss by far more than 1e-4 through quantization alone, so an
    honest client gets rejected. The uncertainty we actually have is *across
    windows*, and it is much smaller when the comparison is paired on the same
    windows because the window difficulty cancels.

    Returns ``(margin, mean_delta)`` where ``mean_delta = mean(cand - base)``.
    ``bf16_band`` is the loss change the same delta produces from bf16 rounding
    alone (measured by the caller); the margin never drops below it.
    """
    pairs = [(float(b), float(c)) for b, c in zip(base_windows, cand_windows)
             if math.isfinite(float(b)) and math.isfinite(float(c))]
    n = len(pairs)
    if n == 0:
        return max(float(floor), float(bf16_band)), float("inf")
    deltas = [c - b for b, c in pairs]
    mean_delta = sum(deltas) / n
    if n < 2:
        return max(float(floor), float(bf16_band)), mean_delta
    var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    stderr = math.sqrt(var / n)
    # max(), not sum(): the bf16 band is measured on these same windows, so it
    # already contains their sampling noise; adding would double count it.
    return max(float(floor), float(sigmas) * stderr, float(bf16_band)), mean_delta


def judge_paired(base_windows: Sequence[float], cand_windows: Sequence[float], *,
                 sigmas: float = DEFAULT_SIGMAS, floor: float = DEFAULT_FLOOR,
                 bf16_band: float = 0.0) -> tuple[bool, float, float]:
    """Accept unless the candidate regresses beyond the paired margin.

    Returns ``(accepted, mean_delta, margin)``. Equality and improvement are
    accepted; a delta the measurement cannot distinguish from noise is accepted.
    """
    margin, mean_delta = paired_margin(base_windows, cand_windows, sigmas=sigmas,
                                       floor=floor, bf16_band=bf16_band)
    if not math.isfinite(mean_delta):
        return False, mean_delta, margin
    return (mean_delta <= margin), mean_delta, margin


def bf16_band_from_replay(loss_fp32: float, loss_bf16: float, *, margin_factor: float = 1.0) -> float:
    """Loss change the same delta produces from bf16 rounding alone.

    A delta is stored and shipped as bf16, so the model a peer actually trained
    and the model the server replays differ by rounding. Measuring both replays
    turns that into a number instead of leaving it inside a hand-picked
    threshold: with ``margin_factor=1`` anything inside the band is treated as
    "the quantization did this, not the peer".
    """
    if not (math.isfinite(float(loss_fp32)) and math.isfinite(float(loss_bf16))):
        return 0.0
    return abs(float(loss_bf16) - float(loss_fp32)) * float(max(0.0, margin_factor))


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


# A round is only *verified* when the gate can replay the delta AND has enough
# windows to estimate how much a loss may move for a legitimate reason. Below
# this many windows the paired margin degenerates towards its floor (measured:
# 1e-4 at one window, 2.1e-3 at four, 3.9e-3 at eight), so the verdict is a
# constant dressed up as a measurement.
MIN_PAIRED_WINDOWS = 4


def gate_state(*, evaluate, base_windows=None, bf16_band: float = 0.0,
               min_windows: int = MIN_PAIRED_WINDOWS) -> dict:
    """Report what the objective gate can actually prove, in one dict.

    A gate that accepts everything is indistinguishable from a gate that
    accepts everything *honestly*: both produce "accepted" for every delta. The
    only defence is to state the gate's own capability next to the verdicts, so
    ``level == "unmeasured"`` is visible in the log and in the status endpoint
    instead of being inferred from a server that never rejects anything.

    ``level`` is ``"ok"`` (paired margin on enough windows), ``"degraded"``
    (it can measure, but the margin is weak) or ``"unmeasured"`` (nothing can
    be rejected).
    """
    windows = len(list(base_windows or ()))
    state = {"measured": evaluate is not None, "paired": False, "windows": windows,
             "bf16Band": float(bf16_band), "minWindows": int(min_windows)}
    if evaluate is None:
        return {**state, "level": "unmeasured", "warning":
                "no replay evaluator registered: every delta is ACCEPTED without being measured"}
    if windows == 0:
        return {**state, "level": "degraded", "warning":
                "replay returns a single loss: the margin is a fixed floor with no window noise "
                f"(pass per-window losses and >= {int(min_windows)} base windows)"}
    if windows < int(min_windows):
        return {**state, "level": "degraded", "warning":
                f"paired on {windows} window(s) < {int(min_windows)}: the margin can collapse to "
                "its floor, so a small honest regression may pass unjudged"}
    return {**state, "level": "ok", "paired": True, "warning": None}


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
        "unmeasured": sum(1 for v in verdicts if v.reason == "accepted_without_replay_evaluator"),
    }


def select_by_objective(paths, *, open_delta, model_size, evaluate=None,
                       base_loss=None, margin=None, base_windows=None,
                       bf16_band=0.0):
    """Pick the deltas a round keeps, by the objective and nothing else.

    Pure so it can be tested without importing ``server`` (which binds a port):
    ``open_delta`` is the caller's delta parser and ``evaluate(path)`` replays a
    delta onto the base weights and returns held-out loss.

    Returns ``(kept_paths, weights, rejected_paths, unreadable_paths, verdicts)``.
    With no evaluator every readable delta is kept, because a server that cannot
    measure a delta has no business rejecting it.

    ``base_windows`` switches the judgement to the paired rule: if ``evaluate``
    returns a sequence of per-window losses of the same length, the margin comes
    from the spread of the *paired* per-window deltas rather than from a fixed
    floor. That matters because repeated evaluations of identical weights on a
    fixed slice are identical, so the old ``estimate_margin`` collapsed to its
    1e-4 floor -- and with only one window it still does (measured: 3-sigma =
    1e-4 exactly at n=1, 2.1e-3 at n=4, 3.9e-3 at n=8). ``bf16_band`` adds the
    measured cost of shipping the delta as bf16 so rounding can widen the margin
    instead of silently deciding the verdict.
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
        if evaluate is None or (base_loss is None and base_windows is None):
            verdicts.append(accept_unmeasured(sid, advisory=advisory))
            continue
        if base_windows:
            try:
                cand = evaluate(sid)
            except Exception as exc:                      # a delta we cannot replay
                verdicts.append(Verdict(sid, False, "replay_failed",
                                        advisory={**advisory, "error": str(exc)}))
                continue
            if isinstance(cand, (list, tuple)) and len(cand) == len(base_windows):
                accepted, mean_delta, margin_used = judge_paired(
                    base_windows, cand, bf16_band=bf16_band)
                verdicts.append(Verdict(
                    sid, accepted, "accepted" if accepted else "held_out_regression",
                    base_loss=sum(base_windows) / len(base_windows),
                    candidate_loss=sum(cand) / len(cand),
                    regression=mean_delta, margin=margin_used,
                    advisory={**advisory, "windows": len(cand), "paired": True,
                              "bf16Band": bf16_band}))
                continue
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


def check_round_modules(declarations):
    """Check the module declarations of one round against each other.

    ``declarations`` is any iterable of mappings carrying what a client sends in its
    submission metadata: ``submissionId``, ``soloModule``, ``soloModules`` and
    ``soloSplit``. Returns ``{"problems": [...], "modules": {...}, "missing": [...],
    "splits": [...]}``.

    This is deliberately **advisory**. A delta is one flat vector with no per-tensor
    offsets, so the server cannot prove which module a peer actually trained; what it
    can prove is that the round's declarations are internally consistent — one shared
    split, no module claimed twice, and a record of which modules nobody covered.
    """
    claimed = {}
    splits = {}
    problems = []
    for declaration in declarations or ():
        if not declaration:
            continue
        sid = str(declaration.get("submissionId") or declaration.get("submission_id") or "?")
        module = declaration.get("soloModule")
        if module is None:
            continue
        try:
            module = int(module)
            count = int(declaration.get("soloModules"))
        except (TypeError, ValueError):
            problems.append(f"{sid}: unreadable SOLO module declaration")
            continue
        if not 0 <= module < count:
            problems.append(f"{sid}: module {module} is outside 0..{count - 1}")
            continue
        try:
            split = tuple((int(lo), int(hi)) for lo, hi in (declaration.get("soloSplit") or []))
        except (TypeError, ValueError):
            problems.append(f"{sid}: unreadable SOLO split")
            continue
        if len(split) != count:
            problems.append(f"{sid}: split has {len(split)} ranges but declares {count} modules")
            continue
        splits.setdefault(split, []).append(sid)
        claimed.setdefault(module, []).append(sid)

    if len(splits) > 1:
        problems.append(
            "peers disagree about the split: "
            + "; ".join(f"{sorted(sids)} use {list(split)}" for split, sids in splits.items())
        )
    for module, sids in sorted(claimed.items()):
        if len(sids) > 1:
            problems.append(f"module {module} claimed by more than one peer: {', '.join(sorted(sids))}")
    count = len(next(iter(splits))) if splits else 0
    return {
        "problems": problems,
        "modules": {module: sorted(sids) for module, sids in sorted(claimed.items())},
        "missing": [module for module in range(count) if module not in claimed],
        "splits": [list(split) for split in splits],
    }


def check_round_paradigms(declarations):
    """Check that a round's submissions agree about how they were trained.

    ``declarations`` is any iterable of mappings carrying ``submissionId``,
    ``paradigm`` and ``dilocoInner`` (what a v2 client puts in ``submit_extra``).
    Returns ``{"problems": [...], "paradigms": {...}, "innerSteps": {...}}``.

    Mixing training rules inside one pool is not a security question, it is a
    correctness one: the pool averages uploads, and a ReLoRA adapter delta, a
    MOSAIC slice delta and a whole-model backprop delta do not mean the same
    thing. DiLoCo inner steps matter for the same reason — a peer that took 500
    local steps contributed a different-sized step than one that took 1.

    Advisory only, and deliberately so: a delta arrives as one flat vector with
    no per-tensor offsets, so the server cannot prove which paradigm produced
    it. What it can prove is that the round's declarations are consistent. The
    return value carries ``verified: False`` and a reason so a caller cannot
    mistake agreement for proof — closing that honestly needs a per-tensor
    checksum plus a recomputation of one step from the declared rule, which is
    TODO V2I-14, not a nonce.
    """
    problems = []
    paradigms = {}
    inner = {}
    for decl in declarations:
        if not decl:
            continue
        sid = decl.get("submissionId") or decl.get("submission_id") or "?"
        name = str(decl.get("paradigm") or "backprop").strip().lower()
        if not name:
            name = "backprop"
        paradigms.setdefault(name, []).append(sid)
        try:
            steps = int(decl.get("dilocoInner") or 0)
        except (TypeError, ValueError):
            problems.append(f"{sid}: unreadable dilocoInner declaration")
            continue
        inner.setdefault(steps, []).append(sid)
    if len(paradigms) > 1:
        parts = "; ".join(f"{n}: {', '.join(sorted(set(v)))}" for n, v in sorted(paradigms.items()))
        problems.append(f"peers disagree about the training paradigm: {parts}")
    if len(inner) > 1:
        parts = "; ".join(f"inner={n}: {', '.join(sorted(set(v)))}" for n, v in sorted(inner.items()))
        problems.append(f"peers disagree about DiLoCo inner steps: {parts}")
    return {"problems": problems,
            "paradigms": {n: sorted(set(v)) for n, v in paradigms.items()},
            "innerSteps": {n: sorted(set(v)) for n, v in inner.items()},
            # Machine-readable: nothing here proves the training rule. A caller
            # that gates money, reputation or a round's merge on this would be
            # trusting a string a peer chose for itself.
            "verified": False,
            "verification": ("declarations agree; the wire is one flat vector, so the "
                             "server cannot tell which rule produced it")}
