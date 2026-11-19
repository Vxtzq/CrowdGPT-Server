"""Periodic full-model anchoring: the aggregator's own step on the true loss.

Why this exists (measured, not assumed)
---------------------------------------
A MOSAIC round trains each slice against a *local* objective: a non-terminal
slice minimises the loss of its own SOLO head, never the terminal loss. That is
the greedy gap, and it does not respond to any local hyper-parameter -- seven
candidates were tried and refuted at 65 M (head lr x3/x10/x30, an input-skip
head, head recalibration, overlapping slices at the cut, a decaying outer rate,
fresh local batches, a wider head). Raising the local step size, adding slices
past three, and doubling the data all fail the same way.

There is exactly one participant that holds the whole model and can compute the
true terminal loss: the aggregator. It also already holds a held-out corpus,
because the objective cross-check needs one. So let it take one ordinary
gradient step every N rounds.

Measured at 65 M parameters, 488 rounds (2 M round tokens), local configuration
K=2 / H=5 / inner 1e-3 / outer 0.15 / share_trunk:

    no anchor          5.8121-5.8516 (five-run spread 0.0395)
    anchor every 50    5.8240
    anchor every 10    5.7822   (anchor lr 1e-3: 5.7741)
    anchor every 3     5.7218
    anchor every 2     5.6732
    anchor every 1     5.5854   <-- better than backprop at 2 M tokens (5.6340)

and at 4 M round tokens, anchor every 2: 5.3642 vs 5.6188 without, i.e. within
0.045 of backprop at 4 M (5.3192) instead of 0.30 behind.

What it costs, honestly: one anchor step is exactly one backprop step (one batch,
one update of every parameter). At equal *total tokens* backprop is still ahead
(5.6732 vs 5.5582 at 2.5 M; 5.5854 vs 5.4341 at 3 M; 5.3642 vs 5.0831 at 6 M).
The case for anchoring rests on the budget this system actually pays: a round is
the unit of volunteer wall-clock, and at 488 rounds the anchored configuration is
5.5854 against backprop's 5.8370/5.8440 at the same tokens per round. The anchor
spends the *aggregator's* compute and its own corpus, never volunteer bandwidth.

This module is deliberately standard-library only. The step itself is supplied by
the deployment (``set_anchor_step(fn)``), because only the deployment knows where
its data and optimizer live; the cadence, the env override and the "configured
but no function" warning live here so they can be tested without torch.
"""
from __future__ import annotations

import math

# Which completed step triggers an anchor: every N steps, so N=1 anchors after
# every round, N=0 disables it. Clamped to a sane upper bound so a typo cannot
# silently mean "never" while appearing configured.
MAX_EVERY = 10_000


def resolve_every(raw, *, default: int = 0) -> int:
    """Parse an anchor cadence from an env var, falling back to ``default``.

    ``None`` / empty / non-numeric / NaN fall back rather than raising: a bad env
    value must not take the coordinator down, and the fallback is logged by the
    caller. Negative values mean "off" (0), which is also the shipped default.
    """
    if raw is None:
        return max(0, min(int(default or 0), MAX_EVERY))
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return max(0, min(int(default or 0), MAX_EVERY))
    if not math.isfinite(value):
        return max(0, min(int(default or 0), MAX_EVERY))
    return max(0, min(int(value), MAX_EVERY))


def should_anchor(completed_step: int, every: int) -> bool:
    """True when the round that just finished is an anchor round.

    ``completed_step`` is the 1-based index of the round that just completed, so
    ``every=2`` anchors after rounds 2, 4, 6, ... and ``every=0`` never does.
    """
    every = resolve_every(every)
    if every <= 0:
        return False
    return int(completed_step) > 0 and int(completed_step) % every == 0


def anchor_state(*, every: int, has_step: bool, rounds: int = 0,
                 last_loss: float | None = None) -> dict:
    """Machine-readable state for ``/fl/round_status`` and the pool log.

    ``level`` is ``ok`` when a cadence and a step function are both present,
    ``disabled`` when the cadence is 0, and ``misconfigured`` when a cadence is
    configured but no step function was registered -- which would otherwise look
    exactly like a working anchor that never fires.
    """
    every = resolve_every(every)
    if every <= 0:
        level = "disabled"
        warning = None
    elif not has_step:
        level = "misconfigured"
        warning = ("CROWDGPT_ANCHOR_EVERY is set but no anchor step function is registered: "
                   "the greedy gap stays open and nothing will be measured")
    else:
        level = "ok"
        warning = None
    return {"enabled": bool(every > 0 and has_step), "every": every, "level": level,
            "rounds": int(rounds), "lastLoss": last_loss, "warning": warning}
