"""Cohort policy for matchmaking: target size, and the rescue seat.

The coordinator used to gate on a fixed cohort of three, which was fine while
several volunteers were always online and painful the moment one was not: a
peer could sit in `Pool matchmaking 2 (waiting for peers)` for hours while the
work it could already do went undone.

This module holds the arithmetic of the replacement, kept separate and
stdlib-only so both the coordinator and the standard-library-only CI job can
test it without importing FastAPI or Torch:

* the cohort **target** is `pool_size` (two by default);
* a pool may grow to a hard cap of `pool_size + overflow_max`;
* a seat above the target is a **rescue seat**: it only opens for a peer that
  has itself been waiting at least `starved_admit_s`, so a fresh arrival can
  never pad an existing pair into a trio just by being fast;
* activation is allowed one seat past the target *only while* a compatible
  starved peer is actually waiting, and never past `forming_timeout_s` — a
  rescue that nobody takes must not hold the round open forever.

Everything here is pure arithmetic on numbers the coordinator passes in, so the
behaviour is testable without a server, a socket or a clock.
"""

from __future__ import annotations

MAX_OVERFLOW = 8          # nobody needs more than a handful of rescue seats
MAX_STARVED_S = 24 * 60 * 60


def resolve_overflow_max(raw) -> int:
    """Parse `CROWDGPT_POOL_OVERFLOW_MAX`. Unset or junk means 1 seat."""
    default = 1
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    if value < 0:
        return 0
    return min(value, MAX_OVERFLOW)


def resolve_starved_s(raw) -> int:
    """Parse `CROWDGPT_STARVED_ADMIT_S`. Zero is legal: rescue immediately."""
    default = 600
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    if value < 0:
        return 0
    return min(value, MAX_STARVED_S)


def hard_cap(pool_size: int, overflow_max: int) -> int:
    """Largest cohort a single pool may reach."""
    return max(0, int(pool_size)) + max(0, int(overflow_max))


def admits(n_pool: int, waited_s: float, *, pool_size: int, overflow_max: int,
           starved_s: int) -> bool:
    """May one more peer join a pool that already holds `n_pool` clients?

    Below the target: yes, as always. Between the target and the hard cap: only
    if *this* peer has been waiting long enough, which is what stops a pool from
    being padded by whoever happens to poll next.
    """
    n = int(n_pool)
    if n < int(pool_size):
        return True
    if n >= hard_cap(pool_size, overflow_max):
        return False
    return float(waited_s) >= float(max(0, int(starved_s)))


def activation_target(*, pool_size: int, overflow_max: int, starved_waiting: bool,
                      forming_age_s: float, forming_timeout_s: float) -> int:
    """How many clients a pool needs before the round may start.

    Always the target. It used to rise by one while a compatible starved peer was
    waiting, so that peer's next poll could land as a rescue seat; in production
    that stalled every full pair at 2/2 for up to `forming_timeout_s` (30 min)
    whenever one straggler was waiting, which costs the whole cohort more than the
    straggler waiting for the next pool. A complete pool never waits. The rescue
    seat still exists for pools that are forming anyway -- see `admits()`.
    """
    return max(0, int(pool_size))


def describe(*, pool_size: int, overflow_max: int, starved_s: int) -> dict:
    """One-line policy summary for the startup log and `/fl/round_status`."""
    return {
        "poolSize": int(pool_size),
        "overflowMax": int(overflow_max),
        "hardCap": hard_cap(pool_size, overflow_max),
        "starvedAdmitSeconds": int(starved_s),
    }
