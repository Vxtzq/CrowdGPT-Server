# Two-peer pools, with overflow for starved peers

Status: scoping (not implemented). Target: `CrowdGPT-Server`.

## The problem this solves

`MATCHMAKING["POOL_SIZE"]` defaults to **3** (`CROWDGPT_POOL_SIZE`, floor 2). A
volunteer that arrives alone sits at `X-Status: wait` until two more peers appear,
or until its solo pool dissolves (`BF16_SOLO_DISSOLVE_S`, 30 min). On 2026-11-22 the
ROCm box sat at `Pool matchmaking 2 (waiting for peers)` for as long as it was
watched. With a pool of 2 the common case is two volunteers, and the third slot is
pure latency.

Two changes, one mechanism:

1. **Default cohort = 2.** Lower the default so a pair can train.
2. **Overflow seat for starved peers.** A peer that has waited longer than a
   threshold is admitted into a pool that is already at its target size, up to
   `POOL_SIZE + POOL_OVERFLOW_MAX`. Its work then gets validated instead of idling.

## Why it is not just "set POOL_SIZE=2"

The admission rule closes a pool at `len(pool["clients"]) >= POOL_SIZE` (server.py
~2043) and a pool of 2 activates immediately, so it is `training` — not `forming` —
within milliseconds. A starved peer therefore has no window to join. Lowering the
default alone would trade one wait (for a third peer) for another (for the next
formation).

So the overflow seat needs an explicit insertion point, and the natural one is
**dispatch**: at the moment a pool is about to activate, if a compatible peer has
been starved past the threshold, add it as the overflow seat. That cannot delay a
round and cannot create a cohort that never forms.

## Config

| env | default | meaning |
|---|---|---|
| `CROWDGPT_POOL_SIZE` | **2** (was 3) | target cohort; floor 2 stays |
| `CROWDGPT_POOL_OVERFLOW_MAX` | 1 | extra seats available to starved peers |
| `CROWDGPT_STARVED_ADMIT_S` | 600 | wait before a peer may take an overflow seat |
| `CROWDGPT_OVERFLOW_GRACE_S` | 0 | optional pause before dispatch while a starved peer exists (0 = never delay) |

`MATCHMAKING` gains the last three; `POOL_SIZE`'s default changes and the floor
check stays (`_env_int(..., floor=2)`).

## Mechanism

1. **Track starvation server-side.** `client_wait_since[client_key]` is stamped the
   first time `/fl/task` answers `wait` for that client and cleared when it is
   assigned a task. Client-declared wait times are never trusted.

2. **Rescue at dispatch.** Where a forming pool is activated, before the task is
   materialised:
   ```
   while len(pool["clients"]) < POOL_SIZE + OVERFLOW_MAX:
       peer = _starved_candidate(pool)      # wait >= STARVED_ADMIT_S, compatible hw,
                                            # not same actor, not already in a pool
       if peer is None: break
       attach(peer, pool)                   # logs an overflow admission
   ```
   Prefer peers that have waited longest. Attach at most
   `POOL_OVERFLOW_MAX` extra clients; never overflow a pool that already has one.

3. **Keep the existing guards.** `_pools_compatible(hw_sig, ...)` and
   `_pool_contains_same_actor` still decide eligibility. Overflow must not become a
   way to pack a pool with one actor's accounts — that is exactly what the objective
   gate exists to catch, and the actor check is what stops it at the door.

4. **Optional grace.** When `OVERFLOW_GRACE_S > 0` and a starved peer exists, delay
   activation by that many seconds (bounded by `FORMING_TIMEOUT_S`). Off by
   default: the dispatch-time rescue already covers the case without adding latency
   to everyone else's round.

5. **Make it visible.** `/fl/task` responses gain `X-Pool-Overflow: 1` for an
   overflow seat; the client prints one line so a volunteer can see they were
   rescued rather than matched. `pool_start`'s public event carries
   `overflow=<n>`.

## Accounting to check while implementing

Grep for `POOL_SIZE` and for `== len(pool["clients"])` and decide each site:

- **Aggregation weights** are token-weighted already, so a 3-peer pool needs no
  arithmetic change — but assert that the weights still sum to 1 with an unexpected
  cohort size.
- **Completeness / straggler logic** must accept `len(clients) <= POOL_SIZE +
  OVERFLOW_MAX` rather than equality with the target.
- **Reporting**: `cohortSize` / `maxCohortSize` in `/fl/round_status` should report
  the target and the hard cap separately, so a 3-peer round is explainable after
  the fact.

## Tests (server-side, no torch)

1. Two compatible peers form a pool and it activates at size 2.
2. A third compatible peer that has waited `< STARVED_ADMIT_S` is not rescued.
3. The same peer, once past the threshold, is attached at dispatch → cohort of 3,
   `overflow=1`, and the round proceeds.
4. A fourth peer is refused even when starved (`OVERFLOW_MAX=1`).
5. A starved peer on incompatible hardware is not rescued.
6. `_pool_contains_same_actor` still blocks a second account from one actor.
7. Weights sum to 1 for a 3-peer cohort.
8. `POOL_SIZE=2` with no starved peer never produces a 3-peer cohort, and
   `POOL_SIZE=3` still works with overflow disabled.

## What this does not do

- It does not make a lone peer train alone — the solo dissolve path is unchanged
  (30 min bf16 / 60 min fp32), and a solo round is still the fallback.
- It does not change aggregation mathematics, the objective gate, or the wire
  format.
- It does not remove the need for a third *volunteer*; it removes the need for a
  third *simultaneous* volunteer in the same compatible pool.

## Effort

Half a day for the mechanism plus tests, plus an hour of grepping the `POOL_SIZE`
call sites. Env-only rollout: setting `CROWDGPT_POOL_SIZE=3` and
`CROWDGPT_POOL_OVERFLOW_MAX=0` restores today's behaviour exactly.
