# Standby coordinator

The coordinator stays centralised. A standby exists so a dead primary is a
short outage, not a lost project — **not** so strangers can serve rounds.

## What is implemented

`orchestrator/standby.py`, both on the existing SQLite state store:

| Primitive | Rule |
|---|---|
| `CoordinatorLease` | one row per slot; `acquire` takes `term = previous + 1`; a live lease held by another node refuses with `LeaseHeld`; `renew`/`release` with a stale term raise `StaleTerm` |
| `DeltaLedger` | primary key `(round_id, client_id)` plus the digest: the same bytes are a `replay`, different bytes in the same round are a `conflict` |

The monotonic term is the safety property. A promoted standby holds a term
higher than any writer that could have observed the old one, so a primary that
comes back from a pause is told `StaleTerm` instead of writing over the new
owner. Both tables are created on first use and are plain SQL, so they survive
a restart and work with one process or two.

## Promotion procedure (manual, deliberately)

1. Primary is unhealthy (process down, or `/api/stats` failing) for longer than
   the lease `ttl_s` (default 30 s).
2. Standby calls `CoordinatorLease(store).acquire(node_id, ttl_s)` and gets a
   term strictly greater than the previous one.
3. Standby serves only while `may_serve(node_id)` is true; it must call
   `renew()` on an interval shorter than the TTL.
4. Clients already point at a list of orchestrators (`CROWDGPT_ORCHESTRATORS`,
   client PR #108), so they rotate onto the new primary without a config change.
5. `release(node_id, term)` when stepping down. Releasing is one row update,
   so a rollback is instant.

There is no voting and no quorum. If both nodes believe they are primary, the
term comparison makes one of them stop; nothing is merged.

## Drill (not yet run)

N4's gate is RTO < 10 minutes with zero duplicate deltas, demonstrated by
killing the primary during a live round:

1. Start primary and standby against the same state file.
2. Primary acquires the lease and begins a round; standby polls and stays idle.
3. `kill -9` the primary mid-round.
4. Standby acquires within one TTL, resumes the round from the state store, and
   accepts the clients' deltas.
5. Re-submit one delta the standby already recorded: the ledger answers
   `replay`, and the round's accepted count does not change.
6. Restart the old primary: it must refuse to serve (`StaleTerm`), not write.

Steps 5 and 6 are covered by unit tests today (`tests/test_orchestrator_standby.py`,
18 passing). Steps 1–4 need a running server and a second host, so the drill
itself is still outstanding.

## Not in scope

- Permissionless operation, quorum, or consensus among independent operators.
- Automatic failover of the SQLite file itself. The state store has to live on
  storage both nodes can reach (shared volume or a replicated filesystem);
  that is a deployment decision, and the lease does not pretend to solve it.
