# Orchestrator modules and durable state (P1.1)

The HTTP layer in `server.py` is now a transport adapter over a small,
HTTP-free package:

| Module | Responsibility |
|---|---|
| `orchestrator/state.py` | SQLite connection, WAL mode, schema, transactions, event log |
| `orchestrator/admission.py` | issue / inspect / consume / expire tickets |
| `orchestrator/scheduling.py` | rounds, pools, membership, pool statistics |
| `orchestrator/submission.py` | upload bookkeeping and idempotent submissions |
| `orchestrator/verification.py` | one verdict per submission, with evidence |
| `orchestrator/publication.py` | append-only round publications + JSONL mirror |
| `orchestrator/service.py` | `Orchestrator` facade that wires the services |

No module imports FastAPI. `tests/test_orchestrator_modules.py` asserts that in
a subprocess (`fastapi` must not appear in `sys.modules`) and exercises every
service directly.

## Storage

Default path: `orchestrator_state.sqlite3` beside `server.py`; override with
`CROWDGPT_ORCHESTRATOR_DB`. The database runs in WAL mode with
`synchronous=NORMAL` and a 5 s busy timeout. One connection is shared through a
re-entrant lock, which is sufficient for the current single-process
orchestrator; P3 replaces it with one writer per node.

Tables: `rounds`, `pools`, `pool_members`, `tickets`, `uploads`,
`submissions`, `verifications`, `publications`, `events`, `schema_meta`.

`submissions.content_sha256` is `UNIQUE`: retrying the same content returns the
original row instead of creating a second submission. `publications.round` is a
primary key: the first published record for a round wins, and the JSONL mirror
is appended only when the row was actually inserted.

## Server integration

At import, `server.py` opens the store and mirrors the mutations that matter
after a restart:

- ticket issue/consume → `admission`
- round creation and close, force-end, participating-client set → `scheduling`
- pool create/join/activate/status/remove → `scheduling`
- accepted delta path + token/step/loss metadata → `submission`
- cross-check survivors/outliers → `verification`
- round close → `publication` (record also mirrored to
  `data/public_rounds.jsonl`)

On boot, `startup()` calls `_restore_orchestrator_state()`. A round is restored
only when its persisted state is `open` and its number is `global_step + 1`;
otherwise the server starts the next round as before. Restore rebuilds:

- `round_state` (number, start time, duration, last aggregation, participating
  clients)
- `pools`, `client_pool_map`, and their member lists
- `db["tasks"]` from each pool's task metadata
- live `task_tickets` (unconsumed and unexpired)
- each pool's `deltas` mapping for submissions whose delta file still exists

The legacy sweeping of `checkpoints/pending_deltas` is skipped when a round was
restored, so the reconstructed delta paths remain valid.

## Evidence

`cd CrowdGPT-Server && pytest tests -q` → 7 passed.

`cd Crowdgpt-ng && pytest tests/test_ng_server_harness.py -q` → 2 passed,
including a simulated restart that asserts pools, tasks, membership, and live
tickets are reconstructed rather than dropped.

The server repository has no writable fork yet, so this commit is local.

## P1.2 — signed tickets and idempotent finalize

Tickets are Ed25519-signed by a persistent node key
(`orchestrator_node_key.pem`, mode 0600, created on first boot). The signing
payload is canonical JSON with schema `ng-ticket-1` and binds:

`ticketId`, `clientKey`, `round`, `taskId`, `poolId`, `createdAt`,
`expiresAt`, `stepBudget`, `tokenBudget`, `nonce`, `baseWeights`.

`baseWeights` is `step:<global_step>` for the current round, so a ticket issued
before an aggregation is rejected with `base_weights_changed` instead of being
applied to a different model.

The signed object is included in the `/fl/task` metadata as `ticket`; the
client can ignore it today. The server validates expiry, task id, ticket id,
consumption, and base weights before accepting an upload. Rejections use
`HTTPException(403, detail={"reason": "<machine-readable>"})`.

Uploads now have durable rows from `/fl/submit/init`, per-chunk receipts, and a
stored finalize response. A retried `finalize` after the in-memory upload
context is gone returns the original body instead of `404`. Submission
registration uses `sha256(ticketId + upload checksum)` as the unique content
key, so a duplicate finalize cannot create a second delta.

Evidence:

- `pytest tests -q` in the server checkout → 8 passed.
- `pytest tests/test_ng_server_harness.py -q` → 6 passed: duplicate finalize,
  out-of-order chunks, expired ticket (`{"reason":"ticket_expired"}`), and
  finalize replay after a simulated restart.
