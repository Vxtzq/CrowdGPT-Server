# CrowdGPT-Server

The coordinator for [CrowdGPT](https://github.com/Vxtzq/CrowdGPT). It hands out training tasks to volunteer clients, checks and aggregates the deltas they send back, and publishes the updated weights to Hugging Face ([Vxtzq/Crowd-v1](https://huggingface.co/Vxtzq/Crowd-v1)).

The client lives in the main repository. For the model and training setup, see [ARCHITECTURE.md](https://github.com/Vxtzq/CrowdGPT/blob/main/docs/ARCHITECTURE.md).

## How a round works

1. **Matchmaking.** Clients ask for a task and are grouped into pools of 3 (`CROWDGPT_POOL_SIZE`). Pools prefer clients with the same backend, precision and optimizer, and fall back to looser matches when no strict match is forming.
2. **Task.** Every member of a pool gets the same shard, seed, batch size and step count, so honest members produce nearly identical deltas.
3. **Upload.** Clients send their delta in chunks (`/fl/submit/init`, `/chunk`, `/finalize`).
4. **Cross-check.** When the pool has all its deltas (or times out), the server compares them. Outliers are dropped and flagged; a client with 3 flags is banned (`CROWDGPT_BYZANTINE_BAN_THRESHOLD`).
5. **Aggregation.** Surviving deltas from all pools are applied to the global model at the end of the round (4 h at most). Submissions outside a pool go through Krum instead.
6. **Publish.** The new checkpoint is pushed to Hugging Face, where clients download it for the next round.

### How the aggregate is applied

A round must be a *descent step*, not a consensus of endpoints. Three rules produce that
(see `approach-fix/PLAN.md` in the audit sidecar):

- **Pin tokens per optimizer step; let the round size decide the step count.** `CROWDGPT_TARGET_TOKENS_PER_STEP`
  (default `131072` = 32 micro-batches of 2×2048) is v1's parameterisation: the inner batch is
  fixed, so `accum` is 32 on every card and the optimizer-step count falls out of the round
  (`500` micro-steps ⇒ 16 steps, `2000` ⇒ 63). Same batch, same step size, same trajectory as
  v1 — but derived and reported instead of hardcoded, which is what lets A3 normalise a client
  whose round ended up a different length. `CROWDGPT_TARGET_OPT_STEPS` remains as a step-mode
  fallback (dictate the steps, derive `accum`).
- **The round size is the training-efficiency lever — and it costs uploads.** With the step count
  derived, shortening the round shrinks `accum` and buys more optimizer updates per token: 125
  micro-steps (`accum=8`) measured −0.17 held-out BPB at equal tokens versus 500 micro-steps, and
  reached the same loss in ~4× fewer tokens. That trade multiplies uploads/merges by 4, so the
  default stays at v1's pacing (`ROUND_BASE_STEPS=500` × `VRAM/8`, capped at 2500); lower
  `CROWDGPT_ROUND_BASE_STEPS` (e.g. `125`) and `CROWDGPT_ROUND_HW_MULT_MAX` (e.g. `1`) deliberately
  if you'd rather spend bandwidth than tokens.
- **Normalise a delta by the steps it took, against *its own pool's* round.** Clients report
  `optimizerSteps` and `accum`; the server scales each delta by `clamp(n_ref / steps, 0.5, 2.0)`
  with `n_ref = ceil(target_steps / accum)` for that pool (falling back to the count its survivors
  agree on). One global nominal reference would rescale an entire healthy pool on a big card — a
  2000 micro-step round runs 63 steps, and `clamp(16/63) = 0.5` would halve its delta every round.
  Legacy deltas without `optimizerSteps` are applied unscaled.

`MAX_DELTA_APPLY_NORM` (default `12000`) is a tripwire: a delta above it is clipped and a
warning is logged. It is not a routine nonlinearity.

## Requirements

- Python 3 and a machine with enough disk for checkpoints and pending deltas. The server refuses new tasks below 10 GB free and new uploads below 2 GB.
- `fastapi`, `uvicorn`, `torch`, `numpy`, `requests`, `slowapi`, `huggingface_hub`, `safetensors`. `psutil` is optional.

## Running

```bash
pip install -r requirements.txt

export CROWDGPT_ADMIN_SECRET="$(openssl rand -hex 32)"
export HF_TOKEN="hf_..."        # without it, Hugging Face uploads are disabled

python server.py                 # listens on 0.0.0.0:5006
```

Or run it in a container:

```bash
cp .env.example .env            # set CROWDGPT_ADMIN_SECRET and CROWDGPT_PUBLIC_URL
docker compose up -d --build
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for systemd, persistence and reverse-proxy notes.

Logs go to stdout and to `server.log` (rotated at 50 MB, 5 files). Under Docker, configure rotation on the JSON log driver.

If the server sits behind a reverse proxy or tunnel, set `CROWDGPT_TRUSTED_PROXIES` to the proxy's address range. Otherwise every client appears to come from the proxy, which breaks per-IP limits and bans.

## Configuration

All settings are environment variables. Defaults are the values used when the variable is unset.

**Rounds and pools**

| Variable | Default | Description |
|---|---|---|
| `CROWDGPT_POOL_SIZE` | `3` | Clients per pool |
| `CROWDGPT_POOL_TIMEOUT_S` | `14400` | A training pool completes with whatever deltas it has after this |
| `CROWDGPT_FORMING_TIMEOUT_S` | `1800` | How long a pool waits to fill |
| `CROWDGPT_TIER0` / `TIER1` / `TIER2` | `1` | Matching levels: backend + precision + optimizer / backend + precision / backend only |
| `CROWDGPT_TARGET_BATCH_SIZE` | `2` | Batch size given to pool members |
| `CROWDGPT_TARGET_ACCUM_STEPS` | `32` | Gradient accumulation steps; sent to clients as `0` whenever a step or token budget is live, so the client derives it |
| `CROWDGPT_TARGET_LOCAL_STEPS` | `500` | Micro-steps per task (fallback; the live budget is `ROUND_BASE_STEPS` × hardware factor) |
| `CROWDGPT_ROUND_BASE_STEPS` | `500` | Round budget before the hardware factor (v1's pacing) |
| `CROWDGPT_ROUND_HW_MULT_MAX` | `5.0` | Upper clamp on that factor (`500×5 = 2500` = v1's cap). Set `1.0` to stop big cards getting longer rounds |
| `CROWDGPT_TARGET_TOKENS_PER_STEP` | `131072` | Tokens per optimizer step (v1's batch); the step count is derived from the round budget |
| `CROWDGPT_TARGET_OPT_STEPS` | `0` | Step-mode fallback: dictate the optimizer steps per round and derive `accum`. `0` = tokens/step decides |
| `CROWDGPT_TARGET_AUTOCAST` | `any` | Force a precision for pools (`bf16`, `fp32`, `any`) |
| `CROWDGPT_INIT_MODE` | `ln_identity` | Fresh-run init: `ln_identity` = N(0,0.02) + LayerNorm 1/0 (`‖W‖ ≈ 581.7`), `flat` = N(0,0.02) for every parameter, LayerNorm included (`‖W‖ ≈ 512.9`, what v1 ran with) |
| `CROWDGPT_MAX_DELTA_APPLY_NORM` | `12000` | Per-delta clip at aggregation time. A tripwire that logs a warning, not a normaliser |
| `CROWDGPT_MAX_CHECKPOINTS` | `5` | Local checkpoints to keep |

**Verification**

| Variable | Default | Description |
|---|---|---|
| `CROWDGPT_CROSS_CHECK_ENABLED` | `1` | Compare deltas within a pool |
| `CROWDGPT_CROSS_CHECK_THRESHOLD` | `2.0` | Distance ratio above which a delta is an outlier |
| `CROWDGPT_BYZANTINE_BAN_THRESHOLD` | `3` | Flags before a ban |
| `CROWDGPT_KRUM_F` | `1` | Byzantine clients tolerated by Krum |
| `CROWDGPT_LOSS_CHECK_MODE` | `off` | External loss check: `off`, `log` or `enforce` |
| `CROWDGPT_LOSS_CHECK_URL` / `_TOKEN` | | Loss check service and its shared secret |
| `CROWDGPT_LOSS_FAIL_OPEN` | `1` | Accept deltas when the loss check is unreachable |
| `CROWDGPT_REQUIRE_TASK_TICKET` | `0` | Require a task ticket on submissions |

**Limits**

| Variable | Default | Description |
|---|---|---|
| `CROWDGPT_MIN_TRAIN_TIME_S` | `900` | Minimum time between task and submission |
| `CROWDGPT_MAX_SUBMISSIONS_PER_SUBNET_ROUND` | `500` | Per-subnet cap per round |
| `CROWDGPT_MAX_CONCURRENT_UPLOADS` | `2` | Uploads processed at once |
| `CROWDGPT_BLOCK_UNVERSIONED` | `0` | Reject clients that don't send a version |

**Admin and publishing**

| Variable | Default | Description |
|---|---|---|
| `CROWDGPT_ADMIN_SECRET` | | Secret for `/api/admin/*`. Always set it |
| `HF_TOKEN` | | Hugging Face write token |
| `CROWDGPT_PUBLIC_URL` | `https://server.crowdgpt.net` | URL the server advertises to clients |

A few more tuning variables are read in `server.py` (search for `_env_int` and `_env_float`).

## API

Routes used by the client:

| Route | Purpose |
|---|---|
| `GET /fl/task` | Join matchmaking and get a task (weights inline, or metadata only) |
| `GET /fl/heartbeat` | Keep-alive; with `taskId`, reports whether the task still exists |
| `GET /fl/round_status` | Current round, timing and pool counts |
| `POST /fl/submit/init`, `/chunk`, `/finalize` | Chunked delta upload |
| `POST /auth/register`, `/auth/login`, `GET /auth/me` | Accounts |
| `GET /api/stats`, `/api/leaderboard`, `/api/model_info` | Public stats |

Routes under `/api/admin/` require the admin secret.

## Data on disk

| Path | Contents |
|---|---|
| `checkpoints/` | Global model and engram, `global_state.json`, loss check state |
| `checkpoints/pending_deltas/` | Deltas waiting for aggregation |
| `pending_uploads/` | Chunked uploads in progress (expire after 6 h) |
| `users.json` | Accounts, stats and flags |

Tasks and pools are kept in memory only: a restart drops every task in progress, and clients get a fresh one on their next request.

## Security

Please don't report vulnerabilities in public issues. Use GitHub's private vulnerability reporting for this repository.

## Vibe coding notice

I know this code is vibe-coded slop. It is the only way to maintain a constantly changing project I'm sorry about that 😭

## License

GPL-3.0. See [LICENSE](LICENSE).
