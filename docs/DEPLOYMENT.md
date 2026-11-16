# Deploying an orchestrator

## What lives where

`server.py` derives every path from its own location: the SQLite database, the
node key, checkpoints, in-progress uploads, `users.json` and `server.log` are
created next to the script. `CROWDGPT_ORCHESTRATOR_DB` and `CROWDGPT_OBJECT_DIR`
can move the database and the local object store elsewhere; nothing else is
configurable.

State that must be backed up:

| Path | Contents |
|---|---|
| `orchestrator_state.sqlite3` (+ `-wal`, `-shm`) | rounds, pools, tickets, registry |
| `orchestrator_node_key.pem` | node identity — losing it changes the node id |
| `orchestrator_object_secret` | signs object-store URLs |
| `checkpoints/` | model and engram checkpoints |
| `data/` | public round records, dataset cache, object store |
| `pending_uploads/`, `work_tmp/` | in-flight deltas (expire; safe to drop when stopped) |
| `users.json` | accounts, stats and flags |

Nothing outside those paths is written.

## Docker (recommended)

```bash
cp .env.example .env
# set CROWDGPT_ADMIN_SECRET and CROWDGPT_PUBLIC_URL
docker compose up -d --build
docker compose logs -f orchestrator
curl -fsS http://127.0.0.1:5006/api/stats >/dev/null && echo ok
```

The compose file puts the orchestrator database and the object store on the
named volume `crowdgpt-data` (`/data`). Checkpoints, `data/`, `pending_uploads/`,
`work_tmp/`, `users.json` and logs are written next to `server.py` in `/app`
and do **not** survive container recreation. To keep them, add bind mounts:

```yaml
# docker-compose.override.yml
services:
  orchestrator:
    volumes:
      - crowdgpt-data:/data
      - ./state/checkpoints:/app/checkpoints
      - ./state/data:/app/data
      - ./state/users.json:/app/users.json
```

Create the host paths and give them to UID 10001 first:

```bash
mkdir -p state/checkpoints state/data
touch state/users.json
sudo chown -R 10001:10001 state
```

The image's own `/data` is owned by UID 10001; a root-owned bind mount makes
SQLite fail with `unable to open database file`.

Put a TLS reverse proxy in front of port 5006. Set `CROWDGPT_TRUSTED_PROXIES`
to the proxy range. Do not publish 5006 directly to the internet. The image
runs as UID 10001.

## systemd (no Docker)

```bash
sudo useradd --system --home /var/lib/crowdgpt --shell /usr/sbin/nologin crowdgpt
sudo mkdir -p /opt/crowdgpt /var/lib/crowdgpt /etc/crowdgpt
sudo rsync -a --exclude .git --exclude checkpoints --exclude data \
  --exclude pending_uploads --exclude work_tmp ./ /opt/crowdgpt/
sudo python3.12 -m venv /opt/crowdgpt/.venv
sudo /opt/crowdgpt/.venv/bin/pip install -r /opt/crowdgpt/requirements.txt
sudo cp deploy/crowdgpt-orchestrator.service /etc/systemd/system/
sudo tee /etc/crowdgpt/orchestrator.env >/dev/null <<'ENV'
CROWDGPT_ADMIN_SECRET=change-me
CROWDGPT_PUBLIC_URL=https://node.example
CROWDGPT_ORCHESTRATOR_DB=/var/lib/crowdgpt/orchestrator_state.sqlite3
CROWDGPT_OBJECT_DIR=/var/lib/crowdgpt/objects
CROWDGPT_TRUSTED_PROXIES=127.0.0.1/32
ENV
sudo chown -R crowdgpt:crowdgpt /opt/crowdgpt /var/lib/crowdgpt
sudo systemctl daemon-reload
sudo systemctl enable --now crowdgpt-orchestrator
curl -fsS http://127.0.0.1:5006/api/stats >/dev/null && echo ok
```

The unit runs as `crowdgpt` with `ProtectSystem=strict`; `ReadWritePaths`
covers both `/opt/crowdgpt` (checkpoints, `users.json`, logs) and
`/var/lib/crowdgpt` (database, object store). If you move other state, extend
that list.

Upgrades: stop the service, `rsync` the new tree over `/opt/crowdgpt` (with the
same excludes), reinstall requirements, start it again. The database migrates
in place.

## Reachability

Trainers make outbound HTTPS requests only; the orchestrator never connects
back to them. The orchestrator needs one inbound HTTPS endpoint — a public IP
behind a reverse proxy, a tunnel, or a private mesh. TLS is not built in;
terminate it at Caddy/Nginx.

## Backup

With the service stopped, copy the state paths listed above:

```bash
sudo systemctl stop crowdgpt-orchestrator
sudo rsync -a --delete /var/lib/crowdgpt/ /backup/crowdgpt/var-lib/
sudo rsync -a --delete /opt/crowdgpt/checkpoints /opt/crowdgpt/data \
  /opt/crowdgpt/users.json /backup/crowdgpt/app/
sudo systemctl start crowdgpt-orchestrator
```

For Docker, back up the `crowdgpt-data` volume (and the bind mounts, if used)
the same way. Copying a live SQLite WAL database can produce an inconsistent
snapshot, so stop the process first.

The node key is identity: restoring it restores the node id.
