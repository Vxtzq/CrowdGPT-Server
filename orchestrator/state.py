"""Durable orchestrator state.

One SQLite database, WAL mode, no ORM.  Every write goes through
``StateStore.transaction`` so the services in this package never see a
half-written round, pool, or ticket.

The schema is intentionally small: the orchestrator owns scheduling metadata,
not bulk model bytes.  Delta paths in ``submissions`` point at files that live
outside SQLite.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round INTEGER PRIMARY KEY,
    state TEXT NOT NULL,
    started_at REAL NOT NULL,
    duration_hours REAL NOT NULL,
    last_aggregation_at REAL NOT NULL DEFAULT 0,
    force_end_at REAL,
    global_step INTEGER NOT NULL DEFAULT 0,
    participating TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pools (
    pool_id TEXT PRIMARY KEY,
    round INTEGER NOT NULL,
    status TEXT NOT NULL,
    hardware_sig TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    last_activity REAL NOT NULL,
    activated_at REAL,
    chunk_idx INTEGER,
    task_id TEXT,
    target_seed INTEGER,
    target_autocast TEXT,
    target_optimizer TEXT,
    target_steps INTEGER,
    power INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pool_members (
    pool_id TEXT NOT NULL,
    client_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    joined_at REAL NOT NULL,
    PRIMARY KEY (pool_id, client_key)
);
CREATE INDEX IF NOT EXISTS idx_pool_members_client ON pool_members(client_key);
CREATE TABLE IF NOT EXISTS tickets (
    client_key TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL UNIQUE,
    round INTEGER NOT NULL,
    pool_id TEXT,
    task_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    step_budget INTEGER NOT NULL,
    token_budget INTEGER NOT NULL,
    nonce TEXT NOT NULL,
    base_weights TEXT,
    signature TEXT,
    public_key TEXT,
    consumed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tickets_round ON tickets(round);
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    ticket_id TEXT,
    round INTEGER NOT NULL,
    pool_id TEXT,
    client_key TEXT NOT NULL,
    user_uuid TEXT,
    content_sha256 TEXT NOT NULL UNIQUE,
    delta_path TEXT,
    tokens INTEGER NOT NULL DEFAULT 0,
    steps INTEGER NOT NULL DEFAULT 0,
    loss_reported TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    finalized_at REAL
);
CREATE INDEX IF NOT EXISTS idx_submissions_round ON submissions(round);
CREATE INDEX IF NOT EXISTS idx_submissions_pool ON submissions(pool_id);
CREATE TABLE IF NOT EXISTS uploads (
    upload_id TEXT PRIMARY KEY,
    ticket_id TEXT,
    client_key TEXT NOT NULL,
    expected_bytes INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    content_sha256 TEXT,
    state TEXT NOT NULL,
    received_chunks TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    submission_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    reason TEXT,
    evidence TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS publications (
    round INTEGER PRIMARY KEY,
    record TEXT NOT NULL,
    published_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS client_keys (
    client_key TEXT PRIMARY KEY,
    pubkey TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    client_key TEXT NOT NULL,
    round INTEGER NOT NULL,
    payload TEXT NOT NULL,
    signature TEXT NOT NULL,
    pubkey TEXT NOT NULL,
    verdict TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_receipts_round ON receipts(round);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT NOT NULL,
    pubkey TEXT NOT NULL,
    record TEXT NOT NULL,
    valid_from REAL NOT NULL,
    valid_until REAL,
    status TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (node_id, pubkey)
);
CREATE INDEX IF NOT EXISTS idx_nodes_active ON nodes(status, valid_until);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS votes (
    proposal_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    weight INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (proposal_id, actor)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, ts);
"""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class StateStore:
    """Thread-safe SQLite state store in WAL mode."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=10.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(SCHEMA_VERSION),),
            )

    def _migrate(self) -> None:
        """Add columns introduced after the first deployment."""
        wanted = {
            "tickets": {"base_weights": "TEXT", "signature": "TEXT", "public_key": "TEXT"},
            "uploads": {"result": "TEXT"},
            "submissions": {"user_uuid": "TEXT"},
        }
        for table, columns in wanted.items():
            have = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def event(self, kind: str, payload: Optional[dict] = None) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO events(ts, kind, payload) VALUES(?,?,?)",
                (time.time(), kind, _json(payload or {})),
            )
            return int(cur.lastrowid)

    def events(self, kind: Optional[str] = None, limit: int = 100) -> list[dict]:
        if kind:
            rows = self.query(
                "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [
            {"id": r["id"], "ts": r["ts"], "kind": r["kind"],
             "payload": json.loads(r["payload"])}
            for r in rows
        ]

    def schema_version(self) -> int:
        row = self.query_one("SELECT value FROM schema_meta WHERE key='schema_version'")
        return int(row["value"]) if row else 0


__all__ = ["SCHEMA_VERSION", "StateStore"]
