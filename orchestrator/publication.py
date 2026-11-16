"""Append-only round publications.

The SQLite row is the source of truth; the JSONL file is the public mirror that
a static site or another orchestrator can consume without database access.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .state import StateStore


class PublicationService:
    def __init__(self, store: StateStore, jsonl_path: Optional[str | Path] = None):
        self.store = store
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def publish_round(self, round_no: int, record: dict) -> dict:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        now = time.time()
        created = False
        with self.store.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO publications(round,record,published_at) VALUES(?,?,?) "
                "ON CONFLICT(round) DO NOTHING",
                (int(round_no), encoded, now),
            )
            created = cur.rowcount == 1
            if created:
                conn.execute(
                    "INSERT INTO events(ts,kind,payload) VALUES(?,?,?)",
                    (now, "round_published", json.dumps({"round": int(round_no)}, sort_keys=True)),
                )
        row = self.store.query_one("SELECT * FROM publications WHERE round=?", (int(round_no),))
        result = {"round": row["round"], "record": json.loads(row["record"]),
                  "published_at": row["published_at"]}
        if self.jsonl_path and created:
            line = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            fd = os.open(self.jsonl_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        return result

    def get_round(self, round_no: int) -> Optional[dict]:
        row = self.store.query_one("SELECT * FROM publications WHERE round=?", (int(round_no),))
        if not row:
            return None
        return {"round": row["round"], "record": json.loads(row["record"]),
                "published_at": row["published_at"]}

    def all_rounds(self) -> list[dict]:
        return [{"round": r["round"], "record": json.loads(r["record"]),
                 "published_at": r["published_at"]}
                for r in self.store.query("SELECT * FROM publications ORDER BY round")]


__all__ = ["PublicationService"]
