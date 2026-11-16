"""Upload bookkeeping and idempotent submission registration."""
from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from .domain import SubmissionView
from .state import StateStore


class SubmissionService:
    def __init__(self, store: StateStore, upload_ttl_s: float = 6 * 3600):
        self.store = store
        self.upload_ttl_s = float(upload_ttl_s)

    def begin_upload(self, client_key: str, *, expected_bytes: int, chunk_size: int,
                     chunks: int, ticket_id: Optional[str] = None,
                     content_sha256: Optional[str] = None,
                     ttl_s: Optional[float] = None,
                     upload_id: Optional[str] = None) -> str:
        upload_id = upload_id or uuid.uuid4().hex
        now = time.time()
        ttl = self.upload_ttl_s if ttl_s is None else float(ttl_s)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO uploads(upload_id,ticket_id,client_key,expected_bytes,chunk_size,"
                "chunks,content_sha256,state,received_chunks,created_at,expires_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (upload_id, ticket_id, client_key, int(expected_bytes), int(chunk_size),
                 int(chunks), content_sha256, "open", "[]", now, now + max(1.0, ttl), now),
            )
        return upload_id

    def get_upload(self, upload_id: str) -> Optional[dict]:
        row = self.store.query_one("SELECT * FROM uploads WHERE upload_id=?", (upload_id,))
        if not row:
            return None
        data = dict(row)
        data["received_chunks"] = json.loads(data.get("received_chunks") or "[]")
        return data

    def mark_chunk(self, upload_id: str, index: int, *, size: int = 0) -> dict:
        now = time.time()
        row = self.store.query_one("SELECT * FROM uploads WHERE upload_id=?", (upload_id,))
        if not row:
            raise KeyError("unknown upload")
        if row["state"] != "open":
            raise ValueError("upload not open")
        if index < 0 or index >= row["chunks"]:
            raise ValueError("chunk out of range")
        received = set(json.loads(row["received_chunks"] or "[]"))
        received.add(int(index))
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE uploads SET received_chunks=?, updated_at=? WHERE upload_id=?",
                (json.dumps(sorted(received)), now, upload_id),
            )
        return {"index": int(index), "size": int(size), "receivedChunks": len(received),
                "totalChunks": row["chunks"]}

    def finalize_upload(self, upload_id: str) -> dict:
        now = time.time()
        row = self.store.query_one("SELECT * FROM uploads WHERE upload_id=?", (upload_id,))
        if not row:
            raise KeyError("unknown upload")
        if row["state"] != "open":
            raise ValueError("upload not open")
        if row["expires_at"] < now:
            raise ValueError("upload expired")
        received = json.loads(row["received_chunks"] or "[]")
        if len(received) != row["chunks"]:
            raise ValueError("missing chunks")
        with self.store.transaction() as conn:
            conn.execute("UPDATE uploads SET state='finalized', updated_at=? WHERE upload_id=?",
                         (now, upload_id))
        data = dict(row)
        data["received_chunks"] = received
        return data

    def complete_upload(self, upload_id: str, result: dict) -> None:
        """Persist the response body so a retried finalize can replay it."""
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE uploads SET state='completed', result=?, updated_at=? WHERE upload_id=?",
                (json.dumps(result, sort_keys=True, separators=(",", ":")), time.time(), upload_id),
            )

    def get_upload_result(self, upload_id: str) -> Optional[dict]:
        row = self.store.query_one(
            "SELECT state,result FROM uploads WHERE upload_id=?", (upload_id,)
        )
        if not row or row["state"] != "completed" or not row["result"]:
            return None
        try:
            return json.loads(row["result"])
        except ValueError:
            return None

    def record_submission(self, *, client_key: str, round_no: int,
                          content_sha256: str, status: str = "accepted",
                          ticket_id: Optional[str] = None, pool_id: Optional[str] = None,
                          delta_path: Optional[str] = None, tokens: int = 0,
                          steps: int = 0, loss_reported: Optional[str] = None,
                          user_uuid: Optional[str] = None) -> tuple[SubmissionView, bool]:
        """Register one finalized delta.  The content hash is the idempotency key.

        Returns ``(submission, created)``.  A retry with the same content gets
        the original row and ``created=False``.
        """
        existing = self.store.query_one(
            "SELECT * FROM submissions WHERE content_sha256=?", (content_sha256,)
        )
        if existing:
            return _submission(existing), False
        now = time.time()
        submission_id = uuid.uuid4().hex
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO submissions(submission_id,ticket_id,round,pool_id,client_key,"
                "user_uuid,content_sha256,delta_path,tokens,steps,loss_reported,status,"
                "created_at,finalized_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (submission_id, ticket_id, int(round_no), pool_id, client_key,
                 user_uuid, content_sha256, delta_path, int(tokens), int(steps),
                 loss_reported, status, now, now),
            )
        row = self.store.query_one("SELECT * FROM submissions WHERE submission_id=?", (submission_id,))
        return _submission(row), True

    def load_submissions(self, *, round_no: Optional[int] = None,
                         pool_id: Optional[str] = None) -> list[SubmissionView]:
        where, params = [], []
        if round_no is not None:
            where.append("round=?")
            params.append(int(round_no))
        if pool_id is not None:
            where.append("pool_id=?")
            params.append(pool_id)
        sql = "SELECT * FROM submissions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at"
        return [_submission(r) for r in self.store.query(sql, tuple(params))]

    def purge_expired_uploads(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        with self.store.transaction() as conn:
            cur = conn.execute("DELETE FROM uploads WHERE expires_at < ? AND state='open'", (now,))
            return int(cur.rowcount)


def _submission(row) -> SubmissionView:
    return SubmissionView(
        submission_id=row["submission_id"], client_key=row["client_key"], round=row["round"],
        content_sha256=row["content_sha256"], user_uuid=row["user_uuid"] if "user_uuid" in row.keys() else None,
        status=row["status"], created_at=row["created_at"],
        finalized_at=row["finalized_at"], ticket_id=row["ticket_id"], pool_id=row["pool_id"],
        delta_path=row["delta_path"], tokens=row["tokens"], steps=row["steps"],
        loss_reported=row["loss_reported"],
    )


__all__ = ["SubmissionService"]
