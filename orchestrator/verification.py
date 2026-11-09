"""Verification verdicts attached to one submission."""
from __future__ import annotations

import json
import time
from typing import Optional

from .domain import VerificationView
from .state import StateStore


class VerificationService:
    def __init__(self, store: StateStore):
        self.store = store

    def record(self, submission_id: str, verdict: str, reason: Optional[str] = None,
               evidence: Optional[dict] = None) -> VerificationView:
        if verdict not in ("accepted", "rejected", "flagged", "unverified"):
            raise ValueError(f"invalid verdict: {verdict}")
        now = time.time()
        payload = json.dumps(evidence or {}, sort_keys=True, separators=(",", ":"))
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO verifications(submission_id,verdict,reason,evidence,created_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(submission_id) DO UPDATE SET "
                "verdict=excluded.verdict, reason=excluded.reason, evidence=excluded.evidence, "
                "created_at=excluded.created_at",
                (submission_id, verdict, reason, payload, now),
            )
            conn.execute(
                "INSERT INTO events(ts,kind,payload) VALUES(?,?,?)",
                (now, "verification", json.dumps({"submission": submission_id,
                                                  "verdict": verdict}, sort_keys=True)),
            )
        return VerificationView(submission_id, verdict, reason, evidence or {}, now)

    def get(self, submission_id: str) -> Optional[VerificationView]:
        row = self.store.query_one(
            "SELECT * FROM verifications WHERE submission_id=?", (submission_id,)
        )
        return _verification(row) if row else None

    def all(self) -> list[VerificationView]:
        return [_verification(r) for r in self.store.query(
            "SELECT * FROM verifications ORDER BY created_at")]


def _verification(row) -> VerificationView:
    try:
        evidence = json.loads(row["evidence"] or "{}")
    except ValueError:
        evidence = {}
    return VerificationView(row["submission_id"], row["verdict"], row["reason"],
                            evidence, row["created_at"])


__all__ = ["VerificationService"]
