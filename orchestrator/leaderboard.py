"""Leaderboard derived from stored submissions, not self-reported profile totals.

A submission exists only after the coordinator accepted a finalized delta with
a content hash. Duplicate content is one row, so a client cannot inflate the
ledger by replaying the same bytes under a new upload id.
"""
from __future__ import annotations

from typing import Optional

from .state import StateStore


class LeaderboardService:
    def __init__(self, store: StateStore):
        self.store = store

    def verified(self, limit: int = 100) -> dict:
        rows = self.store.query(
            "SELECT COALESCE(user_uuid, 'client:' || client_key) AS actor, "
            "CASE WHEN user_uuid IS NULL THEN 'anonymous' ELSE 'account' END AS kind, "
            "COUNT(*) AS submissions, COALESCE(SUM(tokens), 0) AS tokens, "
            "COUNT(DISTINCT round) AS rounds, MIN(round) AS first_round, "
            "MAX(round) AS last_round "
            "FROM submissions WHERE status='accepted' "
            "GROUP BY actor, kind ORDER BY tokens DESC, submissions DESC LIMIT ?",
            (max(1, int(limit)),))
        entries = [{"actor": r["actor"], "identity": r["kind"],
                    "submissions": int(r["submissions"]), "tokens": int(r["tokens"]),
                    "rounds": int(r["rounds"]), "firstRound": int(r["first_round"]),
                    "lastRound": int(r["last_round"])} for r in rows]
        total = sum(e["tokens"] for e in entries)
        return {"source": "verified", "entries": entries,
                "totals": {"actors": len(entries), "tokens": total,
                           "submissions": sum(e["submissions"] for e in entries)}}

    def weight_for(self, actor: str) -> int:
        row = self.store.query_one(
            "SELECT COALESCE(SUM(tokens), 0) AS tokens FROM submissions "
            "WHERE status='accepted' AND COALESCE(user_uuid, 'client:' || client_key)=?",
            (str(actor),))
        return int(row["tokens"]) if row else 0

    def legacy(self, entries: list[dict]) -> dict:
        """Wrap the old profile-total list so its provenance is explicit."""
        return {"source": "reported", "entries": entries,
                "note": "self-reported profile totals; not verified against submissions"}


__all__ = ["LeaderboardService"]
