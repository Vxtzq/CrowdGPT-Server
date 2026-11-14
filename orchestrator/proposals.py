"""Proposal and vote surface for "what improves next" (PM-4).

Voting weight is verified tokens from accepted submissions, so a fresh account
with no stored work has no say. One actor has at most one vote per proposal;
voting again replaces the weight rather than adding a second ballot.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from .leaderboard import LeaderboardService
from .state import StateStore


class ProposalService:
    def __init__(self, store: StateStore):
        self.store = store
        self.leaderboard = LeaderboardService(store)

    def create(self, title: str, body: str, author: str) -> dict:
        title = str(title).strip()
        if not title or len(title) > 200:
            raise ValueError("title must be 1..200 characters")
        if len(str(body)) > 4000:
            raise ValueError("body is too long")
        pid = uuid.uuid4().hex
        now = time.time()
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO proposals(proposal_id,title,body,author,status,created_at) "
                "VALUES(?,?,?,?,?,?)", (pid, title, str(body), str(author), "open", now))
        return self.get(pid)

    def get(self, proposal_id: str) -> Optional[dict]:
        row = self.store.query_one("SELECT * FROM proposals WHERE proposal_id=?",
                                   (str(proposal_id),))
        if not row: return None
        return {"proposalId": row["proposal_id"], "title": row["title"],
                "body": row["body"], "author": row["author"],
                "status": row["status"], "createdAt": row["created_at"],
                "votes": self.tally(str(proposal_id))}

    def list(self, limit: int = 100) -> list[dict]:
        rows = self.store.query(
            "SELECT proposal_id FROM proposals ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),))
        return [self.get(r["proposal_id"]) for r in rows]

    def vote(self, proposal_id: str, actor: str, weight: int) -> dict:
        if self.get(proposal_id) is None:
            raise KeyError("proposal not found")
        weight = int(weight)
        if weight <= 0:
            raise PermissionError("verified_contributor_required")
        now = time.time()
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO votes(proposal_id,actor,weight,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(proposal_id,actor) DO UPDATE SET weight=excluded.weight, "
                "created_at=excluded.created_at",
                (str(proposal_id), str(actor), weight, now))
        return self.tally(str(proposal_id))

    def vote_with_ledger(self, proposal_id: str, actor: str) -> dict:
        return self.vote(proposal_id, actor, self.leaderboard.weight_for(actor))

    def tally(self, proposal_id: str) -> dict:
        rows = self.store.query(
            "SELECT actor, weight FROM votes WHERE proposal_id=?", (str(proposal_id),))
        return {"count": len(rows), "weight": sum(int(r["weight"]) for r in rows),
                "voters": sorted(r["actor"] for r in rows)}


__all__ = ["ProposalService"]
