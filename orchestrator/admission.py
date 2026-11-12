"""Ticket admission: who may train, on which task, for how long."""
from __future__ import annotations

import json
import secrets
import time
import uuid
from typing import Optional

from .domain import TicketView
from .identity import NodeIdentity
from .state import StateStore


class AdmissionService:
    def __init__(self, store: StateStore, ticket_ttl_s: float = 4 * 3600,
                 step_budget: int = 500, token_budget: int = 10_000_000,
                 identity: Optional[NodeIdentity] = None):
        self.store = store
        self.ticket_ttl_s = float(ticket_ttl_s)
        self.step_budget = int(step_budget)
        self.token_budget = int(token_budget)
        self.identity = identity or NodeIdentity.generate()

    def issue(self, client_key: str, task_id: str, round_no: int,
              pool_id: Optional[str] = None,
              step_budget: Optional[int] = None,
              token_budget: Optional[int] = None,
              ttl_s: Optional[float] = None,
              base_weights: Optional[str] = None) -> TicketView:
        now = time.time()
        existing = self.get(client_key)
        if (existing and existing.task_id == task_id and not existing.consumed
                and existing.expires_at > now):
            return existing
        ttl = self.ticket_ttl_s if ttl_s is None else float(ttl_s)
        ticket = TicketView(
            ticket_id=uuid.uuid4().hex,
            client_key=client_key,
            round=int(round_no),
            task_id=str(task_id),
            created_at=now,
            expires_at=now + max(0.001, ttl),
            step_budget=int(self.step_budget if step_budget is None else step_budget),
            token_budget=int(self.token_budget if token_budget is None else token_budget),
            nonce=secrets.token_hex(16),
            pool_id=pool_id,
            base_weights=base_weights,
        )
        signature = self.identity.sign(ticket.signing_payload())
        ticket = TicketView(**{**ticket.__dict__, "signature": signature,
                               "public_key": self.identity.public_key_hex})
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO tickets(client_key,ticket_id,round,pool_id,task_id,created_at,"
                "expires_at,step_budget,token_budget,nonce,base_weights,signature,public_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(client_key) DO UPDATE SET ticket_id=excluded.ticket_id,"
                "round=excluded.round,pool_id=excluded.pool_id,task_id=excluded.task_id,"
                "created_at=excluded.created_at,expires_at=excluded.expires_at,"
                "step_budget=excluded.step_budget,token_budget=excluded.token_budget,"
                "nonce=excluded.nonce,base_weights=excluded.base_weights,"
                "signature=excluded.signature,"
                "public_key=excluded.public_key,consumed_at=NULL",
                (ticket.client_key, ticket.ticket_id, ticket.round, ticket.pool_id,
                 ticket.task_id, ticket.created_at, ticket.expires_at,
                 ticket.step_budget, ticket.token_budget, ticket.nonce,
                 ticket.base_weights, ticket.signature, ticket.public_key),
            )
            conn.execute(
                "INSERT INTO events(ts,kind,payload) VALUES(?,?,?)",
                (now, "ticket_issued",
                 json.dumps({"client": client_key, "round": round_no, "ticket": ticket.ticket_id},
                            sort_keys=True, separators=(",", ":"))),
            )
        return ticket

    def get(self, client_key: str) -> Optional[TicketView]:
        row = self.store.query_one("SELECT * FROM tickets WHERE client_key=?", (client_key,))
        return _ticket(row) if row else None

    def get_by_id(self, ticket_id: str) -> Optional[TicketView]:
        row = self.store.query_one("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))
        return _ticket(row) if row else None

    def consume(self, client_key: str, task_id: Optional[str] = None) -> bool:
        now = time.time()
        row = self.store.query_one("SELECT * FROM tickets WHERE client_key=?", (client_key,))
        if not row:
            return False
        if task_id is not None and row["task_id"] != task_id:
            return False
        if row["consumed_at"] is not None:
            return False
        if row["expires_at"] < now:
            return False
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE tickets SET consumed_at=? WHERE client_key=? AND consumed_at IS NULL",
                (now, client_key),
            )
        return True

    def purge_expired(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        with self.store.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM tickets WHERE expires_at < ? AND consumed_at IS NULL", (now,)
            )
            return int(cur.rowcount)

    def verify(self, ticket: TicketView) -> bool:
        if not ticket.signature or not ticket.public_key:
            return False
        return self.identity.verify(ticket.signing_payload(), ticket.signature, ticket.public_key)

    def load_all(self, round_no: Optional[int] = None) -> dict[str, TicketView]:
        if round_no is None:
            rows = self.store.query("SELECT * FROM tickets ORDER BY created_at")
        else:
            rows = self.store.query("SELECT * FROM tickets WHERE round=? ORDER BY created_at", (round_no,))
        return {r["client_key"]: _ticket(r) for r in rows}


def _ticket(row) -> TicketView:
    return TicketView(
        ticket_id=row["ticket_id"], client_key=row["client_key"], round=row["round"],
        task_id=row["task_id"], created_at=row["created_at"], expires_at=row["expires_at"],
        step_budget=row["step_budget"], token_budget=row["token_budget"], nonce=row["nonce"],
        pool_id=row["pool_id"], consumed_at=row["consumed_at"],
        base_weights=row["base_weights"] if "base_weights" in row.keys() else None,
        signature=row["signature"] if "signature" in row.keys() else None,
        public_key=row["public_key"] if "public_key" in row.keys() else None,
    )


__all__ = ["AdmissionService"]
