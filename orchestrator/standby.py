"""Standby coordination primitives: a lease, and a replay ledger.

The coordinator stays centralised; N4 only makes it *available*. Two nodes may
not both serve rounds, so a standby takes a lease before it answers, and every
accepted delta is recorded once so a promoted standby cannot apply the same
submission twice.

Both pieces are plain SQL on the existing :class:`StateStore`, so they work
whether the deployment runs one process or two, and they are unit-testable
without FastAPI, torch or a network.

Deliberately not here: quorum, voting, or permissionless admission. Promotion
is a lease with a monotonic term, not an election among strangers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .state import StateStore

LEASE_TABLE = "coordinator_lease"
DELTA_TABLE = "delta_receipts"

STANDBY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {LEASE_TABLE} (
    slot TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    term INTEGER NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
    round_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (round_id, client_id)
);
"""


def _ensure_schema(store: StateStore) -> None:
    """Create the two tables outside a transaction.

    ``executescript`` issues an implicit COMMIT, which would break a caller's
    ``StateStore.transaction()`` block; each statement is cheap and idempotent,
    so they run one at a time instead.
    """

    for statement in STANDBY_SCHEMA.split(";"):
        if statement.strip():
            store.execute(statement)


class LeaseError(RuntimeError):
    """Base class for lease refusals."""


class LeaseHeld(LeaseError):
    """Another node currently holds a live lease."""

    def __init__(self, holder: str, expires_at: float, term: int) -> None:
        super().__init__(f"lease held by {holder} until {expires_at:.0f} (term {term})")
        self.holder = holder
        self.expires_at = expires_at
        self.term = term


class StaleTerm(LeaseError):
    """The caller's term is not the current one, so it must stop serving."""


@dataclass(frozen=True)
class Lease:
    node_id: str
    term: int
    acquired_at: float
    expires_at: float

    def expired(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at


class CoordinatorLease:
    """One lease slot, one holder, strictly increasing terms.

    The term is what makes a promotion safe: a standby that acquires term ``n``
    can prove it is newer than any writer that saw term ``< n``, and a node
    that wakes up with a stale term is told to stop instead of writing over the
    new primary.
    """

    def __init__(self, store: StateStore, *, slot: str = "primary", clock=time.time) -> None:
        self.store = store
        self.slot = slot
        self._clock = clock
        _ensure_schema(store)

    def current(self) -> Optional[Lease]:
        row = self.store.query_one(f"SELECT * FROM {LEASE_TABLE} WHERE slot=?", (self.slot,))
        if row is None:
            return None
        return Lease(node_id=row["node_id"], term=int(row["term"]),
                     acquired_at=float(row["acquired_at"]), expires_at=float(row["expires_at"]))

    def may_serve(self, node_id: str) -> bool:
        lease = self.current()
        return lease is not None and lease.node_id == node_id and not lease.expired(self._clock())

    def acquire(self, node_id: str, ttl_s: float = 30.0) -> Lease:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        now = self._clock()
        with self.store.transaction() as conn:
            row = conn.execute(f"SELECT * FROM {LEASE_TABLE} WHERE slot=?", (self.slot,)).fetchone()
            if row is not None and float(row["expires_at"]) > now and row["node_id"] != node_id:
                raise LeaseHeld(row["node_id"], float(row["expires_at"]), int(row["term"]))
            term = 1 if row is None else int(row["term"]) + 1
            acquired = now if row is None or float(row["expires_at"]) <= now else float(row["acquired_at"])
            conn.execute(
                f"INSERT INTO {LEASE_TABLE}(slot, node_id, term, acquired_at, expires_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(slot) DO UPDATE SET node_id=excluded.node_id, term=excluded.term, "
                "acquired_at=excluded.acquired_at, expires_at=excluded.expires_at",
                (self.slot, node_id, term, acquired, now + ttl_s),
            )
        self.store.event("lease_acquired", {"slot": self.slot, "node": node_id, "term": term})
        return Lease(node_id=node_id, term=term, acquired_at=acquired, expires_at=now + ttl_s)

    def renew(self, node_id: str, term: int, ttl_s: float = 30.0) -> Lease:
        now = self._clock()
        with self.store.transaction() as conn:
            row = conn.execute(f"SELECT * FROM {LEASE_TABLE} WHERE slot=?", (self.slot,)).fetchone()
            if row is None:
                raise LeaseError("no lease to renew")
            if int(row["term"]) != int(term) or row["node_id"] != node_id:
                raise StaleTerm(f"term {term} of {node_id} is not current")
            conn.execute(f"UPDATE {LEASE_TABLE} SET expires_at=? WHERE slot=?", (now + ttl_s, self.slot))
            acquired = float(row["acquired_at"])
        return Lease(node_id=node_id, term=int(term), acquired_at=acquired, expires_at=now + ttl_s)

    def release(self, node_id: str, term: int) -> None:
        with self.store.transaction() as conn:
            row = conn.execute(f"SELECT * FROM {LEASE_TABLE} WHERE slot=?", (self.slot,)).fetchone()
            if row is None:
                return
            if int(row["term"]) != int(term) or row["node_id"] != node_id:
                raise StaleTerm(f"term {term} of {node_id} is not current")
            conn.execute(f"UPDATE {LEASE_TABLE} SET expires_at=0 WHERE slot=?", (self.slot,))
        self.store.event("lease_released", {"slot": self.slot, "node": node_id, "term": int(term)})


@dataclass(frozen=True)
class Claim:
    accepted: bool
    reason: str
    recorded_at: Optional[float] = None


class DeltaLedger:
    """One accepted delta per (round, client); a replay is refused.

    Keyed by round **and** client, and storing the digest, so:

    * the same bytes posted twice are a replay,
    * a different delta from the same client in the same round is a conflict,
    * the next round starts clean without deleting anything.
    """

    def __init__(self, store: StateStore) -> None:
        self.store = store
        _ensure_schema(store)

    def claim(self, round_id: str, client_id: str, digest: str, *, now: Optional[float] = None) -> Claim:
        if not round_id or not client_id or not digest:
            raise ValueError("round_id, client_id and digest are required")
        stamp = now if now is not None else time.time()
        with self.store.transaction() as conn:
            row = conn.execute(
                f"SELECT digest, recorded_at FROM {DELTA_TABLE} WHERE round_id=? AND client_id=?",
                (round_id, client_id),
            ).fetchone()
            if row is not None:
                if row["digest"] == digest:
                    return Claim(False, "replay", float(row["recorded_at"]))
                return Claim(False, "conflict: different delta already recorded", float(row["recorded_at"]))
            conn.execute(
                f"INSERT INTO {DELTA_TABLE}(round_id, client_id, digest, recorded_at) VALUES(?,?,?,?)",
                (round_id, client_id, digest, stamp),
            )
        return Claim(True, "accepted", stamp)

    def count(self, round_id: str) -> int:
        row = self.store.query_one(f"SELECT COUNT(*) AS n FROM {DELTA_TABLE} WHERE round_id=?", (round_id,))
        return int(row["n"]) if row else 0
