"""Node registry with key rotation overlap."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from .identity import NodeIdentity
from .records import build_node_record, canonical_json, digest_of
from .state import StateStore


@dataclass(frozen=True)
class Node:
    node_id: str
    pubkey: str
    record: dict
    valid_from: float
    valid_until: Optional[float]
    status: str


class NodeRegistry:
    def __init__(self, store: StateStore):
        self.store = store

    def register(self, record: dict, *, valid_from: Optional[float] = None,
                 status: str = "active") -> Node:
        now = time.time()
        start = now if valid_from is None else float(valid_from)
        body = dict(record)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO nodes(node_id,pubkey,record,valid_from,valid_until,status,updated_at) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(node_id,pubkey) DO UPDATE SET "
                "record=excluded.record, valid_from=excluded.valid_from, status=excluded.status, "
                "updated_at=excluded.updated_at",
                (body["node_id"], body["pubkey"], canonical_json(body), start, None, status, now),
            )
        return Node(body["node_id"], body["pubkey"], body, start, None, status)

    def rotate(self, old: Node, new_record: dict, *, overlap_s: float = 3600) -> Node:
        now = time.time()
        with self.store.transaction() as conn:
            conn.execute("UPDATE nodes SET valid_until=?, updated_at=? WHERE node_id=? AND pubkey=?",
                         (now + float(overlap_s), now, old.node_id, old.pubkey))
        return self.register(new_record, valid_from=now)

    def resolve(self, node_id: str, pubkey: str, at: Optional[float] = None) -> Optional[Node]:
        at = time.time() if at is None else at
        row = self.store.query_one(
            "SELECT * FROM nodes WHERE node_id=? AND pubkey=? AND valid_from<=? "
            "AND (valid_until IS NULL OR valid_until>=?)", (node_id, pubkey, at, at))
        if not row:
            return None
        return Node(row["node_id"], row["pubkey"], json.loads(row["record"]),
                    row["valid_from"], row["valid_until"], row["status"])

    def active(self, at: Optional[float] = None) -> list[Node]:
        now = time.time() if at is None else at
        rows = self.store.query(
            "SELECT * FROM nodes WHERE status='active' AND valid_from<=? "
            "AND (valid_until IS NULL OR valid_until>=?) ORDER BY valid_from", (now, now))
        return [Node(r["node_id"], r["pubkey"], json.loads(r["record"]),
                     r["valid_from"], r["valid_until"], r["status"]) for r in rows]


def node_id_for(identity: NodeIdentity) -> str:
    return "node-" + identity.public_key_hex[:16]


def local_node_record(identity: NodeIdentity, *, endpoints: list[str],
                      roles: list[str], software_version: str,
                      started_at: str) -> dict:
    return build_node_record(node_id=node_id_for(identity), pubkey=identity.public_key_hex,
                             endpoints=endpoints, roles=roles,
                             software_version=software_version, started_at=started_at)


__all__ = ["Node", "NodeRegistry", "local_node_record", "node_id_for"]
