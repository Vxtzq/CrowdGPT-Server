"""Append-only audit view over published round records."""
from __future__ import annotations

from typing import Optional

from .publication import PublicationService
from .records import merkle_proof, merkle_root, record_hash, verify_merkle_proof


class AuditService:
    def __init__(self, publication: PublicationService):
        self.publication = publication

    def rounds(self) -> list[dict]:
        return self.publication.all_rounds()

    def checkpoint(self, upto: Optional[int] = None) -> dict:
        rows = [r for r in self.publication.all_rounds()
                if upto is None or r["round"] <= upto]
        leaves = [record_hash(r["record"]) for r in rows]
        return {"from_round": rows[0]["round"] if rows else None,
                "to_round": rows[-1]["round"] if rows else None,
                "count": len(rows), "root": merkle_root(leaves),
                "record_hashes": leaves}

    def inclusion(self, round_no: int) -> dict:
        rows = self.publication.all_rounds()
        leaves = [record_hash(r["record"]) for r in rows]
        leaf = next((r["record"] and record_hash(r["record"]) for r in rows
                     if r["round"] == round_no), None)
        if leaf is None: raise KeyError(f"round {round_no} is not published")
        sorted_leaves = sorted(leaves)
        idx = sorted_leaves.index(leaf)
        root = merkle_root(leaves)
        proof = merkle_proof(leaves, idx)
        return {"round": round_no, "leaf": leaf, "root": root, "proof": proof,
                "valid": verify_merkle_proof(leaf, proof, root)}

    def verify_chain(self) -> dict:
        rows = self.publication.all_rounds()
        prev = None; broken = []
        for r in rows:
            rec = r["record"]
            if prev is not None and rec.get("prev_record_hash") != prev:
                broken.append(r["round"])
            prev = record_hash(rec)
        return {"rounds": len(rows), "broken": broken, "ok": not broken}


__all__ = ["AuditService"]
