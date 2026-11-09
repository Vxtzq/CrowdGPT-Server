"""Signed client step receipts."""
from __future__ import annotations

import json
import time
from typing import Optional

from .records import canonical_json, digest_of, verify_client_receipt
from .state import StateStore


class ReceiptService:
    def __init__(self, store: StateStore):
        self.store = store

    def register_key(self, client_key: str, pubkey: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO client_keys(client_key,pubkey,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(client_key) DO UPDATE SET pubkey=excluded.pubkey, "
                "updated_at=excluded.updated_at", (client_key, pubkey.lower(), time.time()))

    def pubkey(self, client_key: str) -> Optional[str]:
        row = self.store.query_one("SELECT pubkey FROM client_keys WHERE client_key=?", (client_key,))
        return row["pubkey"] if row else None

    def verify_and_record(self, client_key: str, receipt: dict, signature: str,
                          pubkey: str) -> tuple[bool, str]:
        if not verify_client_receipt(receipt, pubkey, signature):
            return False, "receipt_signature_invalid"
        registered = self.pubkey(client_key)
        if registered and registered != pubkey.lower():
            return False, "receipt_key_mismatch"
        if receipt.get("ticket_id") != receipt.get("ticket_id", "").lower() or len(receipt.get("ticket_id", "")) != 32:
            return False, "receipt_ticket_invalid"
        rid = digest_of({"ticket": receipt.get("ticket_id"), "client": client_key,
                         "nonce": receipt.get("nonce")})
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO receipts(receipt_id,client_key,round,payload,signature,pubkey,"
                "verdict,created_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(receipt_id) DO NOTHING",
                (rid, client_key, int(receipt.get("round", 0)), canonical_json(receipt),
                 signature, pubkey.lower(), "accepted", time.time()))
        return True, "ok"

    def get(self, receipt_id: str) -> Optional[dict]:
        row = self.store.query_one("SELECT * FROM receipts WHERE receipt_id=?", (receipt_id,))
        if not row: return None
        d = dict(row); d["payload"] = json.loads(d["payload"]); return d

    def all(self, round_no: Optional[int] = None) -> list[dict]:
        rows = (self.store.query("SELECT * FROM receipts WHERE round=? ORDER BY created_at", (round_no,))
                if round_no is not None else self.store.query("SELECT * FROM receipts ORDER BY created_at"))
        out = []
        for row in rows:
            d = dict(row); d["payload"] = json.loads(d["payload"]); out.append(d)
        return out


__all__ = ["ReceiptService"]
