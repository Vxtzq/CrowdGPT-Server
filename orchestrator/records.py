"""Canonical, signed round records and Merkle roots.

This mirrors the ``ng-1`` wire objects in ``crowdgpt_ng/protocol.py`` without
importing the client package: the server must be deployable on its own.
Hashes are SHA-256 over sorted-key compact JSON; losses are decimal strings.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional, Sequence

from .identity import NodeIdentity

SCHEMA = "ng-1"
ZERO_ROOT = "0" * 64


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def merkle_root(leaves: Iterable[str]) -> str:
    """Sorted-leaf Merkle root with duplicate-last, matching ng protocol."""
    nodes = sorted(str(x).lower() for x in leaves)
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in nodes]
    while len(level) > 1:
        if len(level) % 2: level.append(level[-1])
        level = [hashlib.sha256(bytes.fromhex(level[i] + level[i + 1])).hexdigest()
                 for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: Iterable[str], index: int) -> list[dict]:
    nodes = sorted(str(x).lower() for x in leaves)
    if not nodes: raise ValueError("empty tree")
    if not 0 <= index < len(nodes): raise ValueError("index out of range")
    level = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in nodes]
    pos = index
    proof = []
    while len(level) > 1:
        if len(level) % 2: level.append(level[-1])
        sibling = pos ^ 1
        proof.append({"position": "right" if pos % 2 == 0 else "left",
                      "hash": level[sibling]})
        level = [hashlib.sha256(bytes.fromhex(level[i] + level[i + 1])).hexdigest()
                 for i in range(0, len(level), 2)]
        pos //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: Sequence[dict], root: str) -> bool:
    cur = hashlib.sha256(str(leaf).lower().encode("utf-8")).hexdigest()
    for step in proof:
        h = step["hash"]
        cur = (hashlib.sha256(bytes.fromhex(cur + h)).hexdigest()
               if step["position"] == "right"
               else hashlib.sha256(bytes.fromhex(h + cur)).hexdigest())
    return cur == root


def build_node_record(*, node_id: str, pubkey: str, endpoints: Sequence[str],
                      roles: Sequence[str], software_version: str, started_at: str,
                      bond_rounds: int = 0, status: str = "active") -> dict:
    return {"schema": SCHEMA, "node_id": node_id, "pubkey": pubkey,
            "endpoints": list(endpoints), "roles": list(roles),
            "software_version": software_version, "started_at": started_at,
            "bond_rounds": int(bond_rounds), "status": status}


def build_step_receipt(*, ticket_id: str, round_no: int, client_id: str,
                       base_weights: str, tokens: int, steps: int,
                       loss_samples: Sequence[str], loss_mean: Optional[str],
                       fingerprint_root: Optional[str], code_version: str,
                       device: dict, started_at: str, finished_at: str,
                       nonce: str) -> dict:
    return {"schema": SCHEMA, "ticket_id": ticket_id, "round": int(round_no),
            "client_id": client_id, "base_weights": base_weights,
            "tokens": int(tokens), "steps": int(steps),
            "loss_samples": list(loss_samples), "loss_mean": loss_mean,
            "fingerprint_root": fingerprint_root, "code_version": code_version,
            "device": dict(device), "started_at": started_at,
            "finished_at": finished_at, "nonce": nonce}


def build_round_record(*, round_no: int, opened_at: str, closed_at: str,
                       base_weights: str, aggregate_weights: str, leader: str,
                       witnesses: Sequence[str], tickets_root: str,
                       submissions_root: str, deltas_root: str,
                       prev_record_hash: Optional[str], eval: Optional[dict],
                       bytes_down: int, bytes_up: int,
                       code_versions: Sequence[str]) -> dict:
    return {"schema": SCHEMA, "round": int(round_no), "opened_at": opened_at,
            "closed_at": closed_at, "base_weights": base_weights,
            "aggregate_weights": aggregate_weights, "leader": leader,
            "witnesses": list(witnesses), "tickets_root": tickets_root,
            "submissions_root": submissions_root, "deltas_root": deltas_root,
            "prev_record_hash": prev_record_hash, "eval": eval,
            "bytes": {"down": int(bytes_down), "up": int(bytes_up)},
            "code_versions": list(code_versions), "certificate": []}


def signing_payload(record: dict) -> dict:
    body = dict(record)
    body.pop("certificate", None)
    return body


def record_hash(record: dict) -> str:
    return digest_of(signing_payload(record))


def sign_record(record: dict, identity: NodeIdentity, node_id: str) -> dict:
    sig = identity.sign(canonical_json(signing_payload(record)).encode("utf-8"))
    cert = {"node": node_id, "pubkey": identity.public_key_hex, "sig": sig}
    body = dict(record)
    body["certificate"] = list(body.get("certificate") or []) + [cert]
    return body


def verify_certificate(record: dict, cert: dict, *, expected_pubkey: Optional[str] = None) -> bool:
    if expected_pubkey and cert.get("pubkey") != expected_pubkey:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(cert["pubkey"]))
        pub.verify(bytes.fromhex(cert["sig"]),
                   canonical_json(signing_payload(record)).encode("utf-8"))
        return True
    except Exception:
        return False


def verify_client_receipt(receipt: dict, pubkey: str, signature: str) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey))
        pub.verify(bytes.fromhex(signature),
                   canonical_json(receipt).encode("utf-8"))
        return True
    except Exception:
        return False


__all__ = [
    "SCHEMA", "ZERO_ROOT", "build_node_record", "build_round_record",
    "build_step_receipt", "canonical_json", "digest_of", "merkle_root",
    "merkle_proof", "record_hash", "sign_record", "signing_payload",
    "verify_certificate", "verify_client_receipt", "verify_merkle_proof",
]
