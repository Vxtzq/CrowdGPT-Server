import time

import pytest

from orchestrator.identity import NodeIdentity
from orchestrator.nodes import NodeRegistry, local_node_record, node_id_for
from orchestrator.records import (ZERO_ROOT, build_round_record, merkle_root,
                                  record_hash, sign_record, verify_certificate)
from orchestrator.state import StateStore


def _record(leader="node-aaaaaaaaaaaaaaaa"):
    return build_round_record(
        round_no=5, opened_at="2026-11-09T00:00:00Z", closed_at="2026-11-09T04:00:00Z",
        base_weights="a" * 64, aggregate_weights="b" * 64, leader=leader,
        witnesses=(), tickets_root="1" * 64, submissions_root="2" * 64,
        deltas_root="3" * 64, prev_record_hash=ZERO_ROOT, eval=None,
        bytes_down=100, bytes_up=200, code_versions=("3.2",),
    )


def test_leader_signature_and_tamper_rejection():
    leader = NodeIdentity.generate()
    node = node_id_for(leader)
    signed = sign_record(_record(node), leader, node)
    assert len(signed["certificate"]) == 1
    assert verify_certificate(signed, signed["certificate"][0]) is True
    tampered = dict(signed); tampered["bytes"] = {"down": 1, "up": 200}
    assert verify_certificate(tampered, signed["certificate"][0]) is False
    wrong = NodeIdentity.generate()
    assert verify_certificate(signed, signed["certificate"][0],
                              expected_pubkey=wrong.public_key_hex) is False


def test_witness_cosignature():
    leader = NodeIdentity.generate(); witness = NodeIdentity.generate()
    rec = _record(node_id_for(leader))
    rec = sign_record(rec, leader, node_id_for(leader))
    rec = sign_record(rec, witness, node_id_for(witness))
    assert len(rec["certificate"]) == 2
    assert all(verify_certificate(rec, c) for c in rec["certificate"])


def test_ticket_and_round_record_hash_stability():
    rec = _record()
    assert record_hash(rec) == record_hash(dict(rec))
    other = dict(rec); other["round"] = 6
    assert record_hash(rec) != record_hash(other)


def test_merkle_root_duplicate_last():
    assert merkle_root([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    a = merkle_root(["a"])
    ab = merkle_root(["a", "b"])
    assert a != ab
    # sorted leaves: order does not matter
    assert merkle_root(["b", "a"]) == ab


def test_node_rotation_overlap_window(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    reg = NodeRegistry(store)
    old = NodeIdentity.generate(); new = NodeIdentity.generate()
    old_rec = local_node_record(old, endpoints=["https://old.example"],
                                roles=["leader"], software_version="3.2",
                                started_at="2026-11-09T00:00:00Z")
    old_node = reg.register(old_rec, valid_from=time.time() - 100)
    rec = local_node_record(new, endpoints=["https://new.example"],
                            roles=["leader"], software_version="3.2",
                            started_at="2026-11-09T00:00:00Z")
    now = time.time()
    new_node = reg.rotate(old_node, rec, overlap_s=2.0)
    assert reg.resolve(old_node.node_id, old_node.pubkey, at=now + 1).pubkey == old_node.pubkey
    assert reg.resolve(new_node.node_id, new_node.pubkey, at=now + 1).pubkey == new_node.pubkey
    assert reg.resolve(old_node.node_id, old_node.pubkey, at=now + 10) is None
    assert {n.pubkey for n in reg.active(at=now + 10)} == {new_node.pubkey}
    store.close()
