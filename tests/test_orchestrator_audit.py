from orchestrator.audit import AuditService
from orchestrator.publication import PublicationService
from orchestrator.records import build_round_record, record_hash
from orchestrator.state import StateStore
from orchestrator.records import ZERO_ROOT


def _record(r, prev):
    return build_round_record(
        round_no=r, opened_at="2026-11-09T00:00:00Z", closed_at="2026-11-09T01:00:00Z",
        base_weights="a" * 64, aggregate_weights="b" * 64, leader="node-aaaaaaaaaaaaaaaa",
        witnesses=(), tickets_root="1" * 64, submissions_root="2" * 64,
        deltas_root="3" * 64, prev_record_hash=prev, eval=None,
        bytes_down=1, bytes_up=2, code_versions=("3.2",))


def test_inclusion_proof_and_chain(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    pub = PublicationService(store, tmp_path / "rounds.jsonl")
    prev = None
    for r in range(1, 5):
        rec = _record(r, prev); prev = record_hash(rec)
        pub.publish_round(r, rec)
    audit = AuditService(pub)
    cp = audit.checkpoint()
    assert cp["count"] == 4 and len(cp["root"]) == 64
    inc = audit.inclusion(3)
    assert inc["valid"] is True
    assert audit.verify_chain()["ok"] is True
    # Editing one record breaks its inclusion proof.
    edited = pub.get_round(3)["record"]; edited["bytes"]["up"] = 999
    assert record_hash(edited) != inc["leaf"]
    store.close()
