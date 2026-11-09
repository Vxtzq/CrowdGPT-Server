import hashlib

from orchestrator.adversarial import plausible_loss, same_origin
from orchestrator.identity import NodeIdentity
from orchestrator.records import build_round_record, sign_record, verify_certificate
from orchestrator.state import StateStore
from orchestrator.admission import AdmissionService
from orchestrator.submission import SubmissionService


def test_fake_loss_is_rejected():
    assert plausible_loss(0.001) is False
    assert plausible_loss(0.079) is False
    assert plausible_loss(float("nan")) is False
    assert plausible_loss(2.9) is True
    assert plausible_loss(11.9) is True


def test_sybil_origin_pair():
    assert same_origin({"subnet": "10.0.0.0/24"}, {"subnet": "10.0.0.0/24"}) is True
    assert same_origin({"device": "d1"}, {"device": "d1"}) is True
    assert same_origin({"subnet": "10.0.0.0/24"}, {"subnet": "10.0.1.0/24"}) is False


def test_replayed_ticket_is_rejected(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    adm = AdmissionService(store); t = adm.issue("ip:a", "task", 1)
    assert adm.consume("ip:a", "task") is True
    assert adm.consume("ip:a", "task") is False
    assert adm.get("ip:a").consumed is True
    store.close()


def test_duplicate_delta_content_is_excluded(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3"); sub = SubmissionService(store)
    key = hashlib.sha256(b"same-delta").hexdigest()
    a, ca = sub.record_submission(client_key="ip:a", round_no=1, content_sha256=key,
                                  delta_path="/tmp/a.bin")
    b, cb = sub.record_submission(client_key="ip:b", round_no=1, content_sha256=key,
                                  delta_path="/tmp/b.bin")
    assert ca is True and cb is False
    assert a.submission_id == b.submission_id
    assert len(sub.load_submissions(round_no=1)) == 1
    store.close()


def test_tampered_record_rejected():
    key = NodeIdentity.generate()
    rec = build_round_record(
        round_no=1, opened_at="2026-11-09T00:00:00Z", closed_at="2026-11-09T01:00:00Z",
        base_weights="a" * 64, aggregate_weights="b" * 64, leader="node-aaaaaaaaaaaaaaaa",
        witnesses=(), tickets_root="1" * 64, submissions_root="2" * 64,
        deltas_root="3" * 64, prev_record_hash=None, eval=None,
        bytes_down=1, bytes_up=2, code_versions=("3.2",))
    signed = sign_record(rec, key, "node-aaaaaaaaaaaaaaaa")
    assert verify_certificate(signed, signed["certificate"][0]) is True
    signed["round"] = 2
    assert verify_certificate(signed, signed["certificate"][0]) is False
