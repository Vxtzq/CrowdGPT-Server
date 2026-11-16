import hashlib

from orchestrator.identity import NodeIdentity
from orchestrator.records import canonical_json
from orchestrator.state import StateStore
from orchestrator.receipts import ReceiptService


def _receipt():
    return {"schema": "ng-1", "ticket_id": "a" * 32, "round": 7,
            "client_id": "device-1", "base_weights": hashlib.sha256(b"step:42").hexdigest(),
            "tokens": 1234, "steps": 7, "loss_samples": ["2.900000"],
            "loss_mean": "2.900000", "fingerprint_root": "c" * 64,
            "code_version": "3.2", "device": {"backend": "CUDA", "gpu": "RTX 4090"},
            "started_at": "2026-11-09T00:00:00Z", "finished_at": "2026-11-09T00:10:00Z",
            "nonce": "d" * 64}


def test_receipt_signature_and_key_binding(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3"); svc = ReceiptService(store)
    key = NodeIdentity.generate(); receipt = _receipt()
    sig = key.sign(canonical_json(receipt).encode())
    svc.register_key("ip:a", key.public_key_hex)
    ok, reason = svc.verify_and_record("ip:a", receipt, sig, key.public_key_hex)
    assert ok and reason == "ok"
    assert len(svc.all(7)) == 1
    # duplicate nonce is idempotent
    ok2, _ = svc.verify_and_record("ip:a", receipt, sig, key.public_key_hex)
    assert ok2 and len(svc.all(7)) == 1
    # tampered field
    bad = dict(receipt); bad["tokens"] = 999999
    ok3, reason3 = svc.verify_and_record("ip:a", bad, sig, key.public_key_hex)
    assert not ok3 and reason3 == "receipt_signature_invalid"
    # a different registered key cannot claim the client
    other = NodeIdentity.generate(); svc.register_key("ip:a", other.public_key_hex)
    ok4, reason4 = svc.verify_and_record("ip:a", receipt, sig, key.public_key_hex)
    assert not ok4 and reason4 == "receipt_key_mismatch"
    store.close()
