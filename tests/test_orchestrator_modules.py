import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import Orchestrator, StateStore


def test_services_import_without_http():
    code = (
        "import sys; import orchestrator; "
        "assert 'fastapi' not in sys.modules, 'orchestrator must not import fastapi'; "
        "print(orchestrator.SCHEMA_VERSION)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


def test_state_store_uses_wal(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        mode = store.query_one("PRAGMA journal_mode")["journal_mode"]
        assert mode.lower() == "wal"
        assert store.schema_version() == 1
    finally:
        store.close()


def test_ticket_lifecycle_and_expiry(tmp_path):
    orch = Orchestrator.open(tmp_path / "state.sqlite3")
    try:
        t1 = orch.admission.issue("ip:10.0.0.1", "round_7_p1", 7, pool_id="p1")
        again = orch.admission.issue("ip:10.0.0.1", "round_7_p1", 7, pool_id="p1")
        assert again.ticket_id == t1.ticket_id
        assert again.consumed is False
        assert orch.admission.consume("ip:10.0.0.1", "round_7_p1") is True
        assert orch.admission.get("ip:10.0.0.1").consumed is True
        assert orch.admission.consume("ip:10.0.0.1", "round_7_p1") is False

        short = orch.admission.issue("ip:10.0.0.2", "round_7_p1", 7, ttl_s=0.01)
        time.sleep(0.03)
        assert orch.admission.get("ip:10.0.0.2").expires_at < time.time()
        assert orch.admission.purge_expired() == 1
        assert orch.admission.get("ip:10.0.0.2") is None
    finally:
        orch.close()


def test_round_and_pool_survive_restart(tmp_path):
    path = tmp_path / "state.sqlite3"
    orch = Orchestrator.open(path)
    try:
        orch.scheduling.save_round(12, state="open", started_at=1000.0, duration_hours=4.0,
                                   last_aggregation_at=1001.0, global_step=41,
                                   participating={"ip:a": {"submitted": True}})
        orch.scheduling.create_pool("p1", 12, "ip:a", hardware_sig=("CUDA", 1, "bf16", "adamw"))
        orch.scheduling.join_pool("p1", "ip:b", 1)
        orch.scheduling.activate_pool("p1", chunk_idx=9, task_id="round_12_p1",
                                      target_seed=123, target_autocast="bf16",
                                      target_optimizer="adamw", target_steps=500)
        orch.scheduling.set_pool_status("p1", "training")
    finally:
        orch.close()

    orch = Orchestrator.open(path)
    try:
        rnd = orch.scheduling.load_round()
        assert rnd.round == 12 and rnd.global_step == 41
        assert rnd.participating["ip:a"]["submitted"] is True
        pools = orch.scheduling.load_pools(round_no=12)
        assert len(pools) == 1
        pool = pools[0]
        assert pool.status == "training"
        assert pool.clients == ("ip:a", "ip:b")
        assert pool.task_id == "round_12_p1"
        assert pool.target_seed == 123
        assert pool.hardware_sig == ("CUDA", 1, "bf16", "adamw")
        assert orch.scheduling.pool_stats(12) == {
            "activePools": 1, "formingPools": 0, "trainingPools": 1,
        }
    finally:
        orch.close()


def test_submission_is_idempotent_by_content_hash(tmp_path):
    orch = Orchestrator.open(tmp_path / "state.sqlite3")
    try:
        upload_id = orch.submission.begin_upload(
            "ip:a", expected_bytes=10, chunk_size=5, chunks=2, ticket_id="t1",
            content_sha256="a" * 64,
        )
        assert orch.submission.mark_chunk(upload_id, 1, size=5)["receivedChunks"] == 1
        assert orch.submission.mark_chunk(upload_id, 0, size=5)["receivedChunks"] == 2
        up = orch.submission.finalize_upload(upload_id)
        assert up["state"] == "open"
        assert orch.submission.get_upload(upload_id)["state"] == "finalized"

        first, created = orch.submission.record_submission(
            client_key="ip:a", round_no=12, content_sha256="b" * 64,
            ticket_id="t1", pool_id="p1", delta_path="/tmp/delta.bin",
            tokens=100, steps=7, loss_reported="3.100000",
        )
        second, created_again = orch.submission.record_submission(
            client_key="ip:a", round_no=12, content_sha256="b" * 64,
            ticket_id="t1", pool_id="p1", delta_path="/tmp/delta.bin",
            tokens=100, steps=7, loss_reported="3.100000",
        )
        assert created is True and created_again is False
        assert second.submission_id == first.submission_id
        assert len(orch.submission.load_submissions(round_no=12)) == 1
        assert orch.submission.load_submissions(pool_id="p1")[0].tokens == 100
    finally:
        orch.close()


def test_verification_and_publication(tmp_path):
    orch = Orchestrator.open(tmp_path / "state.sqlite3",
                             publication_jsonl=tmp_path / "public" / "rounds.jsonl")
    try:
        v = orch.verification.record("s1", "rejected", reason="delta_norm_mismatch",
                                     evidence={"reported": 1.0, "actual": 9.0})
        assert v.verdict == "rejected"
        assert orch.verification.get("s1").evidence["actual"] == 9.0
        with pytest.raises(ValueError):
            orch.verification.record("s1", "maybe")

        published = orch.publication.publish_round(12, {"round": 12, "loss": "3.100000"})
        assert published["record"]["loss"] == "3.100000"
        again = orch.publication.publish_round(12, {"round": 12, "loss": "9.999999"})
        assert again["record"]["loss"] == "3.100000"  # first publication wins
        lines = (tmp_path / "public" / "rounds.jsonl").read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["record"]["round"] == 12
        assert [r["round"] for r in orch.publication.all_rounds()] == [12]
    finally:
        orch.close()


def test_two_connections_see_one_state(tmp_path):
    path = tmp_path / "state.sqlite3"
    a = Orchestrator.open(path)
    b = Orchestrator.open(path)
    try:
        a.scheduling.save_round(3, state="open", started_at=time.time(), global_step=2)
        assert b.scheduling.load_round().round == 3
        a.scheduling.create_pool("p9", 3, "ip:a")
        assert b.scheduling.get_pool("p9").clients == ("ip:a",)
    finally:
        a.close(); b.close()
