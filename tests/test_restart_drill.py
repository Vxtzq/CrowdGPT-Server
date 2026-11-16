"""P1.5 restart drills: SIGKILL a real process at each durability boundary."""
from __future__ import annotations

import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest

from orchestrator import Orchestrator
from orchestrator.objects import LocalObjectStore

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGKILL"),
                                reason="SIGKILL drill requires POSIX")


def _child_committed(db: str, ready: str):
    orch = Orchestrator.open(db, identity_path=Path(db).with_name("node.pem"))
    orch.scheduling.save_round(7, state="open", started_at=100.0, global_step=6)
    orch.scheduling.create_pool("p1", 7, "ip:a")
    orch.scheduling.join_pool("p1", "ip:b", 1)
    orch.admission.issue("ip:a", "round_7_p1", 7, pool_id="p1")
    Path(ready).write_text("ok")
    os.kill(os.getpid(), signal.SIGKILL)


def _child_uncommitted(db: str, ready: str):
    orch = Orchestrator.open(db)
    with orch.store.transaction() as conn:
        conn.execute(
            "INSERT INTO rounds(round,state,started_at,duration_hours,last_aggregation_at,"
            "global_step,participating,updated_at) VALUES(9,'open',1,4,0,8,'{}',1)")
        Path(ready).write_text("mid-transaction")
        time.sleep(60)
    os.kill(os.getpid(), signal.SIGKILL)


def _child_half_upload(db: str, obj: str, ready: str):
    orch = Orchestrator.open(db)
    orch.submission.begin_upload("ip:a", expected_bytes=10, chunk_size=5, chunks=2)
    store = LocalObjectStore(Path(obj), secret=b"x" * 32)
    part = Path(obj) / "delta.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"half")
    Path(ready).write_text("half")
    os.kill(os.getpid(), signal.SIGKILL)


def _run(child, *args):
    ready = Path(args[-1])
    p = multiprocessing.Process(target=child, args=args)
    p.start()
    deadline = time.time() + 20
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "child did not reach the drill boundary"
    p.join(timeout=5)
    if p.is_alive():
        p.kill(); p.join(timeout=5)
    return p.exitcode


def test_sigkill_after_commit_keeps_round_pool_and_ticket(tmp_path):
    db = tmp_path / "state.sqlite3"
    _run(_child_committed, str(db), str(tmp_path / "ready"))
    orch = Orchestrator.open(db, identity_path=tmp_path / "node.pem")
    try:
        assert orch.scheduling.load_round().round == 7
        pool = orch.scheduling.get_pool("p1")
        assert pool.clients == ("ip:a", "ip:b")
        assert orch.admission.get("ip:a").task_id == "round_7_p1"
    finally:
        orch.close()


def test_sigkill_mid_transaction_rolls_back(tmp_path):
    db = tmp_path / "state.sqlite3"
    _run(_child_uncommitted, str(db), str(tmp_path / "ready"))
    orch = Orchestrator.open(db)
    try:
        assert orch.scheduling.load_round(9) is None
    finally:
        orch.close()


def test_sigkill_during_upload_leaves_no_committed_object(tmp_path):
    db = tmp_path / "state.sqlite3"
    obj = tmp_path / "objects"
    _run(_child_half_upload, str(db), str(obj), str(tmp_path / "ready"))
    orch = Orchestrator.open(db)
    try:
        row = orch.store.query_one("SELECT upload_id FROM uploads")
        up = orch.submission.get_upload(row["upload_id"])
        assert up["state"] == "open"
        assert not (obj / "delta").exists()
    finally:
        orch.close()
