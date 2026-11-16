"""Round, pool, and membership state.

Scheduling owns the durable lifecycle.  Matchmaking policy still lives in the
HTTP layer; this module records the decisions so a restart can reconstruct the
same view.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, Optional

from .domain import PoolView, RoundView
from .state import StateStore


class SchedulingService:
    def __init__(self, store: StateStore):
        self.store = store

    # ---- rounds ---------------------------------------------------------
    def save_round(self, round_no: int, state: str = "open", started_at: Optional[float] = None,
                   duration_hours: float = 4.0, last_aggregation_at: float = 0.0,
                   force_end_at: Optional[float] = None, global_step: int = 0,
                   participating: Optional[dict] = None) -> RoundView:
        now = time.time()
        started_at = now if started_at is None else float(started_at)
        view = RoundView(
            round=int(round_no), state=state, started_at=started_at,
            duration_hours=float(duration_hours), last_aggregation_at=float(last_aggregation_at),
            force_end_at=force_end_at, global_step=int(global_step),
            participating=dict(participating or {}),
        )
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO rounds(round,state,started_at,duration_hours,last_aggregation_at,"
                "force_end_at,global_step,participating,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(round) DO UPDATE SET state=excluded.state,"
                "started_at=excluded.started_at,duration_hours=excluded.duration_hours,"
                "last_aggregation_at=excluded.last_aggregation_at,"
                "force_end_at=excluded.force_end_at,global_step=excluded.global_step,"
                "participating=excluded.participating,updated_at=excluded.updated_at",
                (view.round, view.state, view.started_at, view.duration_hours,
                 view.last_aggregation_at, view.force_end_at, view.global_step,
                 json.dumps(view.participating, sort_keys=True), now),
            )
        return view

    def load_round(self, round_no: Optional[int] = None) -> Optional[RoundView]:
        if round_no is None:
            row = self.store.query_one("SELECT * FROM rounds ORDER BY round DESC LIMIT 1")
        else:
            row = self.store.query_one("SELECT * FROM rounds WHERE round=?", (round_no,))
        if not row:
            return None
        try:
            participating = json.loads(row["participating"] or "{}")
        except ValueError:
            participating = {}
        return RoundView(
            round=row["round"], state=row["state"], started_at=row["started_at"],
            duration_hours=row["duration_hours"], last_aggregation_at=row["last_aggregation_at"],
            force_end_at=row["force_end_at"], global_step=row["global_step"],
            participating=participating,
        )

    # ---- pools ----------------------------------------------------------
    def create_pool(self, pool_id: str, round_no: int, client_key: str,
                    hardware_sig: Iterable = (), status: str = "forming",
                    created_at: Optional[float] = None) -> PoolView:
        now = time.time()
        created = now if created_at is None else float(created_at)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO pools(pool_id,round,status,hardware_sig,created_at,last_activity,"
                "updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(pool_id) DO NOTHING",
                (pool_id, int(round_no), status, json.dumps(list(hardware_sig)),
                 created, now, now),
            )
            conn.execute(
                "INSERT INTO pool_members(pool_id,client_key,position,joined_at) "
                "VALUES(?,?,?,?) ON CONFLICT(pool_id,client_key) DO NOTHING",
                (pool_id, client_key, 0, now),
            )
        return self.get_pool(pool_id)

    def join_pool(self, pool_id: str, client_key: str, position: int) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO pool_members(pool_id,client_key,position,joined_at) "
                "VALUES(?,?,?,?) ON CONFLICT(pool_id,client_key) DO UPDATE SET position=excluded.position",
                (pool_id, client_key, int(position), time.time()),
            )
            conn.execute("UPDATE pools SET last_activity=?, updated_at=? WHERE pool_id=?",
                         (time.time(), time.time(), pool_id))

    def leave_pool(self, pool_id: str, client_key: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM pool_members WHERE pool_id=? AND client_key=?",
                         (pool_id, client_key))

    def activate_pool(self, pool_id: str, *, chunk_idx: Optional[int] = None,
                      task_id: Optional[str] = None, target_seed: Optional[int] = None,
                      target_autocast: Optional[str] = None,
                      target_optimizer: Optional[str] = None,
                      target_steps: Optional[int] = None) -> None:
        now = time.time()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE pools SET status='training', activated_at=?, last_activity=?, "
                "chunk_idx=?, task_id=?, target_seed=?, target_autocast=?, "
                "target_optimizer=?, target_steps=?, updated_at=? WHERE pool_id=?",
                (now, now, chunk_idx, task_id, target_seed, target_autocast,
                 target_optimizer, target_steps, now, pool_id),
            )

    def set_pool_status(self, pool_id: str, status: str, *, power: Optional[bool] = None) -> None:
        with self.store.transaction() as conn:
            if power is None:
                conn.execute("UPDATE pools SET status=?, last_activity=?, updated_at=? WHERE pool_id=?",
                             (status, time.time(), time.time(), pool_id))
            else:
                conn.execute("UPDATE pools SET status=?, power=?, last_activity=?, updated_at=? "
                             "WHERE pool_id=?",
                             (status, 1 if power else 0, time.time(), time.time(), pool_id))

    def remove_pool(self, pool_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM pool_members WHERE pool_id=?", (pool_id,))
            conn.execute("DELETE FROM pools WHERE pool_id=?", (pool_id,))

    def get_pool(self, pool_id: str) -> Optional[PoolView]:
        row = self.store.query_one("SELECT * FROM pools WHERE pool_id=?", (pool_id,))
        return self._pool(row) if row else None

    def load_pools(self, round_no: Optional[int] = None,
                   statuses: Optional[Iterable[str]] = None) -> list[PoolView]:
        sql = "SELECT * FROM pools"
        params: list = []
        where = []
        if round_no is not None:
            where.append("round=?")
            params.append(int(round_no))
        if statuses:
            marks = ",".join("?" for _ in statuses)
            where.append(f"status IN ({marks})")
            params.extend(statuses)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at"
        return [self._pool(r) for r in self.store.query(sql, tuple(params))]

    def pool_stats(self, round_no: Optional[int] = None) -> dict:
        pools = self.load_pools(round_no)
        return {
            "activePools": len(pools),
            "formingPools": sum(1 for p in pools if p.status == "forming"),
            "trainingPools": sum(1 for p in pools if p.status == "training"),
        }

    def _pool(self, row) -> PoolView:
        try:
            hardware_sig = tuple(json.loads(row["hardware_sig"] or "[]"))
        except ValueError:
            hardware_sig = ()
        members = self.store.query(
            "SELECT client_key,position FROM pool_members WHERE pool_id=? ORDER BY position,rowid",
            (row["pool_id"],),
        )
        submissions = self.store.query(
            "SELECT client_key,delta_path FROM submissions WHERE pool_id=? AND status='accepted' "
            "AND delta_path IS NOT NULL ORDER BY created_at",
            (row["pool_id"],),
        )
        return PoolView(
            pool_id=row["pool_id"], round=row["round"], status=row["status"],
            clients=tuple(m["client_key"] for m in members), hardware_sig=hardware_sig,
            created_at=row["created_at"], last_activity=row["last_activity"],
            activated_at=row["activated_at"], chunk_idx=row["chunk_idx"], task_id=row["task_id"],
            target_seed=row["target_seed"], target_autocast=row["target_autocast"],
            target_optimizer=row["target_optimizer"], target_steps=row["target_steps"],
            power=bool(row["power"]),
            deltas={s["client_key"]: s["delta_path"] for s in submissions},
        )


__all__ = ["SchedulingService"]
