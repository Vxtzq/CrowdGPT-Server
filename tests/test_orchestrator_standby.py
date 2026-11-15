"""Tests for the standby lease and the delta replay ledger.

These are the two properties a promoted standby must not get wrong: only one
node serves at a time, and a submission is never applied twice. Both are plain
SQL over a temporary ``StateStore``, so they run in CI without FastAPI/torch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.standby import (
    Claim,
    CoordinatorLease,
    DeltaLedger,
    LeaseError,
    LeaseHeld,
    StaleTerm,
)
from orchestrator.state import StateStore


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def store(tmp_path: Path):
    s = StateStore(tmp_path / "state.sqlite3")
    yield s
    s.close()


@pytest.fixture()
def clock():
    return Clock()


def test_first_acquire_takes_term_one(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    assert lease.current() is None
    held = lease.acquire("node-a", ttl_s=30)
    assert held.term == 1
    assert held.node_id == "node-a"
    assert lease.may_serve("node-a")
    assert not lease.may_serve("node-b")


def test_live_lease_refuses_a_second_holder(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    lease.acquire("node-a", ttl_s=30)
    with pytest.raises(LeaseHeld) as err:
        lease.acquire("node-b", ttl_s=30)
    assert err.value.holder == "node-a"


def test_expiry_allows_takeover_with_a_higher_term(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    lease.acquire("node-a", ttl_s=30)
    clock.advance(31)
    taken = lease.acquire("node-b", ttl_s=30)
    assert taken.term == 2
    assert lease.may_serve("node-b")
    assert not lease.may_serve("node-a")


def test_reacquire_by_the_same_node_still_bumps_the_term(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    first = lease.acquire("node-a", ttl_s=30)
    second = lease.acquire("node-a", ttl_s=30)
    assert second.term == first.term + 1


def test_renew_extends_but_does_not_change_the_term(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    held = lease.acquire("node-a", ttl_s=30)
    clock.advance(20)
    renewed = lease.renew("node-a", held.term, ttl_s=30)
    assert renewed.term == held.term
    assert renewed.expires_at == pytest.approx(clock.now + 30)
    assert lease.may_serve("node-a")


def test_a_stale_term_cannot_renew_or_release(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    lease.acquire("node-a", ttl_s=30)
    clock.advance(31)
    lease.acquire("node-b", ttl_s=30)
    with pytest.raises(StaleTerm):
        lease.renew("node-a", 1)
    with pytest.raises(StaleTerm):
        lease.release("node-a", 1)


def test_release_lets_the_standby_take_over_immediately(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    held = lease.acquire("node-a", ttl_s=300)
    lease.release("node-a", held.term)
    assert not lease.may_serve("node-a")
    taken = lease.acquire("node-b", ttl_s=30)
    assert taken.term == held.term + 1


def test_releasing_the_right_term_is_idempotent(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    held = lease.acquire("node-a", ttl_s=30)
    lease.release("node-a", held.term)
    lease.release("node-a", held.term)
    assert lease.current().expires_at <= clock.now


def test_lease_events_are_audited(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    held = lease.acquire("node-a", ttl_s=30)
    lease.release("node-a", held.term)
    kinds = [e["kind"] for e in store.events(limit=10)]
    assert kinds[:2] == ["lease_released", "lease_acquired"]


def test_rejects_a_non_positive_ttl(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    with pytest.raises(ValueError):
        lease.acquire("node-a", ttl_s=0)


def test_same_delta_twice_is_a_replay(store):
    ledger = DeltaLedger(store)
    first = ledger.claim("round-7", "client-1", "a" * 64, now=10.0)
    again = ledger.claim("round-7", "client-1", "a" * 64, now=11.0)
    assert first == Claim(True, "accepted", 10.0)
    assert again.accepted is False
    assert again.reason == "replay"
    assert again.recorded_at == 10.0
    assert ledger.count("round-7") == 1


def test_a_different_delta_in_the_same_round_is_a_conflict(store):
    ledger = DeltaLedger(store)
    ledger.claim("round-7", "client-1", "a" * 64)
    conflict = ledger.claim("round-7", "client-1", "b" * 64)
    assert conflict.accepted is False
    assert conflict.reason.startswith("conflict")


def test_the_next_round_starts_clean(store):
    ledger = DeltaLedger(store)
    assert ledger.claim("round-7", "client-1", "a" * 64).accepted
    assert ledger.claim("round-8", "client-1", "a" * 64).accepted
    assert ledger.count("round-8") == 1


def test_clients_do_not_collide(store):
    ledger = DeltaLedger(store)
    assert ledger.claim("round-7", "client-1", "a" * 64).accepted
    assert ledger.claim("round-7", "client-2", "a" * 64).accepted
    assert ledger.count("round-7") == 2


def test_ledger_requires_identifiers(store):
    ledger = DeltaLedger(store)
    with pytest.raises(ValueError):
        ledger.claim("", "client-1", "a" * 64)


def test_ledger_survives_a_reopen(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = StateStore(path)
    DeltaLedger(first).claim("round-7", "client-1", "a" * 64)
    first.close()
    second = StateStore(path)
    try:
        assert DeltaLedger(second).claim("round-7", "client-1", "a" * 64).reason == "replay"
    finally:
        second.close()


def test_lease_survives_a_reopen(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = StateStore(path)
    held = CoordinatorLease(first).acquire("node-a", ttl_s=300)
    first.close()
    second = StateStore(path)
    try:
        assert CoordinatorLease(second).may_serve("node-a")
        assert CoordinatorLease(second).current().term == held.term
    finally:
        second.close()


def test_lease_and_ledger_share_one_store(store, clock):
    lease = CoordinatorLease(store, clock=clock)
    ledger = DeltaLedger(store)
    held = lease.acquire("node-a", ttl_s=30)
    assert lease.may_serve("node-a")
    assert ledger.claim("round-1", "c", "d" * 64).accepted
    lease.release("node-a", held.term)
    assert not lease.may_serve("node-a")
    assert ledger.count("round-1") == 1
