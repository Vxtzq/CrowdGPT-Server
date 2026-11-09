"""Plain data objects exchanged between orchestrator services."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class RoundView:
    round: int
    state: str
    started_at: float
    duration_hours: float
    last_aggregation_at: float = 0.0
    force_end_at: Optional[float] = None
    global_step: int = 0
    participating: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class PoolView:
    pool_id: str
    round: int
    status: str
    clients: tuple[str, ...]
    hardware_sig: tuple
    created_at: float
    last_activity: float
    activated_at: Optional[float] = None
    chunk_idx: Optional[int] = None
    task_id: Optional[str] = None
    target_seed: Optional[int] = None
    target_autocast: Optional[str] = None
    target_optimizer: Optional[str] = None
    target_steps: Optional[int] = None
    power: bool = False
    deltas: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TicketView:
    ticket_id: str
    client_key: str
    round: int
    task_id: str
    created_at: float
    expires_at: float
    step_budget: int
    token_budget: int
    nonce: str
    pool_id: Optional[str] = None
    consumed_at: Optional[float] = None

    @property
    def consumed(self) -> bool:
        return self.consumed_at is not None


@dataclass(frozen=True)
class SubmissionView:
    submission_id: str
    client_key: str
    round: int
    content_sha256: str
    status: str
    created_at: float
    finalized_at: Optional[float] = None
    ticket_id: Optional[str] = None
    pool_id: Optional[str] = None
    delta_path: Optional[str] = None
    tokens: int = 0
    steps: int = 0
    loss_reported: Optional[str] = None


@dataclass(frozen=True)
class VerificationView:
    submission_id: str
    verdict: str
    reason: Optional[str]
    evidence: dict
    created_at: float


__all__ = [
    "RoundView", "PoolView", "TicketView", "SubmissionView", "VerificationView",
]
