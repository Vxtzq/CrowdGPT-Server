"""Facade that wires the orchestrator services to one durable store."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .admission import AdmissionService
from .audit import AuditService
from .identity import NodeIdentity
from .leaderboard import LeaderboardService
from .proposals import ProposalService
from .publication import PublicationService
from .receipts import ReceiptService
from .scheduling import SchedulingService
from .state import StateStore
from .submission import SubmissionService
from .validation import ValidationService
from .verification import VerificationService


class Orchestrator:
    def __init__(self, store: StateStore, *, publication_jsonl: Optional[str | Path] = None,
                 identity_path: Optional[str | Path] = None):
        self.store = store
        self.identity = (NodeIdentity.load_or_create(identity_path) if identity_path
                         else NodeIdentity.generate())
        self.admission = AdmissionService(store, identity=self.identity)
        self.scheduling = SchedulingService(store)
        self.submission = SubmissionService(store)
        self.verification = VerificationService(store)
        self.receipts = ReceiptService(store)
        self.publication = PublicationService(store, publication_jsonl)
        self.audit = AuditService(self.publication)
        self.leaderboard = LeaderboardService(store)
        self.proposals = ProposalService(store)
        self.validation = ValidationService(store)

    @classmethod
    def open(cls, path: str | Path, *, publication_jsonl: Optional[str | Path] = None,
             identity_path: Optional[str | Path] = None) -> "Orchestrator":
        return cls(StateStore(path), publication_jsonl=publication_jsonl,
                   identity_path=identity_path)

    def close(self) -> None:
        self.store.close()


__all__ = ["Orchestrator"]
