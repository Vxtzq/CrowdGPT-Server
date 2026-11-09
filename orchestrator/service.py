"""Facade that wires the orchestrator services to one durable store."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .admission import AdmissionService
from .publication import PublicationService
from .scheduling import SchedulingService
from .state import StateStore
from .submission import SubmissionService
from .verification import VerificationService


class Orchestrator:
    def __init__(self, store: StateStore, *, publication_jsonl: Optional[str | Path] = None):
        self.store = store
        self.admission = AdmissionService(store)
        self.scheduling = SchedulingService(store)
        self.submission = SubmissionService(store)
        self.verification = VerificationService(store)
        self.publication = PublicationService(store, publication_jsonl)

    @classmethod
    def open(cls, path: str | Path, *, publication_jsonl: Optional[str | Path] = None) -> "Orchestrator":
        return cls(StateStore(path), publication_jsonl=publication_jsonl)

    def close(self) -> None:
        self.store.close()


__all__ = ["Orchestrator"]
