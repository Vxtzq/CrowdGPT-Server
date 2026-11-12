"""Facade that wires the orchestrator services to one durable store."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .admission import AdmissionService
from .identity import NodeIdentity
from .publication import PublicationService
from .scheduling import SchedulingService
from .state import StateStore
from .submission import SubmissionService
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
        self.publication = PublicationService(store, publication_jsonl)

    @classmethod
    def open(cls, path: str | Path, *, publication_jsonl: Optional[str | Path] = None,
             identity_path: Optional[str | Path] = None) -> "Orchestrator":
        return cls(StateStore(path), publication_jsonl=publication_jsonl,
                   identity_path=identity_path)

    def close(self) -> None:
        self.store.close()


__all__ = ["Orchestrator"]
