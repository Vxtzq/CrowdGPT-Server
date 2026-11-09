"""Durable, HTTP-free orchestrator core.

The FastAPI server is a transport adapter.  Everything that decides who trains,
what a ticket permits, which pool a client belongs to, and what gets published
lives here and can be exercised without importing FastAPI.
"""
from .admission import AdmissionService
from .domain import PoolView, RoundView, SubmissionView, TicketView, VerificationView
from .publication import PublicationService
from .scheduling import SchedulingService
from .service import Orchestrator
from .state import SCHEMA_VERSION, StateStore
from .submission import SubmissionService
from .verification import VerificationService

__all__ = [
    "AdmissionService", "Orchestrator", "PoolView", "PublicationService",
    "RoundView", "SCHEMA_VERSION", "SchedulingService", "StateStore",
    "SubmissionService", "SubmissionView", "TicketView", "VerificationService",
    "VerificationView",
]
