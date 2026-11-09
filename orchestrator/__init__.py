"""Durable, HTTP-free orchestrator core.

The FastAPI server is a transport adapter.  Everything that decides who trains,
what a ticket permits, which pool a client belongs to, and what gets published
lives here and can be exercised without importing FastAPI.
"""
from .admission import AdmissionService
from .audit import AuditService
from .domain import PoolView, RoundView, SubmissionView, TicketView, VerificationView
from .proposals import ProposalService
from .publication import PublicationService
from .receipts import ReceiptService
from .scheduling import SchedulingService
from .service import Orchestrator
from .state import SCHEMA_VERSION, StateStore
from .submission import SubmissionService
from .validation import ValidationService
from .verification import VerificationService

__all__ = [
    "AdmissionService", "AuditService", "LeaderboardService", "Orchestrator", "PoolView", "ProposalService", "PublicationService",
    "ReceiptService", "RoundView", "SCHEMA_VERSION", "SchedulingService", "StateStore",
    "SubmissionService", "SubmissionView", "TicketView", "ValidationService", "VerificationService",
    "VerificationView",
]
