# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Self

__all__ = [
    "InsufficientCreditsError",
    "JobCancelledError",
    "JobFailedError",
    "JobStatus",
    "JobTimedOutError",
    "QoroJobError",
]


class JobStatus(StrEnum):
    """Status of a job on the Qoro Service."""

    PENDING = "PENDING"
    """Job is queued and waiting to be processed."""

    SCHEDULED = "SCHEDULED"
    """Job has been assigned to a scheduler."""

    RUNNING = "RUNNING"
    """Job is currently being executed."""

    PAUSED = "PAUSED"
    """Job execution is temporarily paused."""

    COMPLETED = "COMPLETED"
    """Job has finished successfully."""

    FAILED = "FAILED"
    """Job execution encountered an error."""

    CANCELLED = "CANCELLED"
    """Job was cancelled before completion."""

    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    """Job could not run because the billing account lacks credits."""

    TIMED_OUT = "TIMED_OUT"
    """Job exceeded its server-side execution deadline."""

    @classmethod
    def coerce(cls, value: object) -> Self | None:
        """Return the canonical status for a known backend value."""
        try:
            return cls(value)
        except (TypeError, ValueError):
            return None


class QoroJobError(Exception):
    """Base class for an unsuccessful terminal Qoro job."""

    job_id: str
    """Identifier of the job that reached a terminal state."""

    status: JobStatus
    """Terminal status reported by the Qoro Service."""

    reason = "did not complete successfully"

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Qoro job {job_id} {self.reason} ({self.status}).")


class JobFailedError(QoroJobError):
    """A Qoro job ended in the generic ``FAILED`` state."""

    status = JobStatus.FAILED
    reason = "failed"


class JobCancelledError(QoroJobError):
    """The Qoro scheduler cancelled a job without a local cancellation request."""

    status = JobStatus.CANCELLED
    reason = "was cancelled by the scheduler"


class InsufficientCreditsError(QoroJobError):
    """A Qoro job could not run because its billing account lacked credits."""

    status = JobStatus.INSUFFICIENT_CREDITS
    reason = "could not run because the billing account has insufficient credits"


class JobTimedOutError(QoroJobError):
    """A Qoro job exceeded its server-side execution deadline."""

    status = JobStatus.TIMED_OUT
    reason = "timed out during execution"
