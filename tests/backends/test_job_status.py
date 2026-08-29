# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

import pytest

from divi.backends import (
    InsufficientCreditsError,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimedOutError,
    QoroJobError,
)
from divi.backends._job_status import JobStatus as CanonicalJobStatus
from divi.backends.runners import JobStatus as RunnerJobStatus


def test_job_status_is_reexported_without_changing_identity():
    assert JobStatus is CanonicalJobStatus
    assert RunnerJobStatus is CanonicalJobStatus


def test_job_status_matches_usher_contract():
    assert issubclass(JobStatus, StrEnum)
    assert {status.value for status in JobStatus} == {
        "PENDING",
        "SCHEDULED",
        "COMPLETED",
        "FAILED",
        "PAUSED",
        "RUNNING",
        "CANCELLED",
        "INSUFFICIENT_CREDITS",
        "TIMED_OUT",
    }


@pytest.mark.parametrize(
    ("exception_type", "status"),
    [
        (JobFailedError, JobStatus.FAILED),
        (JobCancelledError, JobStatus.CANCELLED),
        (InsufficientCreditsError, JobStatus.INSUFFICIENT_CREDITS),
        (JobTimedOutError, JobStatus.TIMED_OUT),
    ],
)
def test_terminal_job_errors_are_tied_to_their_status(exception_type, status):
    error = exception_type("job-123")

    assert isinstance(error, QoroJobError)
    assert error.job_id == "job-123"
    assert error.status is status
    assert "job-123" in str(error)
    assert status in str(error)
