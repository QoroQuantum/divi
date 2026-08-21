# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


class ExecutionCancelledError(Exception):
    """Signal that a running program, job, or batch was cooperatively cancelled."""


class CharacterizationFailedError(Exception):
    """A characterisation job did not produce a usable report.

    Raised when the server reports the job as failed, or returns it as
    complete with no analysis attached. Carries the ``job_id`` so the job
    can be inspected or re-fetched.
    """

    job_id: str
    """Identifier of the job that failed."""

    status: str
    """Terminal status the server reported."""

    def __init__(self, job_id: str, status: str, detail: str):
        super().__init__(f"Characterization job {job_id} ({status}): {detail}")
        self.job_id = job_id
        self.status = status


class CharacterizationSubmitError(Exception):
    """A created characterisation job did not return a usable response.

    Carries the ``job_id`` and failed ``phase`` so callers can inspect or
    re-fetch the existing job instead of creating a second billable submission.
    The underlying transport or server error is chained as ``__cause__``.
    """

    job_id: str
    """Identifier of the job that existed before the failure."""

    phase: str
    """Operation that failed, such as ``"submission"`` or result retrieval."""

    def __init__(self, job_id: str, cause: Exception, *, phase: str = "submission"):
        super().__init__(
            f"Characterization job {job_id} was created, but {phase} did not "
            f"return safely ({type(cause).__name__}: {cause}). Inspect or fetch "
            f"that job with get_characterization_result({job_id!r}) rather than "
            "submitting the QUBO again."
        )
        self.job_id = job_id
        self.phase = phase
