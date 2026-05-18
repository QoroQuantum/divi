# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for backends that execute circuits as async jobs.

Implemented by backends like :class:`~divi.backends.QoroService` that submit
work to a remote scheduler and return a job handle, then expose polling,
result-fetching, and cancellation against that handle.
"""

from collections.abc import Callable, Mapping
from threading import Event
from typing import Protocol, runtime_checkable

import requests

from divi.backends import ExecutionResult


@runtime_checkable
class AsyncJobBackend(Protocol):
    """Backend that runs circuits as an asynchronous remote job.

    Implementations submit work to a scheduler (cloud HPC, hardware queue,
    etc.) and return an :class:`~divi.backends.ExecutionResult` carrying a
    ``job_id`` rather than circuit results. Callers then poll for status via
    :meth:`poll_job_status`, fetch outcomes with :meth:`get_job_results`,
    and may :meth:`cancel_job` an in-flight handle.
    """

    @property
    def shots(self) -> int:
        """Number of measurement shots applied to sampling-mode circuits."""
        ...

    def submit_circuits(
        self,
        circuits: Mapping[str, str],
        *,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit a batch of QASM circuits and return a handle.

        The returned :class:`~divi.backends.ExecutionResult` carries the
        scheduler-side ``job_id`` but no circuit results; populate it via
        :meth:`get_job_results` once polling reports a terminal status.

        Args:
            circuits: Mapping of unique label → OpenQASM source.
            cancellation_event: When set, the implementation should refuse
                to dispatch (or short-circuit dispatch) and raise
                :class:`~divi.exceptions.ExecutionCancelledError`. The same
                event is honoured by :meth:`poll_job_status` to interrupt
                an in-flight polling loop.
            **kwargs: Backend-specific options (``ham_ops``, ``shot_groups``, …).
        """
        ...

    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        on_complete: Callable[[requests.Response], None] | None = None,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cancellation_event: Event | None = None,
    ):
        """Query the scheduler-side job state; optionally block until terminal.

        Args:
            execution_result: Handle returned by :meth:`submit_circuits`.
            loop_until_complete: If ``True``, poll until a terminal status
                (``COMPLETED`` / ``FAILED`` / ``CANCELLED``); otherwise return
                after a single query.
            on_complete: Invoked with the final HTTP response when a terminal
                status is reached.
            verbose: When ``True``, log per-poll status. Disable when
                rendering progress via ``progress_callback`` so user-facing
                output isn't doubled.
            progress_callback: Called as ``(poll_attempt, status_str)`` for
                progress-bar updates.
            cancellation_event: When set, the loop exits by raising
                :class:`~divi.exceptions.ExecutionCancelledError`. In-flight
                HTTP requests are not interrupted — cancellation latency is
                bounded by the per-request timeout.

        Returns:
            The most recent :class:`~divi.backends.JobStatus`.
        """
        ...

    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult:
        """Fetch results for a completed job and return them populated.

        Must only be called after :meth:`poll_job_status` reports a
        ``COMPLETED`` :class:`~divi.backends.JobStatus`.
        """
        ...

    def cancel_job(self, execution_result: ExecutionResult) -> requests.Response:
        """Request cancellation of an in-flight job.

        Must be idempotent: cancelling a job already in a terminal state is a
        normal race outcome and should not raise (a 409 from the scheduler
        is acceptable to either swallow or surface as a recognisable
        exception that callers can ignore).
        """
        ...
