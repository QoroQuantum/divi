# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across the whole test suite."""

import itertools
import re
from threading import Event

import numpy as np
from qiskit import QuantumCircuit

from divi.backends import AsyncJobBackend, ExecutionResult, JobStatus
from divi.circuits._payloads import bound_circuits
from divi.exceptions import ExecutionCancelledError


class FakeAsyncBackend(AsyncJobBackend):
    """Async backend recording how it is driven; ``poll`` scripts its polling.

    ``poll(backend, execution_result, progress_callback, cancellation_event)``
    replaces the default of reporting ``RUNNING`` once and completing.
    """

    def __init__(self, *, poll=None, max_retries=None, cancel_error=None, run_time=2.5):
        super().__init__(shots=100, polling_interval=0.0, max_retries=max_retries)
        self._poll = poll
        self._cancel_error = cancel_error
        self._run_time = run_time
        self._job_ids = itertools.count(1)
        self._circuits: dict[str, dict[str, str]] = {}
        self.jobs: list[ExecutionResult] = []
        self.poll_calls: list[dict] = []
        self.fetched: list[ExecutionResult] = []
        self.cancelled_jobs: list[ExecutionResult] = []
        self.polling = Event()

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        job = ExecutionResult(job_id=f"job-{next(self._job_ids)}")
        self._circuits[job.job_id] = bound_circuits(payloads)
        self.jobs.append(job)
        return job

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        self.poll_calls.append(
            {
                "loop_until_complete": loop_until_complete,
                "verbose": verbose,
                "cancellation_event": cancellation_event,
            }
        )
        self.polling.set()
        if self._poll is not None:
            return self._poll(
                self, execution_result, progress_callback, cancellation_event
            )
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        return JobStatus.COMPLETED

    def get_job_results(self, execution_result) -> ExecutionResult:
        self.fetched.append(execution_result)
        return ExecutionResult(
            results=[
                {"label": label, "results": {"00": 100}}
                for label in self._circuits[execution_result.job_id]
            ],
            run_time=self._run_time,
        )

    def cancel_job(self, execution_result):
        self.cancelled_jobs.append(execution_result)
        if self._cancel_error is not None:
            raise self._cancel_error


def block_until_cancelled(
    backend, execution_result, progress_callback, cancellation_event
):
    """Polling script: wait for the cancellation event, then raise."""
    if not cancellation_event.wait(timeout=10):
        raise TimeoutError("The cancellation event was never set.")
    raise ExecutionCancelledError("Polling interrupted.")


def raise_after_running(error_type):
    """Polling script: report ``RUNNING``, then raise ``error_type(job_id)``."""

    def poll(backend, execution_result, progress_callback, cancellation_event):
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        raise error_type(execution_result.job_id)

    return poll


def report_status(status: str, attempt: int):
    """Polling script: report ``status`` at ``attempt``, then complete."""

    def poll(backend, execution_result, progress_callback, cancellation_event):
        if progress_callback is not None:
            progress_callback(attempt, status)
        return JobStatus.COMPLETED

    return poll


def exact_match(message: str) -> str:
    """``pytest.raises(match=...)`` pattern matching ``message`` and nothing else."""
    return f"^{re.escape(message)}$"


def unitary(simulator, circuit: QuantumCircuit) -> np.ndarray:
    """``circuit``'s unitary, column ``j`` its output state on basis input ``j``,
    with qubit 0 the least-significant bit."""
    n = circuit.num_qubits
    columns = []
    for index in range(2**n):
        prepared = circuit.copy_empty_like()
        prepared.global_phase = 0
        for qubit in range(n):
            if index >> qubit & 1:
                prepared.x(qubit)
        columns.append(simulator.statevector(prepared.compose(circuit)))
    return np.column_stack(columns)


def same_unitary(
    simulator, a: QuantumCircuit, b: QuantumCircuit, atol: float = 1e-8
) -> bool:
    """Whether circuits ``a`` and ``b`` implement one unitary, up to a global
    phase."""
    return equal_up_to_phase(unitary(simulator, a), unitary(simulator, b), atol)


def equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """Whether matrices or states ``a`` and ``b`` differ only by a global phase."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pivot = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(a[pivot]) < atol:
        return False
    phase = b[pivot] / a[pivot]
    return abs(abs(phase) - 1.0) <= atol and np.allclose(a * phase, b, atol=atol)
