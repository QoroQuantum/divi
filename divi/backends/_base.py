# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The contracts every backend implements, and the result type they return."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from threading import Event
from typing import Protocol, Self, runtime_checkable

import numpy as np
import requests
from qiskit import QuantumCircuit

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits


def normalise_circuit_batch(circuits: CircuitBatch) -> dict[str, str]:
    """Reduce any accepted circuit collection to label → OpenQASM.

    Sequences are labelled by positional index. Qiskit circuits are exported
    to OpenQASM 2.

    Raises:
        TypeError: If a single circuit is passed instead of a collection, or
            an element is neither a string nor a QuantumCircuit.
        ValueError: If a QuantumCircuit cannot be exported to OpenQASM 2.
    """
    return bound_circuits(circuits)


@dataclass(frozen=True)
class ExecutionResult:
    """Result container for circuit execution.

    This class provides a unified return type for all CircuitRunner.submit_circuits()
    methods. For synchronous backends, it contains the results directly. For
    asynchronous backends, it contains the job_id that can be used to fetch results later.

    The class is frozen (immutable) to ensure data integrity. Use the ``with_results()``
    method to create a new instance with results populated from an async ExecutionResult.

    Examples:
        >>> # Synchronous backend
        >>> result = ExecutionResult(results=[{"label": "circuit_0", "results": {"00": 100}}])
        >>> result.is_async()
        False

        >>> # Asynchronous backend
        >>> result = ExecutionResult(job_id="job-12345")
        >>> result.is_async()
        True
        >>> # After fetching results
        >>> result = backend.get_job_results(result)
        >>> result.results is not None
        True
    """

    results: list[dict] | None = None
    """Results for synchronous backends, as a list of dicts each containing
    ``"label"`` (str) and ``"results"`` (dict) keys."""

    job_id: str | None = None
    """Job identifier for asynchronous backends."""

    def is_async(self) -> bool:
        """Check if this result represents an async job.

        Returns:
            bool: True if job_id is not None and results are None (async backend),
                False otherwise (sync backend or results already fetched).
        """
        return self.job_id is not None and self.results is None

    def with_results(self, results: list[dict]) -> Self:
        """Create a new ExecutionResult with results populated.

        This method creates a new instance with results set, effectively converting
        an async ExecutionResult to a completed one.

        Args:
            results: The job results to populate.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results populated
                and job_id preserved.
        """
        return replace(self, results=results)


class CircuitRunner(ABC):
    """
    A generic interface for anything that can "run" quantum circuits.
    """

    def __init__(self, shots: int, track_depth: bool = False):
        if shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {shots}.")

        self._shots = shots
        self.track_depth = track_depth
        self._depth_history: list[list[int]] = []

    @property
    def shots(self):
        """
        Get the number of measurement shots for circuit execution.

        Returns:
            int: Number of shots configured for this runner.
        """
        return self._shots

    @property
    @abstractmethod
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        return False

    @property
    @abstractmethod
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.

        Returns:
            bool: True if the backend returns a job ID and requires polling
                  for results (e.g., QoroService). False if the backend
                  returns results immediately (e.g., QiskitSimulator).
        """
        return False

    @property
    def resolves_parameters(self) -> bool:
        """Whether the backend substitutes parameter values itself.

        When True the pipeline hands over parametric payloads — one circuit plus
        a parameter matrix — and the backend resolves them. When False it
        binds first, so every payload arrives already resolved: no parameters and
        one row per circuit.
        """
        return False

    def set_seed(self, seed: int) -> None:
        """Seed the backend's random number generator, if supported.

        The default implementation is a no-op. Backends that can seed
        their simulation RNG should override this method.

        Args:
            seed: Seed value for the backend's RNG.
        """

    @abstractmethod
    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Submit quantum circuits for execution.

        Args:
            payloads: One :class:`~divi.circuits.CircuitPayload` per circuit
                variant. Each carries its own labelled parameter
                sets, one resolved circuit per row. Backends that do not set
                :attr:`resolves_parameters` receive them already bound —
                no parameters, one row each — which ``bound_circuits``
                flattens to a ``label -> circuit`` mapping. It also accepts the
                plain collections callers may pass in their place: a mapping of
                label → OpenQASM string or
                :class:`~qiskit.circuit.QuantumCircuit`, or a bare sequence
                labelled by positional index.
            cancellation_event: When set, the backend aborts the batch and
                raises :class:`~divi.exceptions.ExecutionCancelledError`.
                Sync backends honour it between items; async backends thread
                it into their poll loop.
            **kwargs: Additional backend-specific parameters for circuit execution.

        Returns:
            ExecutionResult: For synchronous backends, contains results directly.
                For asynchronous backends, contains a job_id that can be used to
                fetch results later.
        """

    @staticmethod
    def _reject_shot_groups_with_ham_ops(ham_ops, shot_groups) -> None:
        """Reject the one combination no backend can honour.

        Expectation values are computed analytically, so a per-circuit shot
        allocation would be silently ignored.
        """
        if ham_ops is not None and shot_groups is not None:
            raise ValueError(
                "shot_groups is incompatible with ham_ops: expectation-value "
                "mode is analytical and ignores shot counts. Pass exactly one."
            )

    def _record_qasm_depths(self, qasm_strings: Iterable[str]) -> None:
        """Append one batch of circuit depths when ``track_depth`` is set."""
        if not self.track_depth:
            return
        self._depth_history.append(
            [QuantumCircuit.from_qasm_str(qasm).depth() for qasm in qasm_strings]
        )

    @property
    def depth_history(self) -> list[list[int]]:
        """Circuit depth per batch when ``track_depth`` is True.

        Each element is a list of depths (one per circuit) for that submission.
        Empty when ``track_depth`` is False or before any circuits have been run.
        """
        return self._depth_history.copy()

    def average_depth(self) -> float:
        """Average circuit depth across all tracked submissions.

        Returns 0.0 when depth history is empty.
        """
        all_depths = [d for batch in self._depth_history for d in batch]
        return float(np.mean(all_depths)) if all_depths else 0.0

    def std_depth(self) -> float:
        """Standard deviation of circuit depth across all tracked submissions.

        Returns 0.0 when depth history is empty or has a single value.
        """
        all_depths = [d for batch in self._depth_history for d in batch]
        return float(np.std(all_depths)) if len(all_depths) > 1 else 0.0

    def clear_depth_history(self) -> None:
        """Clear the depth history. Use when reusing the backend for a new run."""
        self._depth_history.clear()


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
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit a batch of circuits and return a handle.

        The returned :class:`~divi.backends.ExecutionResult` carries the
        scheduler-side ``job_id`` but no circuit results; populate it via
        :meth:`get_job_results` once polling reports a terminal status.

        Args:
            payloads: One :class:`~divi.circuits.CircuitPayload` per circuit
                variant, or a ``{label: circuit}`` mapping of already-resolved
                circuits.
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
        on_complete: Callable[[dict], None] | None = None,
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
            on_complete: Invoked with the decoded final status payload when a
                terminal status is reached. Backends that report no timing
                metadata may skip the call.
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
