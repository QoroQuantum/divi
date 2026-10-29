# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The contracts every backend implements, and the result type they return."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from threading import Event
from typing import Any, Self

import numpy as np
import requests
from qiskit import QuantumCircuit

from divi.backends._job_status import JobStatus
from divi.backends._pauli_serde import group_ham_ops, ham_ops_terms_per_circuit
from divi.backends._shot_allocation import (
    from_per_circuit,
    per_circuit_or_none,
    to_wire,
)
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
    ``"label"`` (str) and ``"results"`` (dict) keys, plus an optional
    ``"metadata"`` dict of whatever else the backend reported for the circuit."""

    job_id: str | None = None
    """Job identifier for asynchronous backends."""

    backend_jobs: int = 1
    """Number of physical backend submissions represented by this result."""

    run_time: float = 0.0
    """Seconds the backend reports spending on execution, summed over the
    circuits or jobs it times separately. ``0.0`` for a backend that reports
    none, and for an asynchronous job until
    :meth:`~divi.backends.AsyncJobBackend.get_job_results` fetches it."""

    def is_async(self) -> bool:
        """Check if this result represents an async job.

        Returns:
            bool: True if job_id is not None and results are None (async backend),
                False otherwise (sync backend or results already fetched).
        """
        return self.job_id is not None and self.results is None

    def with_results(self, results: Sequence[dict], run_time: float = 0.0) -> Self:
        """Create a new ExecutionResult with results populated.

        This method creates a new instance with results set, effectively converting
        an async ExecutionResult to a completed one.

        Args:
            results: The job results to populate.
            run_time: Execution time in seconds the backend reported for the job.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results populated
                and job_id preserved.
        """
        return replace(self, results=results, run_time=run_time)


@dataclass(frozen=True)
class PerCircuitKwarg:
    """A ``submit_circuits`` value with one entry per circuit, carried in one or
    more keyword arguments in a compressed form."""

    names: tuple[str, ...]
    """The keyword arguments that carry the value."""

    decode: Callable[[Mapping[str, Any], int], list | None]
    """One value per circuit from a call's kwargs, or ``None`` when unset."""

    encode: Callable[[Sequence[Any]], dict[str, Any]]
    """The kwargs carrying one value per circuit."""

    fill: Any = None
    """The value a circuit takes when its call leaves this unset. ``None``
    means there is none, so calls that set it and calls that do not cannot be
    merged into one."""

    label: str = ""
    """How a call that sets the value is described in progress output."""

    def is_set(self, kwargs: Mapping[str, Any]) -> bool:
        """Whether ``kwargs`` sets the value."""
        return any(kwargs.get(name) is not None for name in self.names)


def _decode_shots(kwargs: Mapping[str, Any], n_circuits: int) -> list[int] | None:
    return per_circuit_or_none(kwargs.get("shot_groups"), n_circuits)


def _encode_shots(shots: Sequence[int]) -> dict[str, Any]:
    return {"shot_groups": to_wire(from_per_circuit(shots))}


def _decode_observables(kwargs: Mapping[str, Any], n_circuits: int) -> list[str] | None:
    ham_ops = kwargs.get("ham_ops")
    if ham_ops is None:
        return None
    return [
        ";".join(terms)
        for terms in ham_ops_terms_per_circuit(
            n_circuits, ham_ops, kwargs.get("circuit_ham_map")
        )
    ]


def _encode_observables(observables: Sequence[str]) -> dict[str, Any]:
    ham_ops, circuit_ham_map = group_ham_ops(observables)
    if len(circuit_ham_map) == 1:
        return {"ham_ops": ham_ops}
    return {"ham_ops": ham_ops, "circuit_ham_map": circuit_ham_map}


class CircuitRunner(ABC):
    """
    A generic interface for anything that can "run" quantum circuits.
    """

    def __init__(self, shots: int, track_depth: bool = False):
        if not isinstance(shots, Integral) or isinstance(shots, bool) or shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {shots}.")

        self._shots = int(shots)
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

    @property
    def per_circuit_kwargs(self) -> tuple[PerCircuitKwarg, ...]:
        """The ``submit_circuits`` kwargs that carry one value per circuit.

        Merging calls concatenates their circuits, so these are decoded per
        circuit, concatenated and encoded again; every other kwarg applies to
        the whole call. A backend taking further per-circuit kwargs extends this.
        """
        return (
            PerCircuitKwarg(
                names=("shot_groups",),
                decode=_decode_shots,
                encode=_encode_shots,
                fill=self.shots,
            ),
            PerCircuitKwarg(
                names=("ham_ops", "circuit_ham_map"),
                decode=_decode_observables,
                encode=_encode_observables,
                label="expval",
            ),
        )

    def _reject_unknown_options(self, kwargs: Mapping[str, Any]) -> None:
        """Raise on keyword arguments this backend does not take."""
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.submit_circuits() got unexpected keyword "
                f"arguments {sorted(kwargs)}."
            )

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
        return float(np.std(all_depths)) if all_depths else 0.0

    def clear_depth_history(self) -> None:
        """Clear the depth history. Use when reusing the backend for a new run."""
        self._depth_history.clear()


class AsyncJobBackend(CircuitRunner):
    """A :class:`CircuitRunner` that runs circuits as an asynchronous remote job.

    Implementations submit work to a scheduler (cloud HPC, hardware queue,
    etc.), so :meth:`~divi.backends.CircuitRunner.submit_circuits` returns an
    :class:`~divi.backends.ExecutionResult` carrying a ``job_id`` rather than
    circuit results. Callers inspect or wait for state via
    :meth:`poll_job_status`, fetch outcomes with :meth:`get_job_results`, and
    may :meth:`cancel_job` an in-flight handle.
    """

    polling_interval: float
    """Seconds to wait between status polls."""

    max_retries: int | None
    """Maximum number of status polls; ``None`` polls until the job ends."""

    def __init__(
        self,
        shots: int,
        track_depth: bool = False,
        *,
        polling_interval: float,
        max_retries: int | None,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self.polling_interval = polling_interval
        self.max_retries = max_retries

    @abstractmethod
    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cancellation_event: Event | None = None,
    ) -> JobStatus:
        """Inspect once, or wait for success and raise on terminal failure.

        Args:
            execution_result: Handle returned by
                :meth:`~divi.backends.CircuitRunner.submit_circuits`.
            loop_until_complete: When ``False``, return the current status.
                When ``True``, poll until completion or terminal failure.
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
            The current status for a one-shot poll, or ``COMPLETED`` when waiting.

        Raises:
            ~divi.exceptions.ExecutionCancelledError: If local cancellation is
                requested while waiting.
            ~divi.backends.QoroJobError: If a Qoro job reaches an unsuccessful
                terminal status while waiting. Other implementations may raise
                their corresponding backend-specific terminal-job exception.
        """

    @abstractmethod
    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult:
        """Fetch results for a completed job and return them populated, with
        the job's reported ``run_time``.

        Must only be called after :meth:`poll_job_status` reports a
        ``COMPLETED`` :class:`~divi.backends.JobStatus`.
        """

    @abstractmethod
    def cancel_job(self, execution_result: ExecutionResult) -> requests.Response:
        """Request cancellation of an in-flight job.

        Must be idempotent: cancelling a job already in a terminal state is a
        normal race outcome and should not raise (a 409 from the scheduler
        is acceptable to either swallow or surface as a recognisable
        exception that callers can ignore).
        """
