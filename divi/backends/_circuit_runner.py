# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from threading import Event
from typing import Protocol, runtime_checkable

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.qasm2 import QASM2ExportError

from divi.circuits import TemplateEntry

from ._execution_result import ExecutionResult

#: Accepted input to :meth:`CircuitRunner.submit_circuits`: a mapping of label →
#: circuit, or a sequence of circuits labelled by positional index.
CircuitBatch = Mapping[str, str | QuantumCircuit] | Sequence[str | QuantumCircuit]


def _to_qasm(circuit: str | QuantumCircuit, label: str) -> str:
    if isinstance(circuit, str):
        return circuit
    if isinstance(circuit, QuantumCircuit):
        try:
            return qasm2.dumps(circuit)
        except QASM2ExportError as e:
            raise ValueError(
                f"Circuit '{label}' cannot be exported to OpenQASM 2: {e}"
            ) from e
    raise TypeError(
        f"Circuit '{label}' must be an OpenQASM string or a QuantumCircuit, "
        f"got {type(circuit).__name__}."
    )


def normalise_circuit_batch(circuits: CircuitBatch) -> dict[str, str]:
    """Reduce any accepted circuit collection to label → OpenQASM.

    Sequences are labelled by positional index. Qiskit circuits are exported
    to OpenQASM 2.

    Raises:
        TypeError: If a single circuit is passed instead of a collection, or
            an element is neither a string nor a QuantumCircuit.
        ValueError: If a QuantumCircuit cannot be exported to OpenQASM 2.
    """
    if isinstance(circuits, (str, QuantumCircuit)):
        raise TypeError(
            "submit_circuits expects a collection of circuits; wrap a single "
            "circuit in a list."
        )

    if isinstance(circuits, Mapping):
        return {label: _to_qasm(c, label) for label, c in circuits.items()}

    return {str(i): _to_qasm(c, str(i)) for i, c in enumerate(circuits)}


@runtime_checkable
class SupportsCircuitTemplates(Protocol):
    """Capability protocol for backends that resolve parametric QASM
    templates server-side.

    The pipeline's deferred-binding path is gated on
    ``isinstance(backend, SupportsCircuitTemplates)``; a backend opts in
    simply by implementing :meth:`submit_circuit_templates` — no inheritance
    is required, and no capability flag has to be plumbed through the base
    class for backends that have nothing else to share with the protocol.
    """

    def submit_circuit_templates(
        self,
        templates: list[TemplateEntry],
        *,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult: ...


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
        circuits: CircuitBatch,
        *,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Submit quantum circuits for execution.

        This abstract method must be implemented by subclasses to define how
        circuits are executed on their respective backends (simulator, hardware, etc.).
        Implementations must pass ``circuits`` through
        :func:`~divi.backends.normalise_circuit_batch` before use.

        Args:
            circuits: Mapping of circuit label → OpenQASM string or
                :class:`~qiskit.circuit.QuantumCircuit`, or a bare sequence of
                circuits labelled by positional index.
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
