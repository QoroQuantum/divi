# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from queue import Queue
from threading import Event
from typing import Any

from divi.backends import CircuitRunner


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    This class defines the minimal interface that all quantum algorithms
    must implement. It provides no concrete functionality, ensuring
    maximum flexibility for different types of quantum algorithms.

    Subclasses must implement:
        - run(): Execute the quantum algorithm
        - _generate_circuits(): Generate quantum circuits for execution
        - _post_process_results(): Process execution results

    Attributes:
        backend (CircuitRunner): The quantum circuit execution backend.
        _seed (int | None): Random seed for reproducible results.
        _progress_queue (Queue | None): Queue for progress reporting.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        **kwargs,
    ):
        """Initialize the QuantumProgram.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            seed (int | None): Random seed for reproducible results. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.
            **kwargs: Additional keyword arguments for subclasses.
        """
        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue
        self._total_circuit_count = 0
        self._total_run_time = 0.0

    @abstractmethod
    def run(self) -> tuple[int, float]:
        """Execute the quantum algorithm.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: Total number of circuits executed
                - float: Total runtime in seconds
        """
        pass

    @abstractmethod
    def _generate_circuits(self, **kwargs):
        """Generate quantum circuits for execution.

        Args:
            **kwargs: Additional keyword arguments for circuit generation.
        """
        pass

    @abstractmethod
    def _post_process_results(self, results: dict) -> Any:
        """Process execution results.

        Args:
            results (dict): Raw results from circuit execution.

        Returns:
            Any: Processed results specific to the algorithm.
        """
        pass

    def _set_cancellation_event(self, event: Event):
        """Set a cancellation event for graceful program termination.

        This method is called by batch runners to provide a mechanism
        for stopping the optimization loop cleanly when requested.

        Args:
            event (Event): Threading Event object that signals cancellation when set.
        """
        self._cancellation_event = event

    @property
    def total_circuit_count(self) -> int:
        """Get the total number of circuits executed.

        Returns:
            int: Cumulative count of circuits submitted for execution.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self) -> float:
        """Get the total runtime across all circuit executions.

        Returns:
            float: Cumulative execution time in seconds.
        """
        return self._total_run_time
