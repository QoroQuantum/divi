# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np

from divi.backends._execution_result import ExecutionResult


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
                  returns results immediately (e.g., ParallelSimulator).
        """
        return False

    @abstractmethod
    def submit_circuits(self, circuits: dict[str, str], **kwargs) -> ExecutionResult:
        """
        Submit quantum circuits for execution.

        This abstract method must be implemented by subclasses to define how
        circuits are executed on their respective backends (simulator, hardware, etc.).

        Args:
            circuits (dict[str, str]): Dictionary mapping circuit labels to their
                OpenQASM string representations.
            **kwargs: Additional backend-specific parameters for circuit execution.

        Returns:
            ExecutionResult: For synchronous backends, contains results directly.
                For asynchronous backends, contains a job_id that can be used to
                fetch results later.
        """
        pass

    @property
    def depth_history(self) -> list[list[int]]:
        """Circuit depth per batch when :attr:`track_depth` is True.

        Each element is a list of depths (one per circuit) for that submission.
        Empty when track_depth is False or before any circuits have been run.
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
