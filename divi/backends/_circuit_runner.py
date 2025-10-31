# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class CircuitRunner(ABC):
    """
    A generic interface for anything that can "run" quantum circuits.
    """

    def __init__(self, shots: int):
        if shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {shots}.")

        self._shots = shots

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
    def submit_circuits(self, circuits: dict[str, str], **kwargs):
        """
        Submit quantum circuits for execution.

        This abstract method must be implemented by subclasses to define how
        circuits are executed on their respective backends (simulator, hardware, etc.).

        Args:
            circuits (dict[str, str]): Dictionary mapping circuit labels to their
                OpenQASM string representations.
            **kwargs: Additional backend-specific parameters for circuit execution.

        Returns:
            The return type depends on the backend implementation. Typically returns
            measurement results or a job identifier.
        """
        pass
