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
        return self._shots

    @abstractmethod
    def submit_circuits(self, circuits: dict[str, str], **kwargs):
        pass
