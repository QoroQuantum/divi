from abc import ABC, abstractmethod


class CircuitRunner(ABC):
    """
    A generic interface for anything that can "run" quantum circuits.
    """

    @abstractmethod
    def submit_circuits(self, circuits, shots: int, **kwargs):
        pass
