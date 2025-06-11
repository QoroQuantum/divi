from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

from cirq.circuits.circuit import Circuit
from mitiq.zne.inference import Factory


class QEMProtocol(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        pass

    @abstractmethod
    def postprocess_results(self, results: Sequence[float]) -> float:
        pass


class _NoMitigation(QEMProtocol):
    """
    A dummy default mitigation protocol.
    """

    @property
    def name(self):
        return "NoMitigation"

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        # Identity, do nothing
        return [cirq_circuit]

    def postprocess_results(self, results: Sequence[float]) -> float:
        if len(results) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")

        return results[0]


class ZNE(QEMProtocol):
    def __init__(
        self,
        scale_factors: Sequence[float],
        folding_fn: Callable,
        extrapolation_factory: Factory,
    ):

        if not isinstance(scale_factors, Sequence) or not all(
            isinstance(elem, float) for elem in scale_factors
        ):
            raise ValueError("scale_factors is expected to be a sequence of floats.")

        if not isinstance(folding_fn, partial):
            raise ValueError(
                "folding_fn is expected to be of type partial with all parameters "
                "except for the circuit object and the scale factor already set."
            )

        if not isinstance(extrapolation_factory, Factory):
            raise ValueError("extrapolation_fn is expected to be of Factory.")

        self._scale_factors = scale_factors
        self._folding_fn = folding_fn
        self._extrapolation_factory = extrapolation_factory

    @property
    def name(self):
        return "zne"

    @property
    def scale_factors(self):
        return self._scale_factors

    @property
    def folding_fn(self):
        return self._folding_fn

    @property
    def extrapolation_factory(self):
        return self._extrapolation_factory

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        # TODO
        return

    def postprocess_results(self, results: Sequence[float]) -> float:
        # TODO
        return
