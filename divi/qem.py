# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

from cirq.circuits.circuit import Circuit
from mitiq.zne import combine_results, construct_circuits
from mitiq.zne.inference import Factory


class QEMProtocol(ABC):
    """
    Abstract Base Class for Quantum Error Mitigation (QEM) protocols.

    All concrete QEM protocols should inherit from this class and implement
    the abstract methods and properties. This ensures a consistent interface
    across different mitigation techniques.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        """
        Modifies a given Cirq circuit into one or more new circuits
        required by the QEM protocol.

        For example, a Zero Noise Extrapolation (ZNE) protocol might
        produce multiple scaled versions of the input circuit. A simple
        mitigation protocol might return the original circuit unchanged.

        Args:
            cirq_circuit (cirq.Circuit): The input quantum circuit to be modified.

        Returns:
            Sequence[cirq.Circuit]: A sequence (e.g., list or tuple) of
                                    Cirq circuits to be executed.
        """
        pass

    @abstractmethod
    def postprocess_results(self, results: Sequence[float]) -> float:
        """
        Applies post-processing (e.g., extrapolation, filtering) to the
        results obtained from executing the modified circuits.

        This method takes the raw output from quantum circuit executions
        (typically a sequence of expectation values or probabilities) and
        applies the core error mitigation logic to produce a single,
        mitigated result.

        Args:
            results (Sequence[float]): A sequence of floating-point results,
                                       corresponding to the executions of the
                                       circuits returned by `modify_circuit`.

        Returns:
            float: The single, mitigated result after post-processing.
        """
        pass


class _NoMitigation(QEMProtocol):
    """
    A dummy default mitigation protocol.
    """

    @property
    def name(self) -> str:
        return "NoMitigation"

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        # Identity, do nothing
        return [cirq_circuit]

    def postprocess_results(self, results: Sequence[float]) -> float:
        """
        Returns the single result provided, ensuring only one result is given.

        If multiple results are provided, it raises a RuntimeError, as this
        protocol expects a single measurement outcome for its input circuit.

        Args:
            results (Sequence[float]): A sequence containing a single floating-point result.

        Returns:
            float: The single result from the sequence.

        Raises:
            RuntimeError: If more than one result is provided.
        """
        if len(results) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")

        return results[0]


class ZNE(QEMProtocol):
    """
    Implements the Zero Noise Extrapolation (ZNE) quantum error mitigation protocol.

    This protocol uses `Mitiq`'s functionalities to construct noise-scaled
    circuits and then extrapolate to the zero-noise limit based on the
    obtained results.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        folding_fn: Callable,
        extrapolation_factory: Factory,
    ):
        """
        Initializes a ZNE protocol instance.

        Args:
            scale_factors (Sequence[float]): A sequence of noise scale factors
                                             to be applied to the circuits. These
                                             factors typically range from 1.0 upwards.
            folding_fn (Callable): A callable (e.g., a `functools.partial` object)
                                   that defines how the circuit should be "folded"
                                   to increase noise. This function must accept
                                   a `cirq.Circuit` and a `float` (scale factor)
                                   as its first two arguments.
            extrapolation_factory (mitiq.zne.inference.Factory): An instance of
                                                                `Mitiq`'s `Factory`
                                                                class, which provides
                                                                the extrapolation method.

        Raises:
            ValueError: If `scale_factors` is not a sequence of numbers,
                        `folding_fn` is not callable, or `extrapolation_factory`
                        is not an instance of `mitiq.zne.inference.Factory`.
        """
        if (
            not isinstance(scale_factors, Sequence)
            or not all(isinstance(elem, (int, float)) for elem in scale_factors)
            or not all(elem >= 1.0 for elem in scale_factors)
        ):
            raise ValueError(
                "scale_factors is expected to be a sequence of real numbers >=1."
            )

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
    def name(self) -> str:
        return "zne"

    @property
    def scale_factors(self) -> Sequence[float]:
        return self._scale_factors

    @property
    def folding_fn(self):
        return self._folding_fn

    @property
    def extrapolation_factory(self):
        return self._extrapolation_factory

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        return construct_circuits(
            cirq_circuit,
            scale_factors=self._scale_factors,
            scale_method=self._folding_fn,
        )

    def postprocess_results(self, results: Sequence[float]) -> float:
        return combine_results(
            scale_factors=self._scale_factors,
            results=results,
            extrapolation_method=self._extrapolation_factory.extrapolate,
        )
