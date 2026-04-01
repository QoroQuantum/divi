# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from cirq.circuits.circuit import Circuit
from mitiq.zne import combine_results, construct_circuits
from mitiq.zne.inference import Factory

#: Type alias for QEM context data passed between expand and reduce.
#: A plain dict carrying protocol-specific side-channel information.
QEMContext = dict


class QEMProtocol(ABC):
    """Abstract base class for Quantum Error Mitigation protocols.

    Subclasses implement two methods that mirror the pipeline's
    expand/reduce lifecycle:

    * ``expand`` — given a Cirq circuit (and optionally the observable being
      measured), return the circuits to execute on quantum hardware and a
      :class:`QEMContext` carrying any classically-computed side-channel
      data needed during postprocessing.
    * ``reduce`` — given the context from ``expand`` and the quantum results,
      produce a single mitigated expectation value.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def expand(
        self,
        cirq_circuit: Circuit,
        observable: Any | None = None,
    ) -> tuple[tuple[Circuit, ...], QEMContext]:
        """Generate circuits and classical context for error mitigation.

        Args:
            cirq_circuit: The bound quantum circuit to mitigate.
            observable: The observable being measured (e.g. a PennyLane
                operator). Protocols that require observable information
                (like QuEPP) should raise if this is ``None``.

        Returns:
            A tuple of ``(circuits, context)`` where *circuits* are the Cirq
            circuits to execute on the quantum backend, and *context* is a
            :class:`QEMContext` with optional classical side-channel data
            for the reduce phase.
        """

    @abstractmethod
    def reduce(
        self,
        quantum_results: Sequence[float],
        context: QEMContext,
    ) -> float:
        """Combine quantum results with classical context into a mitigated value.

        Args:
            quantum_results: Expectation values from executing the circuits
                returned by ``expand``, in the same order.
            context: The :class:`QEMContext` produced by ``expand``.

        Returns:
            The mitigated expectation value.
        """

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        """Hook called after all per-group ``reduce`` calls in an evaluation.

        Protocols can override this to inspect the collected contexts and
        emit summary diagnostics (e.g. signal-destruction warnings).
        The default implementation is a no-op.

        Args:
            contexts: All :class:`QEMContext` objects from the current
                evaluation batch.
        """


class _NoMitigation(QEMProtocol):
    """A dummy default mitigation protocol."""

    @property
    def name(self) -> str:
        return "NoMitigation"

    def expand(
        self, cirq_circuit: Circuit, observable: Any | None = None
    ) -> tuple[tuple[Circuit, ...], QEMContext]:
        return (cirq_circuit,), {}

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        if len(quantum_results) == 0:
            raise RuntimeError("NoMitigation received an empty results sequence.")
        if len(quantum_results) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")
        return quantum_results[0]


class ZNE(QEMProtocol):
    """Zero Noise Extrapolation (ZNE) quantum error mitigation protocol.

    Uses Mitiq to construct noise-scaled circuits and extrapolate to the
    zero-noise limit.
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
            raise ValueError("extrapolation_factory is expected to be of type Factory.")

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

    def expand(
        self, cirq_circuit: Circuit, observable: Any | None = None
    ) -> tuple[tuple[Circuit, ...], QEMContext]:
        circuits = construct_circuits(
            cirq_circuit,
            scale_factors=self._scale_factors,
            scale_method=self._folding_fn,
        )
        return tuple(circuits), {}

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        return combine_results(
            scale_factors=self._scale_factors,
            results=list(quantum_results),
            extrapolation_method=self._extrapolation_factory.extrapolate,
        )
