# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager

from divi.circuits._qem_passes import GlobalFoldPass

__all__ = [
    "QEMProtocol",
    "ZNE",
    "ZNEExtrapolator",
    "LinearExtrapolator",
    "RichardsonExtrapolator",
]

#: Type alias for QEM context data passed between expand and reduce.
#: A plain dict carrying protocol-specific side-channel information.
QEMContext = dict


class QEMProtocol(ABC):
    """Abstract base class for Quantum Error Mitigation protocols.

    Subclasses implement two methods that mirror the pipeline's
    expand/reduce lifecycle:

    * ``expand`` — given a Qiskit :class:`DAGCircuit` (and optionally the
      observable being measured), return the DAGs to execute on quantum
      hardware and a ``QEMContext`` carrying any classically-computed
      side-channel data needed during postprocessing.
    * ``reduce`` — given the context from ``expand`` and the quantum
      results, produce a single mitigated expectation value.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def expand(
        self,
        dag: DAGCircuit,
        observable: Any | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Generate DAGs and classical context for error mitigation."""

    @abstractmethod
    def reduce(
        self,
        quantum_results: Sequence[float],
        context: QEMContext,
    ) -> float:
        """Combine quantum results with classical context into a mitigated value."""

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        """Hook called after all per-group ``reduce`` calls in an evaluation.

        Protocols can override this to inspect the collected contexts and
        emit summary diagnostics (e.g. signal-destruction warnings).
        """


class _NoMitigation(QEMProtocol):
    """A dummy default mitigation protocol — pass the circuit through."""

    @property
    def name(self) -> str:
        return "NoMitigation"

    def expand(
        self, dag: DAGCircuit, observable: Any | None = None
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        return (dag,), {}

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        if len(quantum_results) == 0:
            raise RuntimeError("NoMitigation received an empty results sequence.")
        if len(quantum_results) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")
        return quantum_results[0]


# ---------------------------------------------------------------------------
# Zero-noise extrapolation
# ---------------------------------------------------------------------------
@runtime_checkable
class ZNEExtrapolator(Protocol):
    """Structural type for zero-noise extrapolation.

    Any object with an ``extrapolate(scale_factors, results) -> float``
    method satisfies this protocol — no subclassing required.
    """

    def extrapolate(
        self,
        scale_factors: Sequence[float],
        results: Sequence[float],
    ) -> float: ...


def _validate_extrapolation_inputs(
    name: str, scale_factors: np.ndarray, results: np.ndarray
) -> None:
    """Guard against non-finite inputs that would silently corrupt extrapolation."""
    if not np.all(np.isfinite(scale_factors)):
        raise ValueError(f"{name}: scale_factors contains NaN or Inf values.")
    if not np.all(np.isfinite(results)):
        raise ValueError(f"{name}: results contains NaN or Inf values.")


class LinearExtrapolator:
    """Fit a line ``y = a + b·s`` and return ``a`` (the intercept at s=0)."""

    def extrapolate(
        self, scale_factors: Sequence[float], results: Sequence[float]
    ) -> float:
        if len(scale_factors) != len(results):
            raise ValueError(
                f"LinearExtrapolator: scale_factors and results lengths disagree "
                f"({len(scale_factors)} vs {len(results)})."
            )
        if len(scale_factors) < 2:
            raise ValueError("LinearExtrapolator requires at least 2 data points.")
        sfs = np.asarray(scale_factors, dtype=float)
        res = np.asarray(results, dtype=float)
        _validate_extrapolation_inputs("LinearExtrapolator", sfs, res)
        _, intercept = np.polyfit(sfs, res, deg=1)
        return float(intercept)


class RichardsonExtrapolator:
    """Richardson (Lagrange) extrapolation through all ``N`` points to s=0.

    Given ``(s_i, y_i)`` pairs, fits the unique polynomial of degree N-1
    passing through them and evaluates at ``s=0`` via Lagrange weights:

    ``P(0) = Σ_i y_i · Π_{j≠i} (-s_j) / (s_i - s_j)``
    """

    def extrapolate(
        self, scale_factors: Sequence[float], results: Sequence[float]
    ) -> float:
        if len(scale_factors) != len(results):
            raise ValueError(
                f"RichardsonExtrapolator: scale_factors and results lengths "
                f"disagree ({len(scale_factors)} vs {len(results)})."
            )
        if len(scale_factors) < 1:
            raise ValueError("RichardsonExtrapolator requires at least 1 data point.")
        sfs = np.asarray(scale_factors, dtype=float)
        if len(sfs) != len(np.unique(sfs)):
            raise ValueError(
                "RichardsonExtrapolator requires unique scale factors; "
                f"got duplicates in {list(scale_factors)}."
            )
        res = np.asarray(results, dtype=float)
        _validate_extrapolation_inputs("RichardsonExtrapolator", sfs, res)
        # Lagrange weights for evaluation at s=0.  When len(sfs)==1 the
        # inner product is empty → np.prod([]) = 1.0 → returns results[0].
        weights = np.array(
            [
                np.prod([-s_j / (s_i - s_j) for j, s_j in enumerate(sfs) if j != i])
                for i, s_i in enumerate(sfs)
            ]
        )
        return float(np.dot(weights, res))


# Type for the folding callable: (DAGCircuit, scale_factor) → DAGCircuit.
FoldingFn = Callable[[DAGCircuit, float], DAGCircuit]


def _default_fold(dag: DAGCircuit, scale: float) -> DAGCircuit:
    """Default folding: global unitary folding via :class:`GlobalFoldPass`."""
    qc = dag_to_circuit(dag)
    folded = PassManager([GlobalFoldPass(scale)]).run(qc)
    return circuit_to_dag(folded)


class ZNE(QEMProtocol):
    """Zero Noise Extrapolation.

    For each scale factor, applies a folding function to produce a
    noise-scaled circuit, then extrapolates the per-scale expectation
    values to ``s=0`` with the provided extrapolator.

    Args:
        scale_factors: Noise scale factors (≥ 1; e.g. ``[1, 3, 5]``).
            The default ``folding_fn`` requires odd integers; custom
            folding functions may accept arbitrary floats.
        folding_fn: ``(DAGCircuit, scale) → DAGCircuit``.
            Defaults to global unitary folding via
            :class:`~divi.circuits._qem_passes.GlobalFoldPass`.  Pass a
            custom callable for local folding, random gate folding, or
            any other noise-scaling strategy.
        extrapolator: Any object with an
            ``extrapolate(scale_factors, results) -> float`` method.
            No subclassing required — just implement the method.
            Defaults to :class:`RichardsonExtrapolator`.

    Example — custom folding + custom extrapolator::

        def my_local_fold(dag, scale):
            ...  # your per-gate folding logic
            return folded_dag

        class MyExtrapolator:
            def extrapolate(self, scale_factors, results):
                ...  # your curve-fitting logic
                return zero_noise_value

        zne = ZNE(
            scale_factors=[1.0, 1.5, 2.0],
            folding_fn=my_local_fold,
            extrapolator=MyExtrapolator(),
        )
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        folding_fn: FoldingFn | None = None,
        extrapolator: ZNEExtrapolator | None = None,
    ):
        if not isinstance(scale_factors, Sequence) or not all(
            isinstance(e, (int, float)) for e in scale_factors
        ):
            raise ValueError("scale_factors must be a sequence of real numbers.")
        if not all(e >= 1.0 for e in scale_factors):
            raise ValueError("All scale factors must be ≥ 1.0.")

        if extrapolator is not None and not isinstance(extrapolator, ZNEExtrapolator):
            raise ValueError(
                f"extrapolator must be a ZNEExtrapolator, got "
                f"{type(extrapolator).__name__}."
            )

        self._scale_factors = scale_factors
        self._folding_fn = folding_fn or _default_fold
        self._extrapolator = extrapolator or RichardsonExtrapolator()

    @property
    def name(self) -> str:
        return "zne"

    @property
    def scale_factors(self) -> Sequence[float]:
        return self._scale_factors

    @property
    def extrapolator(self) -> ZNEExtrapolator:
        return self._extrapolator

    @property
    def folding_fn(self) -> FoldingFn:
        return self._folding_fn

    def expand(
        self, dag: DAGCircuit, observable: Any | None = None
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        folded = tuple(self._folding_fn(dag, s) for s in self._scale_factors)
        return folded, {}

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        return self._extrapolator.extrapolate(
            self._scale_factors, list(quantum_results)
        )
