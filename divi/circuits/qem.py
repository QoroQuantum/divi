# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from qiskit.dagcircuit import DAGCircuit

from divi.circuits._qem_passes import (
    GlobalFoldPass,
    LocalFoldPass,
    _compute_effective_scale,
    _count_foldable_gates,
)

__all__ = [
    "QEMProtocol",
    "ZNE",
    "ZNEExtrapolator",
    "LinearExtrapolator",
    "RichardsonExtrapolator",
    "GlobalFoldPass",
    "LocalFoldPass",
    "FoldingFn",
    "global_fold",
    "local_fold",
]

#: Type alias for QEM context data passed between expand and reduce.
#: A plain dict carrying protocol-specific side-channel information.
QEMContext = dict


class QEMProtocol(ABC):
    """Abstract base class for Quantum Error Mitigation protocols.

    Subclasses implement two methods that mirror the pipeline's
    expand/reduce lifecycle:

    * ``expand`` — given a Qiskit :class:`~qiskit.dagcircuit.DAGCircuit`
      (and optionally the observable being measured), return the DAGs to
      execute on quantum hardware and a ``QEMContext`` carrying any
      classically-computed side-channel data needed during postprocessing.
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
        """Generate DAGs and classical context for error mitigation.

        The input ``dag`` is consumed by this method: implementations may
        mutate it, and callers must not retain it expecting the original
        state.  This matches the broader pipeline convention for
        ``consumes_dag_bodies`` stages (see
        :class:`~divi.pipeline.abc.BundleStage`).
        """

    def dry_expand(
        self,
        dag: DAGCircuit,
        observable: Any | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Analytic counterpart to :meth:`expand` used by dry-run pipelines.

        Must emit the **same number of DAGs** as :meth:`expand` would on the
        same input and populate any context keys that
        :meth:`~divi.pipeline.stages.QEMStage.introspect` inspects
        (``n_rotations``, ``n_paths``, ``symbolic``) so dry-run reports
        render correctly. Implementations
        should skip any computation that only matters at reduction time —
        classical simulation, weight evaluation, deep-copying the DAG for
        each scale factor, etc.

        The default implementation falls back to :meth:`expand`, which is
        correct but not necessarily fast; override on expensive protocols
        (e.g. QuEPP's Clifford simulation).
        """
        return self.expand(dag, observable)

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


#: Type for the folding callable — given a ``DAGCircuit`` and a requested
#: ``scale_factor``, return ``(folded_dag, effective_scale)``.  The second
#: value is the scale actually realised (it may differ from the request
#: when the gate count is too small for the fractional part to round
#: cleanly) and is forwarded to the extrapolator.
#:
#: Contract for implementers:
#:
#: * Callables **consume** their input ``DAGCircuit`` — callers pass a
#:   DAG they no longer need, and the fold is free to mutate it.
#: * By convention ``folding_fn(dag, 1.0)`` returns the DAG unmodified
#:   with ``effective_scale=1.0`` — both built-in folds honor this.
FoldingFn = Callable[[DAGCircuit, float], tuple[DAGCircuit, float]]


def global_fold(dag: DAGCircuit, scale: float) -> tuple[DAGCircuit, float]:
    """Apply :class:`GlobalFoldPass` and return ``(folded_dag, effective_scale)``.

    Mutates ``dag`` in place (deepcopy first if the original is needed).
    """
    effective = _compute_effective_scale(_count_foldable_gates(dag), scale)
    return GlobalFoldPass(scale).run(dag), effective


def local_fold(
    dag: DAGCircuit,
    scale: float,
    *,
    selection: Literal["random", "from_left", "from_right"] = "random",
    exclude: set[str] | None = None,
    rng=None,
) -> tuple[DAGCircuit, float]:
    """Apply :class:`LocalFoldPass` and return ``(folded_dag, effective_scale)``.

    For use with :class:`ZNE` via ``functools.partial`` when customising
    ``selection`` / ``exclude`` / ``rng``::

        from functools import partial
        zne = ZNE(
            scale_factors=[1.0, 1.5, 2.0],
            folding_fn=partial(local_fold, selection="from_left"),
        )

    Mutates ``dag`` in place (deepcopy first if the original is needed).
    """
    pass_ = LocalFoldPass(scale, selection=selection, exclude=exclude, rng=rng)
    effective = pass_.effective_scale(dag)
    return pass_.run(dag), effective


class ZNE(QEMProtocol):
    """Zero Noise Extrapolation.

    For each scale factor, applies a folding function to produce a
    noise-scaled circuit, then extrapolates the per-scale expectation
    values to ``s=0`` with the provided extrapolator.

    **Choosing a folding strategy**

    * :func:`global_fold` (default) — applies ``(U†·U)^k`` + partial tail
      fold at the *circuit* level.  Deterministic, good first choice for
      widely-spaced scales (e.g. ``[1, 3, 5]``).
    * :func:`local_fold` — per-gate folding with fractional-scale
      support.  Use when you need scales close to 1 (``[1.0, 1.25, 1.5]``)
      on deep circuits where global folding would explode the gate count,
      or when you want to skip specific gates during folding
      (``exclude={"cx"}``).  ``global_fold`` has no equivalent exclude
      mechanism — it folds the whole unitary.

    **Effective vs requested scales**

    The achievable scale factors form a discrete grid of granularity
    ``2/d`` (``d`` = foldable gate count).  For small ``d`` a requested
    non-integer scale may snap to a different value.  ``expand`` reports
    the *effective* scale factors via the context and :meth:`reduce`
    forwards them to the extrapolator, so extrapolation stays unbiased.
    A warning is emitted if two requested scales collapse to the same
    effective value.

    Args:
        scale_factors: Noise scale factors (≥ 1; e.g. ``[1, 3, 5]`` or
            ``[1.0, 1.5, 2.0]``).  Arbitrary real values ≥ 1 are
            supported by both default folds.
        folding_fn: ``(DAGCircuit, scale) → (DAGCircuit, effective_scale)``.
            Defaults to :func:`global_fold`.  Pass :func:`local_fold` (or
            ``functools.partial(local_fold, selection=...)``) for local
            folding, or any custom callable.  See
            :data:`~divi.circuits.qem.FoldingFn` for the input-mutation
            contract and the ``scale=1.0`` pass-through convention.
        extrapolator: Any object with an
            ``extrapolate(scale_factors, results) -> float`` method.
            No subclassing required — just implement the method.
            Defaults to :class:`RichardsonExtrapolator`.

    Example — switch to local folding::

        from divi.circuits.qem import ZNE, local_fold

        zne = ZNE(
            scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
            folding_fn=local_fold,
        )

    Example — custom folding + custom extrapolator::

        def my_fold(dag, scale):
            ...  # your folding logic
            return folded_dag, effective_scale  # both required

        class MyExtrapolator:
            def extrapolate(self, scale_factors, results):
                ...
                return zero_noise_value

        zne = ZNE(
            scale_factors=[1.0, 1.5, 2.0],
            folding_fn=my_fold,
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
        self._folding_fn = folding_fn or global_fold
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
        # Expand takes ownership of `dag` (see QEMProtocol.expand).  For
        # S scale factors we need S distinct folded outputs; the input
        # can absorb one of them, so we deepcopy S-1 times instead of S.
        # (Deepcopy of a Rust-backed DAGCircuit is the fastest clone
        # primitive available — see PauliTwirlStage._apply_twirl_substitute
        # for the benchmarking rationale.)
        scales = self._scale_factors
        folded_pairs = [self._folding_fn(copy.deepcopy(dag), s) for s in scales[:-1]]
        folded_pairs.append(self._folding_fn(dag, scales[-1]))
        folded_dags = tuple(pair[0] for pair in folded_pairs)
        effective_scales = tuple(float(pair[1]) for pair in folded_pairs)

        if len(set(effective_scales)) < len(effective_scales):
            warnings.warn(
                f"ZNE: requested scale factors {list(self._scale_factors)} "
                f"collapse to effective scales {list(effective_scales)} — "
                f"the foldable gate count is too small for the requested "
                f"granularity.  Extrapolation may fail or be biased; "
                f"consider fewer scale factors, integer scales only, or a "
                f"circuit with more foldable gates.",
                stacklevel=2,
            )

        return folded_dags, {"effective_scales": effective_scales}

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        scales = context.get("effective_scales", self._scale_factors)
        return self._extrapolator.extrapolate(scales, list(quantum_results))
