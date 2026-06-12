# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-noise extrapolation protocols and folding helpers."""

import copy
import random
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.basepasses import TransformationPass

from divi.circuits.qem import QEMContext, QEMProtocol
from divi.pipeline.abc import ResultFormat

__all__ = [
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


_NON_UNITARY_OP_NAMES = frozenset(("measure", "reset", "barrier"))


def _count_foldable_gates(
    dag: DAGCircuit,
    exclude_names: frozenset[str] = frozenset(),
    exclude_arities: frozenset[int] = frozenset(),
) -> int:
    """Count unitary gates eligible for folding."""
    return sum(
        1
        for node in dag.op_nodes()
        if node.op.name not in _NON_UNITARY_OP_NAMES
        and node.op.name not in exclude_names
        and len(node.qargs) not in exclude_arities
    )


def _compute_fold_plan(d: int, scale_factor: float) -> tuple[int, int]:
    """Return base folds and extra-fold count for ``d`` foldable gates."""
    if d == 0 or scale_factor == 1.0:
        return 0, 0
    k = int((scale_factor - 1) // 2)
    remainder = scale_factor - (1 + 2 * k)
    n = max(0, min(d, round(remainder * d / 2)))
    return k, n


def _compute_effective_scale(d: int, scale_factor: float) -> float:
    """Effective scale factor actually realised by folding ``d`` gates."""
    if d == 0:
        return 1.0
    k, n = _compute_fold_plan(d, scale_factor)
    return 1.0 + 2 * k + 2 * n / d


class GlobalFoldPass(TransformationPass):
    """Global unitary folding with fractional scale-factor support.

    For a target scale factor ``s`` on a circuit of ``d`` unitary gates::

        k         = (s - 1) // 2
        remainder = s - (1 + 2k)
        n         = round(remainder · d / 2)

    The returned DAG is ``U · (U† · U)^k · L† · L`` where ``L`` is the
    sub-circuit of the last ``n`` unitary gates of ``U``.  Non-unitary
    instructions are ignored when counting ``d`` and selecting the tail.

    Args:
        scale_factor: Real number ≥ 1.  ``1.0`` is a pass-through.

    Raises:
        ValueError: If ``scale_factor`` < 1.
    """

    def __init__(self, scale_factor: float):
        super().__init__()
        if scale_factor < 1.0:
            raise ValueError(
                f"GlobalFoldPass: scale_factor must be >= 1, got {scale_factor}."
            )
        self.scale_factor = float(scale_factor)

    def effective_scale(self, dag: DAGCircuit) -> float:
        """Scale factor actually realised on ``dag``."""
        return _compute_effective_scale(_count_foldable_gates(dag), self.scale_factor)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Fold ``dag`` in place and return the mutated DAG."""
        if self.scale_factor == 1.0:
            return dag

        all_ops = [
            (node.op, node.qargs, node.cargs) for node in dag.topological_op_nodes()
        ]
        unitary_ops = [
            entry for entry in all_ops if entry[0].name not in _NON_UNITARY_OP_NAMES
        ]
        d = len(unitary_ops)
        k, n = _compute_fold_plan(d, self.scale_factor)
        if k == 0 and n == 0:
            return dag

        inv_ops = [
            (op.inverse(), qargs, cargs) for op, qargs, cargs in reversed(unitary_ops)
        ]

        for _ in range(k):
            for op, qargs, cargs in inv_ops:
                dag.apply_operation_back(op, qargs, cargs)
            for op, qargs, cargs in unitary_ops:
                dag.apply_operation_back(op, qargs, cargs)

        if n > 0:
            tail = unitary_ops[-n:]
            tail_inv = [
                (op.inverse(), qargs, cargs) for op, qargs, cargs in reversed(tail)
            ]
            for op, qargs, cargs in tail_inv:
                dag.apply_operation_back(op, qargs, cargs)
            for op, qargs, cargs in tail:
                dag.apply_operation_back(op, qargs, cargs)

        return dag


class LocalFoldPass(TransformationPass):
    """Per-gate folding with fractional scale-factor support.

    Each unitary gate ``G`` is replaced by ``G · (G† · G)^k``.  Fractional
    scale factors fold a selected subset of gates one extra time.

    Args:
        scale_factor: Real number ≥ 1.  ``1.0`` is a pass-through.
        selection: Which gates receive the extra fold.
        exclude: Optional op names or arity shorthands to skip.
        rng: Optional random source for ``selection="random"``.

    Raises:
        ValueError: If ``scale_factor`` < 1 or ``selection`` is unknown.
    """

    _VALID_SELECTIONS = ("random", "from_left", "from_right")
    _ARITY_SHORTHANDS = {"single": 1, "double": 2, "triple": 3}

    def __init__(
        self,
        scale_factor: float,
        selection: Literal["random", "from_left", "from_right"] = "random",
        exclude: set[str] | None = None,
        rng: random.Random | None = None,
    ):
        super().__init__()
        if scale_factor < 1.0:
            raise ValueError(
                f"LocalFoldPass: scale_factor must be >= 1, got {scale_factor}."
            )
        if selection not in self._VALID_SELECTIONS:
            raise ValueError(
                f"LocalFoldPass: selection must be one of "
                f"{self._VALID_SELECTIONS}, got {selection!r}."
            )
        self.scale_factor = float(scale_factor)
        self.selection = selection
        self._rng = rng or random.Random()

        exclude = set(exclude) if exclude else set()
        self._exclude_arities = frozenset(
            self._ARITY_SHORTHANDS[e] for e in exclude if e in self._ARITY_SHORTHANDS
        )
        self._exclude_names = frozenset(
            e for e in exclude if e not in self._ARITY_SHORTHANDS
        )

    def _pick_extra_indices(self, d: int, n: int) -> set[int]:
        if n <= 0:
            return set()
        if self.selection == "from_left":
            return set(range(n))
        if self.selection == "from_right":
            return set(range(d - n, d))
        return set(self._rng.sample(range(d), n))

    @staticmethod
    def _folded_sub_dag(node, num_folds: int) -> DAGCircuit:
        n_qubits = len(node.qargs)
        qc = QuantumCircuit(n_qubits)
        qargs = list(range(n_qubits))
        qc.append(node.op, qargs)
        inv_op = node.op.inverse()
        for _ in range(num_folds):
            qc.append(inv_op, qargs)
            qc.append(node.op, qargs)
        return circuit_to_dag(qc)

    def _is_foldable(self, node) -> bool:
        if node.op.name in _NON_UNITARY_OP_NAMES:
            return False
        if node.op.name in self._exclude_names:
            return False
        if len(node.qargs) in self._exclude_arities:
            return False
        return True

    def effective_scale(self, dag: DAGCircuit) -> float:
        """Scale factor actually realised on ``dag``."""
        d = _count_foldable_gates(dag, self._exclude_names, self._exclude_arities)
        return _compute_effective_scale(d, self.scale_factor)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Fold ``dag`` in place and return the mutated DAG."""
        if self.scale_factor == 1.0:
            return dag

        op_nodes = [node for node in dag.op_nodes() if self._is_foldable(node)]
        d = len(op_nodes)
        k, n = _compute_fold_plan(d, self.scale_factor)
        if k == 0 and n == 0:
            return dag

        extra = self._pick_extra_indices(d, n)

        for i, node in enumerate(op_nodes):
            num_folds = k + 1 if i in extra else k
            if num_folds == 0:
                continue
            dag.substitute_node_with_dag(node, self._folded_sub_dag(node, num_folds))
        return dag


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
            folding, or any custom callable.  See :data:`FoldingFn` for
            the input-mutation contract and the ``scale=1.0`` pass-through
            convention.
        extrapolator: Any object with an
            ``extrapolate(scale_factors, results) -> float`` method.
            No subclassing required — just implement the method.
            Defaults to :class:`RichardsonExtrapolator`.
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
        if len(scale_factors) < 2:
            raise ValueError(
                "scale_factors must contain at least two points to extrapolate "
                f"to the zero-noise limit; got {list(scale_factors)}."
            )
        if not all(e >= 1.0 for e in scale_factors):
            raise ValueError("All scale factors must be ≥ 1.0.")
        if len(set(scale_factors)) != len(scale_factors):
            raise ValueError(
                "scale_factors must be unique; got duplicates in "
                f"{list(scale_factors)}."
            )

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

    def applies_to(self, result_format: ResultFormat) -> bool:
        # Extrapolation acts on expectation values; raw sampling has none.
        return result_format is ResultFormat.EXPVALS

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
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
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

        return folded_dags, {
            "effective_scales": effective_scales,
            "dag_indices": list(range(len(folded_dags))),
        }

    def dry_expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """One unfolded alias of ``dag`` per scale factor; never mutates the
        input (dry batches may alias one DAG across entries)."""
        scales = tuple(self._scale_factors)
        n_foldable = _count_foldable_gates(dag)
        effective_scales = tuple(
            float(_compute_effective_scale(n_foldable, s)) for s in scales
        )

        if len(set(effective_scales)) < len(effective_scales):
            warnings.warn(
                f"ZNE: requested scale factors {list(scales)} collapse to "
                f"effective scales {list(effective_scales)} — the foldable "
                f"gate count is too small for the requested granularity. "
                f"Extrapolation may fail or be biased; consider fewer scale "
                f"factors, integer scales only, or a circuit with more "
                f"foldable gates.",
                stacklevel=2,
            )

        return tuple(dag for _ in scales), {
            "effective_scales": effective_scales,
            "dag_indices": list(range(len(scales))),
        }

    def reduce(
        self,
        quantum_results: Sequence[Any],
        context: QEMContext,
    ) -> list[float]:
        """Extrapolate per-observable expectation values to ``s=0``.

        Each entry of ``quantum_results`` is a ``list[float]`` of per-
        observable expectation values from one scale factor.  Extrapolation
        runs independently per observable.
        """
        indices = context.get("dag_indices")
        selected = (
            [quantum_results[i] for i in indices]
            if indices is not None
            else list(quantum_results)
        )
        scales = context.get("effective_scales", self._scale_factors)

        if not selected:
            raise RuntimeError("ZNE received an empty results sequence.")

        if not isinstance(selected[0], (list, tuple)):
            selected = [[v] for v in selected]
        n_obs = len(selected[0])
        return [
            float(self._extrapolator.extrapolate(scales, [row[i] for row in selected]))
            for i in range(n_obs)
        ]
