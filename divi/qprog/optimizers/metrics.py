# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Metric estimators for natural-gradient optimisation.

A :class:`MetricEstimator` turns a program into a set of evaluators (a gradient
and/or a metric) for a natural-gradient optimizer. Each estimator owns all
knowledge of a particular metric — the Hamiltonian pullback metric
(:class:`PullbackMetricEstimator`) or the Fubini–Study metric
(:class:`FubiniStudyMetricEstimator`) — and measures through the program's metric
pipeline. The optimizer injects the estimator; the program stays ignorant of
which metric is in play.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import PauliList, SparsePauliOp

from divi.circuits import MetaCircuit, build_overlap_meta
from divi.pipeline import CircuitPreprocessor, ResultFormat
from divi.pipeline._result_keys_operations import (
    average_by_param_set,
    group_by_branch_and_param_set,
)
from divi.pipeline.abc import ContractViolation
from divi.qprog.mixins import DataBindingMixin

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

__all__ = [
    "FubiniStudyMetricEstimator",
    "MetricEstimator",
    "PullbackMetricEstimator",
    "StochasticFidelityMetricEstimator",
]

# Evaluator callables vary in arity: ``metric_fn``/``jac`` take the parameter
# vector; ``fidelity_fn`` takes ``(theta, perturbations)``. Hence ``Callable[...]``.
Evaluators = dict[str, Callable[..., Any]]
_METRIC_BRANCH_AXES = ("ham", "circuit")


#: Report key for a metric routine. One name for every estimator: what a reader
#: needs is that the optimizer runs a metric routine and what it costs, while
#: *how* the metric is measured is a property of the estimator class they chose.
#: An estimator emitting several routines qualifies each (``metric[block 0]``).
METRIC_ROUTINE = "metric"


def _run_metric_by_branch(
    program: "VariationalQuantumAlgorithm",
    preprocessor: CircuitPreprocessor,
    param_sets: npt.NDArray[np.float64],
) -> dict[tuple, dict[int, npt.NDArray[np.float64]]]:
    """Run a metric preprocessor and keep selected source branches separate."""
    result = program.evaluate(
        np.atleast_2d(param_sets),
        preprocessor,
        preserve_keys=True,
        axes_to_preserve=_METRIC_BRANCH_AXES,
    )
    return group_by_branch_and_param_set(
        result,
        lambda value: np.asarray(value, dtype=np.float64).reshape(-1),
    )


#: Cap on distinct ansatz structures held by one :func:`_memoize_by_ansatz` cache.
#: A fixed ansatz needs one entry; the cap only bounds pathological growth.
_ANSATZ_CACHE_CAP = 128

#: One gate's contribution to an ansatz fingerprint: name, qubit indices, params.
_GateKey = tuple[str, tuple[int, ...], tuple[str, ...]]
#: Full ansatz fingerprint: ordered parameter names plus the gate sequence.
_AnsatzKey = tuple[tuple[str, ...], tuple[_GateKey, ...]]
_T = TypeVar("_T")


def _ansatz_fingerprint(meta: MetaCircuit) -> _AnsatzKey:
    """Structural key for a cost ansatz body, under which
    :func:`_memoize_by_ansatz` reuses what it built from a deterministic ansatz.

    Keyed on parameter *names* (the spec stage may re-instantiate ``Parameter``
    objects between evaluations, but names are stable) in their forward-binding
    order, plus the per-gate name, qubit indices, and params.
    """
    _, dag = meta.circuit_bodies[0]
    bit_index = {bit: i for i, bit in enumerate(dag.qubits)}
    gates = tuple(
        (
            node.op.name,
            tuple(bit_index[q] for q in node.qargs),
            tuple(str(p) for p in node.op.params),
        )
        for node in dag.op_nodes()
    )
    return tuple(str(p) for p in meta.parameters), gates


def _memoize_by_ansatz(
    build: Callable[[MetaCircuit], _T],
) -> Callable[[MetaCircuit], _T]:
    """Wrap ``build`` so each distinct ansatz structure is built once; the cache
    clears when it reaches ``_ANSATZ_CACHE_CAP``."""
    cache: dict[_AnsatzKey, _T] = {}

    def lookup(meta: MetaCircuit) -> _T:
        key = _ansatz_fingerprint(meta)
        if key not in cache:
            if len(cache) >= _ANSATZ_CACHE_CAP:
                cache.clear()
            cache[key] = build(meta)
        return cache[key]

    return lookup


def _overlap_preprocessor(
    overlap_for: Callable[[MetaCircuit], MetaCircuit],
) -> CircuitPreprocessor:
    return CircuitPreprocessor(
        METRIC_ROUTINE,
        preprocess=overlap_for,
        result_format=ResultFormat.PROBS,
        consumes_dag_bodies=True,
        # Built once per ``bind`` and reused across fidelity calls (its
        # ``overlap_for`` closure caches structurally, never resets), so the
        # pipeline is safe to memoize for the run.
        cache_key=METRIC_ROUTINE,
    )


def _run_overlap(
    program: "VariationalQuantumAlgorithm",
    preprocessor: CircuitPreprocessor,
    param_sets: npt.NDArray[np.float64],
    zeros: str,
) -> dict[int, float]:
    """Run an overlap preprocessor and return averaged all-zero probabilities."""
    result = program.evaluate(
        np.atleast_2d(param_sets), preprocessor, preserve_keys=True
    )
    averaged = average_by_param_set(
        result,
        lambda value: np.asarray([_zeros_probability(value, zeros)], dtype=np.float64),
    )
    return {idx: float(value[0]) for idx, value in averaged.items()}


def _zeros_probability(
    value: dict[str, float] | Sequence[dict[str, float]], zeros: str
) -> float:
    """All-zeros probability from one distribution (or the mean of several)."""
    if isinstance(value, dict):
        return float(value.get(zeros, 0.0))
    if not value:
        return 0.0
    return float(np.mean([probs.get(zeros, 0.0) for probs in value]))


def _reject_data_bound(program: "VariationalQuantumAlgorithm", metric: str) -> None:
    """Raise :class:`ContractViolation` for a data-bound program — its ansatz
    state depends on the data input, so the metric geometry is data-dependent."""
    if not isinstance(program, DataBindingMixin) or program.feature_batch is None:
        return
    # Only the unsupervised case has a metric alternative. Recommending pullback
    # unconditionally sent supervised users in a circle, since pullback rejects a
    # per-sample loss and pointed them back here.
    if program.sample_loss is not None:
        raise ContractViolation(
            f"The {metric} metric does not support data-bound programs, and no "
            "metric estimator supports a supervised (labelled) data-binding loss: "
            "the per-sample loss is a non-linear function of the expectation "
            "values. Use a non-metric optimizer such as SPSAOptimizer or a "
            "gradient-free ScipyOptimizer method."
        )
    raise ContractViolation(
        f"The {metric} metric does not support data-bound programs: the ansatz "
        "state depends on the data input, so the metric is data-dependent. Use "
        "QNGOptimizer with its default pullback metric, which supports an "
        "unlabelled data-bound loss, or a non-metric optimizer such as "
        "SPSAOptimizer."
    )


class MetricEstimator(ABC):
    """Strategy that produces natural-gradient evaluators for a program."""

    #: Whether the metric is estimated from resampled fidelity overlaps (the
    #: QN-SPSA stochastic path) rather than a closed-form ``"metric_fn"``.
    uses_fidelity_sampling: bool = False

    @abstractmethod
    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        """Raise :class:`~divi.pipeline.ContractViolation` if this metric cannot be
        applied to ``program``. Called at ``run()`` start so an incompatible pairing
        fails loudly before any optimisation."""
        raise NotImplementedError

    @abstractmethod
    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        """Return the evaluators this metric provides, keyed by name.

        Deterministic estimators (pullback, Fubini–Study) provide ``"metric_fn"``
        — a pure function of the parameters returning the metric matrix — and the
        pullback estimator additionally returns the loss gradient under ``"jac"``.
        The stochastic-fidelity estimator instead provides ``"fidelity_fn"``: the
        QN-SPSA optimizer builds its metric from finite differences of that
        fidelity rather than from a closed-form matrix. The variational algorithm
        forwards whichever keys appear to the optimizer; keys absent fall back to
        the algorithm's parameter-shift defaults.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocessors(
        self, program: "VariationalQuantumAlgorithm"
    ) -> tuple[CircuitPreprocessor, ...]:
        """The routines this estimator measures through.

        The same ones :meth:`bind`'s evaluators submit, built from ``program``
        structure alone so they can be enumerated without running anything. Most
        estimators measure through one; Fubini–Study measures each commuting-gate
        block separately and returns one per block.

        Implementations must call :meth:`check_compatible` first and raise
        :class:`~divi.pipeline.ContractViolation` when the metric cannot apply to
        ``program``.
        """
        raise NotImplementedError


class _MetricOptimizerMixin:
    """Delegation for optimizers holding a ``metric_estimator``: program
    validation, evaluator binding, and the routines they drive all defer to the
    estimator, so an optimizer adds no metric knowledge of its own."""

    metric_estimator: "MetricEstimator"

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        self.metric_estimator.check_compatible(program)

    def build_evaluators(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        return self.metric_estimator.bind(program)

    def preprocessors(
        self, program: "VariationalQuantumAlgorithm"
    ) -> tuple[CircuitPreprocessor, ...]:
        return self.metric_estimator.preprocessors(program)


class PullbackMetricEstimator(MetricEstimator):
    r"""Hamiltonian-aware pullback metric.

    Builds ``G_ij = sum_r |c_r| (d_i <P_r>)(d_j <P_r>)`` from the per-Pauli-term
    expectation gradients of the cost observables ``O_j = sum_r a_jr P_r``;
    an identity term's gradient is exactly zero. Each term's weight in the loss is
    ``c_r = a_jr * dL/d<O_j>``, with the slopes supplied by the program at the
    current parameters: ``1`` when the loss is ``<H>``, the loss's local slope
    otherwise. Weighting by ``|c_r|`` is a heuristic choice, not a Gauss–Newton
    or Fisher metric; it makes ``G`` scale with the loss as the gradient does,
    so rescaling the loss leaves the natural-gradient step unchanged while
    ``mean(diag(G))`` is at least 1. Below that,
    :class:`~divi.qprog.optimizers.QNGOptimizer`'s damping
    stays at ``regularization`` and the step shrinks with the loss. The loss
    gradient ``J @ c`` and the metric share the
    same parameter-shift evaluation, so both are returned from one pass.
    Measurement-only and PSD by construction (rank at most the number of terms).

    Requires a loss built from the cost observables' expectation values (not a
    supervised data-binding loss).

    When the cost fans out into several measurement branches (QDrift sampling,
    a data cohort), the loss gradient averages linearly across branches while
    the metric is the *mean of the per-branch metrics* ``E_b[G_b]``, not the
    metric of the mean Jacobian ``G(E_b[J])``. This is deliberate and is the
    only well-defined choice: each QDrift branch samples a different Hamiltonian
    with its own term set and coefficients, so the branch Jacobians are not
    commensurable to average. ``E_b[G_b]`` is the expected pullback metric over
    the sampling distribution — the same empirical-Fisher averaging a batched
    natural gradient uses. For the single-branch case (all deterministic VQAs)
    the two forms coincide.
    """

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        if program.cost_preprocessor().result_format is not ResultFormat.EXPVALS:
            raise ContractViolation(
                "The pullback metric requires a loss built from expectation "
                "values, but this program's cost reads raw counts. Use the "
                "Fubini–Study estimator (FubiniStudyMetricEstimator) instead."
            )
        if isinstance(program, DataBindingMixin) and program.sample_loss is not None:
            raise ContractViolation(
                "The pullback metric is invalid for a supervised data-binding "
                "loss: the per-sample loss is a non-linear function of the "
                "expectation values. No metric estimator supports this case — the "
                "Fubini–Study estimators reject data-bound programs outright — so "
                "use a non-metric optimizer such as SPSAOptimizer."
            )
        # Structural, so it belongs here rather than in bind(): a dry run only
        # catches ContractViolation from check_compatible.
        if not program.cost_circuit.observable:
            raise ContractViolation(
                "The pullback metric requires the cost circuit to carry a loss "
                "observable."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        shifts, weights = program._grad_shift_rule
        cache: dict[str, Any] = {"key": None, "value": None}

        def grad_and_metric(
            params: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            key = (
                program._evaluation_counter,
                np.asarray(params, dtype=np.float64).tobytes(),
            )
            if cache["key"] != key:
                params = np.asarray(params, dtype=np.float64)
                branch_payloads = _term_expectations(
                    program, shifts + params, _loss_gradient(program, params)
                )
                gradients = []
                metrics = []
                for exp_vals, coeffs in branch_payloads.values():
                    jacobian = weights @ exp_vals  # (m, v): d_i <P_r>
                    gradients.append(jacobian @ coeffs)
                    metrics.append((jacobian * np.abs(coeffs)) @ jacobian.T)
                grad = np.mean(gradients, axis=0)
                metric = np.mean(metrics, axis=0)
                cache["key"] = key
                cache["value"] = (grad, metric)
            return cache["value"]

        return {
            "jac": lambda params: grad_and_metric(params)[0],
            "metric_fn": lambda params: grad_and_metric(params)[1],
        }

    def preprocessors(
        self, program: "VariationalQuantumAlgorithm"
    ) -> tuple[CircuitPreprocessor, ...]:
        self.check_compatible(program)
        return (_all_terms_preprocessor(),)


def _split_into_terms(hamiltonian: SparsePauliOp) -> tuple[SparsePauliOp, ...]:
    """Split into single-term, unit-coefficient observables."""
    return tuple(SparsePauliOp(label) for label in hamiltonian.paulis.to_labels())


def _split_observable_into_terms(meta: MetaCircuit) -> MetaCircuit:
    """Expand a branch's cost observables into their unit-coefficient
    single-Pauli terms as one multi-observable tuple."""
    if not meta.observable:
        raise ContractViolation(
            "The pullback metric requires each cost branch to carry a loss observable."
        )
    terms = tuple(
        term for observable in meta.observable for term in _split_into_terms(observable)
    )
    if not terms:
        raise ContractViolation(
            "The pullback metric requires the cost observables to carry at least "
            "one Pauli term."
        )
    return replace(meta, observable=terms, _was_multi_obs=True)


def _all_terms_preprocessor() -> CircuitPreprocessor:
    """Cacheable preprocessor measuring every Pauli term of the cost observable
    as separate expectation values in one multi-observable pass."""
    return CircuitPreprocessor(
        METRIC_ROUTINE,
        preprocess=_split_observable_into_terms,
        cache_key=METRIC_ROUTINE,
    )


def _loss_gradient(
    program: "VariationalQuantumAlgorithm", params: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The program's ``dL/d<O_j>`` at ``params`` as a finite real vector.

    Raises:
        ContractViolation: If the slopes are complex or not finite.
    """
    slopes = np.asarray(program._loss_gradient_by_observable(params))
    if np.iscomplexobj(slopes) or not np.all(np.isfinite(slopes)):
        raise ContractViolation(
            f"The loss gradient by observable must be finite and real, got {slopes!r}."
        )
    return slopes.astype(np.float64).reshape(-1)


def _term_expectations(
    program: "VariationalQuantumAlgorithm",
    param_sets: npt.NDArray[np.float64],
    loss_gradient: npt.NDArray[np.float64],
) -> dict[tuple, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Per-branch ``(n_sets, n_terms)`` term expectations and each term's weight
    in the loss.

    Measures every Pauli term of each branch's cost observables in one
    multi-observable pass. A term's weight is its coefficient in its observable
    times ``loss_gradient`` at that observable; the coefficients are recovered
    by inspecting the same sampled cohort the cost evaluation used
    (``_post_spec_batch``), rather than smuggled out through a closure.
    """
    by_branch = _run_metric_by_branch(program, _all_terms_preprocessor(), param_sets)
    source = program._post_spec_batch()

    payloads: dict[tuple, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for branch_key, by_param in by_branch.items():
        source_meta = source.get(branch_key)
        if source_meta is None or not source_meta.observable:
            raise ContractViolation(
                "A measured metric branch is absent from the cost cohort or "
                "carries no observable."
            )
        if len(source_meta.observable) != len(loss_gradient):
            raise ContractViolation(
                "The loss gradient has one entry per cost observable; this branch "
                f"has {len(source_meta.observable)} observables and "
                f"{len(loss_gradient)} entries."
            )
        weights = np.concatenate(
            [
                np.real(observable.coeffs) * loss_gradient[j]
                for j, observable in enumerate(source_meta.observable)
            ]
        )
        rows = np.asarray([by_param[i] for i in sorted(by_param)], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(weights):
            raise ContractViolation(
                "Per-term measurement count does not match the branch's "
                "coefficient count."
            )
        payloads[branch_key] = (rows, weights)
    return payloads


#: Hermitian generator of each supported single-parameter rotation gate, as the
#: Pauli string of ``generator = 0.5 * Pauli`` on the gate's wires.
_GATE_GENERATORS = {
    "rx": "X",
    "ry": "Y",
    "rz": "Z",
    "rxx": "XX",
    "rzz": "ZZ",
}

_FS_UNSUPPORTED_GATE = (
    "The Fubini–Study metric supports only single-parameter Pauli-rotation gates "
    "(rx/ry/rz/rxx/rzz) with a bare parameter as the angle; this ansatz uses "
    "{gate!r}{detail}. Use the pullback metric (PullbackMetricEstimator) or a "
    "non-metric optimizer."
)

#: Appended when the gate itself is supported and the *angle* is what is not — a
#: message naming only the gate sends the reader to check it against the allow-list
#: it is already on.
_FS_UNSUPPORTED_ANGLE = (
    " with the angle expression {angle!r}, which is not a bare trainable parameter"
)


class FubiniStudyMetricEstimator(MetricEstimator):
    r"""Block-diagonal Fubini–Study metric (quantum geometric tensor).

    For each block of mutually-commuting parametric gates with Hermitian
    generators ``K_i``, the metric on the pre-block state is
    ``g_ij = 1/2 <{K_i, K_j}> - <K_i><K_j>``. The blocks are stacked block-
    diagonally and pulled back to the parameter basis, so a parameter driving
    several gates collects the contribution of each of its generators. Unlike
    the pullback metric this is independent of the loss
    observable — it is the geometry of the ansatz state — so it applies to any
    program with a supported Pauli-rotation ansatz, whatever its loss. It
    provides only ``metric_fn``; the gradient falls
    back to the program's parameter-shift rule.
    """

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        # Generator extraction raises ContractViolation on any unsupported gate.
        _fs_blocks(program.cost_circuit)
        _reject_data_bound(program, "Fubini–Study")

    @cached_property
    def _layouts_for(self) -> Callable[[MetaCircuit], list["_FSBlockLayout"]]:
        """Block layouts per ansatz structure, shared by bind and preprocessors."""
        return _memoize_by_ansatz(_build_fs_block_layouts)

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        layouts_for = self._layouts_for
        routines = _build_fs_routines(program, layouts_for)
        n_params = len(program.cost_circuit.parameters)

        def metric_fn(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            theta = np.asarray(params, dtype=np.float64).reshape(-1)
            layouts_by_branch = {
                branch_key: layouts_for(meta)
                for branch_key, meta in program._post_spec_batch().items()
            }
            branch_metrics: dict[tuple, npt.NDArray[np.float64]] = {}
            for block_id, (preprocessor, prefix_names, gather) in enumerate(routines):
                measured = _run_metric_by_branch(
                    program, preprocessor, theta[gather][np.newaxis]
                )
                blocks_by_branch = _compute_fs_block_covariance(
                    measured, layouts_by_branch, block_id, prefix_names
                )
                for branch_key, (indices, block) in blocks_by_branch.items():
                    metric = branch_metrics.setdefault(
                        branch_key, np.zeros((n_params, n_params))
                    )
                    # One parameter may drive several gates; they add.
                    np.add.at(metric, np.ix_(indices, indices), block)
            return np.mean(list(branch_metrics.values()), axis=0)

        return {"metric_fn": metric_fn}

    def preprocessors(
        self, program: "VariationalQuantumAlgorithm"
    ) -> tuple[CircuitPreprocessor, ...]:
        self.check_compatible(program)
        routines = _build_fs_routines(program, self._layouts_for)
        return tuple(preprocessor for preprocessor, _, _ in routines)


class StochasticFidelityMetricEstimator(MetricEstimator):
    r"""Stochastic Fubini–Study metric via state-overlap fidelities (QN-SPSA).

    Provides a ``"fidelity_fn"`` evaluator rather than a closed-form
    ``"metric_fn"``: the QN-SPSA optimizer reconstructs the metric from finite
    differences of the state fidelity
    :math:`F(\theta_1,\theta_2)=|\langle\psi(\theta_1)|\psi(\theta_2)\rangle|^2`,
    estimated as the all-zeros probability of the compute-uncompute circuit
    :math:`U(\theta_1)\,U(\theta_2)^\dagger`. Like the Fubini–Study metric it is
    the geometry of the ansatz state — independent of the *loss observable* — so
    it applies to any qiskit-invertible ansatz. The overlap circuits are built by
    a preprocessor from the program's normal post-spec ansatz cohort and averaged
    over preserved pipeline axes.
    """

    uses_fidelity_sampling = True

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        try:
            build_overlap_meta(program.cost_circuit)
        except Exception as exc:
            raise ContractViolation(
                "The stochastic-fidelity metric requires an invertible ansatz "
                "(qiskit QuantumCircuit.inverse()); this program's cost circuit "
                "could not be inverted."
            ) from exc
        _reject_data_bound(program, "stochastic-fidelity")

    def _build_overlap_preprocessor(self) -> CircuitPreprocessor:
        """Build the overlap routine shared by :meth:`bind` and
        :meth:`preprocessors`. Overlap circuits are built lazily and cached by
        ansatz structure, so this does no circuit work until the routine runs.
        """
        return _overlap_preprocessor(_memoize_by_ansatz(build_overlap_meta))

    def preprocessors(
        self, program: "VariationalQuantumAlgorithm"
    ) -> tuple[CircuitPreprocessor, ...]:
        # Validate eagerly so an incompatible ansatz raises ContractViolation
        # before the lazy closure runs.
        self.check_compatible(program)
        return (self._build_overlap_preprocessor(),)

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        # No re-validation: the optimizer's validate_program already ran it.
        preprocessor = self._build_overlap_preprocessor()
        zeros = "0" * program.cost_circuit.n_qubits

        def fidelity_fn(
            theta: npt.NDArray[np.float64],
            perturbations: Sequence[npt.NDArray[np.float64]],
        ) -> npt.NDArray[np.float64]:
            """Fidelities ``F(theta, theta + p)`` for each ``p`` in ``perturbations``.

            The overlap circuit is produced from each member of the post-spec
            ansatz cohort, and overlaps are averaged across preserved cohort axes.
            """
            theta = np.asarray(theta, dtype=np.float64).reshape(-1)
            rows = np.vstack(
                [
                    np.concatenate([theta, theta + np.asarray(p, dtype=np.float64)])
                    for p in perturbations
                ]
            )
            indexed = _run_overlap(program, preprocessor, rows, zeros)
            return np.array([indexed[i] for i in range(len(perturbations))])

        return {"fidelity_fn": fidelity_fn}


def _fs_blocks(
    cost_circuit: MetaCircuit,
) -> tuple[tuple[tuple, ...], list[tuple[int, list]], list[Parameter], int]:
    """Layer the ansatz into blocks of commuting parametric gates.

    Returns ``(tape, blocks, full_params, n_qubits)``: ``tape`` is every
    ``(operation, wire_indices)`` in order, and each block is
    ``(prefix_end, [(param_index, generator)])`` where ``tape[:prefix_end]`` is
    applied before the block (its pre-state) and each generator is the
    ``SparsePauliOp`` of the corresponding parametric gate.
    A block closes on a non-parametric gate or a wire conflict, so within a block
    the generators act on disjoint wires and therefore commute.
    """
    dag = cost_circuit.circuit_bodies[0][1]
    full_params = list(cost_circuit.parameters)
    param_indices = {param: index for index, param in enumerate(full_params)}
    n = dag.num_qubits()

    blocks: list[tuple[int, list]] = []
    prefix_ops: list[tuple] = []
    cur: list[tuple[int, SparsePauliOp]] = []
    cur_wires: set[int] = set()
    cur_prefix_end = 0

    def close() -> None:
        nonlocal cur, cur_wires
        if cur:
            blocks.append((cur_prefix_end, cur))
        cur, cur_wires = [], set()

    # Walk in original circuit insertion order: ``dag.op_nodes()`` yields nodes in
    # insertion order, whereas ``topological_op_nodes()`` ASAP-reschedules and would
    # split a layer's parallel rotations across the entangler staircase before them.
    for node in dag.op_nodes():
        parametric = [p for p in node.op.params if isinstance(p, ParameterExpression)]
        wires = [dag.find_bit(q).index for q in node.qargs]
        wire_set = set(wires)
        if parametric:
            pauli = _GATE_GENERATORS.get(node.op.name)
            angle = node.op.params[0]
            if pauli is None or len(parametric) != 1 or angle not in param_indices:
                detail = (
                    _FS_UNSUPPORTED_ANGLE.format(angle=str(angle))
                    if pauli is not None and angle not in param_indices
                    else ""
                )
                raise ContractViolation(
                    _FS_UNSUPPORTED_GATE.format(gate=node.op.name, detail=detail)
                )
            generator = SparsePauliOp.from_sparse_list(
                [(pauli, wires, 0.5)], num_qubits=n
            )
            if cur and wire_set & cur_wires:
                close()
            if not cur:
                cur_prefix_end = len(prefix_ops)
            cur.append((param_indices[angle], generator))
            cur_wires |= wire_set
        else:
            close()
        prefix_ops.append((node.op, wires))
    close()
    return tuple(prefix_ops), blocks, full_params, n


#: Per Fubini–Study block: the pre-state measurement circuit (every needed
#: label as one observable), then the block's parameter indices, generators
#: and labels.
_FSBlockLayout = tuple[
    MetaCircuit, tuple[int, ...], tuple[SparsePauliOp, ...], tuple[str, ...]
]


def _compute_fs_block_covariance(
    measured: Mapping[tuple, Mapping[int, npt.NDArray[np.float64]]],
    layouts_by_branch: Mapping[tuple, Sequence[_FSBlockLayout]],
    block_id: int,
    reference_prefix_param_names: Sequence[str],
) -> dict[tuple, tuple[tuple[int, ...], npt.NDArray[np.float64]]]:
    """``g_ij = 1/2 <{K_i, K_j}> - <K_i><K_j>`` on each measured branch's
    block pre-state, with the block layouts taken from the cost cohort."""
    blocks: dict[tuple, tuple[tuple[int, ...], npt.NDArray[np.float64]]] = {}
    for branch_key, values_by_param in measured.items():
        layouts = layouts_by_branch.get(branch_key)
        if layouts is None:
            raise ContractViolation(
                "A measured Fubini-Study branch is absent from the cost cohort."
            )
        _, indices, generators, labels = _select_fs_block_layout(
            layouts, block_id, reference_prefix_param_names
        )
        vec = np.asarray(values_by_param[0], dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(labels):
            raise ContractViolation(
                "Fubini-Study per-label measurement count does not match the "
                "branch's label count."
            )
        exp = dict(zip(labels, vec.tolist()))
        blocks[branch_key] = (indices, _fs_block_from_expectations(generators, exp))
    return blocks


def _fs_block_from_expectations(
    generators: Sequence[SparsePauliOp], exp: Mapping[str, float]
) -> npt.NDArray[np.float64]:
    """Block metric ``0.25 * (<P_i P_j> - <P_i><P_j>)`` for generators ``0.5 * P``."""
    single_labels, pair_labels, (rows, cols) = _fs_generator_labels(generators)
    single = np.array([exp[label] for label in single_labels])
    correlation = np.ones((len(generators),) * 2)
    correlation[rows, cols] = correlation[cols, rows] = [
        exp[label] for label in pair_labels
    ]
    return 0.25 * (correlation - np.outer(single, single))


def _fs_generator_labels(
    generators: Sequence[SparsePauliOp],
) -> tuple[list[str], list[str], tuple[np.ndarray, np.ndarray]]:
    """Labels of each generator's Pauli and of every pairwise product.

    A block's generators are ``0.5 * P`` on disjoint wires, so they commute,
    ``{K_i, K_j} = 2 K_i K_j`` and the product Pauli of ``i < j`` is the merge of
    the two supports. Returns ``(single_labels, pair_labels, (rows, cols))`` with
    ``pair_labels[n]`` belonging to the pair ``(rows[n], cols[n])``.
    """
    x = np.concatenate([generator.paulis.x for generator in generators])
    z = np.concatenate([generator.paulis.z for generator in generators])
    rows, cols = np.triu_indices(len(generators), k=1)
    single_labels = list(PauliList.from_symplectic(z, x).to_labels())
    pair_labels = list(
        PauliList.from_symplectic(z[rows] | z[cols], x[rows] | x[cols]).to_labels()
    )
    return single_labels, pair_labels, (rows, cols)


def _fs_needed_labels(generators: Sequence[SparsePauliOp]) -> tuple[str, ...]:
    single_labels, pair_labels, _ = _fs_generator_labels(generators)
    return tuple(sorted({*single_labels, *pair_labels}))


def _build_fs_block_layouts(meta: MetaCircuit) -> list[_FSBlockLayout]:
    """Every block's layout, from one walk of the ansatz."""
    tape, blocks, full_params, n_qubits = _fs_blocks(meta)
    prefix = QuantumCircuit(n_qubits)
    position = 0
    layouts: list[_FSBlockLayout] = []
    for prefix_end, entries in blocks:
        for op, wires in tape[position:prefix_end]:
            prefix.append(op, wires)
        position = prefix_end
        present = set(prefix.parameters)
        generators = tuple(generator for _, generator in entries)
        labels = _fs_needed_labels(generators)
        prefix_meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(prefix)),),
            parameters=tuple(p for p in full_params if p in present),
            observable=tuple(SparsePauliOp(label) for label in labels),
            _was_multi_obs=True,
        )
        layouts.append(
            (prefix_meta, tuple(index for index, _ in entries), generators, labels)
        )
    return layouts


def _select_fs_block_layout(
    layouts: Sequence[_FSBlockLayout],
    block_id: int,
    reference_prefix_param_names: Sequence[str],
) -> _FSBlockLayout:
    """One branch's layout for ``block_id``, checked against the reference ansatz."""
    if block_id >= len(layouts):
        raise ContractViolation(
            "A sampled metric branch has fewer Fubini-Study blocks than "
            "the reference ansatz."
        )
    layout = layouts[block_id]
    if tuple(p.name for p in layout[0].parameters) != tuple(
        reference_prefix_param_names
    ):
        raise ContractViolation(
            "A sampled metric branch has a different Fubini-Study prefix "
            "parameter layout than the reference ansatz."
        )
    return layout


def _build_fs_routines(
    program: "VariationalQuantumAlgorithm",
    layouts_for: Callable[[MetaCircuit], list[_FSBlockLayout]],
) -> list[tuple[CircuitPreprocessor, tuple[str, ...], npt.NDArray[np.intp]]]:
    """Per reference block: its measurement preprocessor, prefix parameter
    names, and the indices that gather those parameters from ``theta``."""
    cost_circuit = program.cost_circuit
    param_index = {p.name: i for i, p in enumerate(cost_circuit.parameters)}
    routines = []
    for block_id, (prefix_meta, *_) in enumerate(layouts_for(cost_circuit)):
        names = tuple(p.name for p in prefix_meta.parameters)
        routines.append(
            (
                _fs_prefix_labels_preprocessor(block_id, names, layouts_for),
                names,
                np.array([param_index[name] for name in names], dtype=np.intp),
            )
        )
    return routines


def _fs_prefix_labels_preprocessor(
    block_id: int,
    reference_prefix_param_names: Sequence[str],
    layouts_for: Callable[[MetaCircuit], list[_FSBlockLayout]],
) -> CircuitPreprocessor:
    """Cacheable preprocessor measuring every Pauli label a Fubini-Study block
    needs on its pre-state in one multi-observable pass."""

    def preprocess(meta: MetaCircuit) -> MetaCircuit:
        return _select_fs_block_layout(
            layouts_for(meta), block_id, reference_prefix_param_names
        )[0]

    return CircuitPreprocessor(
        # One pipeline per block, so each carries the block in its name: the name
        # is the report key, and identically-named routines would collapse into a
        # single entry showing only the last block's circuits.
        f"{METRIC_ROUTINE}[block {block_id}]",
        preprocess=preprocess,
        consumes_dag_bodies=True,
        cache_key=(METRIC_ROUTINE, block_id),
    )
