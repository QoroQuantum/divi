# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Metric estimators for natural-gradient optimization.

A :class:`MetricEstimator` turns a program into a set of evaluators (a gradient
and/or a metric) for a natural-gradient optimizer. Each estimator owns all
knowledge of a particular metric — the Hamiltonian pullback metric
(:class:`PullbackMetricEstimator`) or the Fubini–Study metric
(:class:`FubiniStudyMetricEstimator`) — and measures through the program's metric
pipeline. The optimizer injects the estimator; the program stays ignorant of
which metric is in play.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, build_overlap_meta
from divi.pipeline import CircuitPreprocessor, ResultFormat
from divi.pipeline._result_keys_operations import (
    average_by_param_set,
    group_by_branch_and_param_set,
)
from divi.pipeline.abc import ContractViolation

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

# Evaluator callables vary in arity: ``metric_fn``/``jac`` take the parameter
# vector; ``fidelity_fn`` takes ``(theta, perturbations)``. Hence ``Callable[...]``.
Evaluators = dict[str, Callable[..., Any]]
_METRIC_BRANCH_AXES = ("ham", "circuit")


def _run_metric_by_branch(
    program: "VariationalQuantumAlgorithm",
    preprocessor: CircuitPreprocessor,
    param_sets: npt.NDArray[np.float64],
) -> dict[tuple, dict[int, npt.NDArray[np.float64]]]:
    """Run a metric preprocessor and keep selected source branches separate."""
    result = cast(
        dict[tuple, Any],
        program.evaluate(
            np.atleast_2d(param_sets),
            preprocessor,
            preserve_keys=True,
            axes_to_preserve=_METRIC_BRANCH_AXES,
        ),
    )
    return group_by_branch_and_param_set(
        result,
        lambda value: np.asarray(value, dtype=np.float64).reshape(-1),
    )


#: Cap on distinct overlap circuits cached per fidelity evaluator. A fixed
#: ansatz needs one entry; the cap only bounds pathological growth.
_OVERLAP_CACHE_CAP = 128

#: One gate's contribution to an ansatz fingerprint: name, qubit indices, params.
_GateKey = tuple[str, tuple[int, ...], tuple[str, ...]]
#: Full ansatz fingerprint: ordered parameter names plus the gate sequence.
_AnsatzKey = tuple[tuple[str, ...], tuple[_GateKey, ...]]


def _ansatz_fingerprint(meta: MetaCircuit) -> _AnsatzKey:
    """Structural key for a cost ansatz body — the exact input to
    ``build_overlap_meta`` — so a deterministic ansatz is built once and reused.

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
        for node in dag.topological_op_nodes()
    )
    return tuple(str(p) for p in meta.parameters), gates


def _overlap_preprocessor(
    overlap_for: Callable[[MetaCircuit], MetaCircuit],
) -> CircuitPreprocessor:
    return CircuitPreprocessor(
        "overlap",
        preprocess=overlap_for,
        result_format=ResultFormat.PROBS,
        consumes_dag_bodies=True,
        # Built once per ``bind`` and reused across fidelity calls (its
        # ``overlap_for`` closure caches structurally, never resets), so the
        # pipeline is safe to memoize for the run.
        cache_key="overlap",
    )


def _run_overlap(
    program: "VariationalQuantumAlgorithm",
    preprocessor: CircuitPreprocessor,
    param_sets: npt.NDArray[np.float64],
    zeros: str,
) -> dict[int, float]:
    """Run an overlap preprocessor and return averaged all-zero probabilities."""
    result = cast(
        dict[tuple, Any],
        program.evaluate(np.atleast_2d(param_sets), preprocessor, preserve_keys=True),
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


class MetricEstimator(ABC):
    """Strategy that produces natural-gradient evaluators for a program."""

    @abstractmethod
    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        """Raise :class:`~divi.pipeline.ContractViolation` if this metric cannot be
        applied to ``program``. Called at ``run()`` start so an incompatible pairing
        fails loudly before any optimization."""
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


class PullbackMetricEstimator(MetricEstimator):
    r"""Hamiltonian-aware pullback metric.

    Builds ``G_ij = sum_r a_r^2 (d_i <P_r>)(d_j <P_r>)`` from the per-Pauli-term
    expectation gradients of the loss observable ``H = sum_r a_r P_r``. The
    energy gradient ``J @ a`` and the metric share the same parameter-shift
    evaluation, so both are returned from one pass. Measurement-only and PSD by
    construction (rank at most the number of Hamiltonian terms).

    Requires the program's loss to be the expectation value of its cost
    Hamiltonian (VQE/QAOA, plain or unsupervised-data-bound CustomVQA).

    When the cost fans out into several measurement branches (QDrift sampling,
    a data cohort), the energy gradient averages linearly across branches while
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
                "The pullback metric requires the loss to be the expectation "
                "value of the cost Hamiltonian, but this program's cost computes "
                "a classical objective (e.g. PCE). Use the Fubini–Study estimator "
                "(FubiniStudyMetricEstimator) instead."
            )
        if getattr(program, "_sample_loss_fn", None) is not None:
            raise ContractViolation(
                "The pullback metric is invalid for a supervised data-binding "
                "loss: the per-sample loss is a non-linear function of the "
                "expectation values. Use a non-metric optimizer or the "
                "Fubini–Study estimator."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        shift_mask = program._grad_shift_mask
        cache: dict[str, Any] = {"key": None, "value": None}
        if (
            program.cost_circuit.observable is None
            or len(program.cost_circuit.observable) != 1
        ):
            raise ContractViolation(
                "The pullback metric requires the cost circuit to carry exactly "
                "one loss observable."
            )

        def grad_and_metric(
            params: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            key = (
                program._evaluation_counter,
                np.asarray(params, dtype=np.float64).tobytes(),
            )
            if cache["key"] != key:
                branch_payloads = _term_expectations(program, shift_mask + params)
                gradients = []
                metrics = []
                for exp_vals, coeffs in branch_payloads.values():
                    jacobian = 0.5 * (
                        exp_vals[0::2] - exp_vals[1::2]
                    )  # (m, v): d_i <P_r>
                    gradients.append(jacobian @ coeffs)
                    metrics.append((jacobian * coeffs**2) @ jacobian.T)
                grad = np.mean(gradients, axis=0)
                metric = np.mean(metrics, axis=0)
                cache["key"] = key
                cache["value"] = (grad, metric)
            return cache["value"]

        return {
            "jac": lambda params: grad_and_metric(params)[0],
            "metric_fn": lambda params: grad_and_metric(params)[1],
        }


def _split_into_terms(
    hamiltonian: SparsePauliOp,
) -> tuple[tuple[SparsePauliOp, ...], npt.NDArray[np.float64]]:
    """Split into single-term, unit-coefficient observables (dropping identity
    terms, which have zero gradient) and the matching real coefficients."""
    terms: list[SparsePauliOp] = []
    coeffs: list[float] = []
    for label, coeff in hamiltonian.to_list():
        if set(label) == {"I"}:  # identity term: zero gradient, no contribution
            continue
        terms.append(SparsePauliOp(label))
        coeffs.append(float(np.real(coeff)))

    if not terms:
        raise ValueError(
            "Pullback metric requires a loss observable with at least one "
            "non-identity Pauli term."
        )
    return tuple(terms), np.asarray(coeffs, dtype=np.float64)


def _zero_observable(n_qubits: int) -> SparsePauliOp:
    """All-zeros observable for blocks/terms with no non-identity contribution."""
    return SparsePauliOp("I" * n_qubits, coeffs=np.asarray([0.0]))


def _split_observable_into_terms(meta: MetaCircuit) -> MetaCircuit:
    """Expand a branch's single cost observable into its unit-coefficient
    single-Pauli terms as a multi-observable tuple."""
    if meta.observable is None or len(meta.observable) != 1:
        raise ContractViolation(
            "The pullback metric requires each cost branch to carry exactly "
            "one loss observable."
        )
    terms, _ = _split_into_terms(meta.observable[0])
    return replace(meta, observable=tuple(terms), _was_multi_obs=True)


def _all_terms_preprocessor() -> CircuitPreprocessor:
    """Cacheable preprocessor measuring every Pauli term of the cost observable
    as separate expectation values in one multi-observable pass."""
    return CircuitPreprocessor(
        "metric-terms",
        preprocess=_split_observable_into_terms,
        cache_key="metric-terms",
    )


def _term_expectations(
    program: "VariationalQuantumAlgorithm",
    param_sets: npt.NDArray[np.float64],
) -> dict[tuple, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Per-branch ``(n_sets, n_terms)`` term expectations and matching coefficients.

    Measures every Pauli term of each branch's cost observable in one
    multi-observable pass; the coefficients are recovered by inspecting the same
    sampled cohort the cost evaluation used (``_post_spec_batch``),
    rather than smuggled out through a closure.
    """
    by_branch = _run_metric_by_branch(program, _all_terms_preprocessor(), param_sets)
    source = program._post_spec_batch()

    payloads: dict[tuple, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for branch_key, by_param in by_branch.items():
        source_meta = source.get(branch_key)
        if source_meta is None or source_meta.observable is None:
            raise ContractViolation(
                "A measured metric branch is absent from the cost cohort or "
                "carries no observable."
            )
        _, coeffs = _split_into_terms(source_meta.observable[0])
        rows = np.asarray([by_param[i] for i in sorted(by_param)], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(coeffs):
            raise ContractViolation(
                "Per-term measurement count does not match the branch's "
                "coefficient count."
            )
        payloads[branch_key] = (rows, coeffs)
    return payloads


#: Hermitian generator of each supported single-parameter rotation gate, as the
#: Pauli string of ``generator = 0.5 * Pauli`` on the gate's wires.
_GATE_GENERATORS = {
    "rx": "X",
    "ry": "Y",
    "rz": "Z",
    "rxx": "XX",
    "ryy": "YY",
    "rzz": "ZZ",
}

_FS_UNSUPPORTED_GATE = (
    "The Fubini–Study metric supports only single-parameter Pauli-rotation gates "
    "(rx/ry/rz/rxx/ryy/rzz) with a bare parameter as the angle; this ansatz uses "
    "{gate!r}. Use the pullback metric (PullbackMetricEstimator) or a non-metric "
    "optimizer."
)


class FubiniStudyMetricEstimator(MetricEstimator):
    r"""Block-diagonal Fubini–Study metric (quantum geometric tensor).

    For each block of mutually-commuting parametric gates with Hermitian
    generators ``K_i``, the metric on the pre-block state is
    ``g_ij = 1/2 <{K_i, K_j}> - <K_i><K_j>``. The blocks are stacked block-
    diagonally. Unlike the pullback metric this is independent of the loss
    observable — it is the geometry of the ansatz state — so it applies to any
    program with a supported Pauli-rotation ansatz (including PCE, whose loss is
    a classical objective). It provides only ``metric_fn``; the gradient falls
    back to the program's parameter-shift rule.
    """

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        # Generator extraction raises ContractViolation on any unsupported gate.
        _fs_blocks(program.cost_circuit)
        if getattr(program, "feature_batch", None) is not None:
            raise ContractViolation(
                "The Fubini–Study metric does not support data-bound programs: "
                "the ansatz state depends on the data input, so the metric is "
                "data-dependent. Use a non-metric optimizer."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        blocks, full_params, _ = _fs_blocks(program.cost_circuit)

        def metric_fn(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            theta = np.asarray(params, dtype=np.float64).reshape(-1)
            branch_metrics: dict[tuple, npt.NDArray[np.float64]] = {}
            for block_id, _ in enumerate(blocks):
                blocks_by_branch = _fs_block_covariance(
                    program,
                    full_params,
                    theta,
                    block_id,
                )
                for branch_key, (indices, block) in blocks_by_branch.items():
                    metric = branch_metrics.setdefault(
                        branch_key, np.zeros((len(full_params), len(full_params)))
                    )
                    for a, ia in enumerate(indices):
                        for b, ib in enumerate(indices):
                            metric[ia, ib] = block[a, b]
            return np.mean(list(branch_metrics.values()), axis=0)

        return {"metric_fn": metric_fn}


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

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        try:
            build_overlap_meta(program.cost_circuit)
        except Exception as exc:
            raise ContractViolation(
                "The stochastic-fidelity metric requires an invertible ansatz "
                "(qiskit QuantumCircuit.inverse()); this program's cost circuit "
                "could not be inverted."
            ) from exc
        if getattr(program, "feature_batch", None) is not None:
            raise ContractViolation(
                "The stochastic-fidelity metric does not support data-bound "
                "programs: the ansatz state depends on the data input, so the "
                "fidelity is data-dependent. Use a non-metric optimizer."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        overlap_cache: dict[_AnsatzKey, MetaCircuit] = {}

        def _overlap_for(meta: MetaCircuit) -> MetaCircuit:
            key = _ansatz_fingerprint(meta)
            overlap = overlap_cache.get(key)
            if overlap is None:
                if len(overlap_cache) >= _OVERLAP_CACHE_CAP:
                    overlap_cache.clear()
                overlap = build_overlap_meta(meta)
                overlap_cache[key] = overlap
            return overlap

        preprocessor = _overlap_preprocessor(_overlap_for)
        zeros = "0" * program.cost_circuit.n_qubits

        def fidelity_fn(
            theta: npt.NDArray[np.float64],
            perturbations: list[npt.NDArray[np.float64]],
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
) -> tuple[list[tuple[list, list]], list[Parameter], int]:
    """Layer the ansatz into blocks of commuting parametric gates.

    Returns ``(blocks, full_params, n_qubits)`` where each block is
    ``(prefix_ops, [(param_index, generator)])``: ``prefix_ops`` is the list of
    ``(operation, wire_indices)`` applied before the block (its pre-state), and
    each generator is the ``SparsePauliOp`` of the corresponding parametric gate.
    A block closes on a non-parametric gate or a wire conflict, so within a block
    the generators act on disjoint wires and therefore commute.
    """
    dag = cost_circuit.circuit_bodies[0][1]
    full_params = list(cost_circuit.parameters)
    n = dag.num_qubits()

    blocks: list[tuple[list, list]] = []
    prefix_ops: list[tuple] = []
    cur: list[tuple[int, SparsePauliOp]] = []
    cur_wires: set[int] = set()
    cur_prefix: list[tuple] = []

    def close() -> None:
        nonlocal cur, cur_wires, cur_prefix
        if cur:
            blocks.append((cur_prefix, cur))
        cur, cur_wires, cur_prefix = [], set(), []

    # Walk in original circuit insertion order: ``dag.op_nodes()`` yields nodes in
    # insertion order, whereas ``topological_op_nodes()`` ASAP-reschedules and would
    # split a layer's parallel rotations across the entangler staircase before them.
    for node in dag.op_nodes():
        parametric = [p for p in node.op.params if isinstance(p, ParameterExpression)]
        wires = [dag.find_bit(q).index for q in node.qargs]
        if parametric:
            pauli = _GATE_GENERATORS.get(node.op.name)
            angle = node.op.params[0]
            if pauli is None or len(parametric) != 1 or angle not in full_params:
                raise ContractViolation(_FS_UNSUPPORTED_GATE.format(gate=node.op.name))
            generator = SparsePauliOp.from_sparse_list(
                [(pauli, wires, 0.5)], num_qubits=n
            )
            if cur and set(wires) & cur_wires:
                close()
            if not cur:
                cur_prefix = list(prefix_ops)
            cur.append((full_params.index(angle), generator))
            cur_wires |= set(wires)
        else:
            close()
        prefix_ops.append((node.op, wires))
    close()
    return blocks, full_params, n


def _fs_block_covariance(
    program: "VariationalQuantumAlgorithm",
    full_params: list[Parameter],
    theta: npt.NDArray[np.float64],
    block_id: int,
) -> dict[tuple, tuple[tuple[int, ...], npt.NDArray[np.float64]]]:
    """``g_ij = 1/2 <{K_i, K_j}> - <K_i><K_j>`` on the block's pre-state."""
    exp_by_branch, branch_data = _measure_prefix_paulis(
        program, full_params, theta, block_id
    )

    def spo_exp(spo: SparsePauliOp, exp: dict[str, float]) -> float:
        total = 0.0
        for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
            if abs(coeff) <= 1e-12:
                continue
            total += float(np.real(coeff)) * (
                1.0 if set(label) == {"I"} else exp[label]
            )
        return total

    blocks: dict[tuple, tuple[tuple[int, ...], npt.NDArray[np.float64]]] = {}
    for branch_key, exp in exp_by_branch.items():
        indices, generators = branch_data[branch_key]
        k = len(generators)
        anticommutators = [
            [
                (
                    generators[i] @ generators[j] + generators[j] @ generators[i]
                ).simplify()
                for j in range(k)
            ]
            for i in range(k)
        ]
        single = np.array([spo_exp(generator, exp) for generator in generators])
        block = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                block[i, j] = (
                    0.5 * spo_exp(anticommutators[i][j], exp) - single[i] * single[j]
                )
        blocks[branch_key] = (indices, block)
    return blocks


def _measure_prefix_paulis(
    program: "VariationalQuantumAlgorithm",
    full_params: list[Parameter],
    theta: npt.NDArray[np.float64],
    block_id: int,
) -> tuple[
    dict[tuple, dict[str, float]],
    dict[tuple, tuple[tuple[int, ...], tuple[SparsePauliOp, ...]]],
]:
    """Measure every Pauli label a Fubini-Study block needs on each sampled
    block pre-state in one multi-observable pass.

    Returns ``(exp_by_branch, branch_data)``: the per-branch ``{label: <P>}``
    expectations and the per-branch ``(indices, generators)`` block structure,
    the latter recomputed from the cost cohort (``_post_spec_batch``)
    rather than smuggled out through a closure.
    """
    reference_blocks, _, n_qubits = _fs_blocks(program.cost_circuit)
    if block_id >= len(reference_blocks):
        raise ContractViolation(
            "The Fubini-Study block index is outside the reference ansatz."
        )

    reference_prefix_ops, _ = reference_blocks[block_id]
    reference_prefix_params = _fs_prefix_params(
        reference_prefix_ops, full_params, n_qubits
    )
    reference_prefix_param_names = tuple(p.name for p in reference_prefix_params)
    values = np.array(
        [[theta[full_params.index(p)] for p in reference_prefix_params]],
        dtype=np.float64,
    )

    preprocessor = _fs_prefix_labels_preprocessor(
        block_id, reference_prefix_param_names
    )
    indexed = _run_metric_by_branch(program, preprocessor, values)
    source = program._post_spec_batch()

    exp_by_branch: dict[tuple, dict[str, float]] = {}
    branch_data: dict[tuple, tuple[tuple[int, ...], tuple[SparsePauliOp, ...]]] = {}
    for branch_key, values_by_param in indexed.items():
        source_meta = source.get(branch_key)
        if source_meta is None:
            raise ContractViolation(
                "A measured Fubini-Study branch is absent from the cost cohort."
            )
        _, _, indices, generators, labels = _fs_block_prefix(
            source_meta, block_id, reference_prefix_param_names
        )
        branch_data[branch_key] = (indices, generators)
        vec = np.asarray(values_by_param[0], dtype=np.float64).reshape(-1)
        if labels and vec.shape[0] != len(labels):
            raise ContractViolation(
                "Fubini-Study per-label measurement count does not match the "
                "branch's label count."
            )
        exp_by_branch[branch_key] = {
            label: float(vec[i]) for i, label in enumerate(labels)
        }
    return exp_by_branch, branch_data


def _fs_prefix_params(
    prefix_ops: list[tuple],
    full_params: list[Parameter],
    n_qubits: int,
) -> tuple[Parameter, ...]:
    prefix = QuantumCircuit(n_qubits)
    for op, wires in prefix_ops:
        prefix.append(op, wires)
    return tuple(p for p in full_params if p in prefix.parameters)


def _fs_needed_labels(generators: tuple[SparsePauliOp, ...]) -> tuple[str, ...]:
    needed: set[str] = set()

    def collect(spo: SparsePauliOp) -> None:
        for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
            if abs(coeff) > 1e-12 and set(label) != {"I"}:
                needed.add(label)

    for generator in generators:
        collect(generator)
    for i, left in enumerate(generators):
        for right in generators[i:]:
            collect((left @ right + right @ left).simplify())
    return tuple(sorted(needed))


def _fs_block_prefix(
    meta: MetaCircuit,
    block_id: int,
    reference_prefix_param_names: tuple[str, ...],
) -> tuple[
    QuantumCircuit,
    tuple[Parameter, ...],
    tuple[int, ...],
    tuple[SparsePauliOp, ...],
    tuple[str, ...],
]:
    """Block pre-state circuit and its generator/label data for one FS block.

    Pure function of the (sampled) branch meta — used both to build the
    measurement observable and to recompute per-branch block structure from the
    cost cohort, so no data needs smuggling through a closure.
    """
    blocks, branch_params, n_qubits = _fs_blocks(meta)
    if block_id >= len(blocks):
        raise ContractViolation(
            "A sampled metric branch has fewer Fubini-Study blocks than "
            "the reference ansatz."
        )

    prefix_ops, entries = blocks[block_id]
    indices = tuple(index for index, _ in entries)
    generators = tuple(generator for _, generator in entries)
    labels = _fs_needed_labels(generators)

    prefix = QuantumCircuit(n_qubits)
    for op, wires in prefix_ops:
        prefix.append(op, wires)
    prefix_params = tuple(p for p in branch_params if p in prefix.parameters)
    if tuple(p.name for p in prefix_params) != reference_prefix_param_names:
        raise ContractViolation(
            "A sampled metric branch has a different Fubini-Study prefix "
            "parameter layout than the reference ansatz."
        )
    return prefix, prefix_params, indices, generators, labels


def _fs_prefix_labels_preprocessor(
    block_id: int,
    reference_prefix_param_names: tuple[str, ...],
) -> CircuitPreprocessor:
    """Cacheable preprocessor measuring every Pauli label a Fubini-Study block
    needs on its pre-state in one multi-observable pass. Pure: the per-branch
    block structure is recomputed from the cost cohort by the caller."""

    def preprocess(meta: MetaCircuit) -> MetaCircuit:
        prefix, prefix_params, _, _, labels = _fs_block_prefix(
            meta, block_id, reference_prefix_param_names
        )
        observable = (
            tuple(SparsePauliOp(label) for label in labels)
            if labels
            else (_zero_observable(prefix.num_qubits),)
        )
        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(prefix)),),
            parameters=prefix_params,
            observable=observable,
            _was_multi_obs=True,
        )

    return CircuitPreprocessor(
        "metric-prefix",
        preprocess=preprocess,
        consumes_dag_bodies=True,
        cache_key=("metric-prefix", block_id),
    )
