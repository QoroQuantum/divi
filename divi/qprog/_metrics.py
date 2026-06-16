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
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit, build_overlap_meta
from divi.pipeline import ResultFormat
from divi.pipeline._compilation import _extract_param_set_idx
from divi.pipeline.abc import ContractViolation
from divi.pipeline.stages import CircuitSpecStage, DataBindingStage, MeasurementStage

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

# Evaluator callables vary in arity: ``metric_fn``/``jac`` take the parameter
# vector; ``fidelity_fn`` takes ``(theta, perturbations)``. Hence ``Callable[...]``.
Evaluators = dict[str, Callable[..., Any]]


def _run_metric(
    program: "VariationalQuantumAlgorithm",
    meta_circuit: MetaCircuit,
    param_sets: npt.NDArray[np.float64],
) -> dict[int, npt.NDArray[np.float64]]:
    """Run ``meta_circuit`` through ``program``'s expectation pipeline, returning
    ``{param_set_idx: per-observable expectations}``.

    The single seam the estimators use: it runs the supplied MetaCircuit —
    whatever observables and body it carries (the cost ansatz, or a truncated
    prefix for the Fubini–Study metric) — through the program's generic
    :meth:`~divi.qprog.VariationalQuantumAlgorithm._expectation_pipeline`, built on
    demand so the program never registers a metric-specific pipeline.
    """
    result = program._execute(
        program._expectation_pipeline(),
        meta_circuit,
        param_sets=np.atleast_2d(param_sets),
    )
    return {
        _extract_param_set_idx(key, default=0): np.asarray(
            value, dtype=np.float64
        ).reshape(-1)
        for key, value in result.items()
    }


def _run_overlap(
    program: "VariationalQuantumAlgorithm",
    meta_circuit: MetaCircuit,
    param_sets: npt.NDArray[np.float64],
) -> dict[int, float]:
    """Run a compute-uncompute ``meta_circuit`` as a probability measurement,
    returning ``{param_set_idx: P(all-zeros)}`` — the state-overlap fidelity.

    The probs sibling of :func:`_run_metric`: it assembles the program's PROBS
    measurement pipeline on demand (as :class:`SolutionSamplingMixin` does) and
    reads the all-zeros bitstring probability per parameter row.
    """
    pipeline = program._assemble_pipeline(
        CircuitSpecStage(),
        MeasurementStage(),
        result_format=ResultFormat.PROBS,
    )
    result = program._execute(
        pipeline, meta_circuit, param_sets=np.atleast_2d(param_sets)
    )
    zeros = "0" * meta_circuit.n_qubits
    return {
        _extract_param_set_idx(key, default=0): _zeros_probability(value, zeros)
        for key, value in result.items()
    }


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
    """

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        stages = program._cost_pipeline.stages
        if not any(isinstance(s, MeasurementStage) for s in stages):
            raise ContractViolation(
                "The pullback metric requires the loss to be the expectation "
                "value of the cost Hamiltonian, but this program's cost pipeline "
                "computes a classical objective (e.g. PCE). Use the Fubini–Study "
                "estimator (FubiniStudyMetricEstimator) instead."
            )
        if any(getattr(s, "sample_loss", None) is not None for s in stages):
            raise ContractViolation(
                "The pullback metric is invalid for a supervised data-binding "
                "loss: the per-sample loss is a non-linear function of the "
                "expectation values. Use a non-metric optimizer or the "
                "Fubini–Study estimator."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        shift_mask = program._grad_shift_mask
        supports_expval = bool(getattr(program.backend, "supports_expval", False))
        cache: dict[str, Any] = {"key": None, "value": None}

        def grad_and_metric(
            params: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            key = (
                program._evaluation_counter,
                np.asarray(params, dtype=np.float64).tobytes(),
            )
            if cache["key"] != key:
                gradients = []
                metrics = []
                for source_meta in _cost_source_metas(program):
                    if (
                        source_meta.observable is None
                        or len(source_meta.observable) != 1
                    ):
                        raise ContractViolation(
                            "The pullback metric requires each cost circuit to "
                            "carry exactly one loss observable."
                        )
                    observables, coeffs = _split_into_terms(source_meta.observable[0])
                    exp_vals = _term_expectations(
                        program,
                        observables,
                        supports_expval,
                        shift_mask + params,
                        source_meta=source_meta,
                    )
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


def _term_expectations(
    program: "VariationalQuantumAlgorithm",
    observables: tuple[SparsePauliOp, ...],
    supports_expval: bool,
    param_sets: npt.NDArray[np.float64],
    *,
    source_meta: MetaCircuit | None = None,
) -> npt.NDArray[np.float64]:
    """``(n_sets, n_terms)`` per-term expectations on the cost ansatz.

    On expval-capable backends each term is measured as its own single-observable
    run (analytic, no shot noise); otherwise all terms are measured together in a
    single qubit-wise-commuting run from shot counts.
    """
    if supports_expval:
        columns = []
        for obs in observables:
            indexed = _average_metric_runs(
                [
                    _run_metric(program, meta, param_sets)
                    for meta in _cost_ansatz_metas(
                        program, (obs,), source_meta=source_meta
                    )
                ]
            )
            columns.append(np.array([indexed[k][0] for k in sorted(indexed)]))
        return np.column_stack(columns)

    indexed = _average_metric_runs(
        [
            _run_metric(program, meta, param_sets)
            for meta in _cost_ansatz_metas(
                program, observables, source_meta=source_meta
            )
        ]
    )
    return np.vstack([indexed[k] for k in sorted(indexed)])


def _average_metric_runs(
    runs: list[dict[int, npt.NDArray[np.float64]]],
) -> dict[int, npt.NDArray[np.float64]]:
    """Average identically-indexed metric results across a source cohort."""
    if not runs:
        raise RuntimeError("Metric source cohort cannot be empty.")
    keys = runs[0].keys()
    if any(run.keys() != keys for run in runs[1:]):
        raise ContractViolation("Metric source cohort returned inconsistent indices.")
    return {key: np.mean([run[key] for run in runs], axis=0) for key in keys}


def _cost_source_metas(
    program: "VariationalQuantumAlgorithm",
) -> tuple[MetaCircuit, ...]:
    """Return the cost pipeline's cached spec-stage circuit cohort."""
    return tuple(program._pipeline_source_batch("cost").values())


def _cost_ansatz_metas(
    program: "VariationalQuantumAlgorithm",
    observables: tuple[SparsePauliOp, ...],
    *,
    source_meta: MetaCircuit | None = None,
) -> tuple[MetaCircuit, ...]:
    """Cost ansatz cohort with each member measuring ``observables``."""
    sources = (source_meta,) if source_meta is not None else _cost_source_metas(program)
    return tuple(
        replace(
            meta,
            observable=tuple(observables),
            _was_multi_obs=True,
        )
        for meta in sources
    )


def _cost_ansatz_meta(
    program: "VariationalQuantumAlgorithm",
    observables: tuple[SparsePauliOp, ...],
) -> MetaCircuit:
    """The cost ansatz MetaCircuit with its observable replaced by ``observables``
    — the per-term set the pullback metric measures on the full ansatz."""
    return _cost_ansatz_metas(program, observables)[0]


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
        _fs_blocks(program.meta_circuit_factories["cost_circuit"])
        if any(isinstance(s, DataBindingStage) for s in program._cost_pipeline.stages):
            raise ContractViolation(
                "The Fubini–Study metric does not support data-bound programs: "
                "the ansatz state depends on the data input, so the metric is "
                "data-dependent. Use a non-metric optimizer."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        supports_expval = bool(getattr(program.backend, "supports_expval", False))

        def metric_fn(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            theta = np.asarray(params, dtype=np.float64).reshape(-1)
            cohort_metrics = []
            for source_meta in _cost_source_metas(program):
                blocks, full_params, n_qubits = _fs_blocks(source_meta)
                metric = np.zeros((len(full_params), len(full_params)))
                for prefix_ops, gens in blocks:
                    indices = [pidx for pidx, _ in gens]
                    generators = [g for _, g in gens]
                    block = _fs_block_covariance(
                        program,
                        prefix_ops,
                        full_params,
                        n_qubits,
                        theta,
                        generators,
                        supports_expval,
                    )
                    for a, ia in enumerate(indices):
                        for b, ib in enumerate(indices):
                            metric[ia, ib] = block[a, b]
                cohort_metrics.append(metric)
            return np.mean(cohort_metrics, axis=0)

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
    it applies to any qiskit-invertible ansatz. It is built from the cost-ansatz
    realization captured at construction (``meta_circuit_factories["cost_circuit"]``)
    and does not re-sample with a per-evaluation stochastic trotterization, so for
    QDrift QAOA the metric is the geometry of that one fixed realization
    (intentionally decoupled from the per-evaluation cost cohort) and stays
    consistent across the run.
    """

    def check_compatible(self, program: "VariationalQuantumAlgorithm") -> None:
        try:
            build_overlap_meta(program.meta_circuit_factories["cost_circuit"])
        except Exception as exc:
            raise ContractViolation(
                "The stochastic-fidelity metric requires an invertible ansatz "
                "(qiskit QuantumCircuit.inverse()); this program's cost circuit "
                "could not be inverted."
            ) from exc
        if any(isinstance(s, DataBindingStage) for s in program._cost_pipeline.stages):
            raise ContractViolation(
                "The stochastic-fidelity metric does not support data-bound "
                "programs: the ansatz state depends on the data input, so the "
                "fidelity is data-dependent. Use a non-metric optimizer."
            )

    def bind(self, program: "VariationalQuantumAlgorithm") -> Evaluators:
        overlap_meta = build_overlap_meta(
            program.meta_circuit_factories["cost_circuit"]
        )

        def fidelity_fn(
            theta: npt.NDArray[np.float64],
            perturbations: list[npt.NDArray[np.float64]],
        ) -> npt.NDArray[np.float64]:
            """Fidelities ``F(theta, theta + p)`` for each ``p`` in ``perturbations``,
            measured in a single batched submission (one row per overlap point)."""
            theta = np.asarray(theta, dtype=np.float64).reshape(-1)
            rows = np.vstack(
                [
                    np.concatenate([theta, theta + np.asarray(p, dtype=np.float64)])
                    for p in perturbations
                ]
            )
            indexed = _run_overlap(program, overlap_meta, rows)
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
    prefix_ops: list[tuple],
    full_params: list[Parameter],
    n_qubits: int,
    theta: npt.NDArray[np.float64],
    generators: list[SparsePauliOp],
    supports_expval: bool,
) -> npt.NDArray[np.float64]:
    """``g_ij = 1/2 <{K_i, K_j}> - <K_i><K_j>`` on the block's pre-state."""
    k = len(generators)
    anticommutators = [
        [
            (generators[i] @ generators[j] + generators[j] @ generators[i]).simplify()
            for j in range(k)
        ]
        for i in range(k)
    ]

    needed: set[str] = set()

    def collect(spo: SparsePauliOp) -> None:
        for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
            if abs(coeff) > 1e-12 and set(label) != {"I"}:
                needed.add(label)

    for generator in generators:
        collect(generator)
    for row in anticommutators:
        for ac in row:
            collect(ac)

    exp = _measure_prefix_paulis(
        program,
        prefix_ops,
        full_params,
        n_qubits,
        theta,
        sorted(needed),
        supports_expval,
    )

    def spo_exp(spo: SparsePauliOp) -> float:
        total = 0.0
        for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
            if abs(coeff) <= 1e-12:
                continue
            total += float(np.real(coeff)) * (
                1.0 if set(label) == {"I"} else exp[label]
            )
        return total

    single = np.array([spo_exp(generator) for generator in generators])
    block = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            block[i, j] = 0.5 * spo_exp(anticommutators[i][j]) - single[i] * single[j]
    return block


def _measure_prefix_paulis(
    program: "VariationalQuantumAlgorithm",
    prefix_ops: list[tuple],
    full_params: list[Parameter],
    n_qubits: int,
    theta: npt.NDArray[np.float64],
    needed: list[str],
    supports_expval: bool,
) -> dict[str, float]:
    """Measure each needed Pauli label on the block's pre-state, returning
    ``{label: expectation}``. Expval backends measure each Pauli as its own
    analytic single-observable run; sampling backends measure them together in
    one qubit-wise-commuting run."""
    if not needed:
        return {}

    prefix = QuantumCircuit(n_qubits)
    for op, wires in prefix_ops:
        prefix.append(op, wires)
    prefix_params = tuple(p for p in full_params if p in prefix.parameters)
    values = np.array(
        [[theta[full_params.index(p)] for p in prefix_params]], dtype=np.float64
    )
    prefix_dag = circuit_to_dag(prefix)

    def meta(observable: tuple[SparsePauliOp, ...]) -> MetaCircuit:
        return MetaCircuit(
            circuit_bodies=(((), prefix_dag),),
            parameters=prefix_params,
            observable=observable,
            _was_multi_obs=True,
        )

    if supports_expval:
        return {
            label: float(
                _run_metric(program, meta((SparsePauliOp(label),)), values)[0][0]
            )
            for label in needed
        }

    vals = _run_metric(
        program, meta(tuple(SparsePauliOp(label) for label in needed)), values
    )[0]
    return {label: float(vals[i]) for i, label in enumerate(needed)}
