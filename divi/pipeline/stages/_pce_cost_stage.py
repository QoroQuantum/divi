# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage for PCE Z-basis measurement and binary-polynomial reduction."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageToken,
)
from divi.pipeline.stages._numba_kernels import (
    _compute_hard_cvar_energy_jit,
    _eval_poly_1d_jit,
    _eval_poly_2d_jit,
    compile_problem,
)
from divi.typing import BinaryPolynomialProblem

# Type alias for the tuple returned by ``compile_problem``.
CompiledProblem = tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64], float
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PCE_MEAS_AXIS = "pce_meas"
# Axis name for the single Z-basis measurement circuit emitted by PCECostStage.

# ---------------------------------------------------------------------------
# PCE energy helpers
# ---------------------------------------------------------------------------


def _evaluate_binary_polynomial(
    x_vals: npt.NDArray[np.float64],
    problem: BinaryPolynomialProblem,
    _compiled: CompiledProblem | None = None,
) -> npt.NDArray[np.float64] | float:
    """Evaluate binary polynomial energy for one or many assignments.

    Degree-1 terms are evaluated as ``c * x_i²`` rather than ``c * x_i`` to
    undo the linearisation (``x_i² → x_i``) applied during polynomial
    normalisation.  This is a no-op for binary values (``x² = x``) but
    produces correct energies for continuous soft-relaxed values.

    Args:
        x_vals: Variable assignments. Shape ``(n_vars,)`` for one assignment
            or ``(n_vars, n_states)`` for many.
        problem: Canonical binary polynomial problem.
        _compiled: Pre-compiled CSR arrays from :func:`compile_problem`.
            When provided the Numba JIT kernel is used instead of the
            Python loop.
    """
    if _compiled is not None:
        term_indices, term_offsets, coeffs, constant = _compiled
        x = np.ascontiguousarray(x_vals, dtype=np.float64)
        if x.ndim == 1:
            return float(
                _eval_poly_1d_jit(x, term_indices, term_offsets, coeffs, constant)
            )
        return _eval_poly_2d_jit(x, term_indices, term_offsets, coeffs, constant)

    is_single = x_vals.ndim == 1
    energy = 0.0 if is_single else np.zeros(x_vals.shape[1], dtype=np.float64)

    for term, coeff in problem.terms.items():
        coeff = float(coeff)
        if coeff == 0:
            continue
        if len(term) == 0:
            energy = energy + coeff
            continue

        indices = [problem.variable_to_idx[var] for var in term]
        if len(term) == 1:
            # De-linearise: evaluate as c * x_i² instead of c * x_i.
            idx = indices[0]
            monomial = x_vals[idx] ** 2 if is_single else x_vals[idx, :] ** 2
        elif is_single:
            monomial = np.prod(x_vals[indices])
        else:
            monomial = np.prod(x_vals[indices, :], axis=0)
        energy = energy + (coeff * monomial)

    return float(energy) if is_single else energy


def _compute_soft_energy(
    parities: npt.NDArray[np.uint8],
    probs: npt.NDArray[np.float64],
    alpha: float,
    problem: BinaryPolynomialProblem,
    _compiled: CompiledProblem | None = None,
) -> float:
    """Compute the relaxed (soft) energy from parity expectations."""
    mean_parities = parities.dot(probs)
    z_expectations = 1.0 - (2.0 * mean_parities)
    x_soft = 0.5 * (1.0 + np.tanh(alpha * z_expectations))
    return float(_evaluate_binary_polynomial(x_soft, problem, _compiled=_compiled))


def _compute_hard_cvar_energy(
    parities: npt.NDArray[np.uint8],
    counts: npt.NDArray[np.float64],
    total_shots: float,
    problem: BinaryPolynomialProblem,
    alpha_cvar: float = 0.25,
    _compiled: CompiledProblem | None = None,
) -> float:
    """Compute CVaR energy from sampled hard assignments."""
    x_vals = np.ascontiguousarray(1.0 - parities.astype(np.float64))

    if _compiled is not None:
        term_indices, term_offsets, coeffs, constant = _compiled
        return float(
            _compute_hard_cvar_energy_jit(
                x_vals,
                np.ascontiguousarray(counts, dtype=np.float64),
                float(total_shots),
                float(alpha_cvar),
                term_indices,
                term_offsets,
                coeffs,
                constant,
            )
        )

    energies = _evaluate_binary_polynomial(x_vals, problem)

    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    sorted_counts = counts[sorted_indices]

    cutoff_count = int(np.ceil(alpha_cvar * total_shots))
    accumulated_counts = np.cumsum(sorted_counts)
    limit_idx = np.searchsorted(accumulated_counts, cutoff_count)

    cvar_energy = 0.0
    count_sum = 0
    if limit_idx > 0:
        cvar_energy += np.sum(sorted_energies[:limit_idx] * sorted_counts[:limit_idx])
        count_sum += np.sum(sorted_counts[:limit_idx])

    remaining = cutoff_count - count_sum
    cvar_energy += sorted_energies[limit_idx] * remaining
    return float(cvar_energy / cutoff_count)


class PCECostStage(BundleStage):
    """Pipeline stage that emits a single Z-basis measurement and computes
    nonlinear binary-polynomial energy from shot histograms.

    PCE only needs raw bitstring counts (not expectation values), so this
    stage bypasses MeasurementStage's observable grouping entirely.  Expand
    generates a single "measure all qubits" QASM per circuit spec, and
    reduce applies the soft tanh or hard CVaR energy formula.

    Args:
        problem: Canonical binary polynomial problem used for objective evaluation.
        alpha: Scaling factor for the tanh activation.
        use_soft_objective: If True, compute relaxed (soft) energy;
            otherwise compute hard CVaR energy.
        decode_parities_fn: Function mapping (state_strings, masks) → parities.
        variable_masks_u64: Precomputed uint64 masks for each QUBO variable.
        alpha_cvar: CVaR tail fraction (only used when use_soft_objective is False).
    """

    def __init__(
        self,
        *,
        problem: BinaryPolynomialProblem,
        alpha: float,
        use_soft_objective: bool,
        decode_parities_fn: Callable,
        variable_masks_u64: npt.NDArray[np.uint64],
        alpha_cvar: float = 0.25,
    ) -> None:
        super().__init__(name="PCECostStage")
        self._problem = problem
        self._alpha = alpha
        self._soft = use_soft_objective
        self._decode = decode_parities_fn
        self._masks = variable_masks_u64
        self._alpha_cvar = alpha_cvar
        self._compiled = compile_problem(problem)

    @property
    def axis_name(self) -> str:
        return PCE_MEAS_AXIS

    @property
    def handles_measurement(self) -> bool:
        return True

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Emit a single Z-basis measurement circuit per circuit spec.

        Generates "measure all qubits" QASM and sets the result format to
        COUNTS so raw shot histograms reach reduce.
        """
        env.result_format = ResultFormat.COUNTS

        out = {}
        for key, meta in batch.items():
            n_qubits = len(meta.source_circuit.wires)
            measure_qasm = "".join(
                f"measure q[{i}] -> c[{i}];\n" for i in range(n_qubits)
            )
            tagged = ((((PCE_MEAS_AXIS, 0),), measure_qasm),)
            out[key] = meta.set_measurement_bodies(tagged)

        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Compute polynomial energy from shot histograms.

        Each param set has a single histogram (no observable-group merging
        needed).  Applies the soft tanh or hard CVaR energy formula.
        """
        reduced: dict[object, Any] = {}
        for key, histogram in results.items():
            base_key = tuple(ax for ax in key if ax[0] != PCE_MEAS_AXIS)

            state_strings = list(histogram.keys())
            counts = np.array(list(histogram.values()), dtype=float)
            total_shots = counts.sum()
            parities = self._decode(state_strings, self._masks)

            if self._soft:
                probs = counts / total_shots
                reduced[base_key] = _compute_soft_energy(
                    parities,
                    probs,
                    self._alpha,
                    self._problem,
                    _compiled=self._compiled,
                )
            else:
                reduced[base_key] = _compute_hard_cvar_energy(
                    parities,
                    counts,
                    total_shots,
                    self._problem,
                    self._alpha_cvar,
                    _compiled=self._compiled,
                )

        return reduced
