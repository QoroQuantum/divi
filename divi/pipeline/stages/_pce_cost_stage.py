# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage for PCE Z-basis measurement and binary-polynomial reduction."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.hamiltonians import (
    BinaryPolynomialProblem,
    CompiledBinaryPolynomial,
    compile_problem,
)
from divi.hamiltonians._polynomial import (
    _compute_hard_cvar_energy_jit,
    _evaluate_binary_polynomial,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageOutput,
    StageToken,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PCE_MEAS_AXIS = "pce_meas"
# Axis name for the single Z-basis measurement circuit emitted by PCECostStage.


@dataclass(frozen=True)
class _PCEMeasToken:
    """Carries the expected full register width from expand to reduce."""

    n_qubits: int


# ---------------------------------------------------------------------------
# PCE energy reducers (parity → polynomial energy aggregation)
# ---------------------------------------------------------------------------


def _compute_soft_energy(
    parities: npt.NDArray[np.uint8],
    probs: npt.NDArray[np.float64],
    alpha: float,
    problem: BinaryPolynomialProblem,
    _compiled: CompiledBinaryPolynomial | None = None,
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
    _compiled: CompiledBinaryPolynomial | None = None,
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

    energies = np.atleast_1d(_evaluate_binary_polynomial(x_vals, problem))

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
    generates a single Z-basis measurement QASM per circuit spec (over the
    mask-relevant qubits, or all qubits when ``measure_all``), and reduce
    applies the soft tanh or hard CVaR energy formula.

    Args:
        problem: Canonical binary polynomial problem used for objective evaluation.
        alpha: Scaling factor for the tanh activation.
        use_soft_objective: If True, compute relaxed (soft) energy;
            otherwise compute hard CVaR energy.
        decode_parities_fn: Function mapping (state_strings, masks) → parities.
        variable_masks_u64: Precomputed uint64 masks for each QUBO variable.
        alpha_cvar: CVaR tail fraction (only used when use_soft_objective is False).
        measure_all: When ``False`` (default), measure only the qubits that
            appear in some variable mask (qubits in no mask never contribute to
            a parity, so the energy is unchanged). ``True`` measures every qubit.
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
        measure_all: bool = False,
    ) -> None:
        super().__init__(name="PCECostStage")
        self._problem = problem
        self._alpha = alpha
        self._soft = use_soft_objective
        self._decode = decode_parities_fn
        self._masks = variable_masks_u64
        self._alpha_cvar = alpha_cvar
        self._compiled = compile_problem(problem)
        self._measure_all = measure_all
        # Wires touched by at least one mask (bit i ↔ wire i); others need no
        # measurement. Masks are 1-D uint64 (<=64 qubits) or 2-D limbs.
        if variable_masks_u64.size == 0:
            mask_union = 0
        else:
            limbs = np.atleast_1d(np.bitwise_or.reduce(variable_masks_u64, axis=0))
            mask_union = sum(int(limb) << (64 * j) for j, limb in enumerate(limbs))
        self._relevant_wires = tuple(
            i for i in range(mask_union.bit_length()) if (mask_union >> i) & 1
        )

    @property
    def axis_name(self) -> str:
        return PCE_MEAS_AXIS

    @property
    def handles_measurement(self) -> bool:
        return True

    @property
    def consumes_dag_bodies(self) -> bool:
        # Reads only ``meta.n_qubits`` to build the Z-basis measurement
        # QASM — never inspects body gate content.
        return False

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Emit a Z-basis measurement circuit per spec, result format COUNTS.

        With ``measure_all=False`` only mask-relevant qubits are measured; the
        rest stay 0 in the full-width register and are ignored by the decoder.
        """
        out = {}
        for key, meta in batch.items():
            if self._measure_all or not self._relevant_wires:
                wires = range(meta.n_qubits)
            else:
                wires = self._relevant_wires
            measure_qasm = "".join(f"measure q[{i}] -> c[{i}];\n" for i in wires)
            tagged = ((((PCE_MEAS_AXIS, 0),), measure_qasm),)
            out[key] = meta.set_measurement_bodies(tagged).set_result_format(
                ResultFormat.COUNTS
            )

        sample = next(iter(batch.values()), None)
        token = _PCEMeasToken(n_qubits=sample.n_qubits if sample is not None else 0)
        return StageOutput(batch=out, token=token)

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
            # Parity decoding assumes full-width keys; narrowed keys would
            # misalign every mask (mirrors the _batched_expectation guard).
            if (
                isinstance(token, _PCEMeasToken)
                and state_strings
                and len(state_strings[0]) != token.n_qubits
            ):
                raise ValueError(
                    f"Backend returned {len(state_strings[0])}-bit PCE histogram "
                    f"keys for an {token.n_qubits}-qubit circuit; expected "
                    f"full-width keys (creg c[{token.n_qubits}]). "
                    "Partial-measurement circuits must still report all "
                    "classical bits. If your backend cannot, set "
                    "measure_all_qubits=True to measure the full register."
                )
            counts = np.array(list(histogram.values()), dtype=float)
            total_shots = counts.sum()
            parities = self._decode(state_strings, self._masks)

            if self._soft:
                probs = counts / total_shots
                reduced[base_key] = [
                    _compute_soft_energy(
                        parities,
                        probs,
                        self._alpha,
                        self._problem,
                        _compiled=self._compiled,
                    )
                ]
            else:
                reduced[base_key] = [
                    _compute_hard_cvar_energy(
                        parities,
                        counts,
                        total_shots,
                        self._problem,
                        self._alpha_cvar,
                        _compiled=self._compiled,
                    )
                ]

        return reduced
