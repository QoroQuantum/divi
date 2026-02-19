# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage for PCE observable grouping and QUBO energy reduction."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.pipeline.abc import ChildResults, PipelineEnv, ResultFormat, StageToken
from divi.pipeline.stages._measurement_stage import (
    OBS_GROUP_AXIS,
    MeasurementStage,
)
from divi.pipeline.transformations import group_by_base_key

# ---------------------------------------------------------------------------
# PCE energy helpers (moved from qprog.algorithms._pce)
# ---------------------------------------------------------------------------


def _compute_soft_energy(
    parities: npt.NDArray[np.uint8],
    probs: npt.NDArray[np.float64],
    alpha: float,
    qubo_matrix: npt.NDArray[np.float64] | np.ndarray,
) -> float:
    """Compute the relaxed (soft) QUBO energy from parity expectations."""
    mean_parities = parities.dot(probs)
    z_expectations = 1.0 - (2.0 * mean_parities)
    x_soft = 0.5 * (1.0 + np.tanh(alpha * z_expectations))
    Qx = qubo_matrix @ x_soft
    return float(np.dot(x_soft, Qx))


def _compute_hard_cvar_energy(
    parities: npt.NDArray[np.uint8],
    counts: npt.NDArray[np.float64],
    total_shots: float,
    qubo_matrix: npt.NDArray[np.float64] | np.ndarray,
    alpha_cvar: float = 0.25,
) -> float:
    """Compute CVaR energy from sampled hard assignments."""
    x_vals = 1.0 - parities.astype(float)
    Qx = qubo_matrix @ x_vals
    energies = np.einsum("ij,ij->j", x_vals, Qx)

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


class PCECostStage(MeasurementStage):
    """MeasurementStage variant whose reduce computes QUBO energy.

    Expand is inherited from MeasurementStage (sets up Z-basis measurement QASMs)
    but overrides ``result_format`` to ``COUNTS`` so raw shot histograms reach
    the reduce step for nonlinear energy computation.

    Reduce aggregates histograms across observable groups for each param set
    and applies the PCE nonlinear energy formula instead of the standard
    linear Hamiltonian combination.

    Args:
        qubo_matrix: The QUBO cost matrix.
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
        qubo_matrix: npt.NDArray[np.float64],
        alpha: float,
        use_soft_objective: bool,
        decode_parities_fn: Callable,
        variable_masks_u64: npt.NDArray[np.uint64],
        alpha_cvar: float = 0.25,
    ) -> None:
        # No grouping — each Z observable gets its own measurement circuit.
        super().__init__(
            grouping_strategy=None,
            result_format_override=ResultFormat.COUNTS,
        )
        self._qubo = qubo_matrix
        self._alpha = alpha
        self._soft = use_soft_objective
        self._decode = decode_parities_fn
        self._masks = variable_masks_u64
        self._alpha_cvar = alpha_cvar

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Aggregate observable-group results and compute PCE QUBO energy.

        For shot-based backends: merge histograms across observable groups,
        then compute parity-based soft or hard CVaR energy.

        For expval backends: collect per-Z expectation values, then apply
        the PCE soft energy formula.
        """
        grouped = group_by_base_key(results, OBS_GROUP_AXIS, indexed=True)

        reduced: dict[object, Any] = {}
        for base_key, indexed_items in grouped.items():
            # indexed_items: dict[obs_group_idx, value]
            # Sort by index to preserve observable order for expval path.
            values = [indexed_items[k] for k in sorted(indexed_items)]

            if env.result_format == ResultFormat.EXPVALS:
                # Each value is a scalar Z expectation.
                z_expectations = np.array(values, dtype=np.float64)
                x_soft = 0.5 * (1.0 + np.tanh(self._alpha * z_expectations))
                reduced[base_key] = float(np.dot(x_soft, self._qubo @ x_soft))
            else:
                # Each value is a histogram dict {bitstring: count}.
                # Merge all histograms for this param set.
                shots_dict: dict[str, int] = {}
                for histogram in values:
                    for bitstring, count in histogram.items():
                        shots_dict[bitstring] = shots_dict.get(bitstring, 0) + count

                state_strings = list(shots_dict.keys())
                counts = np.array(list(shots_dict.values()), dtype=float)
                total_shots = counts.sum()
                parities = self._decode(state_strings, self._masks)

                if self._soft:
                    probs = counts / total_shots
                    reduced[base_key] = _compute_soft_energy(
                        parities, probs, self._alpha, self._qubo
                    )
                else:
                    reduced[base_key] = _compute_hard_cvar_energy(
                        parities, counts, total_shots, self._qubo, self._alpha_cvar
                    )

        return reduced
