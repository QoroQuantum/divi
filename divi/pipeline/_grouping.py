# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal

import numpy as np
import pennylane as qml
from pennylane.transforms.core.transform_program import TransformProgram


def _extract_coeffs(obs: qml.operation.Operator) -> list[float]:
    """Extract coefficients from an observable, including nested scalar products."""
    coeff = 1.0
    base = obs
    while isinstance(base, qml.ops.SProd):
        coeff *= base.scalar
        base = base.base
    if isinstance(base, (qml.Hamiltonian, qml.ops.Sum)):
        coeffs, _ = base.terms()
        return [coeff * c for c in coeffs]
    return [coeff]


TRANSFORM_PROGRAM = TransformProgram()
TRANSFORM_PROGRAM.add_transform(qml.transforms.split_to_single_terms)


def _wire_grouping(measurements: list[qml.measurements.MeasurementProcess]):
    """
    Groups a list of PennyLane MeasurementProcess objects by mutually non-overlapping wires.

    Each group contains measurements whose wires do not overlap with those of any other
    measurement in the same group. This enables parallel measurement of compatible observables,
    e.g., for grouped execution or more efficient sampling.

    Returns:
        partition_indices (list[list[int]]): Indices of the original measurements in each group.
        mp_groups (list[list[MeasurementProcess]]): Grouped MeasurementProcess objects.
    """
    mp_groups = []
    wires_for_each_group = []
    group_mapping = {}  # original_index -> (group_idx, pos_in_group)

    for i, mp in enumerate(measurements):
        added = False
        for group_idx, wires in enumerate(wires_for_each_group):
            if not qml.wires.Wires.shared_wires([wires, mp.wires]):
                mp_groups[group_idx].append(mp)
                wires_for_each_group[group_idx] += mp.wires
                group_mapping[i] = (group_idx, len(mp_groups[group_idx]) - 1)
                added = True
                break
        if not added:
            mp_groups.append([mp])
            wires_for_each_group.append(mp.wires)
            group_mapping[i] = (len(mp_groups) - 1, 0)

    partition_indices = [[] for _ in range(len(mp_groups))]
    for original_idx, (group_idx, _) in group_mapping.items():
        partition_indices[group_idx].append(original_idx)

    return partition_indices, mp_groups


def _create_final_postprocessing_fn(coefficients, partition_indices, num_total_obs):
    """Create a wrapper fn that reconstructs the flat results list and computes the final energy."""
    reverse_map = [None] * num_total_obs
    for group_idx, indices_in_group in enumerate(partition_indices):
        for idx_within_group, original_flat_idx in enumerate(indices_in_group):
            reverse_map[original_flat_idx] = (group_idx, idx_within_group)

    missing_indices = [i for i, v in enumerate(reverse_map) if v is None]
    if missing_indices:
        raise RuntimeError(
            f"partition_indices does not cover all observable indices. Missing indices: {missing_indices}"
        )

    def final_postprocessing_fn(grouped_results):
        """
        Takes grouped results, flattens them to the original order,
        multiplies by coefficients, and sums to get the final energy.
        """
        if len(grouped_results) != len(partition_indices):
            raise RuntimeError(
                f"Expected {len(partition_indices)} grouped results, but got {len(grouped_results)}."
            )
        flat_results = np.zeros(num_total_obs, dtype=np.float64)
        for original_flat_idx in range(num_total_obs):
            group_idx, idx_within_group = reverse_map[original_flat_idx]

            group_result = grouped_results[group_idx]
            if isinstance(group_result, dict):
                val = group_result[idx_within_group]
            else:
                # Scalar from custom execute_fn or single-obs shorthand.
                val = group_result
            flat_results[original_flat_idx] = float(np.asarray(val).flat[0])

        # Perform the final summation using the efficient dot product method.
        return np.dot(coefficients, flat_results)

    return final_postprocessing_fn


GroupingStrategy = Literal["wires", "default", "qwc", "_backend_expval"] | None


def compute_measurement_groups(
    measurement: qml.measurements.MeasurementProcess,
    strategy: GroupingStrategy,
) -> tuple[
    tuple[tuple[qml.operation.Operator, ...], ...],
    list[list[int]],
    Callable,
]:
    """Compute measurement groups, partition indices, and postprocessing function.

    Args:
        measurement: PennyLane MeasurementProcess (e.g. expval(obs) or probs()).
        strategy: Grouping strategy: "wires", "qwc", "default", "_backend_expval", or None.

    Returns:
        tuple of (measurement_groups, partition_indices, postprocessing_fn).
        - measurement_groups: Groups of observables for each circuit.
        - partition_indices: Indices mapping original observables to groups.
        - postprocessing_fn: Callable to combine grouped results into final value.
    """
    # For probs() or measurement with no obs: single group, identity postprocessing.
    if not hasattr(measurement, "obs") or measurement.obs is None:
        return ((),), [[0]], lambda x: x

    # Step 1: Split to single-term measurements.
    measurements_only_tape = qml.tape.QuantumScript(measurements=[measurement])
    s_tapes, _ = TRANSFORM_PROGRAM((measurements_only_tape,))
    single_term_mps = s_tapes[0].measurements

    # Step 2: Extract coefficients and build groups.
    obs = measurement.obs
    coeffs = _extract_coeffs(obs)

    if strategy in ("qwc", "default"):
        obs_list = [m.obs for m in single_term_mps]
        partition_indices_raw = qml.pauli.compute_partition_indices(obs_list)
        partition_indices = [list(group) for group in partition_indices_raw]
        measurement_groups = tuple(
            tuple(single_term_mps[i].obs for i in group) for group in partition_indices
        )
    elif strategy == "wires":
        partition_indices, grouped_mps = _wire_grouping(single_term_mps)
        measurement_groups = tuple(tuple(m.obs for m in group) for group in grouped_mps)
    elif strategy is None:
        measurement_groups = tuple(tuple([m.obs]) for m in single_term_mps)
        partition_indices = [[i] for i in range(len(single_term_mps))]
    elif strategy == "_backend_expval":
        measurement_groups = ((),)
        partition_indices = [list(range(len(single_term_mps)))]
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    postprocessing_fn = _create_final_postprocessing_fn(
        coeffs, partition_indices, len(single_term_mps)
    )

    return measurement_groups, partition_indices, postprocessing_fn
