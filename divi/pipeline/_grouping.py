# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Observable grouping for measurement-stage fan-out.

All grouping operates on Qiskit :class:`SparsePauliOp` — no PennyLane
dependency.  Groups are returned as tuples of big-endian Pauli label
strings (qubit 0 on the left), matching the bitstring convention used by
backends and :func:`~divi.pipeline._postprocessing._batched_expectation`.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from qiskit.quantum_info import SparsePauliOp

GroupingStrategy = Literal["wires", "default", "qwc", "_backend_expval"] | None


def _wire_grouping_from_labels(
    labels: list[str],
) -> list[list[int]]:
    """Group Pauli labels by non-overlapping active qubits (wire-disjoint).

    Two labels are wire-disjoint when they have no qubit position where
    both are non-I.  Returns partition indices (list of lists of original
    term indices).
    """

    def _active_qubits(label: str) -> set[int]:
        return {i for i, c in enumerate(label) if c != "I"}

    groups: list[list[int]] = []
    wires_per_group: list[set[int]] = []

    for idx, label in enumerate(labels):
        active = _active_qubits(label)
        placed = False
        for g_idx, g_wires in enumerate(wires_per_group):
            if not (active & g_wires):
                groups[g_idx].append(idx)
                g_wires |= active
                placed = True
                break
        if not placed:
            groups.append([idx])
            wires_per_group.append(active)

    return groups


def _create_postprocessing_fn(
    coefficients: np.ndarray,
    partition_indices: list[list[int]],
    n_terms: int,
) -> Callable:
    """Build a callable that reassembles per-group results into a scalar expval."""
    reverse_lookup: dict[int, tuple[int, int]] = {}
    for group_idx, indices in enumerate(partition_indices):
        for pos, orig_idx in enumerate(indices):
            reverse_lookup[orig_idx] = (group_idx, pos)

    missing = [i for i in range(n_terms) if i not in reverse_lookup]
    if missing:
        raise RuntimeError(
            f"partition_indices does not cover all term indices. Missing: {missing}"
        )

    reverse_map: list[tuple[int, int]] = [reverse_lookup[i] for i in range(n_terms)]

    def postprocessing_fn(grouped_results):
        if len(grouped_results) != len(partition_indices):
            raise RuntimeError(
                f"Expected {len(partition_indices)} grouped results, "
                f"got {len(grouped_results)}."
            )
        flat = np.zeros(n_terms, dtype=np.float64)
        for orig_idx in range(n_terms):
            g_idx, pos = reverse_map[orig_idx]
            val = grouped_results[g_idx]
            if isinstance(val, dict):
                val = val[pos]
            flat[orig_idx] = float(np.asarray(val).flat[0])
        return np.dot(coefficients, flat)

    return postprocessing_fn


def compute_measurement_groups(
    observable: SparsePauliOp,
    strategy: GroupingStrategy,
    n_qubits: int,
) -> tuple[tuple[tuple[str, ...], ...], list[list[int]], Callable]:
    """Group an observable's Pauli terms for measurement fan-out.

    Args:
        observable: The Hermitian observable to group.
        strategy: ``"qwc"``, ``"default"``, ``"wires"``,
            ``"_backend_expval"``, or ``None``.
        n_qubits: Total qubit count in the circuit.

    Returns:
        ``(measurement_groups, partition_indices, postprocessing_fn)``

        *measurement_groups*: tuple of tuples of big-endian Pauli label
        strings (qubit 0 on the left).  Each inner tuple is one
        commuting group.

        *partition_indices*: ``list[list[int]]`` mapping original term
        indices to groups.

        *postprocessing_fn*: callable that recombines per-group
        expectation values into the final scalar using the observable's
        coefficients.
    """
    # Qiskit labels are little-endian; we store big-endian internally.
    le_labels = observable.paulis.to_labels()
    be_labels = [label[::-1] for label in le_labels]
    n_terms = len(be_labels)
    coeffs = np.real(observable.coeffs).astype(np.float64)

    # Pad labels to n_qubits if the observable acts on fewer qubits.
    if len(be_labels[0]) < n_qubits:
        pad = n_qubits - len(be_labels[0])
        be_labels = [label + "I" * pad for label in be_labels]

    if strategy in ("qwc", "default"):
        grouped_ops = observable.group_commuting(qubit_wise=True)
        # Reconstruct partition_indices by matching labels back to originals.
        partition_indices: list[list[int]] = []
        grouped_be_labels: list[tuple[str, ...]] = []
        used: set[int] = set()
        for group_op in grouped_ops:
            group_le = group_op.paulis.to_labels()
            indices = []
            group_labels = []
            for gl in group_le:
                gl_be = gl[::-1]
                if len(gl_be) < n_qubits:
                    gl_be = gl_be + "I" * (n_qubits - len(gl_be))
                # Find the matching original index (handle duplicates).
                for orig_idx, orig_label in enumerate(be_labels):
                    if orig_idx not in used and orig_label == gl_be:
                        indices.append(orig_idx)
                        used.add(orig_idx)
                        group_labels.append(gl_be)
                        break
            partition_indices.append(indices)
            grouped_be_labels.append(tuple(group_labels))
        measurement_groups = tuple(grouped_be_labels)

    elif strategy == "wires":
        partition_indices = _wire_grouping_from_labels(be_labels)
        measurement_groups = tuple(
            tuple(be_labels[i] for i in group) for group in partition_indices
        )

    elif strategy is None:
        partition_indices = [[i] for i in range(n_terms)]
        measurement_groups = tuple((label,) for label in be_labels)

    elif strategy == "_backend_expval":
        partition_indices = [list(range(n_terms))]
        measurement_groups = ((),)

    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    postprocessing_fn = _create_postprocessing_fn(coeffs, partition_indices, n_terms)
    return measurement_groups, partition_indices, postprocessing_fn
