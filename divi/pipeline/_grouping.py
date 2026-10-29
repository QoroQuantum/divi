# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Observable grouping for measurement-stage fan-out.

All grouping operates on Qiskit :class:`SparsePauliOp` — no PennyLane
dependency.  Groups are returned as tuples of big-endian Pauli label
strings (qubit 0 on the left), matching the bitstring convention used by
backends and :func:`~divi.pipeline._postprocessing._batched_expectation`.
"""

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from operator import itemgetter
from typing import Literal

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.circuits._core import flatten_observable_tuple

#: Internal grouping strategy: the backend evaluates the whole observable itself,
#: so no basis-change circuits are built. Not user-passable.
BACKEND_EXPVAL = "_backend_expval"

#: The ``measurement_groups`` value :func:`build_measurement_groups` emits under
#: :data:`BACKEND_EXPVAL` — one empty group, since there is no basis to partition.
#: Structural marker for consumers that cannot see the strategy that produced it.
BACKEND_EXPVAL_GROUPS = ((),)

#: How a sampling backend measures an observable's Pauli terms; each value is
#: described in the "Measuring Observables on a Sampling Backend" guide.
GroupingStrategy = (
    Literal[
        "qwc", "default", "wires", "shadows", "biased_shadows", "derandomised_shadows"
    ]
    | None
)

#: A grouping strategy after the measurement stage resolves it for the backend.
_ResolvedGroupingStrategy = GroupingStrategy | Literal["_backend_expval"]


def _wire_grouping_from_labels(
    labels: Sequence[str],
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
    observables: Sequence[SparsePauliOp],
    per_obs_term_indices: Sequence[list[int]],
    partition_indices: Sequence[list[int]],
    n_union_terms: int,
) -> Callable:
    """Build a callable that reassembles per-group results into per-observable expvals.

    Returns a function that takes per-group results (one entry per
    measurement group) and returns a ``list[float]`` of per-observable
    expectation values. A union term measured in several groups takes the
    sum of its per-group results.
    """
    slots_by_group = [
        np.asarray(indices, dtype=np.intp) for indices in partition_indices
    ]
    getters = [
        itemgetter(*range(len(slots))) if len(slots) else None
        for slots in slots_by_group
    ]
    obs_coeffs = [np.real(o.coeffs).astype(np.float64) for o in observables]
    obs_indices = [
        np.asarray(indices, dtype=np.intp) for indices in per_obs_term_indices
    ]

    def postprocessing_fn(grouped_results):
        if len(grouped_results) != len(partition_indices):
            raise RuntimeError(
                f"Expected {len(partition_indices)} grouped results, "
                f"got {len(grouped_results)}."
            )
        flat = np.zeros(n_union_terms, dtype=np.float64)
        for slots, getter, val in zip(slots_by_group, getters, grouped_results):
            if getter is None:
                continue
            if isinstance(val, dict):
                flat[slots] += np.asarray(getter(val), dtype=np.float64).reshape(
                    len(slots), -1
                )[:, 0]
            else:
                flat[slots] += float(np.asarray(val).flat[0])
        return [
            float(np.dot(coeffs, flat[indices]))
            for coeffs, indices in zip(obs_coeffs, obs_indices)
        ]

    return postprocessing_fn


def _union_labels(
    observable: Sequence[SparsePauliOp], n_qubits: int
) -> tuple[SparsePauliOp, list[list[int]], list[str]]:
    """Flatten an observable tuple into its deduplicated union.

    Returns ``(union, per_obs_term_indices, labels)``, where ``labels`` are the
    union's big-endian Pauli labels (qubit 0 on the left) padded to ``n_qubits``.
    """
    if not observable:
        raise ValueError("The observable tuple is empty.")

    union, per_obs_term_indices = flatten_observable_tuple(observable)
    # Qiskit labels are little-endian; we store big-endian internally.
    labels = [label[::-1].ljust(n_qubits, "I") for label in union.paulis.to_labels()]
    return union, per_obs_term_indices, labels


def _compute_measurement_groups(
    observable: Sequence[SparsePauliOp],
    strategy: _ResolvedGroupingStrategy,
    n_qubits: int,
) -> tuple[
    tuple[tuple[str, ...], ...], list[list[int]], SparsePauliOp, list[list[int]]
]:
    """Group an observable tuple's Pauli terms for measurement fan-out.

    Operates on the union of every observable's Pauli terms — overlapping
    Paulis (across or within observables) are deduplicated, and QWC
    grouping shares measurement circuits across observables when their
    terms are mutually qubit-wise commuting.

    Args:
        observable: Non-empty ``tuple[SparsePauliOp, ...]``.
        strategy: ``"qwc"``, ``"default"``, ``"wires"``,
            ``"_backend_expval"``, or ``None``.  ``"_backend_expval"`` evaluates
            each observable in the tuple analytically (single- or
            multi-observable).
        n_qubits: Total qubit count in the circuit.

    Returns:
        ``(measurement_groups, partition_indices, union, per_obs_term_indices)``

        *measurement_groups*: tuple of tuples of big-endian Pauli label
        strings (qubit 0 on the left), one tuple per commuting group over
        the union of Pauli terms.

        *partition_indices*: ``list[list[int]]`` mapping union term
        indices to groups.

        *union*: the deduplicated union ``SparsePauliOp`` (the same value
        :func:`flatten_observable_tuple` would return) — exposed so
        callers needing shot-allocation weights can avoid recomputing it.

        *per_obs_term_indices*: for each input observable, the union term
        indices of its Pauli terms, as :func:`_create_postprocessing_fn`
        takes them.
    """
    union, per_obs_term_indices, be_labels = _union_labels(observable, n_qubits)
    n_union_terms = len(be_labels)

    if strategy in ("qwc", "default"):
        grouped_ops = union.group_commuting(qubit_wise=True)
        # Reconstruct partition_indices by matching labels back to originals.
        # One queue of union indices per label, consumed in order, so repeated
        # labels are handed out once each without rescanning the union.
        indices_by_label: dict[str, deque[int]] = defaultdict(deque)
        for index, label in enumerate(be_labels):
            indices_by_label[label].append(index)

        partition_indices: list[list[int]] = []
        grouped_be_labels: list[tuple[str, ...]] = []
        for group_op in grouped_ops:
            indices = []
            group_labels = []
            for gl in group_op.paulis.to_labels():
                gl_be = gl[::-1].ljust(n_qubits, "I")
                queue = indices_by_label.get(gl_be)
                if queue:
                    indices.append(queue.popleft())
                    group_labels.append(gl_be)
            partition_indices.append(indices)
            grouped_be_labels.append(tuple(group_labels))
        measurement_groups = tuple(grouped_be_labels)

    elif strategy == "wires":
        partition_indices = _wire_grouping_from_labels(be_labels)
        measurement_groups = tuple(
            tuple(be_labels[i] for i in group) for group in partition_indices
        )

    elif strategy is None:
        partition_indices = [[i] for i in range(n_union_terms)]
        measurement_groups = tuple((label,) for label in be_labels)

    elif strategy == BACKEND_EXPVAL:
        partition_indices = [list(range(n_union_terms))]
        measurement_groups = BACKEND_EXPVAL_GROUPS

    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    covered = np.zeros(n_union_terms, dtype=bool)
    for indices in partition_indices:
        covered[indices] = True
    if not covered.all():
        raise ValueError("The partition does not cover every union term.")

    return measurement_groups, partition_indices, union, per_obs_term_indices
