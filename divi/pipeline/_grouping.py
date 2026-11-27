# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Observable grouping for measurement-stage fan-out.

All grouping operates on Qiskit :class:`SparsePauliOp` — no PennyLane
dependency.  Groups are returned as tuples of big-endian Pauli label
strings (qubit 0 on the left), the order backends key histograms in.
"""

from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Literal

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import PauliList, SparsePauliOp
from scipy.sparse import csr_matrix

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


#: Groups scanned per vectorised conflict check in QWC first-fit grouping.
_QWC_GROUP_CHUNK = 2048

#: Pauli letters as bytes, indexed by :func:`_encode_paulis` codes.
_PAULI_LETTERS = np.frombuffer(b"IXZY", dtype=np.uint8)


def _encode_paulis(paulis: PauliList) -> npt.NDArray[np.uint8]:
    """Per-term, per-qubit Pauli codes ``x + 2z``: 0 I, 1 X, 2 Z, 3 Y."""
    return paulis.x.astype(np.uint8) + 2 * paulis.z.astype(np.uint8)


def _pack_bit_planes(bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint64]:
    """Rows of ``bits`` packed into 64-bit words, ``(n_rows, ceil(n_cols / 64))``."""
    packed = np.packbits(bits, axis=1)
    padding = (-packed.shape[1]) % 8
    packed = np.pad(packed, ((0, 0), (0, padding)))
    return np.ascontiguousarray(packed).view(np.uint64)


def _group_qubit_wise_commuting(codes: npt.NDArray[np.uint8]) -> list[list[int]]:
    """Partition Pauli terms into qubit-wise commuting groups by first fit.

    Terms are visited heaviest first (stable in input order) and join the first
    group whose letters they do not contradict on any shared qubit. Returns
    partition indices into ``codes``, each group in visit order.
    """
    acts = codes != 0
    support = _pack_bit_planes(acts)
    low = _pack_bit_planes((codes & 1).astype(bool))
    high = _pack_bit_planes((codes >> 1).astype(bool))
    group_support = np.zeros_like(support)
    group_low = np.zeros_like(low)
    group_high = np.zeros_like(high)
    groups: list[list[int]] = []

    for term in np.argsort(-acts.sum(axis=1), kind="stable"):
        target = len(groups)
        for start in range(0, len(groups), _QWC_GROUP_CHUNK):
            stop = min(start + _QWC_GROUP_CHUNK, len(groups))
            mismatch = (group_low[start:stop] ^ low[term]) | (
                group_high[start:stop] ^ high[term]
            )
            conflicts = (group_support[start:stop] & support[term] & mismatch).any(
                axis=1
            )
            fits = np.flatnonzero(~conflicts)
            if fits.size:
                target = start + int(fits[0])
                break
        if target == len(groups):
            groups.append([])
        group_support[target] |= support[term]
        group_low[target] |= low[term]
        group_high[target] |= high[term]
        groups[target].append(int(term))
    return groups


def _create_postprocessing_fn(
    observables: Sequence[SparsePauliOp],
    per_obs_term_indices: Sequence[list[int]],
    partition_indices: Sequence[list[int]],
    n_union_terms: int,
) -> Callable[[Sequence[Mapping[int, float]]], list[float]]:
    """Build a callable that reassembles per-group results into per-observable expvals.

    Returns a function that takes per-group results (one entry per
    measurement group) and returns a ``list[float]`` of per-observable
    expectation values. A union term measured in several groups takes the
    sum of its per-group results.
    """
    group_sizes = [len(indices) for indices in partition_indices]
    n_slots = sum(group_sizes)
    union_index_of_slot = np.fromiter(
        chain.from_iterable(partition_indices), dtype=np.intp, count=n_slots
    )
    # Row i holds observable i's coefficients at its union-term columns.
    weights = csr_matrix(
        (
            np.concatenate([np.real(o.coeffs) for o in observables]),
            np.concatenate(
                [np.asarray(indices, dtype=np.intp) for indices in per_obs_term_indices]
            ),
            np.cumsum([0, *map(len, per_obs_term_indices)]),
        ),
        shape=(len(observables), n_union_terms),
    )
    scatter = csr_matrix(
        (np.ones(n_slots), (union_index_of_slot, np.arange(n_slots))),
        shape=(n_union_terms, n_slots),
    )
    slot_weights = (weights @ scatter).tocsr()

    def postprocessing_fn(
        grouped_results: Sequence[Mapping[int, float]],
    ) -> list[float]:
        if len(grouped_results) != len(partition_indices):
            raise RuntimeError(
                f"Expected {len(partition_indices)} grouped results, "
                f"got {len(grouped_results)}."
            )
        values = np.fromiter(
            chain.from_iterable(
                map(result.__getitem__, range(size))
                for result, size in zip(grouped_results, group_sizes)
            ),
            dtype=np.float64,
            count=n_slots,
        )
        return (slot_weights @ values).tolist()

    return postprocessing_fn


def _build_observable_union(
    observable: Sequence[SparsePauliOp], n_qubits: int
) -> tuple[SparsePauliOp, list[list[int]], list[str]]:
    """Flatten an observable tuple into its deduplicated union.

    Returns ``(union, per_obs_term_indices, labels)``, where ``labels`` are the
    union's big-endian Pauli labels (qubit 0 on the left) padded to ``n_qubits``.
    """
    if not observable:
        raise ValueError("The observable tuple is empty.")

    union, per_obs_term_indices = flatten_observable_tuple(observable)
    return (
        union,
        per_obs_term_indices,
        _label_paulis(_encode_paulis(union.paulis), n_qubits),
    )


def _label_paulis(codes: npt.NDArray[np.uint8], n_qubits: int) -> list[str]:
    """Big-endian Pauli labels (qubit 0 on the left) of :func:`_encode_paulis`
    ``codes``, padded to ``n_qubits``."""
    return [
        row.tobytes().decode().ljust(n_qubits, "I") for row in _PAULI_LETTERS[codes]
    ]


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
    union, per_obs_term_indices, be_labels = _build_observable_union(
        observable, n_qubits
    )
    n_union_terms = len(union)
    if strategy == BACKEND_EXPVAL:
        return (
            BACKEND_EXPVAL_GROUPS,
            [list(range(n_union_terms))],
            union,
            per_obs_term_indices,
        )

    if strategy in ("qwc", "default"):
        partition_indices = _group_qubit_wise_commuting(_encode_paulis(union.paulis))
    elif strategy == "wires":
        partition_indices = _wire_grouping_from_labels(be_labels)
    elif strategy is None:
        partition_indices = [[i] for i in range(n_union_terms)]
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    measurement_groups = tuple(
        tuple(be_labels[i] for i in group) for group in partition_indices
    )

    covered = np.zeros(n_union_terms, dtype=bool)
    for indices in partition_indices:
        covered[indices] = True
    if not covered.all():
        raise ValueError("The partition does not cover every union term.")

    return measurement_groups, partition_indices, union, per_obs_term_indices
