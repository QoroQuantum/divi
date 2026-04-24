# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Post-processing functions applied between execute and reduce.

These convert raw backend results (shot counts) into the value format expected
by the reduce chain (expectation values or probability distributions).
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache, reduce
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.circuits._core import MetaCircuit
from divi.pipeline.abc import ChildResults

# ---------------------------------------------------------------------------
# Expectation-value computation helpers
# ---------------------------------------------------------------------------

# X, Y, Z are isospectral (eigenvalues [+1, -1]); I has [+1, +1].
_EIGVAL_MAP = {
    "X": np.array([1, -1], dtype=np.int8),
    "Y": np.array([1, -1], dtype=np.int8),
    "Z": np.array([1, -1], dtype=np.int8),
    "I": np.array([1, 1], dtype=np.int8),
}


@lru_cache(maxsize=512)
def _eigvals_for_label(label: str) -> npt.NDArray[np.int8]:
    """Eigenvalues for a big-endian Pauli label (e.g. ``'ZIZ'``)."""
    active = tuple(c for c in label if c != "I")
    if not active:
        return np.array([1], dtype=np.int8)
    return reduce(np.kron, (_EIGVAL_MAP[c] for c in active))


def _batched_expectation(
    shots_dicts: Sequence[Mapping[str, int]],
    pauli_labels: Sequence[str],
    n_qubits: int,
) -> npt.NDArray[np.float64]:
    """Vectorised counts → expectation values for multiple Pauli observables.

    Args:
        shots_dicts: Per-circuit shot histograms (big-endian bitstrings).
        pauli_labels: Big-endian Pauli label strings (qubit 0 on the left),
            e.g. ``["ZII", "IZI", "ZZI"]``.
        n_qubits: Total qubit count.

    Returns:
        Array of shape ``(n_observables, n_histograms)``.
    """
    n_histograms = len(shots_dicts)
    n_observables = len(pauli_labels)

    # 1. Aggregate unique measured states.
    all_measured_bitstrings: set[str] = set()
    for sd in shots_dicts:
        all_measured_bitstrings.update(sd.keys())

    unique_bitstrings = sorted(all_measured_bitstrings)
    n_unique_states = len(unique_bitstrings)
    bitstring_to_idx_map = {bs: i for i, bs in enumerate(unique_bitstrings)}

    # 2. Build reduced eigenvalue matrix (n_observables × n_unique_states).
    if n_qubits <= 64:
        unique_states_int = np.array(
            [int(bs, 2) for bs in unique_bitstrings], dtype=np.uint64
        )
        bitstring_chars = None
    else:
        unique_states_int = None
        bitstring_chars = np.array([list(bs) for bs in unique_bitstrings], dtype="U1")

    reduced_eigvals_matrix = np.zeros((n_observables, n_unique_states))
    powers_cache: dict[int, npt.NDArray] = {}

    for obs_idx, label in enumerate(pauli_labels):
        # Active qubit positions (non-I) — big-endian, so position i = qubit i.
        active_positions = [i for i, c in enumerate(label) if c != "I"]
        n_active = len(active_positions)

        if n_active == 0:
            # Pure identity — expectation value is always 1.
            reduced_eigvals_matrix[obs_idx, :] = 1.0
            continue

        if n_active in powers_cache:
            powers = powers_cache[n_active]
        else:
            powers = 2 ** np.arange(n_active - 1, -1, -1, dtype=np.intp)
            powers_cache[n_active] = powers

        positions = np.array(active_positions, dtype=np.uint32)
        eigvals = _eigvals_for_label(label)

        if unique_states_int is not None:
            shifts = n_qubits - 1 - positions
            bits = ((unique_states_int[:, np.newaxis] >> shifts) & 1).astype(np.intp)
        elif bitstring_chars is not None:
            bits = bitstring_chars[:, positions].astype(np.intp)
        else:
            raise RuntimeError(
                "unreachable: unique_states_int or bitstring_chars must be set"
            )

        obs_state_indices = np.dot(bits, powers)
        reduced_eigvals_matrix[obs_idx, :] = eigvals[obs_state_indices]

    # 3. Build reduced probability matrix (n_histograms × n_unique_states).
    reduced_prob_matrix = np.zeros((n_histograms, n_unique_states), dtype=np.float32)
    for i, shots_dict in enumerate(shots_dicts):
        total = sum(shots_dict.values())
        for bitstring, count in shots_dict.items():
            col_idx = bitstring_to_idx_map[bitstring]
            reduced_prob_matrix[i, col_idx] = count / total

    # 4. Final (n_observables, n_histograms).
    return (reduced_prob_matrix @ reduced_eigvals_matrix.T).T


def _counts_to_expvals(
    raw: ChildResults,
    batch: dict[Any, MetaCircuit],
) -> ChildResults:
    """Convert shot counts → numeric expectation values.

    Called between *execute* and *_reduce* when ``env.result_format`` is
    ``EXPVALS`` and the backend does not support expval natively.  Each
    branch key is matched to its originating batch key (by subset match),
    then the correct observable group and wire order are looked up from
    the MetaCircuit to compute the expectation value.

    Returns an ``{obs_idx: float}`` dict per branch key.
    """
    # Pre-compute per-batch-key lookup tables.
    batch_keys = set(batch.keys())
    labels_by_bk: dict[tuple, dict[int, tuple[str, ...]]] = {}
    nq_by_bk: dict[tuple, int] = {}
    for bk, node in batch.items():
        nq_by_bk[bk] = node.n_qubits
        labels_by_bk[bk] = dict(enumerate(node.measurement_groups))

    out: ChildResults = {}
    for branch_key, counts in raw.items():
        bk = _find_batch_key(branch_key, batch_keys)
        axis_values = dict(branch_key)
        obs_group_idx = int(axis_values.get("obs_group", 0))

        group_labels = labels_by_bk[bk][obs_group_idx]
        n_qubits = nq_by_bk[bk]

        expvals = _batched_expectation([counts], list(group_labels), n_qubits)
        col = expvals[:, 0]
        out[branch_key] = (
            float(col[0]) if len(col) == 1 else {i: float(v) for i, v in enumerate(col)}
        )

    return out


def _expval_dicts_to_indexed(raw: ChildResults, ham_ops: str) -> ChildResults:
    """Normalise ``{pauli_str: float}`` dicts from expval-native backends.

    Converts each result value into the same format that
    ``_counts_to_expvals`` produces: ``float`` for single-observable groups
    and ``{obs_idx: float}`` for multi-observable groups.  The ordering is
    determined by the semicolon-separated *ham_ops* string.

    Duck-type check: only converts when values are actually Pauli dicts
    (custom execute_fns may return other dict types like probability dicts).
    """
    ops = ham_ops.split(";")
    sample = next(iter(raw.values()), None)
    if not isinstance(sample, dict) or ops[0] not in sample:
        return raw

    out: ChildResults = {}
    for key, pauli_dict in raw.items():
        if len(ops) == 1:
            out[key] = float(pauli_dict[ops[0]])
        else:
            out[key] = {i: float(pauli_dict[op]) for i, op in enumerate(ops)}
    return out


def _counts_to_probs(
    raw: ChildResults,
    shots: int,
) -> ChildResults:
    """Normalise shot counts → probability distributions.

    Reverses bitstring endianness (PennyLane convention → standard MSB-first)
    and divides by total shots.
    """
    out: ChildResults = {}

    for branch_key, counts in raw.items():
        if not isinstance(counts, dict):
            out[branch_key] = counts
            continue
        out[branch_key] = {
            bitstring[::-1]: count / shots for bitstring, count in counts.items()
        }

    return out


def _find_batch_key(branch_key: tuple, batch_keys: set[tuple]) -> tuple:
    """Find the batch key whose axis labels are a subset of *branch_key*."""
    branch_axes = set(branch_key)
    for bk in batch_keys:
        if set(bk).issubset(branch_axes):
            return bk

    raise KeyError(
        f"No batch key matches branch key {branch_key}; "
        f"known batch keys: {batch_keys}"
    )
