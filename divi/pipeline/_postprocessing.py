# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Value post-processing: convert raw backend results into reduce-chain values.

These turn shot counts (or expval-native dicts) into the value format the reduce
chain expects — expectation values, probability distributions, or shot-noise
variance. Operating on result *keys* (parsing, grouping, routing) lives in
:mod:`divi.pipeline._result_keys_operations`.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.circuits import MetaCircuit
from divi.pipeline._result_keys_operations import _find_batch_key
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
    result = _EIGVAL_MAP[active[0]]
    for c in active[1:]:
        result = np.kron(result, _EIGVAL_MAP[c])
    return result.astype(np.int8, copy=False)


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
    labels_by_bk: dict[tuple, dict[int, tuple[object, ...]]] = {}
    nq_by_bk: dict[tuple, int] = {}
    for bk, node in batch.items():
        nq_by_bk[bk] = node.n_qubits
        labels_by_bk[bk] = {i: g for i, g in enumerate(node.measurement_groups)}

    out: ChildResults = {}
    for branch_key, counts in raw.items():
        bk = _find_batch_key(branch_key, batch_keys)
        axis_values = dict(branch_key)
        obs_group_idx = int(axis_values.get("obs_group", 0))

        group_labels = labels_by_bk[bk][obs_group_idx]
        n_qubits = nq_by_bk[bk]

        # Little-endian backend counts → big-endian, matching the labels.
        # Any bitstring→count mapping (incl. qiskit's dict-subclass Counts) is
        # reversed; a non-mapping value crashes loudly below in
        # _batched_expectation rather than silently skipping the reversal.
        if isinstance(counts, Mapping):
            counts = {bitstring[::-1]: count for bitstring, count in counts.items()}

        expvals = _batched_expectation(
            [counts], [str(p) for p in group_labels], n_qubits
        )
        col = expvals[:, 0]
        out[branch_key] = (
            float(col[0]) if len(col) == 1 else {i: float(v) for i, v in enumerate(col)}
        )

    return out


def _observable_coeff_map(
    node: MetaCircuit,
) -> dict[str, float] | None:
    """Big-endian ``{pauli_label: real_coeff}`` for a single-observable cost.

    Returns ``None`` when the node carries multiple observables (a multi-output
    program), for which a single scalar cost variance is undefined. Labels are
    padded to ``node.n_qubits`` to match the stored measurement groups.
    """
    observable = node.observable
    if not isinstance(observable, tuple) or len(observable) != 1:
        return None
    op = observable[0]
    le_labels = op.paulis.to_labels()
    coeffs = np.real(np.asarray(op.coeffs)).astype(np.float64)
    n_qubits = node.n_qubits
    coeff_map: dict[str, float] = {}
    for le_label, coeff in zip(le_labels, coeffs, strict=True):
        be_label = le_label[::-1].ljust(n_qubits, "I")
        coeff_map[be_label] = coeff_map.get(be_label, 0.0) + float(coeff)
    return coeff_map


def _counts_to_cost_variance(
    raw: ChildResults,
    batch: dict[Any, MetaCircuit],
) -> dict[tuple, float]:
    """Estimate the shot-noise variance of each cost value from raw counts.

    For a Hamiltonian ``H = Σ_i c_i P_i`` measured group-wise, the variance of
    the (group-wise independent) mean estimator is approximated by
    ``Var(<H>) = Σ_i c_i² (1 − <P_i>²) / M_g``, where ``M_g`` is the shot count
    for the group containing ``P_i``. This drops intra-group Pauli covariance —
    a standard, cheap first approximation. Each Pauli's ``<P_i>`` and the group
    shot count come from the same counts :func:`_counts_to_expvals` consumes.

    Returns ``{base_key: variance}`` summed over a spec's measurement groups,
    where ``base_key`` is the branch key with the ``obs_group`` axis stripped —
    matching the post-reduce cost keys. Specs with multiple observables yield
    ``nan`` (single-scalar variance undefined).
    """
    batch_keys = set(batch.keys())
    labels_by_bk: dict[tuple, dict[int, tuple[object, ...]]] = {}
    nq_by_bk: dict[tuple, int] = {}
    coeffs_by_bk: dict[tuple, dict[str, float] | None] = {}
    for bk, node in batch.items():
        nq_by_bk[bk] = node.n_qubits
        labels_by_bk[bk] = {i: g for i, g in enumerate(node.measurement_groups)}
        coeffs_by_bk[bk] = _observable_coeff_map(node)

    out: dict[tuple, float] = {}
    for branch_key, counts in raw.items():
        bk = _find_batch_key(branch_key, batch_keys)
        base_key = tuple(ax for ax in branch_key if ax[0] != "obs_group")
        coeff_map = coeffs_by_bk[bk]
        if coeff_map is None:
            out[base_key] = float("nan")
            continue

        axis_values = dict(branch_key)
        obs_group_idx = int(axis_values.get("obs_group", 0))
        group_labels = labels_by_bk[bk][obs_group_idx]
        n_qubits = nq_by_bk[bk]

        if isinstance(counts, Mapping):
            counts = {bitstring[::-1]: count for bitstring, count in counts.items()}
            shots = sum(counts.values())
        else:
            # Non-count payload (shouldn't reach here on the shot path); skip.
            out[base_key] = out.get(base_key, 0.0) + float("nan")
            continue

        if shots <= 0:
            # Degenerate group with no measurements: invalidate the whole cost
            # variance (nan) rather than silently omitting it, which would
            # deflate the summed estimate. Matches the multi-observable and
            # non-count paths above.
            out[base_key] = float("nan")
            continue

        expvals = _batched_expectation(
            [counts], [str(p) for p in group_labels], n_qubits
        )[:, 0]

        group_var = 0.0
        for label, exp in zip(group_labels, expvals, strict=True):
            coeff = coeff_map.get(str(label), 0.0)
            if coeff == 0.0:
                continue
            # ``exp`` is a convex combination of ±1 eigenvalues, so 1−exp² ≥ 0
            # mathematically; clamp to guard float rounding when |exp| ≈ 1.
            group_var += coeff * coeff * max(0.0, 1.0 - float(exp) ** 2) / shots

        prev = out.get(base_key, 0.0)
        # Once a base key is nan (multi-observable / degenerate group) keep it nan.
        out[base_key] = prev if np.isnan(prev) else prev + group_var

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
