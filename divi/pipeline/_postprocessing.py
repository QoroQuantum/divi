# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Value post-processing: convert raw backend results into reduce-chain values.

These turn shot counts (or expval-native dicts) into the value format the reduce
chain expects — expectation values, probability distributions, or shot-noise
variance. Operating on result *keys* (parsing, grouping, routing) lives in
:mod:`divi.pipeline._result_keys_operations`.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from divi.circuits import MetaCircuit
from divi.pipeline._result_keys_operations import OBS_GROUP_AXIS, _find_batch_key
from divi.pipeline.abc import ChildResults

# ---------------------------------------------------------------------------
# Expectation-value computation helpers
# ---------------------------------------------------------------------------

_BIT_CHARS = frozenset("01")
_PAULI_CHARS = frozenset("IXYZ")


def _bitstring_eigenvalues(
    bitstrings: Sequence[str], pauli_labels: Sequence[str], n_qubits: int
) -> npt.NDArray[np.float64]:
    """Return Pauli eigenvalues for each measured bitstring."""
    offending = next((bs for bs in bitstrings if len(bs) != n_qubits), None)
    if offending is not None:
        raise ValueError(
            f"Backend returned {len(offending)}-bit histogram keys "
            f"for the {n_qubits}-qubit circuit; expected full-width keys "
            f"(creg c[{n_qubits}]). Partial-measurement circuits must still "
            f"report all classical bits. If your backend cannot, set "
            f"measure_all_qubits=True to measure the full register."
        )
    malformed = next((bs for bs in bitstrings if not set(bs) <= _BIT_CHARS), None)
    if malformed is not None:
        raise ValueError(f"Backend returned a non-binary histogram key: {malformed!r}.")

    n_states = len(bitstrings)
    state_bits = np.frombuffer(
        "".join(bitstrings).encode("ascii"), dtype=np.uint8
    ).reshape(n_states, n_qubits) - np.uint8(ord("0"))

    for label in pauli_labels:
        if len(label) != n_qubits:
            raise ValueError(
                f"Expected a {n_qubits}-character Pauli label, got {len(label)}."
            )
        if not set(label) <= _PAULI_CHARS:
            raise ValueError(f"Pauli label {label!r} contains characters outside IXYZ.")

    label_chars = np.frombuffer(
        "".join(pauli_labels).encode("ascii"), dtype=np.uint8
    ).reshape(len(pauli_labels), n_qubits)
    # Active-qubit counts are integers far below float32's exact limit of 2**24.
    active = (label_chars != ord("I")).astype(np.float32)
    parity = (active @ state_bits.T.astype(np.float32)) % 2
    return 1.0 - 2.0 * parity.astype(np.float64)


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

    # 1. Aggregate unique measured states.
    unique_bitstrings = sorted(set(chain.from_iterable(shots_dicts)))
    n_unique_states = len(unique_bitstrings)
    bitstring_to_idx_map = {bs: i for i, bs in enumerate(unique_bitstrings)}
    eigenvalues = _bitstring_eigenvalues(unique_bitstrings, pauli_labels, n_qubits)

    totals = np.ones(n_histograms)
    dense_size = n_histograms * n_unique_states
    n_counts = sum(map(len, shots_dicts))
    index_size = np.dtype(np.intp).itemsize
    dense_bytes = dense_size * np.dtype(np.float64).itemsize
    sparse_bytes = (
        n_counts * (np.dtype(np.float64).itemsize + index_size)
        + (n_histograms + 1) * index_size
    )
    use_sparse = dense_size > 1_000_000 and sparse_bytes < dense_bytes

    if not use_sparse:
        count_matrix = np.zeros((n_histograms, n_unique_states))
        for i, shots_dict in enumerate(shots_dicts):
            totals[i] = sum(shots_dict.values()) or 1.0
            for bitstring, count in shots_dict.items():
                count_matrix[i, bitstring_to_idx_map[bitstring]] = count
    else:
        # Large batches usually contain sparse histograms. Build CSR storage in
        # one allocation instead of materialising every histogram/state pair.
        indices = np.empty(n_counts, dtype=np.intp)
        data = np.empty(n_counts, dtype=np.float64)
        indptr = np.empty(n_histograms + 1, dtype=np.intp)
        indptr[0] = 0
        offset = 0
        for i, shots_dict in enumerate(shots_dicts):
            size = len(shots_dict)
            end = offset + size
            indices[offset:end] = [
                bitstring_to_idx_map[bitstring] for bitstring in shots_dict
            ]
            data[offset:end] = list(shots_dict.values())
            totals[i] = sum(shots_dict.values()) or 1.0
            offset = end
            indptr[i + 1] = end
        count_matrix = csr_matrix(
            (data, indices, indptr), shape=(n_histograms, n_unique_states)
        )

    return (count_matrix @ eigenvalues.T / totals[:, None]).T


def _big_endian_histogram(branch_key: tuple, counts: Any) -> dict[str, int]:
    """Validate a backend histogram and flip its little-endian bitstrings.

    Raises:
        TypeError: If ``counts`` is not a mapping keyed by bitstrings.
    """
    if not isinstance(counts, Mapping):
        found = type(counts).__name__
    elif not all(isinstance(bitstring, str) for bitstring in counts):
        found = f"a {type(counts).__name__} not keyed by bitstrings"
    else:
        return {bitstring[::-1]: count for bitstring, count in counts.items()}
    raise TypeError(
        "Expected a bitstring→count histogram from the backend for branch "
        f"{branch_key!r}, got {found}."
    )


def _group_lookups(
    batch: Mapping[Any, MetaCircuit],
) -> tuple[set, dict[tuple, dict[int, tuple[object, ...]]], dict[tuple, int]]:
    """Per-batch-key tables for resolving a branch key's measurement group."""
    labels_by_bk: dict[tuple, dict[int, tuple[object, ...]]] = {}
    nq_by_bk: dict[tuple, int] = {}
    for bk, node in batch.items():
        nq_by_bk[bk] = node.n_qubits
        labels_by_bk[bk] = dict(enumerate(node.measurement_groups))
    return set(batch.keys()), labels_by_bk, nq_by_bk


def _resolve_group(
    branch_key: tuple,
    batch_keys: set,
    labels_by_bk: Mapping[tuple, dict[int, tuple[object, ...]]],
    nq_by_bk: Mapping[tuple, int],
) -> tuple[tuple, tuple[object, ...], int]:
    """Return ``(batch_key, group_labels, n_qubits)`` for one branch key."""
    bk = _find_batch_key(branch_key, batch_keys)
    obs_group_idx = int(dict(branch_key)[OBS_GROUP_AXIS])
    return bk, labels_by_bk[bk][obs_group_idx], nq_by_bk[bk]


def _counts_to_expvals(
    raw: ChildResults,
    batch: Mapping[Any, MetaCircuit],
) -> ChildResults:
    """Convert shot counts → numeric expectation values.

    Called between *execute* and *_reduce* when ``env.result_format`` is
    ``EXPVALS`` and the backend does not support expval natively.  Each
    branch key is matched to its originating batch key (by subset match),
    then the correct observable group and wire order are looked up from
    the MetaCircuit to compute the expectation value.

    Returns an ``{obs_idx: float}`` dict per branch key.
    """
    batch_keys, labels_by_bk, nq_by_bk = _group_lookups(batch)

    members_by_group: defaultdict[
        tuple[tuple[str, ...], int], list[tuple[tuple, dict[str, int]]]
    ] = defaultdict(list)
    for branch_key, counts in raw.items():
        _bk, group_labels, n_qubits = _resolve_group(
            branch_key, batch_keys, labels_by_bk, nq_by_bk
        )
        group = (tuple(map(str, group_labels)), n_qubits)
        members_by_group[group].append(
            (branch_key, _big_endian_histogram(branch_key, counts))
        )

    values: ChildResults = {}
    for (labels, n_qubits), members in members_by_group.items():
        expvals = _batched_expectation(
            [counts for _, counts in members], labels, n_qubits
        )
        for (branch_key, _), col in zip(members, expvals.T):
            values[branch_key] = (
                float(col[0])
                if len(col) == 1
                else {i: float(v) for i, v in enumerate(col)}
            )

    return {branch_key: values[branch_key] for branch_key in raw}


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
    batch: Mapping[Any, MetaCircuit],
    term_factors_by_spec: (
        Mapping[tuple, Mapping[int, float | npt.NDArray[np.float64]]] | None
    ) = None,
) -> dict[tuple, float]:
    """Estimate the shot-noise variance of each cost value from raw counts,
    for groups whose shot counts are fixed in advance.

    Group ``g`` scales term ``P`` by its factor ``f_gP`` in
    ``term_factors_by_spec`` (``1`` when absent), so the
    cost is ``Σ_g mean_g(y)`` over the per-shot values
    ``y = Σ_{P∈g} c_P f_gP λ_P`` of ±1 eigenvalues ``λ_P``. Groups are
    independent, so the variance is ``Σ_g Var_g(y) / M_g`` for ``M_g`` shots,
    covariance between a group's terms included. ``Var_g(y)`` is the mean of
    ``y²`` less the square of ``Σ_P c_P f_gP P̂``, where ``P̂ = Σ_g f_gP <P>_g``
    pools every group measuring ``P``, so a group of a single shot still
    contributes.

    Returns ``{base_key: variance}``, where ``base_key`` is the branch key with
    the ``obs_group`` axis stripped — matching the post-reduce cost keys. Specs
    with multiple observables, or a group with no shots, yield ``nan``.
    """
    batch_keys, labels_by_bk, nq_by_bk = _group_lookups(batch)
    coeffs_by_bk = {bk: _observable_coeff_map(node) for bk, node in batch.items()}
    factors_by_spec = term_factors_by_spec or {}
    factor_keys = set(factors_by_spec)

    # Per cost key: each group's (labels, c·f, eigenvalues, counts, shots).
    groups_by_base: defaultdict[tuple, list[tuple]] = defaultdict(list)
    estimates_by_base: defaultdict[tuple, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    invalid_keys: set[tuple] = set()
    for branch_key, counts in raw.items():
        bk, group_labels, n_qubits = _resolve_group(
            branch_key, batch_keys, labels_by_bk, nq_by_bk
        )
        base_key = tuple(ax for ax in branch_key if ax[0] != OBS_GROUP_AXIS)
        coeff_map = coeffs_by_bk[bk]
        if coeff_map is None:
            invalid_keys.add(base_key)
            continue

        counts = _big_endian_histogram(branch_key, counts)
        shots = sum(counts.values())
        if shots <= 0:
            # A group with no shots invalidates the whole cost variance rather
            # than silently deflating the sum.
            invalid_keys.add(base_key)
            continue

        labels = [str(p) for p in group_labels]
        bitstrings = sorted(counts)
        eigenvalues = _bitstring_eigenvalues(bitstrings, labels, n_qubits)
        frequencies = np.asarray([counts[b] for b in bitstrings], dtype=np.float64)
        factors = np.ones(len(labels))
        if factor_keys:
            group_factors = factors_by_spec[_find_batch_key(base_key, factor_keys)]
            factors = factors * group_factors.get(
                int(dict(branch_key)[OBS_GROUP_AXIS]), 1.0
            )

        estimates = estimates_by_base[base_key]
        for label, factor, expval in zip(
            labels, factors, eigenvalues @ frequencies / shots, strict=True
        ):
            estimates[label] += factor * expval
        coefficients = np.asarray([coeff_map.get(label, 0.0) for label in labels])
        groups_by_base[base_key].append(
            (labels, coefficients * factors, eigenvalues, frequencies, shots)
        )

    out: dict[tuple, float] = dict.fromkeys(invalid_keys, float("nan"))
    for base_key, groups in groups_by_base.items():
        if base_key in invalid_keys:
            continue
        estimates = estimates_by_base[base_key]
        variance = 0.0
        for labels, scaled, eigenvalues, frequencies, shots in groups:
            mean = scaled @ np.asarray([estimates[label] for label in labels])
            mean_square = frequencies @ (scaled @ eigenvalues) ** 2 / shots
            variance += (mean_square - mean**2) / shots
        # A pooled mean can exceed a group's own, so clamp the sum at zero.
        out[base_key] = max(0.0, float(variance))
    return out


def _welford_add(
    state: tuple[int, float, float], value: float, count: int
) -> tuple[int, float, float]:
    """Add ``count`` copies of ``value`` to a ``(samples, mean, M2)`` Welford state."""
    n_samples, mean, m2 = state
    updated = n_samples + count
    delta = value - mean
    return (
        updated,
        mean + delta * count / updated,
        m2 + delta * delta * n_samples * count / updated,
    )


def _counts_to_wrs_cost_variance(
    raw: ChildResults,
    batch: Mapping[Any, MetaCircuit],
    weights_by_spec: Mapping[tuple, Mapping[int, np.ndarray]],
    empty_draws_by_spec: Mapping[tuple, int],
) -> dict[tuple, float]:
    """Estimate the variance of a weighted-random-sampling cost estimator.

    Each measured bitstring contributes ``Σ_P c_P w_P λ_P`` over the selected
    group's terms, where ``w_P`` is the group's weight for term ``P`` (its
    inverse inclusion probability) and ``λ_P`` the term's ±1 eigenvalue. The
    sample variance of these weighted values includes both quantum measurement
    noise and multinomial group-selection noise, including covariance between
    terms measured from the same bitstring. Draws that measured no term
    (``empty_draws_by_spec``) were never submitted and count as zero samples.
    """
    batch_keys, labels_by_bk, nq_by_bk = _group_lookups(batch)
    coeffs_by_bk = {bk: _observable_coeff_map(node) for bk, node in batch.items()}
    weight_keys = set(weights_by_spec)
    # Per-result weighted Welford state: (sample count, mean, centered sum of
    # squares). Updating once per histogram bin avoids expanding individual
    # shots and avoids cancellation in ``sum(y²) - sum(y)² / n``.
    moments: dict[tuple, tuple[int, float, float]] = {}
    invalid_keys: set[tuple] = set()

    for branch_key, counts in raw.items():
        bk, group_labels, n_qubits = _resolve_group(
            branch_key, batch_keys, labels_by_bk, nq_by_bk
        )
        base_key = tuple(ax for ax in branch_key if ax[0] != OBS_GROUP_AXIS)
        coeff_map = coeffs_by_bk[bk]
        if coeff_map is None:
            invalid_keys.add(base_key)
            continue

        counts = _big_endian_histogram(branch_key, counts)
        if sum(counts.values()) <= 0:
            invalid_keys.add(base_key)
            continue
        group_idx = next(value for axis, value in branch_key if axis == OBS_GROUP_AXIS)
        weights = weights_by_spec[_find_batch_key(base_key, weight_keys)][group_idx]

        labels = [str(label) for label in group_labels]
        coefficients = weights * np.asarray(
            [coeff_map.get(label, 0.0) for label in labels], dtype=np.float64
        )
        bitstrings = sorted(counts)
        weighted_values = coefficients @ _bitstring_eigenvalues(
            bitstrings, labels, n_qubits
        )
        state = moments.get(base_key, (0, 0.0, 0.0))
        for bitstring, weighted_value in zip(
            bitstrings, weighted_values.tolist(), strict=True
        ):
            state = _welford_add(state, weighted_value, counts[bitstring])
        moments[base_key] = state

    for base_key, state in moments.items():
        empty_draws = empty_draws_by_spec[_find_batch_key(base_key, weight_keys)]
        if empty_draws:
            moments[base_key] = _welford_add(state, 0.0, empty_draws)

    out: dict[tuple, float] = {}
    for base_key in moments.keys() | invalid_keys:
        if base_key in invalid_keys:
            out[base_key] = float("nan")
            continue
        n_shots, _mean_y, m2_y = moments[base_key]
        if n_shots < 2:
            out[base_key] = float("nan")
            continue
        out[base_key] = max(0.0, m2_y) / (n_shots * (n_shots - 1))

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
        if not isinstance(counts, Mapping):
            out[branch_key] = counts
            continue
        out[branch_key] = {
            bitstring[::-1]: count / shots for bitstring, count in counts.items()
        }

    return out
