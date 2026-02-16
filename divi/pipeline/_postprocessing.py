# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Post-processing functions applied between execute and reduce.

These convert raw backend results (shot counts) into the value format expected
by the reduce chain (expectation values or probability distributions).
"""

from functools import lru_cache, reduce
from typing import Any

import numpy as np
import numpy.typing as npt
import pennylane as qml

from divi.circuits import MetaCircuit
from divi.pipeline.abc import ChildResults

# ---------------------------------------------------------------------------
# Expectation-value computation helpers (moved from qprog._expectation)
# ---------------------------------------------------------------------------


def _get_structural_key(obs: qml.operation.Operation) -> tuple[str, ...]:
    """Generates a hashable, wire-independent key from an observable's structure.

    Maps PauliX/Y to PauliZ because they are all isospectral ([1, -1]).
    """
    name_map = {
        "PauliY": "PauliZ",
        "PauliX": "PauliZ",
        "PauliZ": "PauliZ",
        "Identity": "Identity",
    }

    if isinstance(obs, qml.ops.Prod):
        return tuple(name_map[o.name] for o in obs.operands)

    return (name_map[obs.name],)


@lru_cache(maxsize=512)
def _get_eigvals_from_key(key: tuple[str, ...]) -> npt.NDArray[np.int8]:
    """Computes and caches eigenvalues based on a structural key."""
    eigvals_map = {
        "PauliZ": np.array([1, -1], dtype=np.int8),
        "Identity": np.array([1, 1], dtype=np.int8),
    }

    return reduce(np.kron, (eigvals_map[op] for op in key))


def _batched_expectation(
    shots_dicts: list[dict[str, int]],
    observables: list[qml.operation.Operation],
    wire_order: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Vectorised counts → expectation values for multiple observables/histograms.

    Returns array of shape ``(n_observables, n_histograms)``.
    """
    n_histograms = len(shots_dicts)
    n_total_wires = len(wire_order)
    n_observables = len(observables)

    # 1. Aggregate unique measured states.
    all_measured_bitstrings: set[str] = set()
    for sd in shots_dicts:
        all_measured_bitstrings.update(sd.keys())

    unique_bitstrings = sorted(all_measured_bitstrings)
    n_unique_states = len(unique_bitstrings)
    bitstring_to_idx_map = {bs: i for i, bs in enumerate(unique_bitstrings)}

    # 2. Build reduced eigenvalue matrix (n_observables × n_unique_states).
    if n_total_wires <= 64:
        unique_states_int = np.array(
            [int(bs, 2) for bs in unique_bitstrings], dtype=np.uint64
        )
        use_integer_representation = True
    else:
        bitstring_chars = np.array([list(bs) for bs in unique_bitstrings], dtype="U1")
        use_integer_representation = False

    reduced_eigvals_matrix = np.zeros((n_observables, n_unique_states))
    wire_map = {w: i for i, w in enumerate(wire_order)}
    powers_cache: dict[int, npt.NDArray] = {}

    for obs_idx, observable in enumerate(observables):
        obs_wires = observable.wires
        n_obs_wires = len(obs_wires)

        if n_obs_wires in powers_cache:
            powers = powers_cache[n_obs_wires]
        else:
            powers = 2 ** np.arange(n_obs_wires - 1, -1, -1, dtype=np.intp)
            powers_cache[n_obs_wires] = powers

        obs_wire_indices = np.array([wire_map[w] for w in obs_wires], dtype=np.uint32)
        eigvals = _get_eigvals_from_key(_get_structural_key(observable))
        shifts = n_total_wires - 1 - obs_wire_indices

        if use_integer_representation:
            bits = ((unique_states_int[:, np.newaxis] >> shifts) & 1).astype(np.intp)
        else:
            bits = bitstring_chars[:, shifts].astype(np.intp)

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
    obs_by_bk: dict[tuple, dict[int, tuple]] = {}
    wire_order_by_bk: dict[tuple, tuple[int, ...]] = {}
    for bk, node in batch.items():
        wire_order_by_bk[bk] = tuple(reversed(node.source_circuit.wires))
        obs_by_bk[bk] = {
            idx: observables for idx, observables in enumerate(node.measurement_groups)
        }

    out: ChildResults = {}
    for branch_key, counts in raw.items():
        # Find the batch key whose axes are a subset of the branch key.
        bk = _find_batch_key(branch_key, batch_keys)
        # obs_group index from measurement tag.
        axis_values = dict(branch_key)
        obs_group_idx = int(axis_values.get("obs_group", 0))

        observables = obs_by_bk[bk][obs_group_idx]
        wire_order = wire_order_by_bk[bk]

        expvals = _batched_expectation([counts], list(observables), wire_order)
        col = expvals[:, 0]
        # Single-obs groups → scalar (compatible with QEM/ZNE reduce).
        # Multi-obs groups → {obs_idx: float} dict for final_postprocessing_fn.
        out[branch_key] = (
            float(col[0]) if len(col) == 1 else {i: float(v) for i, v in enumerate(col)}
        )

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
