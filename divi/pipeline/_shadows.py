# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Random Pauli classical-shadow sampling over an observable's Pauli terms.

Each snapshot measures every qubit in a random Pauli basis, drawn per qubit
from that qubit's basis probabilities. A basis *covers* a term when it matches
the term's Pauli on every qubit the term acts on, which happens with
probability ``Π_q β_q(P_q)`` — ``3**-|P|`` for uniform bases. Snapshots that
cover the same terms run the same measurement circuit, so they are merged into
one group.

Basis probabilities are a ``(n_qubits, 4)`` array indexed by Pauli code
(``0`` identity, ``1`` X, ``2`` Z, ``3`` Y), with column ``0`` fixed at ``1``.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import PauliList, SparsePauliOp
from scipy.special import logsumexp

# Fitting stops once no probability moves by more than this in a sweep.
_FIT_TOLERANCE = 1e-10
_MAX_FIT_SWEEPS = 500


def _encode_paulis(paulis: PauliList) -> npt.NDArray[np.uint8]:
    """Per-term, per-qubit Pauli codes ``x + 2z``: 0 I, 1 X, 2 Z, 3 Y."""
    return paulis.x.astype(np.uint8) + 2 * paulis.z.astype(np.uint8)


def _uniform_bases(n_qubits: int) -> npt.NDArray[np.float64]:
    """Basis probabilities drawing X, Y and Z equally on every qubit."""
    return np.tile([1.0, 1 / 3, 1 / 3, 1 / 3], (n_qubits, 1))


def _compute_cover_probabilities(
    codes: npt.NDArray[np.uint8], bases: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Probability ``Π_q β_q(P_q)`` that one random basis covers each term."""
    return bases[np.arange(codes.shape[1]), codes].prod(axis=1)


def _sum_squared_coefficients(
    observables: Sequence[SparsePauliOp],
    per_obs_term_indices: Sequence[list[int]],
    n_terms: int,
) -> npt.NDArray[np.float64]:
    """``Σ_obs c_{obs,P}²`` per union term, merging repeats within an observable."""
    weights = np.zeros(n_terms)
    for observable, indices in zip(observables, per_obs_term_indices):
        coefficients = np.zeros(n_terms)
        np.add.at(coefficients, indices, np.real(observable.coeffs))
        weights += coefficients**2
    return weights


def _fit_biased_bases(
    codes: npt.NDArray[np.uint8], weights: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Basis probabilities minimising ``Σ_P w_P / π_P``, the locally-biased
    shadow variance bound of Hadfield et al. (arXiv:2006.15788).

    ``π_P`` is the cover probability. The bound is convex, and with every other
    qubit fixed it is minimised by ``β_q(a) ∝ sqrt(Σ_{P: P_q = a} w_P β_q(a) /
    π_P)``, so qubits are swept in turn until the probabilities settle. Letters
    no weighted term uses get zero. Cover probabilities are tracked as logs,
    since on wide registers they underflow.
    """
    bases = _uniform_bases(codes.shape[1])
    fitted = weights > 0
    codes, log_weights = codes[fitted], np.log(weights[fitted])
    log_covers = np.log(bases[np.arange(codes.shape[1]), codes]).sum(axis=1)
    active_qubits = np.flatnonzero(codes.any(axis=0))

    for _ in range(_MAX_FIT_SWEEPS):
        largest_change = 0.0
        for qubit in active_qubits:
            letters = codes[:, qubit]
            log_factors = np.log(bases[qubit, letters])
            log_shares = log_weights + log_factors - log_covers
            # Half the log of each letter's summed share: its unnormalised β.
            log_roots = np.array(
                [
                    0.5 * logsumexp(log_shares[letters == letter])
                    if (letters == letter).any()
                    else -np.inf
                    for letter in (1, 2, 3)
                ]
            )
            updated = np.exp(log_roots - log_roots.max())
            updated /= updated.sum()
            largest_change = max(
                largest_change, float(np.abs(updated - bases[qubit, 1:]).max())
            )
            bases[qubit, 1:] = updated
            log_covers += np.log(bases[qubit, letters]) - log_factors
        if largest_change < _FIT_TOLERANCE:
            break
    return bases


def _count_unique_rows(
    rows: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """The distinct rows of a boolean matrix and how often each occurs."""
    packed = np.ascontiguousarray(np.packbits(rows, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1])))[:, 0]
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return rows[first], counts


def _sample_shadow_covers(
    codes: npt.NDArray[np.uint8],
    bases: npt.NDArray[np.float64],
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], int]:
    """Draw ``n_draws`` random Pauli bases and record which terms each covers.

    Returns ``(covers, counts, empty_draws)``: the distinct non-empty cover
    patterns (one row per pattern, one column per term), how many draws
    produced each, and how many draws covered no term.
    """
    active_qubits = np.flatnonzero(codes.any(axis=0))
    thresholds = np.cumsum(bases[active_qubits, 1:3], axis=1)
    # Rounding can leave the X+Z threshold below 1, so pin it when Y is never drawn.
    thresholds[bases[active_qubits, 3] == 0, 1] = np.inf
    draws = rng.random((n_draws, len(active_qubits)))
    letters = 1 + (draws[:, :, None] >= thresholds[None]).sum(axis=2)

    covers = np.ones((n_draws, len(codes)), dtype=bool)
    for column, qubit in enumerate(active_qubits):
        term_codes = codes[:, qubit]
        covers &= (term_codes == 0) | (letters[:, column, None] == term_codes)

    patterns, counts = _count_unique_rows(covers)
    non_empty = patterns.any(axis=1)
    return patterns[non_empty], counts[non_empty], int(counts[~non_empty].sum())
