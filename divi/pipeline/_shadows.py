# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Random Pauli classical-shadow sampling over an observable's Pauli terms.

Each snapshot measures every qubit in a uniformly random Pauli basis. A basis
*covers* a term when it matches the term's Pauli on every qubit the term acts
on, which happens with probability ``3**-|P|``. Snapshots that cover the same
terms run the same measurement circuit, so they are merged into one group.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# Pauli letter -> code; 0 marks identity, bases draw from 1..3.
_PAULI_CODES = np.zeros(256, dtype=np.uint8)
_PAULI_CODES[[ord("X"), ord("Y"), ord("Z")]] = [1, 2, 3]


def _pauli_codes(labels: Sequence[str]) -> npt.NDArray[np.uint8]:
    """Encode big-endian Pauli labels as a ``(n_terms, n_qubits)`` code matrix."""
    chars = np.frombuffer("".join(labels).encode("ascii"), dtype=np.uint8)
    return _PAULI_CODES[chars].reshape(len(labels), -1)


def _shadow_term_weights(labels: Sequence[str]) -> npt.NDArray[np.float64]:
    """Inverse probability ``3**|P|`` that a random basis covers each term."""
    return 3.0 ** np.count_nonzero(_pauli_codes(labels), axis=1)


def _count_unique_rows(
    rows: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """The distinct rows of a boolean matrix and how often each occurs."""
    packed = np.ascontiguousarray(np.packbits(rows, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1])))[:, 0]
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return rows[first], counts


def _sample_shadow_covers(
    labels: Sequence[str], n_draws: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], int]:
    """Draw ``n_draws`` random Pauli bases and record which terms each covers.

    Returns ``(covers, counts, empty_draws)``: the distinct non-empty cover
    patterns (one row per pattern, one column per term), how many draws
    produced each, and how many draws covered no term.
    """
    codes = _pauli_codes(labels)
    active_qubits = np.flatnonzero(codes.any(axis=0))
    bases = rng.integers(1, 4, size=(n_draws, len(active_qubits)), dtype=np.uint8)

    covers = np.ones((n_draws, len(labels)), dtype=bool)
    for column, qubit in enumerate(active_qubits):
        term_codes = codes[:, qubit]
        covers &= (term_codes == 0) | (bases[:, column, None] == term_codes)

    patterns, counts = _count_unique_rows(covers)
    non_empty = patterns.any(axis=1)
    return patterns[non_empty], counts[non_empty], int(counts[~non_empty].sum())
