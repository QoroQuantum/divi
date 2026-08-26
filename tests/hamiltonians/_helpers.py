# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Hamiltonian tests.

Cost and mixer Hamiltonians are checked against the combinatorics they encode
rather than against a second implementation: a cost operator is pinned by the
objective its diagonal scores, a mixer by the basis-state transitions it makes.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def basis_bits(state: int, n_qubits: int) -> list[int]:
    """Bit of each qubit in ``state``, qubit 0 first."""
    return [(state >> qubit) & 1 for qubit in range(n_qubits)]


def diagonal(spo: SparsePauliOp) -> np.ndarray:
    """Real diagonal of a diagonal ``spo``, indexed by basis state."""
    matrix = spo.to_matrix()
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-10)
    return np.real(np.diag(matrix))


def assert_diagonal_scores(
    spo: SparsePauliOp,
    score: Callable[[list[int]], float],
    *,
    offset: float | None = None,
) -> None:
    """Assert the diagonal of ``spo`` scores ``score`` over the basis states.

    A cost Hamiltonian only fixes an objective up to an additive constant, so
    by default the check is that ``score`` and the diagonal differ by one
    shared constant. Pass ``offset`` to pin that constant.
    """
    n_qubits = spo.num_qubits
    actual = diagonal(spo)
    expected = np.array(
        [float(score(basis_bits(state, n_qubits))) for state in range(2**n_qubits)]
    )
    shift = expected - actual
    np.testing.assert_allclose(
        shift, shift[0] if offset is None else offset, atol=1e-10
    )


def assert_transitions(
    spo: SparsePauliOp, transitions: Mapping[tuple[int, int], float]
) -> None:
    """Assert ``spo`` is exactly the given ``{(to, from): amplitude}`` map."""
    n_qubits = spo.num_qubits
    expected = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for (to_state, from_state), amplitude in transitions.items():
        expected[to_state, from_state] = amplitude
    np.testing.assert_allclose(spo.to_matrix(), expected, atol=1e-10)


def add_transition(
    transitions: dict[tuple[int, int], float], to_state: int, from_state: int
) -> None:
    """Accumulate a unit amplitude for one basis-state transition."""
    key = (to_state, from_state)
    transitions[key] = transitions.get(key, 0.0) + 1.0


def single_flip_transitions(n_qubits: int) -> dict[tuple[int, int], float]:
    """Transitions of the X mixer: every single-qubit flip, amplitude one."""
    transitions: dict[tuple[int, int], float] = {}
    for state in range(2**n_qubits):
        for qubit in range(n_qubits):
            add_transition(transitions, state ^ (1 << qubit), state)
    return transitions


def bit_flip_transitions(
    neighbourhoods: Mapping[int, Sequence[int]], n_qubits: int, b: int
) -> dict[tuple[int, int], float]:
    """Transitions of the bit-flip mixer over ``neighbourhoods``.

    A vertex may flip only from basis states whose every neighbour already
    sits at ``b``.
    """
    transitions: dict[tuple[int, int], float] = {}
    for state in range(2**n_qubits):
        bits = basis_bits(state, n_qubits)
        for vertex, neighbours in neighbourhoods.items():
            if all(bits[neighbour] == b for neighbour in neighbours):
                add_transition(transitions, state ^ (1 << vertex), state)
    return transitions
