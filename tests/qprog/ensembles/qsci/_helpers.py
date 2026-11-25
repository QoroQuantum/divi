# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Hamiltonians, configuration sets and oracles shared by the QSCI tests."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.qprog.ensembles.qsci._subspace import ConfigurationSubspace

N_QUBITS = 5

H_SMALL = SparsePauliOp.from_sparse_list(
    [
        ("ZZ", [0, 1], 0.7),
        ("ZZ", [1, 2], -0.4),
        ("X", [0], 0.5),
        ("X", [3], 0.3),
        ("XX", [2, 3], 0.2),
        ("Z", [3], 0.1),
    ],
    num_qubits=4,
)


def build_random_hamiltonian(n_qubits, seed, *, odd_y):
    """Fourteen random Pauli terms; ``odd_y=False`` keeps every Y count even,
    so the CI matrix is real."""
    rng = np.random.default_rng(seed)
    terms = {}
    while len(terms) < 14:
        label = "".join(rng.choice(list("IXYZ"), n_qubits))
        if odd_y or label.count("Y") % 2 == 0:
            terms[label] = rng.normal()
    return SparsePauliOp.from_list(list(terms.items()))


def compute_basis_index(config):
    return sum(1 << q for q, bit in enumerate(config) if bit == "1")


def list_all_configs(n_qubits):
    return [format(i, f"0{n_qubits}b")[::-1] for i in range(2**n_qubits)]


def solve_subspace(hamiltonian, configs, **penalty):
    """A subspace over ``configs`` with its eigensystem solved."""
    subspace = ConfigurationSubspace(hamiltonian, **penalty)
    subspace.add_configurations(configs)
    subspace.extend_hamiltonian_matrix()
    subspace.solve_eigensystem()
    return subspace


def compute_lowest_penalised_state(hamiltonian, penalty, lambda_penalty):
    """The lowest eigenvector of ``H + lambda * D @ D`` over the whole space,
    and its ``<H>``."""
    dense_hamiltonian = hamiltonian.to_matrix()
    dense_penalty = penalty.to_matrix()
    _, vectors = np.linalg.eigh(
        dense_hamiltonian + lambda_penalty * dense_penalty @ dense_penalty
    )
    expected = vectors[:, 0]
    return expected, np.vdot(expected, dense_hamiltonian @ expected).real
