# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RDM observables and assembly in divi.hamiltonians._rdm."""

import itertools

import numpy as np
import pytest

from divi.hamiltonians import qubit_operator_to_spo
from divi.hamiltonians._rdm import assemble_spin_rdms, build_rdm_observables

openfermion = pytest.importorskip("openfermion")


def _sector_state(n_orbitals, n_alpha, n_beta, seed):
    """A random real state on ``2 * n_orbitals`` interleaved qubits with fixed
    alpha and beta electron counts (qubit ``k`` is bit ``k`` of the index)."""
    rng = np.random.default_rng(seed)
    state = np.zeros(4**n_orbitals)
    for index in range(len(state)):
        bits = [(index >> qubit) & 1 for qubit in range(2 * n_orbitals)]
        if sum(bits[0::2]) == n_alpha and sum(bits[1::2]) == n_beta:
            state[index] = rng.normal()
    return state / np.linalg.norm(state)


def _expectation(state, term, n_qubits):
    qubit_operator = openfermion.jordan_wigner(openfermion.FermionOperator(term))
    if not qubit_operator.terms:
        return 0.0
    matrix = qubit_operator_to_spo(qubit_operator, n_qubits).to_matrix(sparse=True)
    return float(np.real(state @ (matrix @ state)))


@pytest.mark.parametrize(
    ("n_alpha", "n_beta"), [(2, 1), (2, 2)], ids=["doublet", "singlet-sector"]
)
def test_assembled_spin_rdms_match_direct_expectations(n_alpha, n_beta):
    n_orbitals = 3
    n_qubits = 2 * n_orbitals
    state = _sector_state(n_orbitals, n_alpha, n_beta, seed=7)
    expectations = [
        float(np.real(state @ (observable.to_matrix(sparse=True) @ state)))
        for observable in build_rdm_observables(n_orbitals)
    ]

    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = assemble_spin_rdms(expectations, n_orbitals)

    orbitals = range(n_orbitals)
    for spin, dm1 in ((0, dm1a), (1, dm1b)):
        for p, q in itertools.product(orbitals, repeat=2):
            expected = _expectation(state, f"{2 * q + spin}^ {2 * p + spin}", n_qubits)
            assert dm1[p, q] == pytest.approx(expected, abs=1e-12)
    for (first, second), dm2 in (((0, 0), dm2aa), ((0, 1), dm2ab), ((1, 1), dm2bb)):
        for p, q, r, s in itertools.product(orbitals, repeat=4):
            term = (
                f"{2 * p + first}^ {2 * r + second}^ {2 * s + second} {2 * q + first}"
            )
            expected = _expectation(state, term, n_qubits)
            assert dm2[p, q, r, s] == pytest.approx(expected, abs=1e-12)


def test_rdm_observables_are_hermitian():
    assert all(
        observable.equiv(observable.adjoint())
        for observable in build_rdm_observables(3)
    )
