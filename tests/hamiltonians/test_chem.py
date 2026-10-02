# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OpenFermion adapters in divi.hamiltonians._chem."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import qubit_operator_to_spo, to_spo
from divi.qprog.algorithms import TimeEvolution
from tests._helpers import exact_match

openfermion = pytest.importorskip("openfermion")
QubitOperator = openfermion.QubitOperator
get_sparse_operator = openfermion.get_sparse_operator


def _two_qubit_qubit_operator():
    return QubitOperator("Z0 Z1", 1.0) + QubitOperator("X0", 0.3)


def test_qubit_operator_to_spo_preserves_qubit_indices():
    for of_term, qiskit_label in {
        "Z0": "IZ",
        "Z1": "ZI",
        "X1": "XI",
        "Y0": "IY",
    }.items():
        spo = qubit_operator_to_spo(QubitOperator(of_term), 2)
        assert spo.equiv(SparsePauliOp(qiskit_label))


def test_qubit_operator_to_spo_matches_openfermion_spectrum():
    qop = (
        QubitOperator("X0 Z1", 0.5) + QubitOperator("Y2", -1.3) + QubitOperator("", 0.7)
    )
    spo = qubit_operator_to_spo(qop)
    of_matrix = get_sparse_operator(qop, n_qubits=spo.num_qubits).toarray()
    np.testing.assert_allclose(
        np.linalg.eigvalsh(spo.to_matrix()),
        np.linalg.eigvalsh(of_matrix),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "of_term, coeff, qiskit_label",
    [("Z3", 1.0, "ZIII"), ("", 0.7, "I"), ("Z0", 1.0, "Z")],
    ids=["highest_qubit_three", "identity_only", "qubit_zero_only"],
)
def test_qubit_operator_to_spo_infers_width(of_term, coeff, qiskit_label):
    spo = qubit_operator_to_spo(QubitOperator(of_term, coeff))
    assert spo.num_qubits == len(qiskit_label)
    assert spo.equiv(SparsePauliOp(qiskit_label, coeff))


def test_qubit_operator_to_spo_rejects_narrow_width():
    with pytest.raises(
        ValueError,
        match=exact_match("n_qubits (2) is smaller than the operator's support (4)."),
    ):
        qubit_operator_to_spo(QubitOperator("Z3", 1.0), 2)


def test_qubit_operator_to_spo_rejects_empty():
    with pytest.raises(ValueError, match=exact_match("QubitOperator has no terms.")):
        qubit_operator_to_spo(QubitOperator())


def test_to_spo_accepts_qubit_operator():
    spo = to_spo(_two_qubit_qubit_operator())
    assert spo.equiv(SparsePauliOp(["ZZ", "IX"], [1.0, 0.3]))


def test_time_evolution_accepts_qubit_operator(dummy_simulator):
    qop = _two_qubit_qubit_operator()
    from_qop = TimeEvolution(hamiltonian=qop, time=1.0, backend=dummy_simulator)
    from_spo = TimeEvolution(hamiltonian=to_spo(qop), time=1.0, backend=dummy_simulator)
    assert from_qop.n_qubits == from_spo.n_qubits == 2
    assert from_qop._hamiltonian.equiv(from_spo._hamiltonian)
