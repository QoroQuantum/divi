# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for chemistry-stack adapters in divi.hamiltonians._chem."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    molecular_hamiltonian_from_pyscf,
    qubit_operator_to_spo,
    to_spo,
)
from divi.qprog.algorithms import TimeEvolution

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


def test_qubit_operator_to_spo_infers_width():
    assert qubit_operator_to_spo(QubitOperator("Z3", 1.0)).num_qubits == 4


def test_qubit_operator_to_spo_rejects_narrow_width():
    with pytest.raises(ValueError, match="smaller than"):
        qubit_operator_to_spo(QubitOperator("Z3", 1.0), 2)


def test_qubit_operator_to_spo_rejects_empty():
    with pytest.raises(ValueError, match="no terms"):
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


def test_molecular_hamiltonian_matches_fci_h2():
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    fci = pytest.importorskip("pyscf.fci")

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    spo, n_electrons = molecular_hamiltonian_from_pyscf(mol)
    assert n_electrons == 2
    assert spo.num_qubits == 4

    e_fci = fci.FCI(scf.RHF(mol).run(verbose=0)).kernel()[0]
    ground = float(np.linalg.eigvalsh(spo.to_matrix())[0])
    assert ground == pytest.approx(e_fci, abs=1e-8)


def test_molecular_hamiltonian_accepts_mean_field():
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    from_mf, _ = molecular_hamiltonian_from_pyscf(scf.RHF(mol).run(verbose=0))
    from_mol, _ = molecular_hamiltonian_from_pyscf(mol)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(from_mf.to_matrix()),
        np.linalg.eigvalsh(from_mol.to_matrix()),
        atol=1e-10,
    )


def test_molecular_hamiltonian_runs_unconverged_mean_field():
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    spo, _ = molecular_hamiltonian_from_pyscf(scf.RHF(mol))  # not run
    ground = float(np.linalg.eigvalsh(spo.to_matrix())[0])
    assert ground == pytest.approx(-1.1372838, abs=1e-5)


def test_molecular_hamiltonian_rejects_open_shell():
    gto = pytest.importorskip("pyscf.gto")

    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        molecular_hamiltonian_from_pyscf(mol)


def test_molecular_hamiltonian_rejects_non_pyscf():
    pytest.importorskip("pyscf")
    with pytest.raises(TypeError, match="pyscf Mole or mean-field"):
        molecular_hamiltonian_from_pyscf(object())
