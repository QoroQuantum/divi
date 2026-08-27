# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for molecule-type dispatch in divi.hamiltonians._molecular."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import molecular_hamiltonian


def test_molecular_hamiltonian_accepts_pennylane_molecule(qp):
    molecule = qp.qchem.Molecule(
        ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.39839733]])
    )

    hamiltonian, n_electrons = molecular_hamiltonian(molecule)

    assert isinstance(hamiltonian, SparsePauliOp)
    assert hamiltonian.num_qubits == 4
    assert n_electrons == 2


def test_molecular_hamiltonian_accepts_pyscf_molecule():
    pytest.importorskip("openfermion")
    gto = pytest.importorskip("pyscf.gto")
    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    hamiltonian, n_electrons = molecular_hamiltonian(molecule)

    assert isinstance(hamiltonian, SparsePauliOp)
    assert hamiltonian.num_qubits == 4
    assert n_electrons == 2


def test_molecular_hamiltonian_rejects_unknown_input():
    with pytest.raises(TypeError, match="PennyLane Molecule or PySCF"):
        molecular_hamiltonian(object())
