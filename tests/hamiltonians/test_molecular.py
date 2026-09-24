# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the molecule frontends and dispatch in divi.hamiltonians._molecular."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import molecular_hamiltonian, molecular_hamiltonian_from_pyscf


@pytest.fixture
def openfermion():
    """OpenFermion, which the PySCF path maps integrals through (``chem`` extra)."""
    return pytest.importorskip("openfermion")


def _ground_energy(spo: SparsePauliOp) -> float:
    return float(np.linalg.eigvalsh(spo.to_matrix())[0])


@pytest.mark.parametrize("frontend", ["pennylane_h2", "pyscf_h2"])
def test_molecular_hamiltonian_accepts_both_frontends(frontend, request):
    molecule = request.getfixturevalue(frontend)
    if frontend == "pyscf_h2":
        pytest.importorskip("openfermion")

    hamiltonian, n_electrons = molecular_hamiltonian(molecule)

    assert isinstance(hamiltonian, SparsePauliOp)
    assert hamiltonian.num_qubits == 4
    assert n_electrons == 2


def test_molecular_hamiltonian_rejects_unknown_input():
    with pytest.raises(TypeError, match="PennyLane Molecule or PySCF"):
        molecular_hamiltonian(object())


def test_molecular_hamiltonian_from_pyscf_matches_fci_h2(pyscf_h2, openfermion):
    scf = pytest.importorskip("pyscf.scf")
    fci = pytest.importorskip("pyscf.fci")

    spo, n_electrons = molecular_hamiltonian_from_pyscf(pyscf_h2)
    assert n_electrons == 2
    assert spo.num_qubits == 4

    e_fci = fci.FCI(scf.RHF(pyscf_h2).run(verbose=0)).kernel()[0]
    assert _ground_energy(spo) == pytest.approx(e_fci, abs=1e-8)


def test_molecular_hamiltonian_from_pyscf_accepts_mean_field(pyscf_h2, openfermion):
    scf = pytest.importorskip("pyscf.scf")

    from_mf, _ = molecular_hamiltonian_from_pyscf(scf.RHF(pyscf_h2).run(verbose=0))
    from_mol, _ = molecular_hamiltonian_from_pyscf(pyscf_h2)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(from_mf.to_matrix()),
        np.linalg.eigvalsh(from_mol.to_matrix()),
        atol=1e-10,
    )


def test_molecular_hamiltonian_from_pyscf_runs_unconverged_mean_field(
    pyscf_h2, openfermion
):
    scf = pytest.importorskip("pyscf.scf")

    spo, _ = molecular_hamiltonian_from_pyscf(scf.RHF(pyscf_h2))  # not run
    assert _ground_energy(spo) == pytest.approx(-1.1372838, abs=1e-5)


def test_molecular_hamiltonian_from_pyscf_rejects_open_shell():
    gto = pytest.importorskip("pyscf.gto")

    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        molecular_hamiltonian_from_pyscf(mol)


def test_molecular_hamiltonian_from_pyscf_rejects_non_pyscf():
    pytest.importorskip("pyscf")
    with pytest.raises(TypeError, match="pyscf Mole or mean-field"):
        molecular_hamiltonian_from_pyscf(object())
