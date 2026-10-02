# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the molecule frontends and dispatch in divi.hamiltonians._molecular."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import molecular_hamiltonian, molecular_hamiltonian_from_pyscf
from divi.hamiltonians._molecular import molecule_integrals
from tests._helpers import exact_match


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
    with pytest.raises(
        TypeError,
        match=exact_match(
            "Expected a PennyLane Molecule or PySCF Mole/mean-field object, "
            "got object."
        ),
    ):
        molecular_hamiltonian(object())


def test_molecular_hamiltonian_from_pyscf_matches_fci_h2(pyscf_h2, openfermion):
    scf = pytest.importorskip("pyscf.scf")
    fci = pytest.importorskip("pyscf.fci")

    spo, n_electrons = molecular_hamiltonian_from_pyscf(pyscf_h2)
    assert n_electrons == 2
    assert spo.num_qubits == 4

    e_fci = fci.FCI(scf.RHF(pyscf_h2).run(verbose=0)).kernel()[0]
    assert _ground_energy(spo) == pytest.approx(e_fci, abs=1e-8)


def test_molecular_hamiltonian_from_pyscf_uses_the_mean_field_orbitals(
    pyscf_h2, openfermion
):
    scf = pytest.importorskip("pyscf.scf")

    mean_field = scf.RHF(pyscf_h2).run(verbose=0)
    mean_field.mo_coeff = mean_field.mo_coeff[:, ::-1]
    from_swapped, _ = molecular_hamiltonian_from_pyscf(mean_field)
    from_mol, _ = molecular_hamiltonian_from_pyscf(pyscf_h2)

    assert not from_swapped.equiv(from_mol)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(from_swapped.to_matrix()),
        np.linalg.eigvalsh(from_mol.to_matrix()),
        atol=1e-10,
    )


def test_molecule_integrals_keep_a_customised_pennylane_basis(qp):
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.46]])
    base = qp.qchem.Molecule(["He", "H"], coordinates, charge=1, basis_name="6-31g")
    molecule = qp.qchem.Molecule(
        ["He", "H"],
        coordinates,
        charge=1,
        basis_name="6-31g",
        alpha=[a * np.linspace(0.9, 1.2, len(a)) for a in base.alpha],
        coeff=[c * np.linspace(1.1, 0.8, len(c)) for c in base.coeff],
    )

    one_body, _, constant = molecule_integrals(molecule)
    expected_constant, expected_one_body, _ = qp.qchem.electron_integrals(molecule)()

    np.testing.assert_allclose(one_body, expected_one_body, atol=1e-12)
    assert constant == pytest.approx(float(np.squeeze(expected_constant)))


def test_molecular_hamiltonian_from_pyscf_runs_unconverged_mean_field(
    pyscf_h2, openfermion
):
    scf = pytest.importorskip("pyscf.scf")

    spo, _ = molecular_hamiltonian_from_pyscf(scf.RHF(pyscf_h2))  # not run
    assert _ground_energy(spo) == pytest.approx(-1.1372838, abs=1e-5)


def test_molecular_hamiltonian_from_pyscf_rejects_open_shell():
    gto = pytest.importorskip("pyscf.gto")

    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
    with pytest.raises(
        NotImplementedError,
        match=exact_match(
            "Only closed-shell (RHF) systems are supported; got an open-shell "
            "molecule with spin=1."
        ),
    ):
        molecular_hamiltonian_from_pyscf(mol)


def test_molecular_hamiltonian_from_pyscf_rejects_non_pyscf():
    pytest.importorskip("pyscf")
    with pytest.raises(
        TypeError,
        match=exact_match("Expected a pyscf Mole or mean-field object, got object."),
    ):
        molecular_hamiltonian_from_pyscf(object())
