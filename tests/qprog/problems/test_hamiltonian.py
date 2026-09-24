# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for HamiltonianProblem and MolecularProblem."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import molecular_hamiltonian
from divi.qprog.problems import (
    HamiltonianProblem,
    MolecularProblem,
    _hamiltonian,
)


def _two_orbital_integrals():
    rng = np.random.default_rng(3)
    one_body = rng.standard_normal((2, 2))
    one_body = one_body + one_body.T
    two_body = rng.standard_normal((2,) * 4)
    for axes in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        two_body = two_body + two_body.transpose(axes)
    return one_body, two_body


def _assert_same_operator(left: SparsePauliOp, right: SparsePauliOp):
    difference = (left - right).simplify(atol=1e-9)
    assert np.allclose(difference.coeffs, 0.0, atol=1e-9)


def test_hamiltonian_problem_accepts_any_observable_form():
    problem = HamiltonianProblem({"ZZ": 1.0, "XI": 0.5})

    assert isinstance(problem.hamiltonian, SparsePauliOp)
    assert problem.hamiltonian.num_qubits == 2
    assert problem.n_electrons is None
    assert problem.n_alpha is None and problem.n_beta is None


def test_hamiltonian_problem_infers_electrons_from_spin_counts():
    problem = HamiltonianProblem(SparsePauliOp("ZZ"), n_alpha=2, n_beta=1)

    assert (problem.n_electrons, problem.n_alpha, problem.n_beta) == (3, 2, 1)


@pytest.mark.parametrize(
    "counts, match",
    [
        ({"n_alpha": 1}, "given together"),
        ({"n_beta": 1}, "given together"),
        ({"n_alpha": -1, "n_beta": 1}, "non-negative"),
        ({"n_electrons": -2}, "non-negative"),
        ({"n_electrons": 4, "n_alpha": 1, "n_beta": 1}, "disagrees"),
    ],
)
def test_hamiltonian_problem_rejects_inconsistent_electron_counts(counts, match):
    with pytest.raises(ValueError, match=match):
        HamiltonianProblem(SparsePauliOp("ZZ"), **counts)


def test_molecular_problem_is_a_hamiltonian_problem():
    one_body, two_body = _two_orbital_integrals()
    problem = MolecularProblem(one_body, two_body, n_alpha=1, n_beta=1)

    assert isinstance(problem, HamiltonianProblem)
    assert problem.molecule is None
    assert problem.n_orbitals == 2
    assert problem.n_electrons == 2
    np.testing.assert_array_equal(problem.one_body_beta, problem.one_body)


def test_molecular_problem_integrals_cannot_drift_from_its_hamiltonian():
    """The cached Hamiltonian stays the image of the integrals: the caller's
    arrays are copied, and the stored ones are read-only."""
    one_body, two_body = _two_orbital_integrals()
    problem = MolecularProblem(one_body, two_body, n_alpha=1, n_beta=1)

    one_body[0, 0] += 1.0
    assert problem.one_body[0, 0] != one_body[0, 0]
    with pytest.raises(ValueError, match="read-only"):
        problem.two_body[0, 0, 0, 0] = 1.0


def test_molecular_problem_maps_its_integrals_to_qubits():
    """The Hamiltonian is the Jordan-Wigner image of the stored integrals, with
    the constant and the beta channel carried through."""
    pytest.importorskip("openfermion")
    one_body, two_body = _two_orbital_integrals()
    one_body_beta = one_body + np.diag([0.3, -0.1])

    problem = MolecularProblem(
        one_body,
        two_body,
        n_alpha=1,
        n_beta=1,
        constant=0.7,
        one_body_beta=one_body_beta,
    )
    spin_blind = MolecularProblem(one_body, two_body, n_alpha=1, n_beta=1, constant=0.7)

    assert problem.hamiltonian.num_qubits == 4
    assert problem.constant == pytest.approx(0.7)
    np.testing.assert_array_equal(problem.one_body_beta, one_body_beta)
    # The beta shift adds shift_p * n_(p, beta), with n = (I - Z) / 2 on qubit 2p + 1.
    beta_shift = SparsePauliOp.from_sparse_list(
        [
            term
            for p, shift in enumerate([0.3, -0.1])
            for term in (("", [], shift / 2), ("Z", [2 * p + 1], -shift / 2))
        ],
        num_qubits=4,
    )
    _assert_same_operator(problem.hamiltonian - spin_blind.hamiltonian, beta_shift)


@pytest.mark.parametrize(
    "one_body, two_body, kwargs, match",
    [
        (np.zeros((2, 3)), np.zeros((2,) * 4), {}, "square"),
        (np.zeros((2, 2)), np.zeros((3,) * 4), {}, "two_body"),
        (
            np.zeros((2, 2)),
            np.zeros((2,) * 4),
            {"one_body_beta": np.zeros((3, 3))},
            "one_body_beta",
        ),
        (np.zeros((2, 2)), np.zeros((2,) * 4), {"n_alpha": 3}, "n_alpha"),
        (np.zeros((2, 2)), np.zeros((2,) * 4), {"n_beta": -1}, "n_beta"),
    ],
)
def test_molecular_problem_rejects_invalid_integrals(one_body, two_body, kwargs, match):
    counts = {"n_alpha": 1, "n_beta": 1, **kwargs}
    with pytest.raises(ValueError, match=match):
        MolecularProblem(one_body, two_body, **counts)


def _jordan_wigner_of_stored_integrals(problem):
    return MolecularProblem(
        problem.one_body,
        problem.two_body,
        n_alpha=problem.n_alpha,
        n_beta=problem.n_beta,
        constant=problem.constant,
    ).hamiltonian


def test_from_molecule_solves_once_on_first_access(pyscf_h2, mocker):
    pytest.importorskip("openfermion")
    integrals = mocker.spy(_hamiltonian, "molecule_integrals")

    problem = MolecularProblem.from_molecule(pyscf_h2)

    assert problem.molecule is pyscf_h2
    assert (problem.n_electrons, problem.n_alpha, problem.n_beta) == (2, 1, 1)
    integrals.assert_not_called()

    first = problem.hamiltonian
    assert problem.hamiltonian is first
    assert problem.two_body.shape == (2,) * 4
    assert problem.one_body.shape == (2, 2)
    integrals.assert_called_once()


@pytest.mark.parametrize("frontend", ["pyscf_h2", "pennylane_h2"])
def test_from_molecule_integrals_reproduce_the_molecular_hamiltonian(frontend, request):
    """Both frontends' integrals Jordan-Wigner back to the Hamiltonian that
    frontend builds itself, which pins PennyLane's two-body index order."""
    pytest.importorskip("openfermion")
    molecule = request.getfixturevalue(frontend)
    problem = MolecularProblem.from_molecule(molecule)

    _assert_same_operator(
        _jordan_wigner_of_stored_integrals(problem), problem.hamiltonian
    )
    _assert_same_operator(problem.hamiltonian, molecular_hamiltonian(molecule)[0])


def test_from_molecule_hamiltonian_shares_the_orbitals_of_its_integrals():
    """N2's degenerate pi orbitals come out in an arbitrary basis on every RHF
    solve, so a Hamiltonian from a second solve would not match the integrals."""
    pytest.importorskip("openfermion")
    gto = pytest.importorskip("pyscf.gto")
    n2 = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)

    problem = MolecularProblem.from_molecule(n2)

    _assert_same_operator(
        problem.hamiltonian, _jordan_wigner_of_stored_integrals(problem)
    )


def test_from_molecule_accepts_differentiable_pennylane_coordinates(qp, pennylane_h2):
    pytest.importorskip("openfermion")
    coordinates = qp.numpy.array(pennylane_h2.coordinates, requires_grad=True)
    problem = MolecularProblem.from_molecule(qp.qchem.Molecule(["H", "H"], coordinates))

    _assert_same_operator(
        problem.hamiltonian, _jordan_wigner_of_stored_integrals(problem)
    )


def test_from_molecule_accepts_a_pyscf_mean_field(pyscf_h2):
    scf = pytest.importorskip("pyscf.scf")
    mean_field = scf.RHF(pyscf_h2).run(verbose=0)

    problem = MolecularProblem.from_molecule(mean_field)

    assert problem.molecule is mean_field
    assert problem.n_electrons == 2


@pytest.mark.parametrize("method", ["UHF", "GHF"])
@pytest.mark.parametrize("solved", [True, False], ids=["solved", "unsolved"])
def test_from_molecule_rejects_a_non_restricted_mean_field(pyscf_h2, method, solved):
    """Rejected up front: an unsolved one would otherwise be replaced by RHF."""
    scf = pytest.importorskip("pyscf.scf")
    mean_field = getattr(scf, method)(pyscf_h2)
    if solved:
        mean_field.run(verbose=0)

    with pytest.raises(NotImplementedError, match="restricted"):
        MolecularProblem.from_molecule(mean_field)


def test_from_molecule_rejects_an_open_shell_pyscf_molecule():
    gto = pytest.importorskip("pyscf.gto")
    doublet = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)

    with pytest.raises(NotImplementedError, match="closed-shell"):
        MolecularProblem.from_molecule(doublet)


def test_from_molecule_rejects_an_open_shell_pennylane_molecule(qp, pennylane_h2):
    triplet = qp.qchem.Molecule(["H", "H"], pennylane_h2.coordinates, mult=3)

    with pytest.raises(NotImplementedError, match="closed-shell"):
        MolecularProblem.from_molecule(triplet)


def test_from_molecule_rejects_other_inputs():
    with pytest.raises(TypeError, match="PennyLane Molecule or PySCF"):
        MolecularProblem.from_molecule(object())
