# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pennylane as qml
import pytest
from qprog_contracts import (
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

from divi.qprog import VQE, Optimizer, VQEAnsatz

pytestmark = pytest.mark.algo


@pytest.fixture
def h2_molecule():
    """Fixture for a simple H2 molecule."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
    return qml.qchem.Molecule(symbols, coordinates)


@pytest.fixture
def h2_hamiltonian(h2_molecule):
    """Fixture for the H2 Hamiltonian."""
    H, _ = qml.qchem.molecular_hamiltonian(h2_molecule)
    return H


def test_vqe_basic_initialization_with_molecule(default_test_simulator, h2_molecule):
    """Test VQE initialization with a molecule object."""
    vqe_problem = VQE(
        molecule=h2_molecule,
        n_layers=2,
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.molecule == h2_molecule
    assert vqe_problem.n_layers == 2
    assert vqe_problem.n_electrons == 2
    assert vqe_problem.n_qubits == 4

    # Check Hamiltonian type
    assert isinstance(
        vqe_problem.cost_hamiltonian, qml.operation.Operator
    ), "Expected a pennylane Operator object for the hamiltonian"

    # Check meta-circuits
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


def test_vqe_basic_initialization_with_hamiltonian(
    default_test_simulator, h2_hamiltonian
):
    """Test VQE initialization with a Hamiltonian object."""
    vqe_problem = VQE(
        hamiltonian=h2_hamiltonian,
        n_electrons=2,
        n_layers=2,
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.n_layers == 2
    assert vqe_problem.n_electrons == 2
    assert vqe_problem.n_qubits == 4

    # Check Hamiltonian type
    assert isinstance(
        vqe_problem.cost_hamiltonian, qml.operation.Operator
    ), "Expected a pennylane Operator object for the hamiltonian"

    # Check meta-circuits
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


def test_clean_hamiltonian_logic(h2_hamiltonian, dummy_simulator):
    """Test that the Hamiltonian is cleaned correctly, separating the constant."""
    # Add a known constant term (Identity) to the Hamiltonian
    constant_value = 5.0
    hamiltonian_with_constant = h2_hamiltonian + qml.Identity(0) * constant_value

    vqe_problem = VQE(
        hamiltonian=hamiltonian_with_constant,
        n_electrons=2,
        backend=dummy_simulator,
    )

    # 1. Check that the constant was extracted correctly
    # The total constant is the manually added one plus the original nuclear repulsion
    coeffs, ops = h2_hamiltonian.terms()
    original_constant = coeffs[ops.index(qml.Identity(0))]
    expected_total_constant = original_constant + constant_value
    assert np.isclose(vqe_problem.loss_constant, expected_total_constant)

    # 2. Check that the final cost Hamiltonian has no Identity operator
    has_identity = any(
        isinstance(op, qml.Identity) for op in vqe_problem.cost_hamiltonian.terms()[1]
    )
    assert not has_identity, "Identity operator should have been removed"


@pytest.mark.parametrize("ansatz", list(VQEAnsatz))
@pytest.mark.parametrize("n_layers", [1, 2])
def test_meta_circuit_qasm(ansatz, n_layers, h2_molecule):
    """Test the QASM representation of the meta circuits."""
    if ansatz == VQEAnsatz.HW_EFFICIENT:
        pytest.skip("Skipping HW_EFFICIENT ansatz")

    vqe_problem = VQE(
        molecule=h2_molecule,
        n_layers=n_layers,
        ansatz=ansatz,
        backend=None,
    )

    meta_circuit_obj = vqe_problem._meta_circuits["cost_circuit"]
    meta_circuit_qasm = meta_circuit_obj.compiled_circuits_bodies[0]

    pattern = r"w_(\d+)_(\d+)"
    matches = re.findall(pattern, meta_circuit_qasm)

    # Check that we have the correct number of unique parameters
    assert len(set(matches)) == n_layers * vqe_problem.n_params
    assert len(set(matches)) // n_layers == vqe_problem.n_params


@pytest.mark.parametrize(
    "vqe_input, n_electrons, error, match_str",
    [
        (
            {},
            None,
            ValueError,
            "Either one of `molecule` and `hamiltonian` must be provided.",
        ),
        (
            {"hamiltonian": qml.Hamiltonian([], [])},
            None,
            ValueError,
            "`n_electrons` is expected to be a non-negative integer.",
        ),
        (
            {"hamiltonian": qml.Hamiltonian([], [])},
            -1,
            ValueError,
            "`n_electrons` is expected to be a non-negative integer.",
        ),
        (
            {"hamiltonian": qml.Hamiltonian([], [])},
            "2",
            ValueError,
            "`n_electrons` is expected to be a non-negative integer.",
        ),
    ],
)
def test_vqe_input_errors(vqe_input, n_electrons, error, match_str):
    """Test that VQE raises errors for invalid inputs."""
    with pytest.raises(error, match=match_str):
        VQE(**vqe_input, n_electrons=n_electrons)


def test_vqe_n_electrons_warning(h2_molecule, dummy_simulator):
    """Test that a warning is issued for inconsistent n_electrons with a molecule."""
    with pytest.warns(
        UserWarning, match="`n_electrons` is provided but not consistent"
    ):
        vqe = VQE(molecule=h2_molecule, n_electrons=1, backend=dummy_simulator)
        # The molecule's electron count should take precedence
        assert vqe.n_electrons == h2_molecule.n_electrons


def test_vqe_fail_with_hw_efficient_ansatz(h2_molecule):
    """Test that HW_EFFICIENT ansatz raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        VQE(
            molecule=h2_molecule,
            ansatz=VQEAnsatz.HW_EFFICIENT,
        )


@pytest.mark.parametrize("optimizer", list(Optimizer))
def test_vqe_correct_circuits_count_and_energies(
    optimizer, dummy_simulator, h2_molecule
):
    """Test circuit counts and energy calculations after a VQE run."""
    vqe_problem = VQE(
        molecule=h2_molecule,
        n_layers=1,
        ansatz=VQEAnsatz.HARTREE_FOCK,
        optimizer=optimizer,
        max_iterations=1,
        backend=dummy_simulator,
    )

    vqe_problem.run()

    verify_correct_circuit_count(vqe_problem)
