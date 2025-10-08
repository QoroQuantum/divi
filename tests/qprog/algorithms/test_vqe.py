# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pennylane as qml
import pytest

from divi.qprog import VQE
from divi.qprog.algorithms import (
    GenericLayerAnsatz,
    HardwareEfficientAnsatz,
    HartreeFockAnsatz,
    QAOAAnsatz,
    UCCSDAnsatz,
)
from tests.qprog.qprog_contracts import (
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
    verify_metacircuit_dict,
)

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


# Ansaetze are now stateless, so we instantiate them once
ANSAETZE_TO_TEST = {
    "argvalues": [
        HartreeFockAnsatz(),
        UCCSDAnsatz(),
        GenericLayerAnsatz([qml.RY, qml.RZ]),
        QAOAAnsatz(),
    ],
    "ids": ["HartreeFock", "UCCSD", "Generic-RYRZ", "QAOA"],
}


def test_vqe_basic_initialization_with_molecule(default_test_simulator, h2_molecule):
    """Test VQE initialization with a molecule object."""
    vqe_problem = VQE(
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=2,  # n_layers is passed to VQE again
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.molecule == h2_molecule
    assert vqe_problem.n_layers == 2  # Assert on VQE instance
    assert vqe_problem.n_electrons == 2
    assert vqe_problem.n_qubits == 4

    assert isinstance(vqe_problem.cost_hamiltonian, qml.operation.Operator)
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


def test_vqe_basic_initialization_with_hamiltonian(
    default_test_simulator, h2_hamiltonian
):
    """Test VQE initialization with a Hamiltonian object."""
    vqe_problem = VQE(
        hamiltonian=h2_hamiltonian,
        n_electrons=2,
        ansatz=HartreeFockAnsatz(),
        n_layers=2,
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.n_layers == 2
    assert vqe_problem.n_electrons == 2
    assert vqe_problem.n_qubits == 4

    assert isinstance(vqe_problem.cost_hamiltonian, qml.operation.Operator)
    verify_metacircuit_dict(vqe_problem, ["cost_circuit"])


def test_clean_hamiltonian_logic(h2_hamiltonian, dummy_simulator):
    """Test that the Hamiltonian is cleaned correctly, separating the constant."""
    constant_value = 5.0
    hamiltonian_with_constant = h2_hamiltonian + qml.Identity(0) * constant_value

    vqe_problem = VQE(
        hamiltonian=hamiltonian_with_constant,
        n_electrons=2,
        ansatz=HartreeFockAnsatz(),
        backend=dummy_simulator,
    )

    coeffs, ops = h2_hamiltonian.terms()
    original_constant = coeffs[ops.index(qml.Identity(0))]
    expected_total_constant = original_constant + constant_value
    assert np.isclose(vqe_problem.loss_constant, expected_total_constant)

    has_identity = any(
        isinstance(op, qml.Identity) for op in vqe_problem.cost_hamiltonian.terms()[1]
    )
    assert not has_identity, "Identity operator should have been removed"


@pytest.mark.parametrize("ansatz_obj", **ANSAETZE_TO_TEST)
@pytest.mark.parametrize("n_layers", [1, 2])
def test_meta_circuit_qasm(ansatz_obj, n_layers, h2_molecule):
    """Test the QASM representation of the meta circuits."""
    vqe_problem = VQE(
        molecule=h2_molecule,
        ansatz=ansatz_obj,
        n_layers=n_layers,
        backend=None,
    )

    meta_circuit_obj = vqe_problem._meta_circuits["cost_circuit"]
    meta_circuit_qasm = meta_circuit_obj.compiled_circuits_bodies[0]

    pattern = r"w_(\d+)_(\d+)"
    matches = re.findall(pattern, meta_circuit_qasm)

    assert len(set(matches)) == vqe_problem.n_params
    assert len(set(matches)) // n_layers == ansatz_obj.n_params_per_layer(
        vqe_problem.n_qubits, n_electrons=vqe_problem.n_electrons
    )


def test_vqe_fail_with_hw_efficient_ansatz(h2_molecule):
    """Test that HW_EFFICIENT ansatz raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        VQE(molecule=h2_molecule, ansatz=HardwareEfficientAnsatz(), backend=None)


@pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
def test_vqe_correct_circuits_count_and_energies(
    optimizer, dummy_simulator, h2_molecule
):
    """Test circuit counts and energy calculations after a VQE run."""
    vqe_problem = VQE(
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=1,
        backend=dummy_simulator,
    )

    vqe_problem.run()
    verify_correct_circuit_count(vqe_problem)
