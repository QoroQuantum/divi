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
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from tests.conftest import CHECKPOINTING_OPTIMIZERS
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
        n_layers=1,  # n_layers is passed to VQE again
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.molecule == h2_molecule
    assert vqe_problem.n_layers == 1  # Assert on VQE instance
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
        n_layers=1,
        backend=default_test_simulator,
    )

    assert vqe_problem.backend.shots == 5000
    assert vqe_problem.n_layers == 1
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


def test_vqe_fail_with_constant_only_hamiltonian(dummy_simulator):
    """Test VQE initialization fails with a constant-only Hamiltonian."""
    hamiltonian = qml.Identity(0) * 5.0
    with pytest.raises(ValueError, match="Hamiltonian contains only constant terms."):
        VQE(
            hamiltonian=hamiltonian,
            n_electrons=2,
            ansatz=HartreeFockAnsatz(),
            backend=dummy_simulator,
        )


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

    meta_circuit_obj = vqe_problem.meta_circuits["cost_circuit"]
    meta_circuit_qasm = meta_circuit_obj._compiled_circuit_bodies[0]

    pattern = r"w_(\d+)_(\d+)"
    matches = re.findall(pattern, meta_circuit_qasm)

    assert len(set(matches)) == vqe_problem.n_params
    assert len(set(matches)) // n_layers == ansatz_obj.n_params_per_layer(
        vqe_problem.n_qubits, n_electrons=vqe_problem.n_electrons
    )


def test_vqe_initialization_with_initial_params(default_test_simulator, h2_molecule):
    """Test VQE initialization with user-provided initial parameters."""

    optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=1)
    ansatz = UCCSDAnsatz()
    n_layers = 1

    temp_n_qubits, temp_n_electrons = 4, 2

    expected_n_params_per_layer = ansatz.n_params_per_layer(
        temp_n_qubits, n_electrons=temp_n_electrons
    )

    expected_shape = (
        optimizer.n_param_sets,
        expected_n_params_per_layer * n_layers,
    )

    dummy_params = np.random.rand(*expected_shape)

    vqe_problem = VQE(
        molecule=h2_molecule,
        ansatz=ansatz,
        n_layers=n_layers,
        optimizer=optimizer,
        initial_params=dummy_params,
        backend=default_test_simulator,
    )

    np.testing.assert_array_equal(vqe_problem.curr_params, dummy_params)


def test_vqe_fail_with_hw_efficient_ansatz(h2_molecule):
    """Test that HW_EFFICIENT ansatz raises NotImplementedError."""

    with pytest.raises(NotImplementedError):
        # Need to access the meta_circuits property to trigger the NotImplementedError
        VQE(
            molecule=h2_molecule, ansatz=HardwareEfficientAnsatz(), backend=None
        ).meta_circuits


@pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
def test_vqe_correct_circuits_count_and_energies(
    optimizer, dummy_simulator, h2_molecule
):
    """Test circuit counts and energy calculations after a VQE run."""
    optimizer = optimizer()  # Create fresh instance
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


@pytest.mark.e2e
@pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
def test_vqe_h2_molecule_e2e_solution(optimizer, default_test_simulator, h2_molecule):
    """Test that VQE finds the correct ground state for the H2 molecule."""

    default_test_simulator.set_seed(1997)

    vqe_problem = VQE(
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer(),
        max_iterations=5,
        backend=default_test_simulator,
        seed=1997,
    )

    vqe_problem.run()

    assert len(vqe_problem.losses_history) == 5

    assert isinstance(vqe_problem.best_loss, float)
    assert isinstance(vqe_problem.best_params, np.ndarray)
    assert vqe_problem.best_params.shape == (vqe_problem.n_params,)

    # The ground state of H2 in this configuration is |1100>
    # This corresponds to occupying the two lowest energy orbitals.
    expected_best_loss = -1.1398024781381293
    assert vqe_problem.best_loss == pytest.approx(expected_best_loss, abs=0.5)
    expected_eigenstate = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(vqe_problem.eigenstate, expected_eigenstate)


@pytest.mark.e2e
@pytest.mark.parametrize("optimizer", **CHECKPOINTING_OPTIMIZERS)
def test_vqe_h2_molecule_e2e_checkpointing_resume(
    optimizer, default_test_simulator, h2_molecule, tmp_path
):
    """Test VQE e2e with checkpointing and resume functionality."""
    optimizer = optimizer()  # Create fresh instance

    checkpoint_dir = tmp_path / "checkpoint_test"
    default_test_simulator.set_seed(1997)

    # Run first half with checkpointing
    vqe_problem1 = VQE(
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=3,  # First half
        backend=default_test_simulator,
        seed=1997,
    )

    vqe_problem1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

    # Verify checkpoint was created
    assert checkpoint_dir.exists()
    checkpoint_path = checkpoint_dir / "checkpoint_003"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "program_state.json").exists()

    # Store state from first run for comparison
    first_run_iteration = vqe_problem1.current_iteration
    first_run_losses_count = len(vqe_problem1.losses_history)
    first_run_best_loss = vqe_problem1.best_loss

    # Load and resume - configuration must be provided by the caller
    vqe_problem2 = VQE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
    )

    # Verify loaded state matches first run
    assert vqe_problem2.current_iteration == first_run_iteration
    assert len(vqe_problem2.losses_history) == first_run_losses_count
    assert vqe_problem2.best_loss == pytest.approx(first_run_best_loss)

    # Continue running to complete the full run
    vqe_problem2.max_iterations = 5
    vqe_problem2.run()

    # Verify final results are correct
    assert len(vqe_problem2.losses_history) == 5

    assert isinstance(vqe_problem2.best_loss, float)
    assert isinstance(vqe_problem2.best_params, np.ndarray)
    assert vqe_problem2.best_params.shape == (vqe_problem2.n_params,)

    # The ground state of H2 in this configuration is |1100>
    expected_best_loss = -1.1398024781381293
    assert vqe_problem2.best_loss == pytest.approx(expected_best_loss, abs=0.5)
    expected_eigenstate = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(vqe_problem2.eigenstate, expected_eigenstate)

    # Verify we completed the full run
    assert vqe_problem2.current_iteration == 5


@pytest.mark.e2e
@pytest.mark.parametrize("optimizer", **CHECKPOINTING_OPTIMIZERS)
def test_vqe_h2_molecule_e2e_multiple_checkpoint_cycles(
    optimizer, default_test_simulator, h2_molecule, tmp_path
):
    """Test VQE e2e with multiple checkpoint/resume cycles.

    Tests checkpoint infrastructure (multiple save/load cycles) with all checkpointing-capable
    optimizers to verify their nuanced checkpoint handling (CMAES generator reinit, DE pop handling).
    """
    optimizer = optimizer()  # Create fresh instance

    checkpoint_dir = tmp_path / "checkpoint_test"
    default_test_simulator.set_seed(1997)

    # First run: iterations 1-2
    vqe_problem1 = VQE(
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=2,
        backend=default_test_simulator,
        seed=1997,
    )
    vqe_problem1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert vqe_problem1.current_iteration == 2
    assert (checkpoint_dir / "checkpoint_002").exists()

    # Second run: resume and run iterations 3-4
    vqe_problem2 = VQE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
    )
    assert vqe_problem2.current_iteration == 2
    vqe_problem2.max_iterations = 4
    vqe_problem2.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert vqe_problem2.current_iteration == 4
    assert (checkpoint_dir / "checkpoint_004").exists()

    # Third run: resume and run iteration 5
    vqe_problem3 = VQE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        molecule=h2_molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
    )
    assert vqe_problem3.current_iteration == 4
    vqe_problem3.max_iterations = 5
    vqe_problem3.run()
    assert vqe_problem3.current_iteration == 5

    # Verify final results are correct
    expected_eigenstate = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(vqe_problem3.eigenstate, expected_eigenstate)
