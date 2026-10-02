# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
import warnings

import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import SparsePauliOp

from divi.qprog import VQE, EarlyStopping
from divi.qprog.algorithms import (
    GenericLayerAnsatz,
    HartreeFockAnsatz,
    LUCJAnsatz,
    QAOAAnsatz,
    QCCAnsatz,
    SuperpositionState,
    UCCSDAnsatz,
    ZerosState,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.problems import HamiltonianProblem, MolecularProblem
from divi.reporting._events import EventKind, TerminalStatus
from tests._helpers import exact_match
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_correct_circuit_count,
    verify_cost_circuit,
)
from tests.qprog.algorithms._helpers import needs_qiskit_nature


@pytest.fixture
def h2_problem(pennylane_h2):
    """H2 through the PennyLane molecule front door."""
    return MolecularProblem.from_molecule(pennylane_h2)


@pytest.fixture
def four_qubit_hamiltonian():
    """Ising chain, transverse field, and one pair-hopping term.

    Four qubits with a two-electron reference — the shape a chemistry
    Hamiltonian has, without needing a chemistry stack to build it.
    """
    return SparsePauliOp.from_sparse_list(
        [("ZZ", [q, q + 1], 1.0) for q in range(3)]
        + [("X", [q], 0.5) for q in range(4)]
        + [("XXYY", [0, 1, 2, 3], 0.25)],
        num_qubits=4,
    )


# Ansaetze are now stateless, so we instantiate them once
ANSAETZE_TO_TEST = {
    "argvalues": [
        HartreeFockAnsatz(),
        pytest.param(UCCSDAnsatz(), marks=needs_qiskit_nature),
        QCCAnsatz(),
        GenericLayerAnsatz([RYGate, RZGate]),
        QAOAAnsatz(),
    ],
    "ids": ["HartreeFock", "UCCSD", "QCC", "Generic-RYRZ", "QAOA"],
}


def test_vqe_initialization_with_pyscf_molecule(
    default_test_simulator, default_optimizer, pyscf_h2
):
    """VQE accepts a PySCF molecule and builds the same H2 problem."""
    pytest.importorskip("openfermion")

    vqe_problem = VQE(
        MolecularProblem.from_molecule(pyscf_h2),
        ansatz=HartreeFockAnsatz(),
        backend=default_test_simulator,
        optimizer=default_optimizer,
    )

    assert vqe_problem.n_qubits == 4
    assert isinstance(vqe_problem.cost_hamiltonian, SparsePauliOp)


def test_vqe_initialization_with_qubit_operator_hamiltonian(
    dummy_simulator, default_optimizer
):
    """VQE accepts a problem built from an OpenFermion QubitOperator."""
    QubitOperator = pytest.importorskip("openfermion").QubitOperator

    qop = QubitOperator("Z0 Z1", 1.0) + QubitOperator("X0", 0.3)
    vqe_problem = VQE(
        HamiltonianProblem(qop, n_electrons=2),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    assert vqe_problem.n_qubits == 2
    assert isinstance(vqe_problem.cost_hamiltonian, SparsePauliOp)


@pytest.fixture(params=["from_molecule", "from_hamiltonian"])
def four_qubit_problem(request):
    if request.param == "from_molecule":
        return request.getfixturevalue("h2_problem")
    return HamiltonianProblem(
        request.getfixturevalue("four_qubit_hamiltonian"), n_electrons=2
    )


def test_vqe_basic_initialization(
    default_test_simulator, four_qubit_problem, default_optimizer
):
    vqe_problem = VQE(
        four_qubit_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        backend=default_test_simulator,
        optimizer=default_optimizer,
    )

    assert vqe_problem.n_layers == 1
    assert vqe_problem.n_qubits == 4
    assert vqe_problem.max_iterations == 10

    assert isinstance(vqe_problem.cost_hamiltonian, SparsePauliOp)
    verify_cost_circuit(vqe_problem)


@pytest.mark.parametrize(
    "ansatz, starts_at_reference",
    [
        (HartreeFockAnsatz(), True),
        pytest.param(UCCSDAnsatz(), True, marks=needs_qiskit_nature),
        (QCCAnsatz(), False),
        (GenericLayerAnsatz([RYGate, RZGate]), False),
    ],
    ids=["HartreeFock", "UCCSD", "QCC", "Generic-RYRZ"],
)
def test_excitation_ansatze_start_at_the_hartree_fock_point(
    dummy_simulator,
    four_qubit_hamiltonian,
    default_optimizer,
    ansatz,
    starts_at_reference,
):
    """All-zero excitation amplitudes prepare the Hartree-Fock reference."""
    vqe = VQE(
        HamiltonianProblem(four_qubit_hamiltonian, n_electrons=2),
        ansatz=ansatz,
        backend=dummy_simulator,
        optimizer=default_optimizer,
        seed=1997,
    )

    params = vqe._initialize_param_sets()

    assert params.shape == vqe.get_expected_param_shape()
    assert bool(np.all(params == 0.0)) is starts_at_reference


def test_vqe_clean_hamiltonian_logic(
    four_qubit_hamiltonian, dummy_simulator, default_optimizer
):
    """Test that the Hamiltonian is cleaned correctly, separating the constant."""
    constant_value = 5.0
    hamiltonian_with_constant = four_qubit_hamiltonian + SparsePauliOp.from_list(
        [("IIII", constant_value)]
    )

    vqe_problem = VQE(
        HamiltonianProblem(hamiltonian_with_constant, n_electrons=2),
        ansatz=HartreeFockAnsatz(),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    # The fixture carries no identity row, so the whole constant is the one added.
    assert np.isclose(vqe_problem.loss_constant, constant_value)

    # ``_clean_hamiltonian_spo`` partitions identity rows out of the SPO and
    # accumulates them into ``loss_constant``; verify no surviving identity
    # row remains.
    labels = vqe_problem.cost_hamiltonian.paulis.to_labels()
    assert not any(
        set(label) == {"I"} for label in labels
    ), "Identity operator should have been removed"


def test_vqe_fail_with_constant_only_hamiltonian(dummy_simulator, default_optimizer):
    """Test VQE initialization fails with a constant-only Hamiltonian."""
    hamiltonian = 5.0 * SparsePauliOp("I")
    with pytest.raises(ValueError, match="Hamiltonian contains only constant terms."):
        VQE(
            HamiltonianProblem(hamiltonian, n_electrons=2),
            ansatz=HartreeFockAnsatz(),
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )


def test_vqe_fail_with_bare_hamiltonian(dummy_simulator, default_optimizer):
    """VQE raises TypeError when given an operator instead of a HamiltonianProblem."""
    with pytest.raises(
        TypeError,
        match=exact_match(
            "problem must be a HamiltonianProblem; got SparsePauliOp. Wrap a bare "
            "operator in HamiltonianProblem, or a molecule in "
            "MolecularProblem.from_molecule."
        ),
    ):
        VQE(
            SparsePauliOp("Z"),
            ansatz=HartreeFockAnsatz(),
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )


def test_vqe_single_term_hamiltonian_succeeds(dummy_simulator, default_optimizer):
    """A one-term Hamiltonian initialises without an operands error."""
    vqe_problem = VQE(
        HamiltonianProblem(0.5 * SparsePauliOp("Z"), n_electrons=1),
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    assert vqe_problem.cost_hamiltonian.equiv(SparsePauliOp("Z", 0.5))
    assert vqe_problem.n_qubits == 1


def test_standalone_sampling_uses_one_direct_progress_session(
    default_test_simulator, default_optimizer, recording_direct_sessions
):
    vqe = VQE(
        HamiltonianProblem(SparsePauliOp("Z"), n_electrons=1),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=default_test_simulator,
        optimizer=default_optimizer,
    )

    vqe.sample_solution(np.array([0.1, 0.2]))

    assert len(recording_direct_sessions) == 1
    session = recording_direct_sessions[0]
    terminal_events = [
        event for event in session.emitted if event.kind is EventKind.FINISH
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0].terminal_status is TerminalStatus.SUCCESS
    assert session.state.get(vqe._progress_key).terminal_status is (
        TerminalStatus.SUCCESS
    )


def _two_qubit_ry_vqe(backend, optimizer, **kwargs):
    """``Z0 + Z1`` under one ``RY`` layer: two parameters, no electrons needed."""
    return VQE(
        HamiltonianProblem(SparsePauliOp(["ZI", "IZ"])),
        ansatz=GenericLayerAnsatz([RYGate]),
        backend=backend,
        optimizer=optimizer,
        **kwargs,
    )


def test_initial_state_is_prepended_to_the_cost_circuit(
    dummy_simulator, default_optimizer
):
    vqe = _two_qubit_ry_vqe(
        dummy_simulator, default_optimizer, initial_state=SuperpositionState()
    )

    circuit = dag_to_circuit(vqe.cost_circuit.circuit_bodies[0][1])

    assert circuit.count_ops()["h"] == 2


def test_chemistry_ansatz_on_the_zeros_state_does_not_warn(
    four_qubit_hamiltonian, dummy_simulator, default_optimizer
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        VQE(
            HamiltonianProblem(four_qubit_hamiltonian, n_electrons=2),
            ansatz=HartreeFockAnsatz(),
            initial_state=ZerosState(),
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )


def test_sample_solution_uses_the_backend_override(
    dummy_simulator, default_optimizer, make_dummy_simulator, mocker
):
    vqe = _two_qubit_ry_vqe(dummy_simulator, default_optimizer)
    override = make_dummy_simulator(100)
    override_submit = mocker.spy(override, "submit_circuits")

    vqe.sample_solution(np.array([0.1, 0.2]), backend=override)

    override_submit.assert_called_once()


def test_loaded_eigenstate_is_int32(dummy_simulator, default_optimizer, tmp_path):
    source = _two_qubit_ry_vqe(dummy_simulator, default_optimizer)
    source.run(
        max_iterations=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
    )

    loaded = VQE.load_state(
        tmp_path,
        backend=dummy_simulator,
        problem=HamiltonianProblem(SparsePauliOp(["ZI", "IZ"])),
        ansatz=GenericLayerAnsatz([RYGate]),
    )

    assert loaded.eigenstate.dtype == np.int32
    np.testing.assert_array_equal(loaded.eigenstate, source.eigenstate)


@needs_qiskit_nature
def test_parameter_frequencies_honour_the_spin_counts(
    dummy_simulator, default_optimizer
):
    """One alpha electron in two spatial orbitals leaves a single excitation;
    the closed-shell split of one electron would be rejected."""
    vqe = VQE(
        HamiltonianProblem(
            SparsePauliOp(["ZIII", "IZII", "IIZI", "IIIZ"]), n_alpha=1, n_beta=0
        ),
        ansatz=UCCSDAnsatz(),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    assert vqe._parameter_frequencies() == [(1.0, 2)]


@pytest.mark.parametrize("ansatz_obj", **ANSAETZE_TO_TEST)
@pytest.mark.parametrize("n_layers", [1, 2])
def test_meta_circuit_qasm(
    ansatz_obj, n_layers, h2_problem, dummy_simulator, default_optimizer
):
    """Test the QASM representation of the meta circuits."""
    vqe_problem = VQE(
        h2_problem,
        ansatz=ansatz_obj,
        n_layers=n_layers,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    meta_circuit_obj = vqe_problem.cost_circuit
    # Parameters are stored as Qiskit ParameterVector elements named "w_i[j]".
    pattern = r"w_(\d+)\[(\d+)\]"
    matches = [re.match(pattern, p.name).groups() for p in meta_circuit_obj.parameters]

    total_params = vqe_problem.n_layers * vqe_problem.n_params_per_layer
    assert len(set(matches)) == total_params
    assert len(set(matches)) // n_layers == ansatz_obj.n_params_per_layer(
        vqe_problem.n_qubits, n_electrons=h2_problem.n_electrons
    )


def test_vqe_correct_circuits_count_and_energies(
    optimizer, dummy_simulator, h2_problem
):
    """Test circuit counts and energy calculations after a VQE run."""
    vqe_problem = VQE(
        h2_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=1,
        backend=dummy_simulator,
    )

    vqe_problem.run()
    verify_correct_circuit_count(vqe_problem)


def test_vqe_lucj_ansatz_runs_to_completion(
    default_test_simulator, default_optimizer, h2_problem
):
    """Regression test: VQE(LUCJAnsatz()) must complete a run.

    LUCJAnsatz emits ``xx_plus_yy``/``rzz`` gates outside the QASM2 body
    emitter's basis; VQE's cost-circuit builder used to hand the DAG to the
    emitter unlowered, raising ``ValueError`` at circuit submission.
    """
    vqe_problem = VQE(
        h2_problem,
        ansatz=LUCJAnsatz(),
        n_layers=1,
        optimizer=default_optimizer,
        max_iterations=2,
        backend=default_test_simulator,
    )

    vqe_problem.run()

    assert len(vqe_problem.losses_history) == 2
    assert np.isfinite(vqe_problem.best_loss)


_H2_FCI_ENERGY = -1.1361891625218803


def _assert_h2_ground_energy(energy):
    """Variational, and within 1e-4 Ha of the exact ground energy."""
    assert _H2_FCI_ENERGY - 1e-9 <= energy <= _H2_FCI_ENERGY + 1e-4


@pytest.mark.e2e
def test_vqe_h2_molecule_e2e_solution(optimizer, default_test_simulator, h2_problem):
    """Test that VQE finds the correct ground state for the H2 molecule."""

    default_test_simulator.set_seed(1997)

    vqe_problem = VQE(
        h2_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=optimizer,
        max_iterations=120,
        early_stopping=EarlyStopping(patience=20, min_delta=1e-6),
        backend=default_test_simulator,
        seed=1997,
    )

    vqe_problem.run()

    assert 1 <= len(vqe_problem.losses_history) <= 120

    assert isinstance(vqe_problem.best_loss, float)
    assert isinstance(vqe_problem.best_params, np.ndarray)
    assert vqe_problem.best_params.shape == (
        vqe_problem.n_layers * vqe_problem.n_params_per_layer,
    )

    _assert_h2_ground_energy(vqe_problem.best_loss)
    expected_eigenstate = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(vqe_problem.eigenstate, expected_eigenstate)


@pytest.mark.e2e
def test_vqe_h2_molecule_e2e_checkpointing_resume(
    checkpointing_optimizer, default_test_simulator, h2_problem, tmp_path
):
    """Test VQE e2e with checkpointing and multiple resume cycles.

    Tests checkpoint infrastructure (multiple save/load cycles) with all checkpointing-capable
    optimizers to verify their nuanced checkpoint handling (CMAES generator reinit, DE pop handling).
    """
    checkpoint_dir = tmp_path / "checkpoint_test"
    default_test_simulator.set_seed(1997)

    # First run: iterations 1-2
    vqe_problem1 = VQE(
        h2_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=checkpointing_optimizer,
        max_iterations=2,
        backend=default_test_simulator,
        seed=1997,
    )
    vqe_problem1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert vqe_problem1.current_iteration == 2

    # Verify checkpoint was created
    checkpoint_path = checkpoint_dir / "checkpoint_002"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "program_state.json").exists()

    # Store state from first run for comparison
    first_run_iteration = vqe_problem1.current_iteration
    first_run_losses_count = len(vqe_problem1.losses_history)
    first_run_best_loss = vqe_problem1.best_loss

    # Second run: resume and run iterations 3-4
    vqe_problem2 = VQE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        problem=h2_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
    )

    # Verify loaded state matches first run
    assert vqe_problem2.current_iteration == first_run_iteration
    assert len(vqe_problem2.losses_history) == first_run_losses_count
    assert vqe_problem2.best_loss == pytest.approx(first_run_best_loss)

    vqe_problem2.max_iterations = 4
    vqe_problem2.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert vqe_problem2.current_iteration == 4
    assert (checkpoint_dir / "checkpoint_004").exists()

    # Third run: resume and run to convergence
    vqe_problem3 = VQE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        problem=h2_problem,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
    )
    assert vqe_problem3.current_iteration == 4
    vqe_problem3.max_iterations = 60
    vqe_problem3.run()
    assert vqe_problem3.current_iteration == 60

    # Verify final results are correct
    assert len(vqe_problem3.losses_history) == 60
    assert isinstance(vqe_problem3.best_loss, float)
    assert isinstance(vqe_problem3.best_params, np.ndarray)
    assert vqe_problem3.best_params.shape == (
        vqe_problem3.n_layers * vqe_problem3.n_params_per_layer,
    )

    _assert_h2_ground_energy(vqe_problem3.best_loss)
    expected_eigenstate = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(vqe_problem3.eigenstate, expected_eigenstate)


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, four_qubit_hamiltonian, dummy_simulator, default_optimizer):
        def _make(**kwargs):
            return VQE(
                HamiltonianProblem(four_qubit_hamiltonian, n_electrons=2),
                ansatz=HartreeFockAnsatz(),
                backend=dummy_simulator,
                optimizer=default_optimizer,
                **kwargs,
            )

        return _make
