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

from divi.backends import MaestroConfig
from divi.hamiltonians._rdm import sum_spin_rdms
from divi.qprog import EarlyStopping
from divi.qprog.algorithms import (
    VQE,
    ElectronCounts,
    GenericLayerAnsatz,
    HartreeFockAnsatz,
    LUCJAnsatz,
    QAOAAnsatz,
    QCCAnsatz,
    UCCSDAnsatz,
    UpCCDAnsatz,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.initial_states import SuperpositionState, ZerosState
from divi.qprog.optimizers import (
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import HamiltonianProblem, MolecularProblem
from divi.reporting._events import EventKind, TerminalStatus
from tests._helpers import WATER_ATOM, exact_match, run_mean_field
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
        (UpCCDAnsatz(), True),
        (QCCAnsatz(), False),
        (GenericLayerAnsatz([RYGate, RZGate]), False),
    ],
    ids=["HartreeFock", "UCCSD", "UpCCD", "QCC", "Generic-RYRZ"],
)
def test_excitation_ansatze_start_at_the_hartree_fock_point(
    dummy_simulator,
    four_qubit_hamiltonian,
    ansatz,
    starts_at_reference,
):
    """All-zero excitation amplitudes prepare the Hartree-Fock reference; the
    other sets of a population spread around it."""
    vqe = VQE(
        HamiltonianProblem(four_qubit_hamiltonian, n_electrons=2),
        ansatz=ansatz,
        backend=dummy_simulator,
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=4),
        seed=1997,
    )

    params = vqe._initialize_param_sets()

    assert params.shape == vqe.get_expected_param_shape()
    assert bool(np.all(params[0] == 0.0)) is starts_at_reference
    assert len(np.unique(params, axis=0)) == len(params)


def test_vqe_cost_includes_the_hamiltonian_constant(
    four_qubit_hamiltonian, default_test_simulator, default_optimizer
):
    def evaluate_cost(hamiltonian):
        vqe = VQE(
            HamiltonianProblem(hamiltonian, n_electrons=2),
            ansatz=HartreeFockAnsatz(),
            backend=default_test_simulator,
            optimizer=default_optimizer,
        )
        params = np.full((1, vqe.n_params), 0.3)
        (value,) = vqe.evaluate(params, vqe.cost_preprocessor()).values()
        return value[0]

    shifted = four_qubit_hamiltonian + SparsePauliOp("IIII", 5.0)

    assert evaluate_cost(shifted) - evaluate_cost(four_qubit_hamiltonian) == (
        pytest.approx(5.0, abs=1e-12)
    )


def test_vqe_with_a_constant_only_hamiltonian_trains_to_the_constant(
    default_test_simulator, default_optimizer
):
    vqe = VQE(
        HamiltonianProblem(5.0 * SparsePauliOp("IIII"), n_electrons=2),
        ansatz=HartreeFockAnsatz(),
        backend=default_test_simulator,
        optimizer=default_optimizer,
        max_iterations=1,
    ).run()

    assert vqe.best_loss == 5.0


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


def test_reassigning_the_cost_hamiltonian_rebuilds_the_cost_circuit(
    dummy_simulator, default_optimizer
):
    vqe = _two_qubit_ry_vqe(dummy_simulator, default_optimizer)
    old_circuit = vqe.cost_circuit
    replacement = SparsePauliOp(["XI", "IX"])

    vqe.cost_hamiltonian = replacement

    assert vqe.cost_circuit is not old_circuit
    assert vqe.cost_circuit.observable[0].equiv(replacement)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            SparsePauliOp.from_list([("XI", 1.0j)]),
            "SparsePauliOp observables must be Hermitian; Pauli coefficients "
            "must be real.",
        ),
        (
            SparsePauliOp(["XII"]),
            "cost_hamiltonian.num_qubits is 3 but the program has 2 qubits; "
            "they must match.",
        ),
    ],
    ids=["non-hermitian", "wrong-width"],
)
def test_assigning_an_invalid_cost_hamiltonian_raises(
    dummy_simulator, default_optimizer, replacement, message
):
    vqe = _two_qubit_ry_vqe(dummy_simulator, default_optimizer)

    with pytest.raises(ValueError, match=exact_match(message)):
        vqe.cost_hamiltonian = replacement


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

    assert vqe.parameter_space.frequencies == ((1.0, 2),)


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
        vqe_problem.n_qubits, ElectronCounts(h2_problem.n_electrons)
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


_H4_CHAIN = "H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7"

#: Problems whose measured RDMs are checked against the cost energy.
_RDM_PROBLEMS = {
    "h4-chain": lambda request: MolecularProblem.from_casci(
        run_mean_field(_H4_CHAIN), 4, 4
    ),
    "water-triplet": lambda request: MolecularProblem.from_casci(
        run_mean_field(WATER_ATOM, spin=2), 4, 4
    ),
    "pennylane-h2": lambda request: request.getfixturevalue("h2_problem"),
}


def _build_uccsd_vqe(problem, backend, optimizer, **kwargs):
    return VQE(
        problem, ansatz=UCCSDAnsatz(), backend=backend, optimizer=optimizer, **kwargs
    )


def _rdm_energy(problem, rdms):
    dm1, dm2 = sum_spin_rdms(rdms)
    return (
        problem.constant
        + np.einsum("pq,pq", problem.one_body, dm1)
        + 0.5 * np.einsum("pqrs,pqrs", problem.two_body, dm2)
    )


@needs_qiskit_nature
@pytest.mark.parametrize(
    "build_problem", _RDM_PROBLEMS.values(), ids=_RDM_PROBLEMS.keys()
)
def test_measured_rdms_reproduce_the_cost_energy(
    request, build_problem, default_test_simulator, default_optimizer
):
    problem = build_problem(request)
    vqe = _build_uccsd_vqe(problem, default_test_simulator, default_optimizer)
    params = np.random.default_rng(3).uniform(-0.2, 0.2, vqe.n_params)

    (cost,) = vqe.evaluate(params, vqe.cost_preprocessor()).values()
    rdms = vqe.measure_rdms(params)

    assert _rdm_energy(problem, rdms) == pytest.approx(cost[0], abs=1e-9)
    (dm1a, dm1b), _ = rdms
    assert np.trace(dm1a) == pytest.approx(problem.n_alpha, abs=1e-9)
    assert np.trace(dm1b) == pytest.approx(problem.n_beta, abs=1e-9)


@needs_qiskit_nature
def test_sampled_rdms_approach_the_exact_ones(
    make_maestro_simulator, default_test_simulator, default_optimizer
):
    problem = MolecularProblem.from_casci(run_mean_field(_H4_CHAIN), 4, 4)
    sampler = make_maestro_simulator(
        shots=4000, force_sampling=True, maestro_config=MaestroConfig(seed=1997)
    )
    exact_vqe = _build_uccsd_vqe(problem, default_test_simulator, default_optimizer)
    params = np.random.default_rng(3).uniform(-0.2, 0.2, exact_vqe.n_params)

    exact = exact_vqe.measure_rdms(params)
    sampled = _build_uccsd_vqe(problem, sampler, default_optimizer).measure_rdms(params)

    (dm1a, dm1b), _ = sampled
    assert np.trace(dm1a) == pytest.approx(problem.n_alpha, abs=0.1)
    assert np.trace(dm1b) == pytest.approx(problem.n_beta, abs=0.1)
    assert _rdm_energy(problem, sampled) == pytest.approx(
        _rdm_energy(problem, exact), abs=0.2
    )


@needs_qiskit_nature
def test_trained_rdms_match_pyscf_fci_for_two_electrons(default_test_simulator):
    mcscf = pytest.importorskip("pyscf.mcscf")
    mean_field = run_mean_field("H 0 0 0; H 0 0 0.74", basis="6-31g")
    vqe = _build_uccsd_vqe(
        MolecularProblem.from_casci(mean_field, 2, 2),
        default_test_simulator,
        ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        max_iterations=50,
    )
    vqe.run(perform_final_computation=False)
    casci = mcscf.CASCI(mean_field, 2, 2)
    casci.verbose = 0
    casci.kernel()

    (dm1s, dm2s) = vqe.measure_rdms()

    (ref1s, ref2s) = casci.fcisolver.make_rdm12s(casci.ci, 2, (1, 1))
    for measured, reference in zip((*dm1s, *dm2s), (*ref1s, *ref2s), strict=True):
        np.testing.assert_allclose(measured, reference, atol=1e-6)


def test_measure_rdms_without_trained_or_explicit_params_raises(
    h2_problem, dummy_simulator, default_optimizer
):
    vqe = VQE(h2_problem, backend=dummy_simulator, optimizer=default_optimizer)

    with pytest.raises(
        RuntimeError,
        match=exact_match(
            "No trained parameters are available. Pass explicit `params` or call "
            "run() first."
        ),
    ):
        vqe.measure_rdms()


def test_measure_rdms_takes_one_parameter_vector(
    h2_problem, dummy_simulator, default_optimizer
):
    vqe = VQE(h2_problem, backend=dummy_simulator, optimizer=default_optimizer)

    with pytest.raises(
        ValueError,
        match=exact_match("measure_rdms takes one parameter vector; got 2 rows."),
    ):
        vqe.measure_rdms(np.zeros((2, vqe.n_params)))


def test_measure_rdms_needs_a_molecular_problem(
    four_qubit_hamiltonian, dummy_simulator, default_optimizer
):
    vqe = VQE(
        HamiltonianProblem(four_qubit_hamiltonian, n_electrons=2),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    with pytest.raises(
        TypeError,
        match=exact_match(
            "measure_rdms needs a MolecularProblem; got HamiltonianProblem."
        ),
    ):
        vqe.measure_rdms(np.zeros(vqe.n_params))


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
