# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import CircuitTag
from divi.qprog import PCE, MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from tests.qprog.qprog_contracts import verify_metacircuit_dict

pytestmark = pytest.mark.algo


@pytest.fixture
def basic_ansatz() -> GenericLayerAnsatz:
    return GenericLayerAnsatz([qml.RY, qml.RZ])


@pytest.fixture
def qubo_identity() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def qubo_small() -> np.ndarray:
    return np.array(
        [
            [1.0, -0.2, 0.0],
            [-0.2, 0.5, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def test_pce_basic_initialization(dummy_simulator, basic_ansatz):
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
    )

    assert pce.n_vars == 2
    assert pce.n_qubits == 2  # ceil(log2(2 + 1)) = 2
    assert pce.alpha == 2.0
    verify_metacircuit_dict(pce, ["cost_circuit", "meas_circuit"])


def test_pce_n_qubits_validation_and_warning(dummy_simulator, basic_ansatz):
    qubo = np.zeros((3, 3))

    with pytest.raises(ValueError, match=r"n_qubits must be >= ceil\(log2\(N \+ 1\)\)"):
        PCE(
            qubo_matrix=qubo,
            n_qubits=1,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
        )

    with pytest.warns(UserWarning, match="n_qubits exceeds the minimum required"):
        PCE(
            qubo_matrix=qubo,
            n_qubits=3,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
        )


def test_pce_soft_energy_post_process_results(dummy_simulator, basic_ansatz):
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=1.0,
    )

    results = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "00": 30,
            "01": 10,
        },
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=1, meas_id=0): {
            "10": 20,
            "11": 40,
        },
    }

    losses = pce._post_process_results(results)

    total_shots = 100
    mean_parities = np.array([(10 + 40) / total_shots, (20 + 40) / total_shots])
    z_expectations = 1.0 - (2.0 * mean_parities)
    x_soft = 0.5 * (1.0 + np.tanh(pce.alpha * z_expectations))
    expected = float(x_soft @ qubo @ x_soft)

    assert losses[0] == pytest.approx(expected)


def test_pce_hard_cvar_energy_post_process_results(dummy_simulator, basic_ansatz):
    qubo = np.diag([1.0, 2.0])

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=6.0,
    )

    results = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "11": 2,
            "10": 3,
            "01": 10,
            "00": 25,
        }
    }
    losses = pce._post_process_results(results)

    energies = []
    counts = []
    for bitstring, count in next(iter(results.values())).items():
        lsb = int(bitstring[-1])
        msb = int(bitstring[-2])
        x = np.array([1 - lsb, 1 - msb], dtype=float)
        energies.append(float(x @ qubo @ x))
        counts.append(float(count))

    order = np.argsort(energies)
    sorted_energies = np.array(energies)[order]
    sorted_counts = np.array(counts)[order]

    cutoff_count = int(np.ceil(0.25 * sum(counts)))
    accumulated_counts = np.cumsum(sorted_counts)
    limit_idx = int(np.searchsorted(accumulated_counts, cutoff_count))

    cvar_energy = 0.0
    count_sum = 0.0
    if limit_idx > 0:
        cvar_energy += np.sum(sorted_energies[:limit_idx] * sorted_counts[:limit_idx])
        count_sum += np.sum(sorted_counts[:limit_idx])

    remaining = cutoff_count - count_sum
    cvar_energy += sorted_energies[limit_idx] * remaining
    expected = float(cvar_energy / cutoff_count)

    assert losses[0] == pytest.approx(expected)


def test_pce_perform_final_computation_sets_solution(
    mocker, dummy_simulator, basic_ansatz, qubo_identity
):

    pce = PCE(
        qubo_matrix=qubo_identity,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    pce._best_probs = {"0_NoMitigation:0_0": {"01": 1.0}}
    mocker.patch.object(pce, "_run_solution_measurement")

    pce._perform_final_computation()

    np.testing.assert_array_equal(pce.solution, np.array([0, 1]))


def test_pce_solution_requires_run(dummy_simulator, basic_ansatz, qubo_identity):
    pce = PCE(
        qubo_matrix=qubo_identity,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    with pytest.raises(RuntimeError, match="Run the VQE optimization first."):
        _ = pce.solution


@pytest.mark.e2e
def test_pce_qubo_e2e_solution(default_test_simulator, basic_ansatz, qubo_small):
    default_test_simulator.set_seed(1997)

    pce = PCE(
        qubo_matrix=qubo_small,
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=5,
        backend=default_test_simulator,
        seed=1997,
    )

    pce.run()

    assert len(pce.losses_history) == 5
    assert isinstance(pce.best_loss, float)
    assert isinstance(pce.best_params, np.ndarray)
    assert pce.solution.shape == (qubo_small.shape[0],)
    assert set(np.unique(pce.solution)).issubset({0, 1})
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


@pytest.mark.e2e
def test_pce_qubo_e2e_checkpointing_resume(
    default_test_simulator, basic_ansatz, qubo_small, tmp_path
):
    default_test_simulator.set_seed(1997)
    checkpoint_dir = tmp_path / "checkpoint_test"

    pce1 = PCE(
        qubo_matrix=qubo_small,
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=MonteCarloOptimizer(population_size=5, n_best_sets=2),
        max_iterations=2,
        backend=default_test_simulator,
        seed=1997,
    )
    pce1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert pce1.current_iteration == 2

    checkpoint_path = checkpoint_dir / "checkpoint_002"
    assert checkpoint_path.exists()
    assert (checkpoint_path / "program_state.json").exists()

    first_run_iteration = pce1.current_iteration
    first_run_losses_count = len(pce1.losses_history)

    pce2 = PCE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        qubo_matrix=qubo_small,
        ansatz=basic_ansatz,
        n_layers=1,
    )
    assert pce2.current_iteration == first_run_iteration
    assert len(pce2.losses_history) == first_run_losses_count

    pce2.max_iterations = 4
    pce2.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
    assert pce2.current_iteration == 4
    assert (checkpoint_dir / "checkpoint_004").exists()

    pce3 = PCE.load_state(
        checkpoint_dir,
        backend=default_test_simulator,
        qubo_matrix=qubo_small,
        ansatz=basic_ansatz,
        n_layers=1,
    )
    assert pce3.current_iteration == 4
    pce3.max_iterations = 5
    pce3.run()
    assert pce3.current_iteration == 5

    assert isinstance(pce3.best_loss, float)
    assert pce3.solution.shape == (qubo_small.shape[0],)


def test_pce_get_top_solutions_decodes_bitstrings(dummy_simulator, basic_ansatz):
    """Test that get_top_solutions decodes encoded qubit states to QUBO variable assignments."""
    # Create a 4-variable QUBO (requires 3 qubits: ceil(log2(4+1)) = 3)
    qubo = np.eye(4)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    # Set up _best_probs with encoded qubit states
    # For 4 variables, we need 3 qubits
    # Encoded state "000" (0) -> parities for vars [1,2,3,4] -> decoded solution
    # Encoded state "001" (1) -> parities for vars [1,2,3,4] -> decoded solution
    # etc.
    pce._best_probs = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "000": 0.5,  # Encoded qubit state
            "001": 0.3,
            "010": 0.15,
            "011": 0.05,
        }
    }
    pce._losses_history = [{0: -1.0}]

    # Get top solutions
    solutions = pce.get_top_solutions(n=4)

    # Verify all solutions are decoded (length should be n_vars=4, not n_qubits=3)
    assert len(solutions) == 4
    for sol in solutions:
        assert len(sol.bitstring) == pce.n_vars  # Decoded QUBO solution length
        assert all(c in "01" for c in sol.bitstring)  # Binary string
        assert 0.0 <= sol.prob <= 1.0
        assert sol.decoded is None  # include_decoded=False by default

    # Verify sorting by probability (descending)
    assert solutions[0].prob == 0.5
    assert solutions[1].prob == 0.3
    assert solutions[2].prob == 0.15
    assert solutions[3].prob == 0.05


def test_pce_get_top_solutions_include_decoded(dummy_simulator, basic_ansatz):
    """Test that get_top_solutions with include_decoded=True returns numpy arrays."""
    qubo = np.eye(3)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    # Set up _best_probs with encoded qubit states (2 qubits for 3 variables)
    pce._best_probs = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "00": 0.6,
            "01": 0.4,
        }
    }
    pce._losses_history = [{0: -1.0}]

    solutions = pce.get_top_solutions(n=2, include_decoded=True)

    assert len(solutions) == 2
    for sol in solutions:
        assert len(sol.bitstring) == pce.n_vars  # Decoded length
        assert sol.decoded is not None
        assert isinstance(sol.decoded, np.ndarray)
        assert sol.decoded.shape == (pce.n_vars,)
        assert sol.decoded.dtype == np.int32
        # Verify decoded array matches bitstring
        expected_array = np.array([int(c) for c in sol.bitstring], dtype=np.int32)
        np.testing.assert_array_equal(sol.decoded, expected_array)


def test_pce_get_top_solutions_min_prob_filtering(dummy_simulator, basic_ansatz):
    """Test that get_top_solutions filters by min_prob correctly."""
    qubo = np.eye(2)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    pce._best_probs = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "00": 0.5,
            "01": 0.3,
            "10": 0.15,
            "11": 0.05,
        }
    }
    pce._losses_history = [{0: -1.0}]

    # Filter by min_prob
    solutions = pce.get_top_solutions(n=10, min_prob=0.2)

    # Only solutions with prob >= 0.2 should be included
    assert len(solutions) == 2
    assert solutions[0].prob == 0.5
    assert solutions[1].prob == 0.3
    assert all(sol.prob >= 0.2 for sol in solutions)


def test_pce_get_top_solutions_decoding_correctness(dummy_simulator, basic_ansatz):
    """Test that decoding produces correct QUBO variable assignments."""
    # For a 2-variable QUBO (2 qubits)
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )

    # Manually verify decoding:
    # Encoded "00" (0): variable masks [1, 2] & 0 = [0, 0], parities = [0, 0], decoded = [1, 1]
    # Encoded "01" (1): variable masks [1, 2] & 1 = [1, 0], parities = [1, 0], decoded = [0, 1]
    # Encoded "10" (2): variable masks [1, 2] & 2 = [0, 2], parities = [0, 1], decoded = [1, 0]
    # Encoded "11" (3): variable masks [1, 2] & 3 = [1, 2], parities = [1, 1], decoded = [0, 0]

    pce._best_probs = {
        CircuitTag(param_id=0, qem_name="NoMitigation", qem_id=0, meas_id=0): {
            "00": 0.4,  # Should decode to "11"
            "01": 0.3,  # Should decode to "01"
            "10": 0.2,  # Should decode to "10"
            "11": 0.1,  # Should decode to "00"
        }
    }
    pce._losses_history = [{0: -1.0}]

    solutions = pce.get_top_solutions(n=4)

    # Verify decoded solutions match expected values
    decoded_bitstrings = [sol.bitstring for sol in solutions]
    assert "11" in decoded_bitstrings  # From "00"
    assert "01" in decoded_bitstrings  # From "01"
    assert "10" in decoded_bitstrings  # From "10"
    assert "00" in decoded_bitstrings  # From "11"

    # Verify all decoded solutions have correct length
    assert all(len(sol.bitstring) == 2 for sol in solutions)
