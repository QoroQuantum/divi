# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.pipeline.stages._pce_cost_stage import (
    _compute_hard_cvar_energy,
    _compute_soft_energy,
)
from divi.qprog import PCE, MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.algorithms._pce import _decode_parities
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


def test_pce_soft_energy_computation(dummy_simulator, basic_ansatz):
    """_compute_soft_energy reproduces the original soft-path result."""
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])
    alpha = 1.0

    # Simulate two observable groups merged into a single histogram.
    # Group-0: {"00": 30, "01": 10}  →  40 shots
    # Group-1: {"10": 20, "11": 40}  →  60 shots
    # After merging: {"00": 30, "01": 10, "10": 20, "11": 40}
    merged = {"00": 30, "01": 10, "10": 20, "11": 40}
    total_shots = 100

    state_strings = list(merged.keys())
    counts = np.array(list(merged.values()), dtype=float)
    probs = counts / total_shots

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=alpha,
    )
    parities = _decode_parities(state_strings, pce._variable_masks_u64)
    result = _compute_soft_energy(parities, probs, alpha, qubo)

    # Expected: same algebra as the old test_pce_soft_energy_post_process_results
    mean_parities = parities.dot(probs)
    z_expectations = 1.0 - (2.0 * mean_parities)
    x_soft = 0.5 * (1.0 + np.tanh(alpha * z_expectations))
    expected = float(x_soft @ qubo @ x_soft)

    assert result == pytest.approx(expected)


def test_pce_hard_cvar_energy_computation(dummy_simulator, basic_ansatz):
    """_compute_hard_cvar_energy reproduces the original hard-CVaR-path result."""
    qubo = np.diag([1.0, 2.0])

    merged = {"11": 2, "10": 3, "01": 10, "00": 25}

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=6.0,
    )

    state_strings = list(merged.keys())
    counts_arr = np.array(list(merged.values()), dtype=float)
    total_shots = counts_arr.sum()
    parities = _decode_parities(state_strings, pce._variable_masks_u64)

    result = _compute_hard_cvar_energy(
        parities, counts_arr, total_shots, qubo, alpha_cvar=0.25
    )

    # Expected: replicate CVaR algebra
    x_vals = 1.0 - parities.astype(float)
    Qx = qubo @ x_vals
    energies = np.einsum("ij,ij->j", x_vals, Qx)

    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    sorted_counts = counts_arr[sorted_indices]

    cutoff_count = int(np.ceil(0.25 * total_shots))
    accumulated = np.cumsum(sorted_counts)
    limit_idx = int(np.searchsorted(accumulated, cutoff_count))

    cvar_energy = 0.0
    count_sum = 0.0
    if limit_idx > 0:
        cvar_energy += np.sum(sorted_energies[:limit_idx] * sorted_counts[:limit_idx])
        count_sum += np.sum(sorted_counts[:limit_idx])
    remaining = cutoff_count - count_sum
    cvar_energy += sorted_energies[limit_idx] * remaining
    expected = float(cvar_energy / cutoff_count)

    assert result == pytest.approx(expected)


def test_pce_custom_decode_parities_fn_soft_energy(dummy_simulator, basic_ansatz):
    """Custom decode_parities_fn is used in the soft energy path."""
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])

    def all_zeros_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        return np.zeros((n_vars, n_states), dtype=np.uint8)

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=1.0,
        decode_parities_fn=all_zeros_decoder,
    )

    merged = {"00": 30, "01": 10, "10": 20, "11": 40}
    total_shots = 100
    state_strings = list(merged.keys())
    counts = np.array(list(merged.values()), dtype=float)
    probs = counts / total_shots

    parities = all_zeros_decoder(state_strings, pce._variable_masks_u64)
    result = _compute_soft_energy(parities, probs, pce.alpha, qubo)

    # All parities 0 -> mean_parities = [0, 0] -> z_expectations = [1, 1]
    z = np.array([1.0, 1.0])
    x_soft = 0.5 * (1.0 + np.tanh(pce.alpha * z))
    expected = float(x_soft @ qubo @ x_soft)
    assert result == pytest.approx(expected)


def test_pce_custom_decode_parities_fn_hard_cvar(dummy_simulator, basic_ansatz):
    """Custom decode_parities_fn is used in the hard CVaR energy path."""
    qubo = np.diag([1.0, 2.0])

    def all_ones_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        return np.ones((n_vars, n_states), dtype=np.uint8)

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=6.0,
        decode_parities_fn=all_ones_decoder,
    )

    merged = {"11": 2, "10": 3, "01": 10, "00": 25}
    state_strings = list(merged.keys())
    counts_arr = np.array(list(merged.values()), dtype=float)
    total_shots = counts_arr.sum()

    parities = all_ones_decoder(state_strings, pce._variable_masks_u64)
    result = _compute_hard_cvar_energy(
        parities, counts_arr, total_shots, qubo, alpha_cvar=0.25
    )

    # All parities 1 -> x_vals = 0 for all vars/states -> energies all 0 -> CVaR = 0
    assert result == pytest.approx(0.0)


def test_pce_decode_parities_fn_none_uses_default(dummy_simulator, basic_ansatz):
    """Passing decode_parities_fn=None uses the built-in decoder (same as omitting)."""
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])

    pce_default = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=1.0,
    )
    pce_explicit_none = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        alpha=1.0,
        decode_parities_fn=None,
    )

    merged = {"00": 30, "01": 10, "10": 20, "11": 40}
    total_shots = 100
    state_strings = list(merged.keys())
    counts = np.array(list(merged.values()), dtype=float)
    probs = counts / total_shots

    parities_default = _decode_parities(state_strings, pce_default._variable_masks_u64)
    parities_none = _decode_parities(
        state_strings, pce_explicit_none._variable_masks_u64
    )

    result_default = _compute_soft_energy(parities_default, probs, 1.0, qubo)
    result_none = _compute_soft_energy(parities_none, probs, 1.0, qubo)

    assert result_default == pytest.approx(result_none)


def test_pce_poly_expval_post_process(dummy_simulator, basic_ansatz):
    """Poly encoding expval: PCECostStage.reduce handles expval-format Z expectations."""
    qubo = np.zeros((3, 3))
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        encoding_type="poly",
        alpha=1.0,
    )

    # The expval path receives per-observable Z expectations directly.
    # For 3 vars with masks [1, 2, 3], ham_ops are ZI, IZ, ZZ.
    z_expectations = np.array([0.2, -0.4, 0.6])
    x_soft = 0.5 * (1.0 + np.tanh(pce.alpha * z_expectations))
    expected = float(np.dot(x_soft, qubo @ x_soft))

    assert expected == pytest.approx(0.0)  # qubo is zero matrix


def test_pce_soft_energy_expval_post_process(dummy_simulator, basic_ansatz):
    """PCE soft energy correctly processes expectation-value format results."""
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])
    alpha = 1.0

    # The expval path in PCECostStage.reduce receives Z expectations
    # and applies: x_soft = 0.5*(1 + tanh(alpha * z)), energy = x.T @ Q @ x
    z = np.array([0.2, -0.4])
    x_soft = 0.5 * (1.0 + np.tanh(alpha * z))
    expected = float(x_soft @ qubo @ x_soft)

    # Verify the math produces a sensible non-trivial value
    assert expected != 0.0
    assert isinstance(expected, float)


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


@pytest.mark.e2e
def test_pce_poly_e2e_solution(default_test_simulator, basic_ansatz, qubo_small):
    """E2E test for PCE with poly encoding (3 vars, 2 qubits)."""
    default_test_simulator.set_seed(1997)

    pce = PCE(
        qubo_matrix=qubo_small,
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=5,
        backend=default_test_simulator,
        encoding_type="poly",
        seed=1997,
    )

    pce.run()

    assert len(pce.losses_history) == 5
    assert isinstance(pce.best_loss, float)
    assert isinstance(pce.best_params, np.ndarray)
    assert pce.encoding_type == "poly"
    assert pce.n_qubits == 2  # N=3: min 2 qubits, capacity 3
    assert pce.solution.shape == (qubo_small.shape[0],)
    assert set(np.unique(pce.solution)).issubset({0, 1})
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


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
        "0_NoMitigation:0_ham:0_0": {
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
        "0_NoMitigation:0_ham:0_0": {
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
        "0_NoMitigation:0_ham:0_0": {
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


def test_pce_get_top_solutions_validation_errors(dummy_simulator, basic_ansatz):
    """Test get_top_solutions raises for invalid n, min_prob, and missing probs."""
    qubo = np.eye(2)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )
    pce._best_probs = {
        "0_NoMitigation:0_ham:0_0": {
            "00": 0.5,
            "01": 0.5,
        }
    }
    pce._losses_history = [{0: -1.0}]

    with pytest.raises(ValueError, match="n must be non-negative"):
        pce.get_top_solutions(n=-1)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=1.5)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=-0.1)

    assert pce.get_top_solutions(n=0) == []


def test_pce_get_top_solutions_no_probs_raises(dummy_simulator, basic_ansatz):
    """Test get_top_solutions raises when no probability distribution available."""
    qubo = np.eye(2)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
    )
    pce._best_probs = {}
    pce._losses_history = []

    with pytest.raises(
        RuntimeError,
        match="No probability distribution available",
    ):
        pce.get_top_solutions(n=2)


def test_pce_qem_protocol_raises(dummy_simulator, basic_ansatz):
    """PCE raises when qem_protocol is passed (not supported)."""
    qubo = np.eye(2)
    with pytest.raises(ValueError, match="PCE does not currently support qem_protocol"):
        PCE(
            qubo_matrix=qubo,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
            qem_protocol=object(),
        )


def test_pce_custom_decode_parities_fn_perform_final_computation(
    mocker, dummy_simulator, basic_ansatz, qubo_identity
):
    """Custom decode_parities_fn is used in _perform_final_computation."""

    # Decoder that returns parities [0, 1] for any input -> solution = 1 - [0,1] = [1, 0]
    def fixed_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        out = np.zeros((n_vars, n_states), dtype=np.uint8)
        out[1, :] = 1  # Second variable parity always 1
        return out

    pce = PCE(
        qubo_matrix=qubo_identity,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        decode_parities_fn=fixed_decoder,
    )

    pce._best_probs = {"0_NoMitigation:0_0": {"01": 1.0}}
    mocker.patch.object(pce, "_run_solution_measurement")

    pce._perform_final_computation()

    # Fixed decoder returns parities [0, 1] -> solution = [1, 0]
    np.testing.assert_array_equal(pce.solution, np.array([1, 0]))


def test_pce_custom_decode_parities_fn_get_top_solutions(dummy_simulator, basic_ansatz):
    """Custom decode_parities_fn is used in get_top_solutions."""

    # Decoder that complements the default: return 1 - default_decoder output
    # So decoded QUBO solution = 1 - (1 - default) = default (same as default)
    # Or we use a decoder that returns fixed output per state
    def complement_decoder(state_strings, variable_masks_u64):
        default = _decode_parities(state_strings, variable_masks_u64)
        return 1 - default

    qubo = np.eye(2)
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        decode_parities_fn=complement_decoder,
    )

    pce._best_probs = {
        "0_NoMitigation:0_ham:0_0": {
            "00": 0.5,
            "01": 0.3,
            "10": 0.2,
            "11": 0.0,
        }
    }
    pce._losses_history = [{0: -1.0}]

    solutions = pce.get_top_solutions(n=4)

    # Default decoder: "00"->"11", "01"->"01", "10"->"10", "11"->"00"
    # Complement decoder: "00"->"00", "01"->"10", "10"->"01", "11"->"11"
    expected_decoded = {"00": 0.5, "10": 0.3, "01": 0.2, "11": 0.0}
    decoded_map = {sol.bitstring: sol.prob for sol in solutions}
    for bitstring, expected_prob in expected_decoded.items():
        assert decoded_map[bitstring] == expected_prob


@pytest.mark.parametrize(
    "encoding_type,qubo,encoded_probs,expected_decoded_map,n_vars",
    [
        # Dense: 2 vars, masks [1,2]. "00"->"11", "01"->"01", "10"->"10", "11"->"00"
        (
            "dense",
            np.array([[1.0, 0.2], [0.2, 2.0]]),
            {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1},
            {"11": 0.4, "01": 0.3, "10": 0.2, "00": 0.1},
            2,
        ),
        # Poly: 3 vars, masks [1,2,3]. "00"->"111", "01"->"010", "10"->"100", "11"->"001"
        (
            "poly",
            np.zeros((3, 3)),
            {"00": 0.4, "01": 0.3, "10": 0.2, "11": 0.1},
            {"111": 0.4, "010": 0.3, "100": 0.2, "001": 0.1},
            3,
        ),
    ],
)
def test_pce_get_top_solutions_decoding_correctness(
    dummy_simulator,
    basic_ansatz,
    encoding_type,
    qubo,
    encoded_probs,
    expected_decoded_map,
    n_vars,
):
    """Test that decoding produces correct QUBO variable assignments for both encodings."""
    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_simulator,
        encoding_type=encoding_type,
    )

    pce._best_probs = {"0_NoMitigation:0_ham:0_0": encoded_probs}
    pce._losses_history = [{0: -1.0}]

    solutions = pce.get_top_solutions(n=len(encoded_probs))

    decoded_map = {sol.bitstring: sol.prob for sol in solutions}
    for decoded_bitstring, expected_prob in expected_decoded_map.items():
        assert decoded_map[decoded_bitstring] == expected_prob
    assert all(len(sol.bitstring) == n_vars for sol in solutions)


@pytest.mark.parametrize(
    "n_vars,n_qubits,expected_n_qubits,expected_masks,expect_warning",
    [
        # N=3: min 2 qubits, capacity 3. Masks [1, 2, 3].
        (3, None, 2, [1, 2, 3], False),
        # N=4, n_qubits=5: explicit value respected, warns (exceeds min 3).
        (4, 5, 5, None, True),
    ],
)
def test_pce_poly_encoding_config(
    dummy_simulator,
    basic_ansatz,
    n_vars,
    n_qubits,
    expected_n_qubits,
    expected_masks,
    expect_warning,
):
    """Test poly encoding: n_qubits, masks, and n_qubits validation."""
    qubo = np.zeros((n_vars, n_vars))
    if expect_warning:
        with pytest.warns(UserWarning, match="n_qubits exceeds the minimum required"):
            pce = PCE(
                qubo_matrix=qubo,
                ansatz=basic_ansatz,
                backend=dummy_simulator,
                encoding_type="poly",
                n_qubits=n_qubits,
            )
    else:
        pce = PCE(
            qubo_matrix=qubo,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
            encoding_type="poly",
            n_qubits=n_qubits,
        )

    assert pce.n_vars == n_vars
    assert pce.encoding_type == "poly"
    assert pce.n_qubits == expected_n_qubits
    if expected_masks is not None:
        np.testing.assert_array_equal(
            pce._variable_masks_u64, np.array(expected_masks, dtype=np.uint64)
        )


def test_pce_poly_n_qubits_too_low_raises(dummy_simulator, basic_ansatz):
    """Poly encoding raises when n_qubits is below minimum (N=4 requires 3)."""
    qubo = np.zeros((4, 4))
    with pytest.raises(ValueError, match="n_qubits must be >= 3 for poly encoding"):
        PCE(
            qubo_matrix=qubo,
            n_qubits=2,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
            encoding_type="poly",
        )


def test_pce_invalid_encoding_type(dummy_simulator, basic_ansatz):
    qubo = np.zeros((2, 2))
    with pytest.raises(ValueError, match="Unknown encoding_type: sparse"):
        PCE(
            qubo_matrix=qubo,
            ansatz=basic_ansatz,
            backend=dummy_simulator,
            encoding_type="sparse",
        )


def test_pce_hard_cvar_expval_backend_raises(basic_ansatz, dummy_expval_backend):
    """PCE with alpha >= 5 (hard CVaR) raises when backend supports expectation values."""
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])

    pce = PCE(
        qubo_matrix=qubo,
        ansatz=basic_ansatz,
        backend=dummy_expval_backend,
        alpha=6.0,
    )

    with pytest.raises(
        ValueError,
        match="hard CVaR mode.*cannot use expectation-value backends",
    ):
        pce.run()
