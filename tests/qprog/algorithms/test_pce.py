# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate

from divi.hamiltonians._polynomial import _evaluate_binary_polynomial
from divi.pipeline.stages import (
    CircuitSpecStage,
    ParameterBindingStage,
    PCECostStage,
    PreprocessStage,
)
from divi.pipeline.stages._pce_cost_stage import (
    _compute_hard_cvar_energy,
    _compute_soft_energy,
)
from divi.qprog import PCE, MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.algorithms._pce import (
    _aggregate_param_group,
    _decode_parities,
    _masks_to_ham_ops,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.problems import BinaryOptimizationProblem
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_cost_circuit,
)
from tests.qprog.problems._helpers import (
    HUBO_CUBIC,
    PCE_QUBO_MATRIX,
    PCE_QUBO_SOLUTION,
    exact_hubo_minima,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_ansatz() -> GenericLayerAnsatz:
    return GenericLayerAnsatz([RYGate, RZGate])


@pytest.fixture
def qubo_identity() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.fixture
def make_pce(basic_ansatz, dummy_simulator, default_optimizer):
    """Build a PCE from standard test defaults.

    ``problem``, ``ansatz``, ``optimizer``, and ``backend`` all have sensible
    defaults; any can be overridden per call, e.g.
    ``make_pce(backend=default_test_simulator, optimizer=ScipyOptimizer(...))``.
    """
    _default_problem = np.array([[1.0, 0.2], [0.2, 2.0]])

    def _make(**kwargs):
        problem = kwargs.pop("problem", _default_problem)
        if not isinstance(problem, BinaryOptimizationProblem):
            problem = BinaryOptimizationProblem(problem)
        kwargs["problem"] = problem
        kwargs.setdefault("ansatz", basic_ansatz)
        kwargs.setdefault("optimizer", default_optimizer)
        kwargs.setdefault("backend", dummy_simulator)
        return PCE(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _histogram_to_probs(histogram):
    """Convert a {state: count} dict to (state_strings, probs)."""
    state_strings = list(histogram.keys())
    counts = np.array(list(histogram.values()), dtype=float)
    probs = counts / counts.sum()
    return state_strings, probs


def _read_solution(pce):
    """Read pce.solution while asserting the expected deprecation warning."""
    with pytest.warns(UserWarning, match="PCE.solution returns the decoded"):
        return pce.solution


def _set_probs(pce, probs_dict, *, key="0_NoMitigation:0_ham:0_0"):
    """Set ``_best_probs`` and ``_losses_history`` on a PCE for get_top_solutions tests."""
    pce._best_probs = {key: probs_dict}
    pce._losses_history = [{0: -1.0}]


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_pce_basic_initialization(make_pce):
    pce = make_pce(
        problem=np.array([[1.0, 0.2], [0.2, 2.0]]),
        n_layers=1,
    )

    assert pce.n_vars == 2
    assert pce.n_qubits == 2  # ceil(log2(2 + 1)) = 2
    assert pce.alpha == 2.0
    verify_cost_circuit(pce)


def test_pce_hubo_basic_initialization(make_pce):
    hubo = {
        ("x0",): -1.0,
        ("x0", "x1"): 0.5,
        ("x0", "x1", "x2"): 2.0,
        (): 0.25,
    }

    pce = make_pce(problem=hubo, n_layers=1)

    assert pce.n_vars == 3
    assert pce.problem.constant == pytest.approx(0.25)
    verify_cost_circuit(pce)


def test_pce_cost_pipeline_uses_custom_stage_stack(make_pce):
    """PCE should own its counts-based cost pipeline explicitly."""
    pce = make_pce(n_layers=1)

    pipeline = pce._build_preprocessor_pipeline(pce.cost_preprocessor())
    assert [type(stage) for stage in pipeline.stages] == [
        CircuitSpecStage,
        PreprocessStage,
        PCECostStage,
        ParameterBindingStage,
    ]


def test_pce_n_qubits_validation_and_warning(make_pce):
    qubo = np.zeros((3, 3))

    with pytest.raises(ValueError, match=r"n_qubits must be >= ceil\(log2\(N \+ 1\)\)"):
        make_pce(problem=qubo, n_qubits=1)

    with pytest.warns(UserWarning, match="n_qubits exceeds the minimum required"):
        make_pce(problem=qubo, n_qubits=3)


def test_pce_default_ansatz_is_hardware_efficient_and_entangling(
    dummy_simulator, default_optimizer
):
    """PCE defaults to an entangling GenericLayerAnsatz, not VQE's chemistry ansatz.

    Regression: VQE's HartreeFockAnsatz default queries n_electrons (None for
    PCE) and raises; PCE must supply its own applicable default.
    """
    pce = PCE(
        problem=BinaryOptimizationProblem(
            np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 1.5]])
        ),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    assert isinstance(pce.ansatz, GenericLayerAnsatz)
    assert pce.ansatz.entangler is not None
    # n_params_per_layer must resolve without an n_electrons-dependent ansatz.
    assert pce.n_params_per_layer > 0

    cost_dag = pce._create_cost_circuit().circuit_bodies[0][1]
    two_qubit_ops = [n.op.name for n in cost_dag.op_nodes() if n.op.num_qubits == 2]
    assert two_qubit_ops, "default PCE ansatz must be entangling"


def test_pce_qem_protocol_raises(make_pce):
    """PCE raises when qem_protocol is passed (not supported)."""
    with pytest.raises(ValueError, match="PCE does not currently support qem_protocol"):
        make_pce(problem=np.eye(2), qem_protocol=object())


def test_pce_invalid_encoding_type(make_pce):
    with pytest.raises(ValueError, match="Unknown encoding_type: sparse"):
        make_pce(problem=np.zeros((2, 2)), encoding_type="sparse")


def test_pce_rejects_non_binary_problem():
    """PCE only accepts a BinaryOptimizationProblem; a raw QUBO raises clearly."""
    with pytest.raises(TypeError, match="requires a BinaryOptimizationProblem"):
        PCE(np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_pce_hard_cvar_expval_backend_raises(
    basic_ansatz, dummy_expval_backend, default_optimizer
):
    """PCE with alpha >= 5 (hard CVaR) raises when backend supports expectation values."""
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=basic_ansatz,
        optimizer=default_optimizer,
        backend=dummy_expval_backend,
        alpha=6.0,
    )

    with pytest.raises(
        ValueError,
        match="hard CVaR mode.*cannot use expectation-value backends",
    ):
        pce.run()


# ---------------------------------------------------------------------------
# Poly encoding
# ---------------------------------------------------------------------------


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
    make_pce,
    n_vars,
    n_qubits,
    expected_n_qubits,
    expected_masks,
    expect_warning,
):
    """Test poly encoding: n_qubits, masks, and n_qubits validation."""
    qubo = np.zeros((n_vars, n_vars))
    kwargs = dict(problem=qubo, encoding_type="poly", n_qubits=n_qubits)
    if expect_warning:
        with pytest.warns(UserWarning, match="n_qubits exceeds the minimum required"):
            pce = make_pce(**kwargs)
    else:
        pce = make_pce(**kwargs)

    assert pce.n_vars == n_vars
    assert pce.encoding_type == "poly"
    assert pce.n_qubits == expected_n_qubits
    if expected_masks is not None:
        np.testing.assert_array_equal(
            pce._variable_masks_u64, np.array(expected_masks, dtype=np.uint64)
        )


def test_pce_poly_n_qubits_too_low_raises(make_pce):
    """Poly encoding raises when n_qubits is below minimum (N=4 requires 3)."""
    with pytest.raises(ValueError, match="n_qubits must be >= 3 for poly encoding"):
        make_pce(problem=np.zeros((4, 4)), n_qubits=2, encoding_type="poly")


# ---------------------------------------------------------------------------
# Energy computation helpers
# ---------------------------------------------------------------------------


def test_pce_hubo_quadratized_objective_helpers(make_pce):
    """Soft HUBO energy for a cubic term via parity expectations.

    hubo = -x0 + 0.25*x1 + 1.5*x0*x1*x2, alpha = 1, histogram (100 shots):
      "000" count=30, "001"=10, "010"=20, "011"=40.
    Parities (x0, x1, x2): 000→[0,0,0], 001→[1,0,1], 010→[0,1,1], 011→[1,1,0].
    mean_parity = [0.5, 0.6, 0.3] → z = [0, -0.2, 0.4].
    x = 0.5*(1 + tanh(z)); linear terms use x² (soft de-linearization).
    energy = -x0² + 0.25*x1² + 1.5*x0*x1*x2.
    """
    hubo = {
        ("x0",): -1.0,
        ("x1",): 0.25,
        ("x0", "x1", "x2"): 1.5,
    }
    pce = make_pce(problem=hubo, alpha=1.0)

    state_strings, probs = _histogram_to_probs(
        {"000": 30, "001": 10, "010": 20, "011": 40}
    )
    parities = _decode_parities(state_strings, pce._variable_masks_u64)

    energy = _compute_soft_energy(parities, probs, pce.alpha, pce.problem)

    x0, x1, x2 = (0.5 * (1.0 + np.tanh(z)) for z in (0.0, -0.2, 0.4))
    expected = -1.0 * x0**2 + 0.25 * x1**2 + 1.5 * x0 * x1 * x2
    assert energy == pytest.approx(expected)


def test_pce_soft_energy_computation(make_pce):
    """_compute_soft_energy returns correct energy for a mixed histogram.

    qubo = diag([1, 2]), alpha = 1, histogram: {"00": 3, "11": 1} (4 shots)
    masks = [1, 2] (one qubit per variable).

    "00"=0: both parities 0.  "11"=3: both parities 1.
    probs = [0.75, 0.25].
    mean_parity = [0*0.75 + 1*0.25, 0*0.75 + 1*0.25] = [0.25, 0.25].
    z = 1 - 2*0.25 = [0.5, 0.5].
    x = 0.5*(1 + tanh(0.5)) = e/(e+1)   [algebraic identity].
    energy = 1*x² + 2*x² = 3*(e/(e+1))².
    """
    pce = make_pce(problem=np.diag([1.0, 2.0]), alpha=1.0)

    states = ["00", "11"]
    counts = np.array([3, 1], dtype=float)
    probs = counts / counts.sum()
    parities = _decode_parities(states, pce._variable_masks_u64)

    result = _compute_soft_energy(parities, probs, 1.0, pce.problem)

    expected = 3.0 * (np.e / (np.e + 1)) ** 2
    assert result == pytest.approx(expected)


def test_pce_hard_cvar_energy_computation(make_pce):
    """_compute_hard_cvar_energy returns correct CVaR for a mixed histogram.

    qubo = diag([1, 2]), alpha_cvar = 0.5, histogram (10 shots):
      "11" (x=[0,0]) → energy=0, count=4
      "10" (x=[1,0]) → energy=1, count=2
      "01" (x=[0,1]) → energy=2, count=3
      "00" (x=[1,1]) → energy=3, count=1

    CVaR(0.5) takes the best ceil(0.5*10)=5 shots:
      4 shots at energy 0  +  1 shot at energy 1  =  1
      CVaR = 1/5 = 0.2
    """
    pce = make_pce(problem=np.diag([1.0, 2.0]), alpha=6.0)

    states = ["11", "10", "01", "00"]
    counts_arr = np.array([4, 2, 3, 1], dtype=float)
    parities = _decode_parities(states, pce._variable_masks_u64)

    result = _compute_hard_cvar_energy(
        parities, counts_arr, 10.0, pce.problem, alpha_cvar=0.5
    )

    assert result == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Custom decode_parities_fn
# ---------------------------------------------------------------------------


def test_pce_custom_decode_parities_fn_soft_energy(make_pce):
    """Custom decode_parities_fn is used in the soft energy path."""

    def all_zeros_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        return np.zeros((n_vars, n_states), dtype=np.uint8)

    pce = make_pce(alpha=1.0, decode_parities_fn=all_zeros_decoder)

    state_strings, probs = _histogram_to_probs({"00": 30, "01": 10, "10": 20, "11": 40})

    parities = all_zeros_decoder(state_strings, pce._variable_masks_u64)
    result = _compute_soft_energy(parities, probs, pce.alpha, pce.problem)

    # All parities 0 -> mean_parities = [0, 0] -> z_expectations = [1, 1]
    z = np.array([1.0, 1.0])
    x_soft = 0.5 * (1.0 + np.tanh(pce.alpha * z))
    qubo = np.array([[1.0, 0.2], [0.2, 2.0]])
    expected = float(x_soft @ qubo @ x_soft)
    assert result == pytest.approx(expected)


def test_pce_custom_decode_parities_fn_hard_cvar(make_pce):
    """Custom decode_parities_fn is used in the hard CVaR energy path."""

    def all_ones_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        return np.ones((n_vars, n_states), dtype=np.uint8)

    pce = make_pce(
        problem=np.diag([1.0, 2.0]), alpha=6.0, decode_parities_fn=all_ones_decoder
    )

    merged = {"11": 2, "10": 3, "01": 10, "00": 25}
    state_strings = list(merged.keys())
    counts_arr = np.array(list(merged.values()), dtype=float)
    total_shots = counts_arr.sum()

    parities = all_ones_decoder(state_strings, pce._variable_masks_u64)
    result = _compute_hard_cvar_energy(
        parities, counts_arr, total_shots, pce.problem, alpha_cvar=0.25
    )

    # All parities 1 -> x_vals = 0 for all vars/states -> energies all 0 -> CVaR = 0
    assert result == pytest.approx(0.0)


def test_pce_decode_parities_fn_none_uses_default(make_pce):
    """Passing decode_parities_fn=None uses the built-in decoder (same as omitting)."""
    pce_default = make_pce(alpha=1.0)
    pce_explicit_none = make_pce(alpha=1.0, decode_parities_fn=None)

    state_strings, probs = _histogram_to_probs({"00": 30, "01": 10, "10": 20, "11": 40})

    parities_default = _decode_parities(state_strings, pce_default._variable_masks_u64)
    parities_none = _decode_parities(
        state_strings, pce_explicit_none._variable_masks_u64
    )

    result_default = _compute_soft_energy(
        parities_default, probs, 1.0, pce_default.problem
    )
    result_none = _compute_soft_energy(
        parities_none, probs, 1.0, pce_explicit_none.problem
    )

    assert result_default == pytest.approx(result_none)


def test_pce_custom_decode_parities_fn_perform_final_computation(
    mocker, make_pce, qubo_identity
):
    """Custom decode_parities_fn is used in sample_solution."""

    # Decoder that returns parities [0, 1] for any input -> solution = 1 - [0,1] = [1, 0]
    def fixed_decoder(state_strings, variable_masks_u64):
        n_vars = len(variable_masks_u64)
        n_states = len(state_strings)
        out = np.zeros((n_vars, n_states), dtype=np.uint8)
        out[1, :] = 1  # Second variable parity always 1
        return out

    pce = make_pce(problem=qubo_identity, decode_parities_fn=fixed_decoder)

    pce._best_params = np.zeros(pce.n_layers * pce.n_params_per_layer)
    pce._best_probs = {"0_NoMitigation:0_0": {"01": 1.0}}
    mocker.patch.object(pce, "_run_solution_measurement_for")

    pce.sample_solution()

    # Fixed decoder returns parities [0, 1] -> solution = [1, 0]
    np.testing.assert_array_equal(_read_solution(pce), np.array([1, 0]))


def test_pce_custom_decode_parities_fn_get_top_solutions(make_pce):
    """Custom decode_parities_fn is used in get_top_solutions."""

    # Decoder that complements the default: return 1 - default_decoder output
    # So decoded QUBO solution = 1 - (1 - default) = default (same as default)
    # Or we use a decoder that returns fixed output per state
    def complement_decoder(state_strings, variable_masks_u64):
        default = _decode_parities(state_strings, variable_masks_u64)
        return 1 - default

    pce = make_pce(problem=np.eye(2), decode_parities_fn=complement_decoder)

    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.2, "11": 0.0})

    solutions = pce.get_top_solutions(n=4)

    # Default decoder: "00"->"11", "01"->"01", "10"->"10", "11"->"00"
    # Complement decoder: "00"->"00", "01"->"10", "10"->"01", "11"->"11"
    expected_decoded = {"00": 0.5, "10": 0.3, "01": 0.2, "11": 0.0}
    decoded_map = {sol.bitstring: sol.prob for sol in solutions}
    for bitstring, expected_prob in expected_decoded.items():
        assert decoded_map[bitstring] == expected_prob


# ---------------------------------------------------------------------------
# sample_solution / solution property
# ---------------------------------------------------------------------------


def test_pce_perform_final_computation_sets_solution(mocker, make_pce, qubo_identity):
    pce = make_pce(problem=qubo_identity)

    pce._best_params = np.zeros(pce.n_layers * pce.n_params_per_layer)
    pce._best_probs = {"0_NoMitigation:0_0": {"01": 1.0}}
    mocker.patch.object(pce, "_run_solution_measurement_for")

    pce.sample_solution()

    np.testing.assert_array_equal(_read_solution(pce), np.array([0, 1]))


def test_pce_solution_requires_run(make_pce, qubo_identity):
    pce = make_pce(problem=qubo_identity)

    with pytest.raises(RuntimeError, match="Run the VQE optimization first."):
        _ = pce.solution


def test_pce_perform_final_computation_none_eigenstate(mocker, make_pce, qubo_identity):
    """When _eigenstate is None, _final_vector is set to None."""
    pce = make_pce(problem=qubo_identity)
    # Stub the measurement step so _best_probs stays empty; VQE then leaves
    # _eigenstate unset, and PCE's sample_solution must handle that path.
    pce._best_params = np.zeros(pce.n_layers * pce.n_params_per_layer)
    mocker.patch.object(pce, "_run_solution_measurement_for")

    pce.sample_solution()

    assert pce._eigenstate is None
    assert pce._final_vector is None


# ---------------------------------------------------------------------------
# get_top_solutions
# ---------------------------------------------------------------------------


def test_pce_get_top_solutions_decodes_bitstrings(make_pce):
    """get_top_solutions decodes encoded qubit states to QUBO variable assignments."""
    # Create a 4-variable QUBO (requires 3 qubits: ceil(log2(4+1)) = 3)
    pce = make_pce(problem=np.eye(4))

    # For 4 variables, we need 3 qubits
    _set_probs(
        pce,
        {
            "000": 0.5,  # Encoded qubit state
            "001": 0.3,
            "010": 0.15,
            "011": 0.05,
        },
    )

    solutions = pce.get_top_solutions(n=4)

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


def test_pce_get_top_solutions_include_decoded(make_pce):
    """get_top_solutions with include_decoded=True returns numpy arrays."""
    pce = make_pce(problem=np.eye(3))

    # 2 qubits for 3 variables
    _set_probs(pce, {"00": 0.6, "01": 0.4})

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


def test_pce_get_top_solutions_decoded_keys_by_variable_name(make_pce):
    """For a string-keyed BQM, decoded is a name-keyed dict, positionally
    consistent with the decoded bitstring (matching .solution)."""
    bqm = dimod.BinaryQuadraticModel(
        {"w": 1.0, "x": 1.0, "y": 1.0}, {}, 0.0, dimod.Vartype.BINARY
    )
    pce = make_pce(problem=bqm)  # 3 variables -> 2 qubits (dense)
    _set_probs(pce, {"00": 0.6, "01": 0.4})

    sol = pce.get_top_solutions(n=1, include_decoded=True)[0]
    assert isinstance(sol.decoded, dict)
    assert set(sol.decoded) == {"w", "x", "y"}
    assert sol.decoded == {
        name: int(bit) for name, bit in zip(pce.problem.variable_order, sol.bitstring)
    }


def test_pce_get_top_solutions_min_prob_filtering(make_pce):
    """get_top_solutions filters by min_prob correctly."""
    pce = make_pce(problem=np.eye(2))

    _set_probs(pce, {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05})

    solutions = pce.get_top_solutions(n=10, min_prob=0.2)

    # Only solutions with prob >= 0.2 should be included
    assert len(solutions) == 2
    assert solutions[0].prob == 0.5
    assert solutions[1].prob == 0.3
    assert all(sol.prob >= 0.2 for sol in solutions)


def test_pce_get_top_solutions_validation_errors(make_pce):
    """get_top_solutions raises for invalid n, min_prob, and missing probs."""
    pce = make_pce(problem=np.eye(2))
    _set_probs(pce, {"00": 0.5, "01": 0.5})

    with pytest.raises(ValueError, match="n must be non-negative"):
        pce.get_top_solutions(n=-1)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=1.5)

    with pytest.raises(ValueError, match="min_prob must be in range"):
        pce.get_top_solutions(n=2, min_prob=-0.1)

    assert pce.get_top_solutions(n=0) == []


def test_pce_get_top_solutions_no_probs_raises(make_pce):
    """get_top_solutions raises when no probability distribution available."""
    pce = make_pce(problem=np.eye(2))
    pce._best_probs = {}
    pce._losses_history = []

    with pytest.raises(
        RuntimeError,
        match="No probability distribution available",
    ):
        pce.get_top_solutions(n=2)


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
    make_pce,
    encoding_type,
    qubo,
    encoded_probs,
    expected_decoded_map,
    n_vars,
):
    """Decoding produces correct QUBO variable assignments for both encodings."""
    pce = make_pce(problem=qubo, encoding_type=encoding_type)

    _set_probs(pce, encoded_probs)

    solutions = pce.get_top_solutions(n=len(encoded_probs))

    decoded_map = {sol.bitstring: sol.prob for sol in solutions}
    for decoded_bitstring, expected_prob in expected_decoded_map.items():
        assert decoded_map[decoded_bitstring] == expected_prob
    assert all(len(sol.bitstring) == n_vars for sol in solutions)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_pce_qubo_e2e_solution(default_test_simulator, basic_ansatz):
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()

    pce = PCE(
        problem=BinaryOptimizationProblem(qubo),
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
    solution = _read_solution(pce)
    assert solution.shape == (qubo.shape[0],)
    assert set(np.unique(solution)).issubset({0, 1})
    np.testing.assert_array_equal(solution, PCE_QUBO_SOLUTION)
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


@pytest.mark.e2e
def test_pce_qubo_e2e_checkpointing_resume(
    default_test_simulator, basic_ansatz, tmp_path
):
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()
    checkpoint_dir = tmp_path / "checkpoint_test"

    pce1 = PCE(
        problem=BinaryOptimizationProblem(qubo),
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
        problem=BinaryOptimizationProblem(qubo),
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
        problem=BinaryOptimizationProblem(qubo),
        ansatz=basic_ansatz,
        n_layers=1,
    )
    assert pce3.current_iteration == 4
    pce3.max_iterations = 5
    pce3.run()
    assert pce3.current_iteration == 5

    assert isinstance(pce3.best_loss, float)
    assert _read_solution(pce3).shape == (qubo.shape[0],)


@pytest.mark.e2e
def test_pce_poly_e2e_solution(default_test_simulator, basic_ansatz):
    """E2E test for PCE with poly encoding (3 vars, 2 qubits)."""
    default_test_simulator.set_seed(1997)
    qubo = PCE_QUBO_MATRIX.copy()

    pce = PCE(
        problem=BinaryOptimizationProblem(qubo),
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
    solution = _read_solution(pce)
    assert solution.shape == (qubo.shape[0],)
    assert set(np.unique(solution)).issubset({0, 1})
    assert all(
        len(bitstring) == pce.n_qubits
        for probs_dict in pce.best_probs.values()
        for bitstring in probs_dict.keys()
    )


@pytest.mark.e2e
def test_pce_hubo_e2e_solution(default_test_simulator, basic_ansatz):
    """PCE solves a small cubic HUBO and reaches exact minimum assignment."""
    default_test_simulator.set_seed(1997)
    _, exact_minima = exact_hubo_minima(HUBO_CUBIC, n_vars=3)

    pce = PCE(
        problem=BinaryOptimizationProblem(HUBO_CUBIC),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=default_test_simulator,
        seed=1997,
    )

    pce.run()
    assert any(np.array_equal(_read_solution(pce), x) for x in exact_minima)


@pytest.mark.e2e
def test_pce_named_variables_decode_to_names(default_test_simulator, basic_ansatz):
    """A string-indexed BQM decodes to name-keyed solutions on both surfaces."""
    default_test_simulator.set_seed(1997)
    bqm = dimod.BinaryQuadraticModel(
        {"w": 10, "x": -3, "y": 2},
        {("w", "x"): -1, ("x", "y"): 1},
        0.0,
        dimod.Vartype.BINARY,
    )
    pce = PCE(
        problem=BinaryOptimizationProblem(bqm),
        ansatz=basic_ansatz,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=3,
        backend=default_test_simulator,
        seed=1997,
    )
    pce.run()

    names = {"w", "x", "y"}
    solution = _read_solution(pce)
    assert isinstance(solution, dict) and set(solution) == names

    top = pce.get_top_solutions(n=1, include_decoded=True)[0]
    assert isinstance(top.decoded, dict) and set(top.decoded) == names


# ---------------------------------------------------------------------------
# _aggregate_param_group
# ---------------------------------------------------------------------------


class TestAggregateParamGroup:
    """Tests for _aggregate_param_group covering L44-55."""

    def test_single_histogram(self):
        """Single param group entry returns its own histogram."""
        group = [("label_0", {"00": 10, "01": 5})]
        states, counts, total = _aggregate_param_group(group)
        assert set(states) == {"00", "01"}
        assert total == 15.0
        assert counts.sum() == 15.0

    def test_merges_multiple_histograms(self):
        """Multiple histograms are merged, overlapping keys summed."""
        group = [
            ("label_0", {"00": 10, "01": 5}),
            ("label_1", {"01": 3, "10": 7}),
        ]
        states, counts, total = _aggregate_param_group(group)

        state_to_count = dict(zip(states, counts.tolist()))
        assert state_to_count["00"] == 10
        assert state_to_count["01"] == 8  # 5 + 3
        assert state_to_count["10"] == 7
        assert total == 25.0

    def test_empty_histograms(self):
        """Empty histograms produce empty results."""
        group = [("label_0", {})]
        states, counts, total = _aggregate_param_group(group)
        assert states == []
        assert len(counts) == 0
        assert total == 0.0


# ---------------------------------------------------------------------------
# _masks_to_ham_ops
# ---------------------------------------------------------------------------


class TestMasksToHamOps:
    """Tests for _masks_to_ham_ops covering L126-140."""

    def test_single_qubit_masks(self):
        """Masks [1, 2] with 2 qubits → ZI and IZ."""
        masks = np.array([1, 2], dtype=np.uint64)
        result = _masks_to_ham_ops(masks, n_qubits=2)
        assert result == "ZI;IZ"

    def test_two_qubit_mask(self):
        """Mask 3 (bits 0 and 1) with 2 qubits → ZZ."""
        masks = np.array([3], dtype=np.uint64)
        result = _masks_to_ham_ops(masks, n_qubits=2)
        assert result == "ZZ"

    def test_three_qubit_poly_masks(self):
        """Poly encoding masks [1, 2, 3] with 2 qubits."""
        masks = np.array([1, 2, 3], dtype=np.uint64)
        result = _masks_to_ham_ops(masks, n_qubits=2)
        # mask 1 = bit 0 → ZI, mask 2 = bit 1 → IZ, mask 3 = bits 0,1 → ZZ
        assert result == "ZI;IZ;ZZ"

    def test_identity_mask(self):
        """Mask 0 → all Identity."""
        masks = np.array([0], dtype=np.uint64)
        result = _masks_to_ham_ops(masks, n_qubits=3)
        assert result == "III"


# ---------------------------------------------------------------------------
# get_top_solutions sort_by
# ---------------------------------------------------------------------------


class TestGetTopSolutionsSortBy:
    """Tests for the sort_by parameter of PCE.get_top_solutions."""

    @pytest.fixture
    def pce_with_probs(self, make_pce):
        """PCE instance with a 2-var diagonal QUBO and pre-set probability distribution.

        qubo = diag([1, 2]), masks = [1, 2], 2 qubits.
        Encoded states and their decoded solutions + energies:
          "00" (int 0) → parities [0, 0] → x = [1, 1] → energy = 1*1 + 2*1 = 3
          "01" (int 1) → parities [1, 0] → x = [0, 1] → energy = 0 + 2*1 = 2
          "10" (int 2) → parities [0, 1] → x = [1, 0] → energy = 1*1 + 0 = 1
          "11" (int 3) → parities [1, 1] → x = [0, 0] → energy = 0
        """
        pce = make_pce(problem=np.diag([1.0, 2.0]))
        _set_probs(
            pce,
            {
                "00": 0.4,  # decoded "11", energy 3
                "01": 0.1,  # decoded "01", energy 2
                "10": 0.2,  # decoded "10", energy 1
                "11": 0.3,  # decoded "00", energy 0
            },
        )
        return pce

    def test_default_sort_by_prob(self, pce_with_probs):
        """Default sort_by='prob' sorts descending by probability."""
        solutions = pce_with_probs.get_top_solutions(n=4)
        probs = [s.prob for s in solutions]
        assert probs == [0.4, 0.3, 0.2, 0.1]

    def test_sort_by_energy_ascending(self, pce_with_probs):
        """sort_by='energy' sorts ascending by energy."""
        solutions = pce_with_probs.get_top_solutions(n=4, sort_by="energy")

        # Verify energy ordering: 0, 1, 2, 3
        energies = []
        for sol in solutions:
            x = np.array([int(c) for c in sol.bitstring], dtype=float)
            energy = _evaluate_binary_polynomial(x, pce_with_probs.problem)
            energies.append(energy)
        assert energies == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_sort_by_energy_includes_energy_field(self, pce_with_probs):
        """sort_by='energy' populates the energy field in SolutionEntry."""
        solutions = pce_with_probs.get_top_solutions(n=4, sort_by="energy")

        assert solutions[0].energy == pytest.approx(0.0)
        assert solutions[1].energy == pytest.approx(1.0)
        assert solutions[2].energy == pytest.approx(2.0)
        assert solutions[3].energy == pytest.approx(3.0)

    def test_sort_by_prob_energy_is_none(self, pce_with_probs):
        """sort_by='prob' leaves the energy field as None."""
        solutions = pce_with_probs.get_top_solutions(n=4)
        for sol in solutions:
            assert sol.energy is None

    def test_sort_by_energy_with_min_prob(self, pce_with_probs):
        """sort_by='energy' respects min_prob filtering."""
        solutions = pce_with_probs.get_top_solutions(
            n=4, sort_by="energy", min_prob=0.15
        )
        # Only probs >= 0.15: 0.4, 0.3, 0.2 → sorted by energy
        assert len(solutions) == 3
        assert all(sol.prob >= 0.15 for sol in solutions)
        # Energy should still be ascending
        assert solutions[0].energy < solutions[1].energy < solutions[2].energy

    def test_sort_by_energy_with_n_limit(self, pce_with_probs):
        """sort_by='energy' with n < total returns only the n lowest-energy solutions."""
        solutions = pce_with_probs.get_top_solutions(n=2, sort_by="energy")
        assert len(solutions) == 2
        assert solutions[0].energy == pytest.approx(0.0)
        assert solutions[1].energy == pytest.approx(1.0)

    def test_sort_by_invalid_raises(self, pce_with_probs):
        """Invalid sort_by value raises ValueError."""
        with pytest.raises(ValueError, match="sort_by must be"):
            pce_with_probs.get_top_solutions(n=4, sort_by="invalid")


# ---------------------------------------------------------------------------
# ObservableMeasuring contracts
# ---------------------------------------------------------------------------


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, make_pce):
        def _make(**kwargs):
            kwargs.setdefault("n_layers", 1)
            return make_pce(**kwargs)

        return _make
