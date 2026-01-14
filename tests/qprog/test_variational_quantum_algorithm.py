# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
from pennylane.measurements import ExpectationMP
from scipy.optimize import OptimizeResult

from divi.circuits.qem import ZNE
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.exceptions import _CancelledError
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _batched_expectation,
)


@pytest.fixture
def mock_backend(mocker):
    """Create a mock backend with required properties."""
    backend = mocker.MagicMock()
    backend.shots = 1000
    backend.is_async = False
    backend.supports_expval = False
    return backend


class SampleVQAProgram(VariationalQuantumAlgorithm):
    def __init__(self, circ_count, run_time, **kwargs):
        self.circ_count = circ_count
        self.run_time = run_time

        self.n_layers = 1
        self._n_params = 4
        self.current_iteration = 0
        self.max_iterations = 0  # Default value to prevent AttributeError

        super().__init__(backend=kwargs.pop("backend", None), **kwargs)

        self._cost_hamiltonian = (
            qml.PauliX(0) + qml.PauliZ(1) + qml.PauliX(0) @ qml.PauliZ(1)
        )
        self.loss_constant = 0.0

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the VQA problem."""
        return self._cost_hamiltonian

    def _create_meta_circuits_dict(self):
        symbols = [sp.Symbol("beta"), sp.symarray("theta", 3)]
        ops = [
            qml.RX(symbols[0], wires=0),
            qml.U3(*symbols[1], wires=1),
            qml.CNOT(wires=[0, 1]),
        ]
        source_circuit = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(self.cost_hamiltonian)]
        )
        meta_circuit = self._meta_circuit_factory(
            source_circuit=source_circuit,
            symbols=symbols,
        )
        return {"cost_circuit": meta_circuit}

    def _generate_circuits(self, **kwargs):
        """Generate circuits - dummy implementation for testing."""
        return []

    def run(
        self,
        perform_final_computation: bool = True,
        **kwargs,
    ) -> tuple[int, float]:
        return super().run(
            perform_final_computation=perform_final_computation,
            **kwargs,
        )

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save SampleVQAProgram-specific state."""
        return {}

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load SampleVQAProgram-specific state."""
        pass

    def _perform_final_computation(self):
        pass


class TestProgram:
    """Test suite for VariationalQuantumAlgorithm core functionality."""

    def _create_sample_program(self, mocker, **kwargs):
        """Helper to create a SampleVQAProgram with common defaults."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._create_mock_optimizer(mocker)
        return SampleVQAProgram(10, 5.5, seed=1997, **kwargs)

    def _create_mock_backend(self, mocker, shots=100, supports_expval=False):
        """Helper to create a mock backend with standard properties."""
        backend = mocker.MagicMock()
        backend.shots = shots
        backend.is_async = False
        backend.supports_expval = supports_expval
        return backend

    def _create_mock_optimizer(self, mocker, n_param_sets=1):
        """Helper to create a mock optimizer."""
        optimizer = mocker.MagicMock()
        optimizer.n_param_sets = n_param_sets
        return optimizer

    def test_correct_random_behavior(self, mocker):
        """Test that random number generation works correctly with seeds."""
        program = self._create_sample_program(mocker)

        assert (
            program._rng.bit_generator.state
            == np.random.default_rng(seed=1997).bit_generator.state
        )

        program._initialize_params()
        first_init = program._curr_params[0]
        assert first_init.shape == (program.n_layers * program.n_params,)

        program._initialize_params()
        second_init = program._curr_params[0]

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, first_init, second_init
        )

    def test_grouping_strategies_and_expectation_values(self, mocker):
        """Test that different grouping strategies produce expected number of groups and identical expectation values."""
        strategies = [(None, 3, 2), ("wires", 2, 2), ("qwc", 1, 1)]
        expvals_collector = []

        for strategy, expected_n_groups, expected_n_diag in strategies:
            program = self._create_sample_program(mocker, grouping_strategy=strategy)
            program.loss_constant = 0.5
            program.backend = self._create_mock_backend(mocker)

            meta_circuit = program.meta_circuits["cost_circuit"]
            assert len(meta_circuit.measurement_groups) == expected_n_groups
            assert (
                len(tuple(filter(lambda x: "h" in x, meta_circuit._measurements)))
                == expected_n_diag
            )

            program._initialize_params()
            fake_shot_histogram = {"00": 23, "01": 27, "10": 15, "11": 35}
            fake_results = {
                f"0_mock-qem:0_{i}": fake_shot_histogram
                for i in range(expected_n_groups)
            }
            expvals_collector.append(program._post_process_results(fake_results)[0])

        assert len(expvals_collector) == 3
        assert all(value == expvals_collector[0] for value in expvals_collector[1:])

    def test_post_process_with_zne(self, mocker):
        """Tests that _post_process_results correctly applies ZNE extrapolation."""
        scale_factors = [1.0, 2.0, 3.0]
        zne_protocol = ZNE(
            folding_fn=partial(fold_global),
            scale_factors=scale_factors,
            extrapolation_factory=LinearFactory(scale_factors=scale_factors),
        )
        program = self._create_sample_program(mocker, qem_protocol=zne_protocol)
        program.loss_constant = 0.0
        program.backend = self._create_mock_backend(mocker)

        mock_shots_per_sf = [
            {"00": 95, "11": 5},
            {"00": 90, "11": 10},
            {"00": 85, "11": 15},
        ]
        mock_results = {}
        n_measurement_groups = 3
        for qem_run_id, shots in enumerate(mock_shots_per_sf):
            for meas_group_id in range(n_measurement_groups):
                key = f"0_zne:{qem_run_id}_{meas_group_id}"
                mock_results[key] = shots

        final_losses = program._post_process_results(mock_results)
        assert np.isclose(final_losses[0], 3.0)

    def test_post_process_with_expectation_values_happy_path(self, mocker):
        """Test _post_process_results when backend supports expectation values with multiple parameter sets."""
        mock_backend = self._create_mock_backend(mocker, supports_expval=True)
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=2)
        program = self._create_sample_program(
            mocker,
            grouping_strategy=None,
            backend=mock_backend,
            optimizer=mock_optimizer,
        )
        program.loss_constant = 0.5

        ham_ops = "XI;IZ;XZ"
        fake_results = {
            "0_NoMitigation:0_0": {"XI": 0.5, "IZ": -0.3, "XZ": 0.2},
            "1_NoMitigation:0_0": {"XI": 0.7, "IZ": -0.1, "XZ": -0.2},
        }

        losses = program._post_process_results(fake_results, ham_ops=ham_ops)

        assert len(losses) == 2
        assert 0 in losses
        assert 1 in losses

        expected_loss_0 = 0.5 + (-0.3) + 0.2 + program.loss_constant
        expected_loss_1 = 0.7 + (-0.1) + (-0.2) + program.loss_constant
        assert np.isclose(losses[0], expected_loss_0)
        assert np.isclose(losses[1], expected_loss_1)

    def test_post_process_with_expectation_values_missing_ham_ops(self, mocker):
        """Test _post_process_results raises error when ham_ops is missing but supports_expval is True."""
        mock_backend = self._create_mock_backend(mocker, supports_expval=True)
        program = self._create_sample_program(
            mocker, grouping_strategy="_backend_expval", backend=mock_backend
        )
        program.loss_constant = 0.0

        fake_results = {"0_NoMitigation:0_0": {"XI": 0.5, "IZ": -0.3}}
        with pytest.raises(
            ValueError,
            match="Hamiltonian operators.*required when using a backend.*supports expectation values",
        ):
            program._post_process_results(fake_results)

    def test_grouping_strategy_warning_with_expval_backend(self, mocker):
        """Test that a warning is issued when grouping_strategy is provided but backend supports expval."""
        mock_backend = self._create_mock_backend(mocker, supports_expval=True)

        with pytest.warns(
            UserWarning,
            match="Backend supports direct expectation value calculation, but a grouping_strategy was provided",
        ):
            program = self._create_sample_program(
                mocker, grouping_strategy="qwc", backend=mock_backend
            )

        # Verify that the grouping strategy was overridden to "_backend_expval"
        assert program._grouping_strategy == "_backend_expval"


class TestBatchedExpectation:
    """Test suite for batched expectation value calculations."""

    def test_matches_pennylane_baseline(self):
        """
        Validates that the optimized batched_expectation function produces results
        identical to PennyLane's standard ExpectationMP processing.
        """
        wire_order = (3, 2, 1, 0)
        shot_histogram = {"0000": 100, "0101": 200, "1011": 300, "1111": 400}
        observables = [
            qml.PauliZ(0),
            qml.PauliZ(2),
            qml.Identity(1),
            qml.PauliZ(1) @ qml.PauliZ(3),
        ]

        baseline_expvals = []
        for obs in observables:
            mp = ExpectationMP(obs)
            expval = mp.process_counts(counts=shot_histogram, wire_order=wire_order)
            baseline_expvals.append(expval)

        optimized_expvals_matrix = _batched_expectation(
            [shot_histogram], observables, wire_order
        )
        optimized_expvals = optimized_expvals_matrix[:, 0]

        assert isinstance(optimized_expvals, np.ndarray)
        np.testing.assert_allclose(optimized_expvals, baseline_expvals)

    def test_with_multiple_histograms(self):
        """
        Tests that batched_expectation correctly processes a list of different
        shot histograms in a single call.
        """
        hist_1 = {"00": 100}
        hist_2 = {"11": 50}
        hist_3 = {"01": 25, "10": 75}
        observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        wire_order = (1, 0)

        expected_1 = np.array([1.0, 1.0, 1.0])
        expected_2 = np.array([-1.0, -1.0, 1.0])
        expected_3 = np.array([0.5, -0.5, -1.0])

        result_matrix = _batched_expectation(
            [hist_1, hist_2, hist_3], observables, wire_order
        )

        assert result_matrix.shape == (3, 3)
        np.testing.assert_allclose(result_matrix[:, 0], expected_1)
        np.testing.assert_allclose(result_matrix[:, 1], expected_2)
        np.testing.assert_allclose(result_matrix[:, 2], expected_3)

    def test_raises_for_unsupported_observable(self):
        """
        Ensures that a KeyError is raised when an observable outside
        the supported set (Pauli, Identity) is provided.
        """
        shots = {"0": 100}
        wire_order = (0,)
        unsupported_observables = [qml.PauliZ(0), qml.Hadamard(0)]

        with pytest.raises(KeyError):
            _batched_expectation(
                shots_dicts=[shots],
                observables=unsupported_observables,
                wire_order=wire_order,
            )

    @pytest.mark.parametrize(
        "n_qubits",
        [1, 10, 32, 64, 65, 100, 150, 200, 500, 1000],
        ids=lambda n: f"{n}-qubits",
    )
    def test_qubit_counts_no_overflow(self, n_qubits):
        """
        Tests that _batched_expectation works correctly across a wide range of qubit counts
        without integer overflow errors, including the critical boundary at 64.

        This test would have caught the bug where 150-qubit circuits caused OverflowError
        when converting bitstrings to uint64.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))

        # Create test bitstrings including edge cases that would overflow uint64
        test_bitstrings = [
            "0" * n_qubits,  # All 0s
            "1" * n_qubits,  # All 1s - would overflow uint64 for >64 qubits
            "1" + "0" * (n_qubits - 1),  # MSB=1, rest 0s
            "0" * (n_qubits - 1) + "1",  # LSB=1, rest 0s
        ]

        shot_histogram = {bs: 100 for bs in test_bitstrings}

        # Test observables on first, middle, and last qubits
        # Ensure mid_qubit is within valid wire range [0, n_qubits-1]
        mid_qubit = n_qubits // 2 if n_qubits > 1 else 0
        observables = [
            qml.PauliZ(0),
            qml.PauliZ(n_qubits - 1),
        ]
        # Add Identity observable if we have a valid middle qubit
        if n_qubits > 1:
            observables.append(qml.Identity(mid_qubit))
        # Add product observable for larger systems
        if n_qubits >= 50:
            observables.append(qml.PauliZ(mid_qubit // 2) @ qml.PauliZ(mid_qubit))

        # Remove duplicates
        observables = list(dict.fromkeys(observables))

        # Should not raise OverflowError
        result = _batched_expectation([shot_histogram], observables, wire_order)

        # Verify results are valid
        assert result.shape == (len(observables), 1)
        assert not np.isnan(result).any(), "Results should not contain NaN"
        assert not np.isinf(result).any(), "Results should not contain Inf"

        # Verify expectation values are in valid range
        for i, obs in enumerate(observables):
            if isinstance(obs, qml.PauliZ) or (
                hasattr(obs, "name") and obs.name == "Prod"
            ):
                assert -1.0 <= result[i, 0] <= 1.0

    def test_boundary_qubit_count_both_paths_work(self):
        """
        Tests that both code paths (<=64 and >64 qubits) work correctly at the boundary.

        This ensures the conditional logic correctly switches between integer and
        character array representations.
        """
        # Test at exactly 64 qubits (uses integer path)
        wire_order_64 = tuple(range(63, -1, -1))
        shot_histogram_64 = {"1" * 64: 100, "0" * 64: 100}
        observables_64 = [qml.PauliZ(0), qml.PauliZ(31), qml.PauliZ(63)]

        result_64 = _batched_expectation(
            [shot_histogram_64], observables_64, wire_order_64
        )

        # Test at 65 qubits (uses character array path)
        wire_order_65 = tuple(range(64, -1, -1))
        shot_histogram_65 = {"1" * 65: 100, "0" * 65: 100}
        observables_65 = [qml.PauliZ(0), qml.PauliZ(32), qml.PauliZ(64)]

        result_65 = _batched_expectation(
            [shot_histogram_65], observables_65, wire_order_65
        )

        # Both should complete without errors and produce valid results
        assert result_64.shape[0] == len(observables_64)
        assert result_65.shape[0] == len(observables_65)
        assert not np.isnan(result_64).any()
        assert not np.isnan(result_65).any()

    @pytest.mark.parametrize(
        "n_qubits",
        [4, 32, 64, 65, 100, 150, 500, 1000],
        ids=lambda n: f"{n}qubits",
    )
    def test_matches_pennylane_baseline(self, n_qubits):
        """
        Validates that results match PennyLane's baseline implementation across
        various qubit counts, ensuring correctness of both code paths.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))
        shot_histogram = {"0" * n_qubits: 50, "1" * n_qubits: 50}
        observables = [qml.PauliZ(0)]

        # Get baseline from PennyLane
        baseline_expvals = []
        for obs in observables:
            mp = ExpectationMP(obs)
            expval = mp.process_counts(counts=shot_histogram, wire_order=wire_order)
            baseline_expvals.append(expval)

        # Get result from our optimized function
        optimized_expvals = _batched_expectation(
            [shot_histogram], observables, wire_order
        )[:, 0]

        # Should match PennyLane's results
        np.testing.assert_allclose(optimized_expvals, baseline_expvals, rtol=1e-10)

    @pytest.mark.parametrize(
        "n_qubits,observable_wires",
        [
            (100, [0, 50]),
            (150, [0, 50, 100]),
            (150, [0, 75, 125]),
            (200, [0, 100, 199]),
            (500, [0, 250, 499]),
            (1000, [0, 500, 999]),
        ],
        ids=["100q_2w", "150q_3w", "150q_3w_alt", "200q_3w", "500q_3w", "1000q_3w"],
    )
    def test_product_observables_large_qubit_counts(self, n_qubits, observable_wires):
        """
        Tests that product observables (multi-qubit) work correctly for large qubit counts.
        """
        wire_order = tuple(range(n_qubits - 1, -1, -1))
        shot_histogram = {
            "0" * n_qubits: 100,
            "1" * n_qubits: 100,
            "1" + "0" * (n_qubits - 1): 100,
        }

        # Create product observable from wire indices
        obs = qml.PauliZ(observable_wires[0])
        for wire in observable_wires[1:]:
            obs = obs @ qml.PauliZ(wire)
        observables = [obs]

        result = _batched_expectation([shot_histogram], observables, wire_order)

        # Verify results
        assert result.shape == (1, 1)
        assert not np.isnan(result).any()
        # Product observables should be in range [-1, 1]
        assert np.abs(result[0, 0]) <= 1.0 + 1e-10


class BaseVariationalQuantumAlgorithmTest:
    """Base test class for VariationalQuantumAlgorithm functionality."""

    def _create_mock_optimizer(self, mocker, n_param_sets=1):
        """Helper to create a mock optimizer with specified n_param_sets."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = n_param_sets
        return mock_optimizer

    def _create_program_with_mock_optimizer(self, mocker, **kwargs):
        """Helper method to create SampleProgram with mocked optimizer."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._create_mock_optimizer(mocker, n_param_sets=1)
        return SampleVQAProgram(circ_count=1, run_time=0.1, **kwargs)

    def _setup_program_with_probs(self, mocker, probs_dict: dict[str, float], **kwargs):
        """Helper to create a program with a synthetic probability distribution."""
        program = self._create_program_with_mock_optimizer(mocker, **kwargs)
        # Wrap in nested structure: {tag: {bitstring: prob}} to match production
        program._best_probs = {"0_NoMitigation:0_0": probs_dict}
        # Mark as having run optimization to avoid warnings
        program._losses_history = [{0: -1.0}]
        return program


class TestParametersBehavior(BaseVariationalQuantumAlgorithmTest):
    """Test suite for parameters functionality."""

    def _setup_mock_optimizer_single_run(self, mocker, program, final_params):
        """Helper to set up a mock optimizer for a single optimization run."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = program.optimizer.n_param_sets

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            loss = cost_fn(final_params)
            result = OptimizeResult(x=final_params, fun=np.array([loss]))
            callback_fn(result)
            return result

        mock_optimizer.optimize.side_effect = mock_optimize_logic
        program.optimizer = mock_optimizer
        return mock_optimizer

    def test_curr_params_returns_numpy_array_not_none(self, mocker):
        """Test that curr_params property always returns actual parameters."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        curr_params = program.curr_params
        assert isinstance(curr_params, np.ndarray)
        assert curr_params.shape == program.get_expected_param_shape()

    def test_curr_params_triggers_lazy_initialization_on_first_access(self, mocker):
        """Test that accessing curr_params triggers parameter generation."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        assert program._curr_params is None
        curr_params = program.curr_params
        assert program._curr_params is not None
        assert isinstance(curr_params, np.ndarray)

    def test_curr_params_setter_stores_custom_parameters(self, mocker):
        """Test that setting curr_params stores user-provided parameters."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        expected_shape = program.get_expected_param_shape()
        custom_curr_params = np.random.uniform(0, 2 * np.pi, expected_shape)
        program.curr_params = custom_curr_params
        assert np.array_equal(program.curr_params, custom_curr_params)
        assert np.array_equal(program._curr_params, custom_curr_params)

    def test_run_validates_parameter_shape(self, mocker):
        """Test that run() validates parameter shape if curr_params is set manually."""
        invalid_params = np.array([[0.1, 0.2]])  # Wrong shape
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=1)
        # Validation happens in constructor, so ValueError should be raised
        with pytest.raises(ValueError, match="Initial parameters must have shape"):
            SampleVQAProgram(
                circ_count=1,
                run_time=0.1,
                optimizer=mock_optimizer,
                initial_params=invalid_params,
                backend=None,
            ).run()

    def test_curr_params_returns_copy_not_reference(self, mocker):
        """Test that curr_params property returns a copy, not a reference."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        curr_params1 = program.curr_params
        curr_params2 = program.curr_params
        assert not np.shares_memory(curr_params1, curr_params2)
        assert not np.shares_memory(curr_params1, program._curr_params)
        assert np.array_equal(curr_params1, curr_params2)

    def test_run_preserves_user_set_curr_params(self, mocker):
        """Test that run() does not overwrite user-set curr_params."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1

        custom_curr_params = np.array([[1.0, 2.0, 3.0, 4.0]])
        program.curr_params = custom_curr_params

        mock_init = mocker.patch.object(program, "_initialize_params")
        mocker.patch.object(
            program, "_run_optimization_circuits", return_value={0: -0.5}
        )

        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        mock_optimizer = self._setup_mock_optimizer_single_run(
            mocker, program, final_params
        )

        program.run()

        mock_init.assert_not_called()
        assert mock_optimizer.optimize.called
        call_args = mock_optimizer.optimize.call_args
        np.testing.assert_array_equal(
            call_args.kwargs["initial_params"], custom_curr_params
        )

    def test_run_initializes_curr_params_when_not_set_by_user(self, mocker):
        """Test that run() initializes parameters when user hasn't set them."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1

        assert program._curr_params is None

        mocker.patch.object(
            program, "_run_optimization_circuits", return_value={0: -0.5}
        )
        mock_init = mocker.spy(program, "_initialize_params")

        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        self._setup_mock_optimizer_single_run(mocker, program, final_params)

        program.run()

        mock_init.assert_called_once()
        assert program._curr_params is not None
        expected_shape = (
            program.optimizer.n_param_sets,
            program.n_layers * program.n_params,
        )
        assert program._curr_params.shape == expected_shape

    @pytest.mark.parametrize(
        "n_param_sets,initial_params_shape",
        [
            (1, (1, 4)),  # Single parameter set
            (2, (2, 4)),  # Multiple parameter sets
        ],
    )
    def test_initial_params_constructor_parameter_sets_curr_params(
        self, mocker, mock_backend, n_param_sets, initial_params_shape
    ):
        """Test that passing initial_params to constructor sets curr_params correctly."""
        custom_params = np.random.uniform(0, 2 * np.pi, initial_params_shape)
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=n_param_sets)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=mock_optimizer,
            initial_params=custom_params,
        )

        # Verify curr_params was set correctly
        np.testing.assert_array_equal(program.curr_params, custom_params)
        assert program._curr_params is not None

    def test_initial_params_constructor_parameter_used_in_run(
        self, mocker, mock_backend
    ):
        """Test that initial_params passed to constructor are used when running."""
        custom_params = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=1)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=mock_optimizer,
            initial_params=custom_params,
        )
        program.max_iterations = 1

        mocker.patch.object(
            program, "_run_optimization_circuits", return_value={0: -0.5}
        )

        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        self._setup_mock_optimizer_single_run(mocker, program, final_params)

        program.run()

        # Verify that the optimizer was called with the initial_params from constructor
        call_args = program.optimizer.optimize.call_args
        np.testing.assert_array_equal(call_args.kwargs["initial_params"], custom_params)


class TestOptimizerBehavior(BaseVariationalQuantumAlgorithmTest):
    """Test suite for optimizer initialization behavior."""

    def test_optimizer_defaults_to_monte_carlo(self, mocker, mock_backend):
        """Test that optimizer defaults to MonteCarloOptimizer when not provided."""
        program = SampleVQAProgram(circ_count=1, run_time=0.1, backend=mock_backend)
        assert isinstance(program.optimizer, MonteCarloOptimizer)
        assert program.optimizer.n_param_sets == 10  # Default population_size

    def test_optimizer_can_be_passed_via_kwargs(self, mocker, mock_backend):
        """Test that optimizer can be passed via kwargs."""
        custom_optimizer = self._create_mock_optimizer(mocker, n_param_sets=5)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=custom_optimizer,
        )
        assert program.optimizer is custom_optimizer
        assert program.optimizer.n_param_sets == 5

    def test_optimizer_is_set_before_super_init(self, mocker, mock_backend):
        """Test that optimizer is available during __init__ before super().__init__()."""
        # This test verifies optimizer is set up early enough for initial_params validation
        custom_params = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=1)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=mock_optimizer,
            initial_params=custom_params,
        )
        # If optimizer wasn't set early, initial_params validation would fail
        assert program.optimizer is mock_optimizer
        np.testing.assert_array_equal(program.curr_params, custom_params)


class TestRunIntegration(BaseVariationalQuantumAlgorithmTest):
    """Test suite for the integration of the run method's components."""

    def setup_mock_optimizer(self, program, mocker, side_effects):
        """Configures a mock optimizer that executes callbacks."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = program.optimizer.n_param_sets

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            last_result = None
            for params, _ in side_effects:
                # Call the actual cost function to trigger the patched _run_optimization_circuits
                actual_loss = cost_fn(params)
                result = OptimizeResult(x=params, fun=actual_loss)
                callback_fn(result)
                last_result = result
            return last_result

        mock_optimizer.optimize.side_effect = mock_optimize_logic
        program.optimizer = mock_optimizer

    def test_run_successful_completion_and_state_tracking(self, mocker):
        """
        Tests that the run method correctly calls the cost function, tracks the best
        loss and parameters via its callback, and sets the final state correctly.
        """
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 3

        mock_run_circuits = mocker.patch.object(program, "_run_optimization_circuits")
        mock_run_circuits.side_effect = [{0: -0.8}, {0: -0.5}, {0: -0.9}]

        curr_params1 = np.array([[0.1, 0.2, 0.3, 0.4]])
        curr_params2 = np.array([[0.5, 0.6, 0.7, 0.8]])
        curr_params3 = np.array([[0.9, 1.0, 1.1, 1.2]])

        self.setup_mock_optimizer(
            program,
            mocker,
            [(curr_params1, -0.8), (curr_params2, -0.5), (curr_params3, -0.9)],
        )

        program.run()

        # Assertions
        assert program.best_loss == -0.9
        np.testing.assert_allclose(program.best_params, curr_params3.flatten())
        np.testing.assert_allclose(program.final_params, curr_params3)
        assert len(program.losses_history) == 3
        assert program.losses_history[0]["0"] == -0.8
        assert program.losses_history[2]["0"] == -0.9
        assert mock_run_circuits.call_count == 3

    def test_run_method_cancellation_handling(self, mocker):
        """Test run method exits gracefully when a cancellation event is set."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1
        mock_event = mocker.MagicMock()
        mock_event.is_set.return_value = True
        program._cancellation_event = mock_event
        program.optimizer.optimize.side_effect = _CancelledError(
            "Cancellation requested"
        )

        circ_count, run_time = program.run()

        assert circ_count == program._total_circuit_count
        assert run_time == program._total_run_time

    def _setup_program_for_final_computation_test(self, mocker):
        """Helper to set up program with mocks for final computation tests."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_run_optimization_circuits", return_value={0: -0.5}
        )
        mock_final_computation = mocker.spy(program, "_perform_final_computation")

        final_params = np.array([[0.1, 0.2, 0.3, 0.4]])
        final_loss = -0.5
        self.setup_mock_optimizer(program, mocker, [(final_params, final_loss)])

        return program, mock_final_computation

    @pytest.mark.parametrize(
        "perform_final_computation,should_call",
        [
            (None, True),  # Default (None means use default)
            (True, True),
            (False, False),
        ],
    )
    def test_run_perform_final_computation(
        self, mocker, perform_final_computation, should_call
    ):
        """Test that _perform_final_computation is called/not called based on parameter."""
        program, mock_final_computation = (
            self._setup_program_for_final_computation_test(mocker)
        )

        if perform_final_computation is None:
            program.run()
        else:
            program.run(perform_final_computation=perform_final_computation)

        if should_call:
            mock_final_computation.assert_called_once()
        else:
            mock_final_computation.assert_not_called()

    @pytest.mark.parametrize(
        "losses_history,expected_min_losses",
        [
            (
                [{0: -0.8}, {0: -0.5}, {0: -0.9}],
                [-0.8, -0.5, -0.9],
            ),  # Single parameter set per iteration
            (
                [
                    {0: -0.8, 1: -0.6, 2: -0.9},  # min is -0.9
                    {0: -0.5, 1: -0.3},  # min is -0.5
                    {0: -0.9, 1: -0.7, 2: -0.4},  # min is -0.9
                ],
                [-0.9, -0.5, -0.9],
            ),  # Multiple parameter sets per iteration
        ],
    )
    def test_min_losses_per_iteration(
        self, mocker, losses_history, expected_min_losses
    ):
        """Test min_losses_per_iteration returns correct minimum losses."""
        program = self._create_program_with_mock_optimizer(mocker)
        program._losses_history = losses_history

        min_losses = program.min_losses_per_iteration

        assert isinstance(min_losses, list)
        assert len(min_losses) == len(expected_min_losses)
        assert min_losses == expected_min_losses

    def test_min_losses_per_iteration_returns_copy(self, mocker):
        """Test that min_losses_per_iteration returns a new list each time."""
        program = self._create_program_with_mock_optimizer(mocker)
        program._losses_history = [
            {0: -0.8},
            {0: -0.5},
        ]

        min_losses1 = program.min_losses_per_iteration
        min_losses2 = program.min_losses_per_iteration

        # Should return new lists (not the same object)
        assert min_losses1 == min_losses2
        assert min_losses1 is not min_losses2

    def test_best_probs_returns_copy(self, mocker):
        """Test that best_probs returns a copy, not a reference."""
        program = self._setup_program_with_probs(mocker, {"00": 0.5, "01": 0.5})

        result = program.best_probs
        # best_probs returns a shallow copy of the nested structure
        # Modifying the outer dict keys doesn't affect original
        original_keys = list(program._best_probs.keys())
        result["new_tag"] = {"11": 1.0}  # Add new key to returned dict

        # Original keys should be unchanged
        assert list(program._best_probs.keys()) == original_keys
        # But modifying nested dicts will affect original (shallow copy)
        # So we test that the outer dict is copied, not the inner dicts
        assert "new_tag" not in program._best_probs


class TestCheckpointing:
    """Tests for VariationalQuantumAlgorithm checkpointing functionality."""

    @pytest.fixture
    def sample_program(self, mock_backend):
        """Create a sample program for testing."""
        program = SampleVQAProgram(circ_count=0, run_time=0.0, backend=mock_backend)
        return program

    def _setup_optimizer_state(self, program, iteration: int = 1):
        """Helper to set optimizer state for testing.

        Sets up internal state so that save_state() can be called without errors.
        Note: This assumes MonteCarloOptimizer since SampleVQAProgram defaults to it.
        """
        program.optimizer._curr_population = np.zeros((10, 4))
        program.optimizer._curr_evaluated_population = np.zeros((10, 4))
        program.optimizer._curr_losses = np.zeros(10)
        program.optimizer._curr_iteration = iteration
        # Set RNG state to avoid None check failure
        program.optimizer._curr_rng_state = np.random.default_rng().bit_generator.state

    def _create_mock_optimize(
        self, program, n_iterations: int = 1, result_x=None, result_fun=None
    ):
        """Helper to create a mock optimize function.

        Args:
            program: The program instance (for optimizer state setup)
            n_iterations: Number of iterations to simulate
            result_x: Optional custom x values for OptimizeResult
            result_fun: Optional custom fun values for OptimizeResult
        """

        def mock_optimize(**kwargs):
            self._setup_optimizer_state(program, iteration=n_iterations)
            callback = kwargs.get("callback_fn")
            if callback:
                if n_iterations > 1:
                    # Simulate multiple iterations
                    for i in range(n_iterations):
                        x = (
                            result_x
                            if result_x is not None
                            else np.array([[0.1, 0.2, 0.3, 0.4]])
                        )
                        fun = (
                            result_fun if result_fun is not None else np.array([0.123])
                        )
                        res = OptimizeResult(x=x, fun=fun, nit=i + 1)
                        callback(res)
                else:
                    # Single iteration
                    x = result_x if result_x is not None else np.zeros((1, 4))
                    fun = result_fun if result_fun is not None else np.array([0.5])
                    res = OptimizeResult(x=x, fun=fun, nit=1)
                    callback(res)

            # Return final result
            final_x = result_x if result_x is not None else np.zeros(4)
            final_fun = result_fun if result_fun is not None else 0.5
            if isinstance(final_x, np.ndarray) and final_x.ndim == 2:
                final_x = final_x[0]
            return OptimizeResult(x=final_x, fun=final_fun)

        return mock_optimize

    def test_save_state_creates_files(self, sample_program, tmp_path, mocker):
        """Test that save_state() creates the expected files."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1)

        checkpoint_dir = tmp_path / "checkpoint"
        sample_program.save_state(CheckpointConfig(checkpoint_dir=checkpoint_dir))

        # Files should be in the per-iteration subdirectory
        assert (
            tmp_path / "checkpoint" / "checkpoint_001" / "program_state.json"
        ).exists()
        assert (
            tmp_path / "checkpoint" / "checkpoint_001" / "optimizer_state.json"
        ).exists()

    def test_save_state_auto_generates_directory(
        self, sample_program, tmp_path, mocker
    ):
        """Test that save_state() generates a directory name if none provided."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1)

        # Change working directory to tmp_path so we don't pollute the repo
        with pytest.MonkeyPatch.context() as m:
            m.chdir(tmp_path)
            path = sample_program.save_state(CheckpointConfig.with_timestamped_dir())
            assert "checkpoint_" in str(path)
            # Path should point to the subdirectory
            assert path.exists()
            assert path.name.startswith("checkpoint_")

    def test_save_load_round_trip(self, sample_program, tmp_path, mocker):
        """Test saving and loading restores state."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(
                sample_program,
                n_iterations=5,
                result_x=np.array([[0.1, 0.2, 0.3, 0.4]]),
                result_fun=np.array([0.123]),
            )
        )
        sample_program.run(max_iterations=5)

        checkpoint_dir = tmp_path / "checkpoint"
        sample_program.save_state(CheckpointConfig(checkpoint_dir=checkpoint_dir))

        # Load state
        # We need to provide required init args: circ_count, run_time
        loaded_program = SampleVQAProgram.load_state(
            checkpoint_dir, backend=sample_program.backend, circ_count=0, run_time=0.0
        )

        assert loaded_program.current_iteration == 5
        assert loaded_program._best_loss == 0.123
        # Note: _curr_params might be different after loading, so we check best_params instead
        # which should be preserved
        assert isinstance(loaded_program.optimizer, MonteCarloOptimizer)

    def test_save_state_raises_error_before_optimization(
        self, sample_program, tmp_path
    ):
        """Test that save_state() raises RuntimeError if optimization hasn't been run."""
        checkpoint_dir = tmp_path / "checkpoint"

        with pytest.raises(RuntimeError, match="optimization has not been run"):
            sample_program.save_state(CheckpointConfig(checkpoint_dir=checkpoint_dir))

    def test_automatic_checkpointing_in_run(self, sample_program, tmp_path, mocker):
        """Test that run() triggers checkpointing."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.save_state = mocker.Mock(wraps=sample_program.save_state)

        checkpoint_dir = tmp_path / "auto_check"
        sample_program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )

        # Should have called save_state
        sample_program.save_state.assert_called()
        # Files should be in the per-iteration subdirectory
        assert (
            tmp_path / "auto_check" / "checkpoint_001" / "program_state.json"
        ).exists()

    def test_automatic_checkpointing_interval(self, sample_program, tmp_path, mocker):
        """Test that checkpoint_interval is respected."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=3)
        )
        sample_program.save_state = mocker.Mock(wraps=sample_program.save_state)

        checkpoint_dir = tmp_path / "interval_check"
        # Save every 2 iterations. Should save at iter 2.
        sample_program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=checkpoint_dir, checkpoint_interval=2
            )
        )

        # In the simulated loop of 3 iterations (1, 2, 3):
        # Iter 1: 1 % 2 != 0 -> No save
        # Iter 2: 2 % 2 == 0 -> Save
        # Iter 3: 3 % 2 != 0 -> No save
        # So save_state should be called exactly once
        assert sample_program.save_state.call_count == 1
        # Should have created checkpoint_002 subdirectory
        assert (
            tmp_path / "interval_check" / "checkpoint_002" / "program_state.json"
        ).exists()

    def test_multiple_checkpoints_and_load_latest(
        self, sample_program, tmp_path, mocker
    ):
        """Test that multiple checkpoints are created and load_state finds the latest."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=5)
        )

        checkpoint_dir = tmp_path / "multi_check"
        # Save every iteration
        sample_program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=checkpoint_dir, checkpoint_interval=1
            )
        )

        # Should have created checkpoints for iterations 1-5
        for i in range(1, 6):
            assert (
                tmp_path / "multi_check" / f"checkpoint_{i:03d}" / "program_state.json"
            ).exists()

        # Load state without specifying subdirectory - should load latest (checkpoint_005)
        loaded_program = SampleVQAProgram.load_state(
            checkpoint_dir, backend=sample_program.backend, circ_count=0, run_time=0.0
        )
        assert loaded_program.current_iteration == 5

        # Load specific checkpoint
        loaded_program_2 = SampleVQAProgram.load_state(
            checkpoint_dir,
            backend=sample_program.backend,
            subdirectory="checkpoint_003",
            circ_count=0,
            run_time=0.0,
        )
        assert loaded_program_2.current_iteration == 3

    def test_resume_with_less_iterations(self, sample_program, tmp_path, mocker):
        """Test resuming with max_iterations less than already completed is a no-op and warns."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=3)
        )
        sample_program.run(
            max_iterations=3,
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=tmp_path / "checkpoint_test"
            ),
        )
        assert sample_program.current_iteration == 3

        # Resume with max_iterations=2 (less than completed)
        loaded_program = SampleVQAProgram.load_state(
            tmp_path / "checkpoint_test",
            backend=sample_program.backend,
            circ_count=0,
            run_time=0.0,
        )
        loaded_program.max_iterations = 2

        # Should warn and not run additional iterations since already completed
        with pytest.warns(
            UserWarning,
            match="max_iterations \\(2\\) is less than current_iteration \\(3\\)",
        ):
            loaded_program.run()

        # Should not run additional iterations since already completed
        assert loaded_program.current_iteration == 3
        assert len(loaded_program.losses_history) == 3


class TestPrecisionFunctionality(BaseVariationalQuantumAlgorithmTest):
    """Test suite for precision functionality in VariationalQuantumAlgorithm."""

    def test_precision_defaults_to_8(self, mock_backend):
        """Test that precision defaults to 8 when not provided."""
        program = SampleVQAProgram(circ_count=1, run_time=0.1, backend=mock_backend)
        assert program._precision == 8

    def test_precision_can_be_passed_as_kwarg(self, mock_backend):
        """Test that precision can be passed as a kwarg."""
        program = SampleVQAProgram(
            circ_count=1, run_time=0.1, backend=mock_backend, precision=12
        )
        assert program._precision == 12

    def test_precision_used_in_qasm_conversion(self, mocker, mock_backend):
        """Test that precision is used when creating QASM circuits."""
        mock_to_openqasm = mocker.patch("divi.circuits._core.to_openqasm")
        mock_to_openqasm.return_value = (["circuit_body"], ["measurement"])

        program = SampleVQAProgram(
            circ_count=1, run_time=0.1, backend=mock_backend, precision=5
        )

        # Access meta_circuits to trigger creation and QASM conversion
        _ = program.meta_circuits

        # Verify to_openqasm was called with precision=5
        assert mock_to_openqasm.called
        call_kwargs = mock_to_openqasm.call_args[1]
        assert call_kwargs["precision"] == 5

    def test_different_precision_values(self, mock_backend):
        """Test that different precision values work correctly."""
        for precision in [1, 4, 8, 12, 16]:
            program = SampleVQAProgram(
                circ_count=1,
                run_time=0.1,
                backend=mock_backend,
                precision=precision,
            )
            assert program._precision == precision

            # Verify it propagates to the factory
            factory_keywords = program._meta_circuit_factory.keywords
            assert factory_keywords["precision"] == precision


class TestPropertyWarnings(BaseVariationalQuantumAlgorithmTest):
    """Test suite for property warnings when accessing uninitialized state."""

    @pytest.mark.parametrize(
        "property_name,expected_warning_msg",
        [
            ("losses_history", "losses_history is empty"),
            ("min_losses_per_iteration", "min_losses_per_iteration is empty"),
            ("best_loss", "best_loss has not been computed yet"),
            ("best_probs", "best_probs is empty"),
        ],
    )
    def test_property_warns_before_optimization(
        self, mocker, property_name, expected_warning_msg
    ):
        """Test that properties warn when accessed before optimization runs."""
        program = self._create_program_with_mock_optimizer(mocker)

        with pytest.warns(UserWarning, match=expected_warning_msg):
            _ = getattr(program, property_name)

    @pytest.mark.parametrize(
        "property_name,expected_warning_msg",
        [
            ("final_params", "final_params is not available"),
            ("best_params", "best_params is not available"),
        ],
    )
    def test_params_warn_before_optimization(
        self, mocker, property_name, expected_warning_msg
    ):
        """Test that final_params and best_params warn when accessed before optimization."""
        program = self._create_program_with_mock_optimizer(mocker)

        with pytest.warns(UserWarning, match=expected_warning_msg):
            # .copy() works on lists, so this won't raise - just warns
            result = getattr(program, property_name)
            assert isinstance(result, list)
            assert len(result) == 0

    def test_best_loss_raises_runtime_error_if_still_infinite_after_optimization(
        self, mocker
    ):
        """Test that best_loss raises RuntimeError if still infinite after optimization."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1

        # Simulate optimization running but best_loss not being updated
        program._losses_history = [{0: 1.0}]  # Optimization has run
        program._best_loss = float("inf")  # But best_loss is still infinite

        with pytest.raises(
            RuntimeError,
            match="best_loss is still infinite after optimization",
        ):
            _ = program.best_loss

    def test_best_probs_warns_when_empty_after_optimization(self, mocker):
        """Test that best_probs warns when empty even after optimization."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_run_optimization_circuits", return_value={0: -0.5}
        )

        final_params = np.array([[0.1, 0.2, 0.3, 0.4]])

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            loss = cost_fn(final_params)
            result = OptimizeResult(x=final_params, fun=np.array([loss]))
            callback_fn(result)
            return result

        program.optimizer.optimize = mocker.Mock(side_effect=mock_optimize_logic)

        # Run without final computation
        program.run(perform_final_computation=False)

        # best_probs should still warn because it's empty
        with pytest.warns(UserWarning, match="best_probs is empty"):
            _ = program.best_probs


class TestTopSolutionsAPI(BaseVariationalQuantumAlgorithmTest):
    """Test suite for get_top_solutions() and best_probs API."""

    def test_get_top_solutions_raises_when_no_probs(self, mocker):
        """Test that get_top_solutions raises RuntimeError when distribution is empty."""
        program = self._create_program_with_mock_optimizer(mocker)
        program._best_probs = {}

        with pytest.raises(
            RuntimeError,
            match="No probability distribution available.*perform_final_computation=True",
        ):
            program.get_top_solutions(n=5)

    def test_decode_solution_fn_default_returns_identity(self, mocker):
        """Test that default decode_solution_fn returns bitstring unchanged."""
        program = self._create_program_with_mock_optimizer(mocker)

        result = program._decode_solution_fn("1010")

        assert result == "1010"

    def test_get_top_solutions_basic_sorting(self, mocker):
        """Test that get_top_solutions sorts by probability descending."""
        probs = {
            "00": 0.1,
            "01": 0.5,
            "10": 0.3,
            "11": 0.1,
        }
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=3)

        assert len(result) == 3
        assert result[0].bitstring == "01"
        assert result[0].prob == 0.5
        assert result[1].bitstring == "10"
        assert result[1].prob == 0.3
        # For tied probabilities (0.1), lexicographic order: "00" < "11"
        assert result[2].bitstring == "00"
        assert result[2].prob == 0.1

    def test_get_top_solutions_deterministic_tie_breaking(self, mocker):
        """Test that get_top_solutions uses lexicographic tie-breaking for equal probabilities."""
        probs = {
            "111": 0.3,
            "000": 0.3,
            "101": 0.2,
            "010": 0.2,
        }
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=4)

        assert len(result) == 4
        # Tied at 0.3: "000" < "111" lexicographically
        assert result[0].bitstring == "000"
        assert result[0].prob == 0.3
        assert result[1].bitstring == "111"
        assert result[1].prob == 0.3
        # Tied at 0.2: "010" < "101" lexicographically
        assert result[2].bitstring == "010"
        assert result[2].prob == 0.2
        assert result[3].bitstring == "101"
        assert result[3].prob == 0.2

    def test_get_top_solutions_respects_n_parameter(self, mocker):
        """Test that get_top_solutions returns at most n solutions."""
        probs = {f"{i:03b}": 0.1 for i in range(8)}  # 8 solutions with equal prob
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=3)

        assert len(result) == 3
        # Should return first 3 in lexicographic order (tie-breaking)
        assert result[0].bitstring == "000"
        assert result[1].bitstring == "001"
        assert result[2].bitstring == "010"

    def test_get_top_solutions_n_exceeds_available(self, mocker):
        """Test that get_top_solutions returns all solutions when n exceeds count."""
        probs = {"00": 0.6, "01": 0.4}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=100)

        assert len(result) == 2  # Only 2 available

    def test_get_top_solutions_n_zero_returns_empty(self, mocker):
        """Test that get_top_solutions returns empty list when n=0."""
        probs = {"00": 0.5, "11": 0.5}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=0)

        assert result == []

    def test_get_top_solutions_n_negative_raises_error(self, mocker):
        """Test that get_top_solutions raises ValueError for negative n."""
        probs = {"00": 1.0}
        program = self._setup_program_with_probs(mocker, probs)

        with pytest.raises(ValueError, match="n must be non-negative"):
            program.get_top_solutions(n=-1)

    def test_get_top_solutions_min_prob_filtering(self, mocker):
        """Test that get_top_solutions filters by min_prob threshold."""
        probs = {
            "00": 0.5,
            "01": 0.3,
            "10": 0.15,
            "11": 0.05,
        }
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=10, min_prob=0.2)

        # Only "00" (0.5) and "01" (0.3) should pass the threshold
        assert len(result) == 2
        assert result[0].bitstring == "00"
        assert result[1].bitstring == "01"

    def test_get_top_solutions_min_prob_invalid_raises_error(self, mocker):
        """Test that get_top_solutions raises ValueError for invalid min_prob."""
        probs = {"00": 1.0}
        program = self._setup_program_with_probs(mocker, probs)

        with pytest.raises(ValueError, match="min_prob must be in range"):
            program.get_top_solutions(min_prob=-0.1)

        with pytest.raises(ValueError, match="min_prob must be in range"):
            program.get_top_solutions(min_prob=1.5)

    def test_get_top_solutions_include_decoded_false(self, mocker):
        """Test that get_top_solutions with include_decoded=False sets decoded to None."""
        probs = {"00": 0.6, "11": 0.4}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=2, include_decoded=False)

        assert len(result) == 2
        assert result[0].decoded is None
        assert result[1].decoded is None

    def test_get_top_solutions_include_decoded_true_calls_decode(self, mocker):
        """Test that get_top_solutions with include_decoded=True uses decode_solution_fn."""
        probs = {"00": 0.6, "11": 0.4}

        # Create a custom decode function
        def mock_decode(bitstring):
            return f"decoded_{bitstring}"

        program = self._setup_program_with_probs(
            mocker, probs, decode_solution_fn=mock_decode
        )

        result = program.get_top_solutions(n=2, include_decoded=True)

        assert len(result) == 2
        assert result[0].decoded == "decoded_00"
        assert result[1].decoded == "decoded_11"

    def test_get_top_solutions_returns_solution_entry_instances(self, mocker):
        """Test that get_top_solutions returns SolutionEntry namedtuple instances."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        probs = {"00": 0.6, "11": 0.4}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=2)

        assert len(result) == 2
        assert isinstance(result[0], SolutionEntry)
        assert isinstance(result[1], SolutionEntry)
        assert result[0].bitstring == "00"
        assert result[0].prob == 0.6

    def test_get_top_solutions_solution_entry_is_frozen(self, mocker):
        """Test that SolutionEntry is frozen (immutable)."""
        pass

        probs = {"00": 1.0}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=1)
        entry = result[0]

        # Attempting to modify immutable namedtuple should raise
        with pytest.raises((AttributeError, TypeError)):
            entry.bitstring = "11"

    def test_get_top_solutions_combined_filters(self, mocker):
        """Test get_top_solutions with both n and min_prob filters."""
        probs = {
            "0000": 0.4,
            "0001": 0.25,
            "0010": 0.15,
            "0011": 0.1,
            "0100": 0.05,
            "0101": 0.03,
            "0110": 0.02,
        }
        program = self._setup_program_with_probs(mocker, probs)

        # Request top 5, but filter out anything below 0.1
        result = program.get_top_solutions(n=5, min_prob=0.1)

        # Should get "0000" (0.4), "0001" (0.25), "0010" (0.15), "0011" (0.1)
        # Even though we asked for 5, only 4 pass the threshold
        assert len(result) == 4
        assert result[0].bitstring == "0000"
        assert result[1].bitstring == "0001"
        assert result[2].bitstring == "0010"
        assert result[3].bitstring == "0011"

    def test_get_top_solutions_empty_after_filtering(self, mocker):
        """Test get_top_solutions returns empty list when all solutions filtered out."""
        probs = {"00": 0.05, "01": 0.03, "10": 0.02}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=10, min_prob=0.1)

        assert result == []

    def test_get_top_solutions_default_parameters(self, mocker):
        """Test get_top_solutions with default parameters."""
        # Create 15 solutions to test default n=10
        probs = {f"{i:04b}": 0.1 for i in range(15)}
        program = self._setup_program_with_probs(mocker, probs)

        result = (
            program.get_top_solutions()
        )  # Should use n=10, min_prob=0.0, include_decoded=False

        assert len(result) == 10  # Default n=10
        # All probabilities equal, so should be in lexicographic order
        for i in range(10):
            assert result[i].bitstring == f"{i:04b}"
            assert result[i].decoded is None  # Default include_decoded=False


class TestSolutionEntryNamedTuple:
    """Test suite for SolutionEntry namedtuple."""

    def test_solution_entry_creation(self):
        """Test basic SolutionEntry creation."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        entry = SolutionEntry(bitstring="101", prob=0.42, decoded=[0, 2])

        assert entry.bitstring == "101"
        assert entry.prob == 0.42
        assert entry.decoded == [0, 2]

    def test_solution_entry_decoded_defaults_to_none(self):
        """Test that decoded defaults to None when not provided."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        entry = SolutionEntry(bitstring="101", prob=0.42)

        assert entry.decoded is None

    def test_solution_entry_is_frozen(self):
        """Test that SolutionEntry is immutable."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        entry = SolutionEntry(bitstring="101", prob=0.42)

        with pytest.raises((AttributeError, TypeError)):
            entry.bitstring = "010"

        with pytest.raises((AttributeError, TypeError)):
            entry.prob = 0.99

    def test_solution_entry_equality(self):
        """Test SolutionEntry equality comparison."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        entry1 = SolutionEntry(bitstring="101", prob=0.42, decoded=[0, 2])
        entry2 = SolutionEntry(bitstring="101", prob=0.42, decoded=[0, 2])
        entry3 = SolutionEntry(bitstring="101", prob=0.42, decoded=None)

        assert entry1 == entry2
        assert entry1 != entry3  # Different decoded value

    def test_solution_entry_decoded_can_be_any_type(self):
        """Test that decoded field accepts various types."""
        from divi.qprog.variational_quantum_algorithm import SolutionEntry

        # List
        entry1 = SolutionEntry(bitstring="101", prob=0.5, decoded=[1, 2, 3])
        assert entry1.decoded == [1, 2, 3]

        # Numpy array
        import numpy as np

        decoded_array = np.array([1, 0, 1])
        entry2 = SolutionEntry(bitstring="101", prob=0.5, decoded=decoded_array)
        np.testing.assert_array_equal(entry2.decoded, decoded_array)

        # String
        entry3 = SolutionEntry(bitstring="101", prob=0.5, decoded="custom_solution")
        assert entry3.decoded == "custom_solution"

        # Dict
        entry4 = SolutionEntry(bitstring="101", prob=0.5, decoded={"nodes": [1, 2]})
        assert entry4.decoded == {"nodes": [1, 2]}
