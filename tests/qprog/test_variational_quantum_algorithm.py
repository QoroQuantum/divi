# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global
from pennylane.measurements import ExpectationMP
from scipy.optimize import OptimizeResult

from divi.circuits import MetaCircuit
from divi.circuits.qem import ZNE
from divi.qprog.exceptions import _CancelledError
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _batched_expectation,
)


class SampleProgram(VariationalQuantumAlgorithm):
    def __init__(self, circ_count, run_time, **kwargs):
        self.circ_count = circ_count
        self.run_time = run_time

        self.n_layers = 1
        self._n_params = 4
        self.current_iteration = 0
        self.max_iterations = 0  # Default value to prevent AttributeError

        super().__init__(backend=None, **kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()
        self.loss_constant = 0.0

    def _create_meta_circuits_dict(self):
        def simple_circuit(params):
            qml.RX(params[0], wires=0)
            qml.U3(*params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(
                qml.PauliX(0) + qml.PauliZ(1) + qml.PauliX(0) @ qml.PauliZ(1)
            )

        symbols = [sp.Symbol("beta"), sp.symarray("theta", 3)]
        meta_circuit = MetaCircuit(
            main_circuit=qml.tape.make_qscript(simple_circuit)(symbols),
            symbols=symbols,
            grouping_strategy=self._grouping_strategy,
        )
        return {"cost_circuit": meta_circuit}

    def _generate_circuits(self, params=None, **kwargs):
        pass

    def run(self, data_file=None):
        return super().run(data_file)

    def _perform_final_computation(self):
        pass


class TestProgram:
    """Test suite for VariationalQuantumAlgorithm core functionality."""

    def test_correct_random_behavior(self, mocker):
        """Test that random number generation works correctly with seeds."""
        program = SampleProgram(10, 5.5, seed=1997)

        program.optimizer = mocker.MagicMock()
        program.optimizer.n_param_sets = 1

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
            program = SampleProgram(10, 5.5, seed=1997, grouping_strategy=strategy)
            program.loss_constant = 0.5
            program.optimizer = mocker.MagicMock()
            program.optimizer.n_param_sets = 1
            program.backend = mocker.MagicMock()
            program.backend.shots = 100

            meta_circuit = program._meta_circuits["cost_circuit"]
            assert len(meta_circuit.measurement_groups) == expected_n_groups
            assert (
                len(tuple(filter(lambda x: "h" in x, meta_circuit.measurements)))
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
        program = SampleProgram(10, 5.5, seed=1997, qem_protocol=zne_protocol)
        program.loss_constant = 0.0
        program.backend = mocker.MagicMock()
        program.backend.shots = 100

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


class BaseVariationalQuantumAlgorithmTest:
    """Base test class for VariationalQuantumAlgorithm functionality."""

    def _create_program_with_mock_optimizer(self, mocker, **kwargs):
        """Helper method to create SampleProgram with mocked optimizer."""
        program = SampleProgram(circ_count=1, run_time=0.1, **kwargs)
        program.optimizer = mocker.MagicMock()
        program.optimizer.n_param_sets = 1
        return program


class TestInitialParameters(BaseVariationalQuantumAlgorithmTest):
    """Test suite for initial parameters functionality."""

    def test_initial_params_returns_numpy_array_not_none(self, mocker):
        """Test that initial_params property always returns actual parameters."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        params = program.initial_params
        assert isinstance(params, np.ndarray)
        assert params is not None
        assert params.shape == program.get_expected_param_shape()

    def test_initial_params_triggers_lazy_initialization_on_first_access(self, mocker):
        """Test that accessing initial_params triggers parameter generation."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        assert program._curr_params is None
        params = program.initial_params
        assert program._curr_params is not None
        assert isinstance(params, np.ndarray)

    def test_initial_params_setter_stores_custom_parameters(self, mocker):
        """Test that setting initial_params stores user-provided parameters."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        expected_shape = program.get_expected_param_shape()
        custom_params = np.random.uniform(0, 2 * np.pi, expected_shape)
        program.initial_params = custom_params
        assert np.array_equal(program.initial_params, custom_params)
        assert np.array_equal(program._curr_params, custom_params)

    def test_initial_params_setter_validates_parameter_shape(self, mocker):
        """Test that setter validates parameter shape matches expected shape."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        wrong_shape_params = np.array([[0.1, 0.2]])
        with pytest.raises(ValueError, match="Initial parameters must have shape"):
            program.initial_params = wrong_shape_params

    def test_initial_params_returns_copy_not_reference(self, mocker):
        """Test that initial_params property returns a copy, not a reference."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        params1 = program.initial_params
        params2 = program.initial_params
        assert not np.shares_memory(params1, params2)
        assert not np.shares_memory(params1, program._curr_params)
        assert np.array_equal(params1, params2)


class TestRunIntegration(BaseVariationalQuantumAlgorithmTest):
    """Test suite for the integration of the run method's components."""

    def setup_mock_optimizer(self, program, mocker, side_effects):
        """Configures a mock optimizer that executes callbacks."""
        mock_optimizer = mocker.MagicMock()

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

        params1 = np.array([[0.1, 0.2, 0.3, 0.4]])
        params2 = np.array([[0.5, 0.6, 0.7, 0.8]])
        params3 = np.array([[0.9, 1.0, 1.1, 1.2]])

        optimizer_side_effects = [
            (params1, -0.8),
            (params2, -0.5),
            (params3, -0.9),
        ]
        self.setup_mock_optimizer(program, mocker, optimizer_side_effects)

        program.run()

        # Assertions
        assert program.best_loss == -0.9
        np.testing.assert_allclose(program.best_params, params3.flatten())
        np.testing.assert_allclose(program.final_params, params3)
        assert len(program.losses_history) == 3
        assert program.losses_history[0][0] == -0.8
        assert program.losses_history[2][0] == -0.9
        assert mock_run_circuits.call_count == 3

    def test_run_method_cancellation_handling(self, mocker):
        """Test run method exits gracefully when a cancellation event is set."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1
        mock_event = mocker.MagicMock()
        mock_event.is_set.return_value = True
        program._cancellation_event = mock_event

        mock_optimizer = program.optimizer = mocker.MagicMock()
        mock_optimizer.optimize.side_effect = _CancelledError("Cancellation requested")

        circ_count, run_time = program.run()

        assert circ_count == program._total_circuit_count
        assert run_time == program._total_run_time

    def test_run_with_data_storage(self, mocker, tmp_path):
        """Test that run calls save_iteration when requested."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1
        data_file = tmp_path / "test_data.pkl"
        mock_save = mocker.patch.object(program, "save_iteration")

        # 1. Mock the function that does the external communication.
        #    It needs to return a list of dicts with 'label' and 'results' keys.
        mock_prepare = mocker.patch.object(program, "_prepare_and_send_circuits")
        mock_prepare.return_value = [{"label": "0_mock:0_0", "results": {"00": 1}}]

        # 2. Mock the function that does complex processing.
        #    This ensures the optimizer gets a valid loss value back.
        final_loss = -0.5
        mocker.patch.object(
            program, "_post_process_results", return_value={0: final_loss}
        )

        final_params = np.array([[0.1, 0.2, 0.3, 0.4]])
        optimizer_side_effects = [(final_params, final_loss)]
        self.setup_mock_optimizer(program, mocker, optimizer_side_effects)

        program.run(data_file=str(data_file))

        mock_save.assert_called_once_with(str(data_file))
