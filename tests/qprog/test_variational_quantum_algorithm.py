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
        def simple_circuit(params):
            qml.RX(params[0], wires=0)
            qml.U3(*params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(self.cost_hamiltonian)

        symbols = [sp.Symbol("beta"), sp.symarray("theta", 3)]
        meta_circuit = MetaCircuit(
            source_circuit=qml.tape.make_qscript(simple_circuit)(symbols),
            symbols=symbols,
            grouping_strategy=self._grouping_strategy,
        )
        return {"cost_circuit": meta_circuit}

    def _generate_circuits(self, **kwargs):
        """Generate circuits - dummy implementation for testing."""
        return []

    def run(
        self,
        perform_final_computation: bool = True,
        data_file: str | None = None,
        **kwargs,
    ) -> tuple[int, float]:
        return super().run(
            perform_final_computation=perform_final_computation,
            data_file=data_file,
            **kwargs,
        )

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
        program.optimizer.optimize.side_effect = _CancelledError(
            "Cancellation requested"
        )

        circ_count, run_time = program.run()

        assert circ_count == program._total_circuit_count
        assert run_time == program._total_run_time

    def test_run_with_data_storage(self, mocker, mock_backend, tmp_path):
        """Test that run calls save_iteration when requested."""
        program = self._create_program_with_mock_optimizer(mocker, backend=mock_backend)
        program.max_iterations = 1
        data_file = tmp_path / "test_data.pkl"

        mocker.patch.object(program, "save_iteration")
        mocker.patch.object(
            program,
            "_prepare_and_send_circuits",
            return_value=[{"label": "0_mock:0_0", "results": {"00": 1}}],
        )

        final_loss = -0.5
        mocker.patch.object(
            program, "_post_process_results", return_value={0: final_loss}
        )

        final_params = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.setup_mock_optimizer(program, mocker, [(final_params, final_loss)])

        program.run(data_file=str(data_file))

        program.save_iteration.assert_called_once_with(str(data_file))

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
            ([], []),  # Empty history
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
