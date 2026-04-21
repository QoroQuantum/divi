# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import sympy as sp
from scipy.optimize import OptimizeResult

from divi.circuits import dag_to_qasm_body, qscript_to_meta
from divi.circuits._conversions import _qscript_to_dag
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.early_stopping import EarlyStopping, StopReason
from divi.qprog.exceptions import _CancelledError
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _compute_parameter_shift_mask,
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
        self._n_params_per_layer = 4
        self.current_iteration = 0
        self.max_iterations = 0  # Default value to prevent AttributeError

        super().__init__(backend=kwargs.pop("backend", None), **kwargs)

        self._cost_hamiltonian = (
            qml.PauliX(0) + qml.PauliZ(1) + qml.PauliX(0) @ qml.PauliZ(1)
        )
        self.loss_constant = 0.0

    def _build_pipelines(self) -> None:
        self._cost_pipeline = self._build_cost_pipeline(CircuitSpecStage())
        self._measurement_pipeline = self._build_measurement_pipeline()

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the VQA problem."""
        return self._cost_hamiltonian

    def _create_meta_circuit_factories(self):
        symbols = [sp.Symbol("beta"), *sp.symarray("theta", 3)]
        ops = [
            qml.RX(symbols[0], wires=0),
            qml.U3(*symbols[1:], wires=1),
            qml.CNOT(wires=[0, 1]),
        ]
        tape = qml.tape.QuantumScript(
            ops=ops, measurements=[qml.expval(self.cost_hamiltonian)]
        )
        meta_circuit = qscript_to_meta(tape, precision=self._precision)
        return {"cost_circuit": meta_circuit}

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

    def _perform_final_computation(self):
        pass


class TestProgram:
    """Test suite for VariationalQuantumAlgorithm core functionality."""

    def _create_sample_program(self, mocker, **kwargs):
        """Helper to create a SampleVQAProgram with common defaults."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._create_mock_optimizer(mocker)
        if "backend" not in kwargs:
            kwargs["backend"] = self._create_mock_backend(mocker)
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

        first_init = program._initialize_param_sets()[0]
        assert first_init.shape == (program.n_layers * program.n_params_per_layer,)

        second_init = program._initialize_param_sets()[0]

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, first_init, second_init
        )

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

    def test_shot_distribution_default_is_none(self, mocker):
        """Spec: omitting shot_distribution leaves the field unset (None)."""
        program = self._create_sample_program(mocker)
        assert program._shot_distribution is None

    def test_shot_distribution_stored_on_program(self, mocker):
        """Spec: explicit shot_distribution is stored verbatim."""
        program = self._create_sample_program(mocker, shot_distribution="weighted")
        assert program._shot_distribution == "weighted"

    def test_shot_distribution_suppresses_backend_expval_autoswitch(self, mocker):
        """Spec: setting shot_distribution prevents the auto-fallback to
        _backend_expval that normally happens on expval-supporting backends."""
        mock_backend = self._create_mock_backend(mocker, supports_expval=True)

        # No warning should be emitted: shot_distribution implies sampling intent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            program = self._create_sample_program(
                mocker,
                grouping_strategy="qwc",
                shot_distribution="uniform",
                backend=mock_backend,
            )
        assert program._grouping_strategy == "qwc"
        assert program._shot_distribution == "uniform"

    def test_shot_distribution_with_explicit_backend_expval_raises(self, mocker):
        """Spec: combining shot_distribution with grouping_strategy='_backend_expval'
        raises ValueError because the backend ignores shots in that mode."""
        mock_backend = self._create_mock_backend(mocker, supports_expval=True)
        with pytest.raises(ValueError, match="incompatible with grouping_strategy"):
            self._create_sample_program(
                mocker,
                grouping_strategy="_backend_expval",
                shot_distribution="weighted",
                backend=mock_backend,
            )

    def test_shot_distribution_threaded_to_measurement_stage(self, mocker):
        """Implementation detail: _build_cost_pipeline forwards shot_distribution
        to MeasurementStage's constructor."""
        program = self._create_sample_program(
            mocker, shot_distribution="weighted_random"
        )
        program._build_pipelines()

        meas_stage = next(
            stage
            for stage in program._cost_pipeline.stages
            if isinstance(stage, MeasurementStage)
        )
        assert meas_stage._shot_distribution == "weighted_random"

    def test_shot_distribution_callable_threaded_through(self, mocker):
        """Implementation detail: callable shot_distribution survives threading."""

        def custom(norms, total):
            return [total] + [0] * (len(norms) - 1)

        program = self._create_sample_program(mocker, shot_distribution=custom)
        program._build_pipelines()

        meas_stage = next(
            stage
            for stage in program._cost_pipeline.stages
            if isinstance(stage, MeasurementStage)
        )
        assert meas_stage._shot_distribution is custom

    def test_program_rng_threaded_into_pipeline_env(self, mocker):
        """Spec: VariationalQuantumAlgorithm._build_pipeline_env populates
        env.rng from self._rng so weighted_random shot allocation is
        reproducible across runs of the same seeded program."""
        program = self._create_sample_program(
            mocker, shot_distribution="weighted_random"
        )
        env = program._build_pipeline_env(param_sets=np.zeros((1, 4)))
        assert env.rng is program._rng

    def test_evaluate_cost_param_sets_uses_initial_spec_hook(self, mocker):
        """Cost evaluation should delegate to the initial-spec hook."""
        program = self._create_sample_program(mocker)
        program._build_pipelines()
        program.loss_constant = 10.0

        initial_spec_mock = mocker.patch.object(
            program, "_get_cost_pipeline_initial_spec", return_value="hook_spec"
        )
        pipeline_run = mocker.patch.object(
            program._cost_pipeline,
            "run",
            return_value={
                (("param_set", 1),): 2.0,
                (("param_set", 0),): 1.0,
            },
        )

        param_sets = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        losses = program._evaluate_cost_param_sets(param_sets)

        initial_spec_mock.assert_called_once_with()
        np.testing.assert_array_equal(
            pipeline_run.call_args.kwargs["env"].param_sets, param_sets
        )
        assert pipeline_run.call_args.kwargs["initial_spec"] == "hook_spec"
        assert list(losses.items()) == [(0, 11.0), (1, 12.0)]


class BaseVariationalQuantumAlgorithmTest:
    """Base test class for VariationalQuantumAlgorithm functionality."""

    def _create_mock_optimizer(self, mocker, n_param_sets=1):
        """Helper to create a mock optimizer with specified n_param_sets."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = n_param_sets
        return mock_optimizer

    def _create_program_with_mock_optimizer(self, mocker, **kwargs):
        """Helper method to create SampleProgram with mocked optimizer and backend."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._create_mock_optimizer(mocker, n_param_sets=1)
        if "backend" not in kwargs:
            backend = mocker.MagicMock()
            backend.shots = 1000
            backend.is_async = False
            backend.supports_expval = False
            kwargs["backend"] = backend
        return SampleVQAProgram(circ_count=1, run_time=0.1, **kwargs)

    def _setup_program_with_probs(self, mocker, probs_dict: dict[str, float], **kwargs):
        """Helper to create a program with a synthetic probability distribution."""
        program = self._create_program_with_mock_optimizer(mocker, **kwargs)
        # Wrap in nested structure: {tag: {bitstring: prob}} to match production
        program._best_probs = {"0_NoMitigation:0_ham:0_0": probs_dict}
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

    def test_run_validates_parameter_shape(self, mocker, mock_backend):
        """Test that run() validates explicit initial parameters."""
        invalid_params = np.array([[0.1, 0.2]])  # Wrong shape
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=1)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            optimizer=mock_optimizer,
            backend=mock_backend,
        )
        # max_iterations=0 would emit a max_iterations-exhausted warning
        # before the shape check; bump it so the validation path runs first.
        program.max_iterations = 1

        with pytest.raises(ValueError, match="Initial parameters must have shape"):
            program.run(initial_params=invalid_params)

    def test_run_uses_explicit_initial_params(self, mocker):
        """Test that run() forwards explicit initial_params to the optimizer."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )

        custom_initial_params = np.array([[1.0, 2.0, 3.0, 4.0]])
        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        mock_optimizer = self._setup_mock_optimizer_single_run(
            mocker, program, final_params
        )

        program.run(initial_params=custom_initial_params)

        assert mock_optimizer.optimize.called
        call_args = mock_optimizer.optimize.call_args
        np.testing.assert_array_equal(
            call_args.kwargs["initial_params"], custom_initial_params
        )

    def test_run_initializes_params_when_not_provided(self, mocker):
        """Test that run() generates fresh initial parameters when needed."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        mock_init = mocker.spy(program, "_initialize_param_sets")

        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        self._setup_mock_optimizer_single_run(mocker, program, final_params)

        program.run()

        mock_init.assert_called_once()
        expected_shape = (
            program.optimizer.n_param_sets,
            program.n_layers * program.n_params_per_layer,
        )
        call_args = program.optimizer.optimize.call_args
        assert call_args.kwargs["initial_params"].shape == expected_shape

    @pytest.mark.parametrize(
        "n_param_sets,initial_params_shape",
        [
            (1, (1, 4)),  # Single parameter set
            (2, (2, 4)),  # Multiple parameter sets
        ],
    )
    def test_run_accepts_initial_params_for_multiple_shapes(
        self, mocker, mock_backend, n_param_sets, initial_params_shape
    ):
        """Test that run() accepts explicit initial_params with the expected shape."""
        custom_params = np.random.uniform(0, 2 * np.pi, initial_params_shape)
        mock_optimizer = self._create_mock_optimizer(mocker, n_param_sets=n_param_sets)
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=mock_optimizer,
        )

        program.max_iterations = 1

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )

        final_params = np.array([[0.5, 1.0, 1.5, 2.0]])
        self._setup_mock_optimizer_single_run(mocker, program, final_params)

        program.run(initial_params=custom_params)

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

    def test_unexpected_constructor_kwargs_raise(self, mocker, mock_backend):
        """Unknown constructor kwargs should fail fast instead of being ignored."""
        with pytest.raises(TypeError, match="Unexpected keyword argument\\(s\\): foo"):
            SampleVQAProgram(
                circ_count=1,
                run_time=0.1,
                backend=mock_backend,
                foo="bar",
            )


class TestRunIntegration(BaseVariationalQuantumAlgorithmTest):
    """Test suite for the integration of the run method's components."""

    def setup_mock_optimizer(self, program, mocker, side_effects):
        """Configures a mock optimizer that executes callbacks."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = program.optimizer.n_param_sets

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            last_result = None
            for params, _ in side_effects:
                # Call the actual cost function to trigger the patched evaluator
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

        mock_run_circuits = mocker.patch.object(program, "_evaluate_cost_param_sets")
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
        assert len(program.param_history()) == 3
        np.testing.assert_allclose(program.param_history()[0], curr_params1)
        np.testing.assert_allclose(program.param_history()[2], curr_params3)
        best_only = program.param_history(mode="best_per_iteration")
        assert len(best_only) == 3
        np.testing.assert_allclose(best_only[0], curr_params1)
        np.testing.assert_allclose(best_only[2], curr_params3)
        assert mock_run_circuits.call_count == 3

    def test_param_history_best_per_iteration_multi_member(self, mocker):
        """best_per_iteration picks the population row with minimum loss each step."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.optimizer.n_param_sets = 2
        program.max_iterations = 2

        mock_run_circuits = mocker.patch.object(program, "_evaluate_cost_param_sets")
        mock_run_circuits.side_effect = [
            {0: 0.0, 1: -1.0},
            {0: -2.0, 1: 0.0},
        ]

        row0 = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        row1 = np.array([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
        self.setup_mock_optimizer(
            program,
            mocker,
            [(row0, None), (row1, None)],
        )

        program.run()

        best = program.param_history(mode="best_per_iteration")
        assert len(best) == 2
        np.testing.assert_allclose(best[0], [[1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_allclose(best[1], [[2.0, 2.0, 2.0, 2.0]])
        full = program.param_history(mode="all_evaluated")
        np.testing.assert_allclose(full[0], row0)
        np.testing.assert_allclose(full[1], row1)

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

        result = program.run()

        assert result is program
        assert program.total_circuit_count == program._total_circuit_count
        assert program.total_run_time == program._total_run_time

    def _setup_program_for_final_computation_test(self, mocker):
        """Helper to set up program with mocks for final computation tests."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
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
        # The mock optimizer controls iteration count; set max_iterations high
        # enough that ``run()``'s default check doesn't short-circuit with a
        # "max_iterations <= current_iteration" warning.
        program.max_iterations = 10
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
        assert isinstance(loaded_program.optimizer, MonteCarloOptimizer)
        assert len(loaded_program._param_history) == 5
        np.testing.assert_allclose(
            loaded_program._param_history[0],
            [[0.1, 0.2, 0.3, 0.4]],
        )

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
            match="max_iterations \\(2\\) is less than or equal to current_iteration \\(3\\)",
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

    def test_precision_used_in_qasm_conversion(self):
        """Precision controls decimal digits in dag_to_qasm_body output."""
        # Circuit with a known numeric constant (not symbolic).
        qscript = qml.tape.QuantumScript(
            ops=[qml.RZ(0.123456789, 0)],
            measurements=[qml.expval(qml.Z(0))],
        )
        dag, _, _ = _qscript_to_dag(qscript)

        body5 = dag_to_qasm_body(dag, precision=5)
        body3 = dag_to_qasm_body(dag, precision=3)
        assert "0.12346" in body5  # 5 digits, rounded
        assert "0.123" in body3  # 3 digits, truncated

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

            # Verify precision propagates to created MetaCircuits
            meta_circuit = program.meta_circuit_factories["cost_circuit"]
            assert meta_circuit.precision == precision


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

    def test_param_history_warn_before_optimization(self, mocker):
        """param_history() warns when called before optimization runs."""
        program = self._create_program_with_mock_optimizer(mocker)
        with pytest.warns(UserWarning, match="Parameter history is unavailable"):
            assert program.param_history() == []
            assert program.param_history(mode="best_per_iteration") == []

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
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
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


class TestEarlyStoppingIntegration(BaseVariationalQuantumAlgorithmTest):
    """Integration tests for early stopping within the full run() loop."""

    def _setup_optimizer_with_flat_losses(
        self, program, mocker, n_iterations, loss_value=-0.5
    ):
        """Configure a mock optimizer that produces constant (flat) losses."""
        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = program.optimizer.n_param_sets

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            last_result = None
            for i in range(n_iterations):
                params = np.full_like(initial_params, float(i))
                _ = cost_fn(params)
                result = OptimizeResult(x=params, fun=np.array([loss_value]))
                callback_fn(result)
                last_result = result
            return last_result

        mock_optimizer.optimize.side_effect = mock_optimize_logic
        program.optimizer = mock_optimizer

    def test_early_stopping_patience_stops_run(self, mocker):
        """Verify that patience-based early stopping terminates the run early."""
        es = EarlyStopping(patience=3, min_delta=0.0)
        program = self._create_program_with_mock_optimizer(
            mocker, seed=42, early_stopping=es
        )
        program.max_iterations = 100  # Would run forever without early stopping

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=100)

        program.run()

        # Should have stopped after patience+1 iterations (1 to set best, then 3 stale)
        assert program.current_iteration == 4
        assert program.stop_reason == StopReason.PATIENCE_EXCEEDED

    def test_no_early_stopping_runs_all_iterations(self, mocker):
        """Verify normal behavior when early_stopping is None."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 5

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=5)

        program.run()

        assert program.current_iteration == 5
        assert program.stop_reason is None

    def test_stop_reason_is_none_before_run(self, mocker):
        """Verify stop_reason is None before run() is called."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        assert program.stop_reason is None

    def test_stop_reason_is_none_when_not_triggered(self, mocker):
        """Verify stop_reason stays None when loss keeps improving."""
        es = EarlyStopping(patience=3, min_delta=0.0)
        program = self._create_program_with_mock_optimizer(
            mocker, seed=42, early_stopping=es
        )
        program.max_iterations = 5

        # Each iteration produces a better loss
        losses = [{0: -0.1 * (i + 1)} for i in range(5)]
        mocker.patch.object(program, "_evaluate_cost_param_sets", side_effect=losses)

        mock_optimizer = mocker.MagicMock()
        mock_optimizer.n_param_sets = program.optimizer.n_param_sets

        def mock_optimize_logic(cost_fn, initial_params, callback_fn, **kwargs):
            last_result = None
            for i in range(5):
                params = np.full_like(initial_params, float(i))
                actual_loss = cost_fn(params)
                result = OptimizeResult(x=params, fun=actual_loss)
                callback_fn(result)
                last_result = result
            return last_result

        mock_optimizer.optimize.side_effect = mock_optimize_logic
        program.optimizer = mock_optimizer

        program.run()

        assert program.current_iteration == 5
        assert program.stop_reason is None

    def test_final_computation_still_runs_after_early_stop(self, mocker):
        """Verify _perform_final_computation is called after early stopping."""
        es = EarlyStopping(patience=2, min_delta=0.0)
        program = self._create_program_with_mock_optimizer(
            mocker, seed=42, early_stopping=es
        )
        program.max_iterations = 100

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=100)
        mock_final = mocker.spy(program, "_perform_final_computation")

        program.run()

        assert program.stop_reason == StopReason.PATIENCE_EXCEEDED
        mock_final.assert_called_once()


class TestComputeParameterShiftMask:
    """Spec: _compute_parameter_shift_mask produces a valid parameter shift rule matrix."""

    @pytest.mark.parametrize("n_params", [1, 2, 3, 5, 8])
    def test_shape(self, n_params):
        mask = _compute_parameter_shift_mask(n_params)
        assert mask.shape == (2 * n_params, n_params)

    @pytest.mark.parametrize("n_params", [1, 2, 3, 5, 8])
    def test_row_pairs_are_opposite_signs(self, n_params):
        """Positive and negative shifts must be mirrors of each other."""
        mask = _compute_parameter_shift_mask(n_params)
        for i in range(n_params):
            np.testing.assert_array_equal(mask[2 * i], -mask[2 * i + 1])

    @pytest.mark.parametrize("n_params", [1, 2, 3, 4, 5, 8])
    def test_each_row_pair_shifts_exactly_one_parameter(self, n_params):
        """Each row pair should shift exactly one parameter at column i."""
        mask = _compute_parameter_shift_mask(n_params)
        for i in range(n_params):
            row = mask[2 * i]
            nonzero_cols = np.nonzero(row)[0]
            assert (
                len(nonzero_cols) == 1
            ), f"Row pair {i} shifts {len(nonzero_cols)} params: columns {nonzero_cols.tolist()}"
            assert nonzero_cols[0] == i

    @pytest.mark.parametrize("n_params", [1, 2, 3, 5, 8])
    def test_shift_magnitude_is_half_pi(self, n_params):
        mask = _compute_parameter_shift_mask(n_params)
        nonzero = mask[mask != 0]
        np.testing.assert_allclose(np.abs(nonzero), 0.5 * np.pi)

    def test_known_values_n_equals_1(self):
        mask = _compute_parameter_shift_mask(1)
        expected = np.array([[np.pi / 2], [-np.pi / 2]])
        np.testing.assert_array_equal(mask, expected)

    def test_known_values_n_equals_2(self):
        mask = _compute_parameter_shift_mask(2)
        h = np.pi / 2
        expected = np.array(
            [
                [h, 0],
                [-h, 0],
                [0, h],
                [0, -h],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestGradientFunction(BaseVariationalQuantumAlgorithmTest):
    """Spec: grad_fn correctly computes parameter-shift gradients from pipeline results."""

    def _create_lbfgsb_program(self, mocker, n_params_per_layer=4, **kwargs):
        """Create a SampleVQAProgram with L-BFGS-B optimizer."""
        optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        backend = mocker.MagicMock()
        backend.shots = 1000
        backend.is_async = False
        backend.supports_expval = False
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            optimizer=optimizer,
            backend=backend,
            seed=42,
            **kwargs,
        )
        program._n_params_per_layer = n_params_per_layer
        program.max_iterations = 2
        return program

    def test_gradient_with_known_return_values(self, mocker):
        """grad_fn produces 0.5 * (positive_shift_values - negative_shift_values)."""
        program = self._create_lbfgsb_program(mocker, n_params_per_layer=3)
        n_params = program.n_layers * program.n_params_per_layer

        # Predetermined values: index i returns float(i)
        # Even indices (0, 2, 4) are positive shifts, odd (1, 3, 5) are negative
        mock_values = {i: float(i) for i in range(2 * n_params)}
        expected_grads = np.array(
            [
                0.5 * (mock_values[2 * i] - mock_values[2 * i + 1])
                for i in range(n_params)
            ]
        )

        grad_call_count = [0]

        def mock_run(param_sets, **kwargs):
            n_sets = np.atleast_2d(param_sets).shape[0]
            if n_sets == 1:
                return {0: -0.5}
            else:
                grad_call_count[0] += 1
                return {i: mock_values[i] for i in range(n_sets)}

        mocker.patch.object(program, "_evaluate_cost_param_sets", side_effect=mock_run)
        program.run(perform_final_computation=False)

        assert grad_call_count[0] >= 1, "grad_fn was never called"
        # L-BFGS-B stores the jacobian in optimize_result.jac
        np.testing.assert_allclose(program.optimize_result.jac, expected_grads)

    def test_gradient_is_zero_when_all_shifts_equal(self, mocker):
        """When all shifted evaluations return the same value, gradient is zero."""
        program = self._create_lbfgsb_program(mocker, n_params_per_layer=3)

        def mock_run(param_sets, **kwargs):
            n_sets = np.atleast_2d(param_sets).shape[0]
            return {i: -0.5 for i in range(n_sets)}

        mocker.patch.object(program, "_evaluate_cost_param_sets", side_effect=mock_run)
        program.run(perform_final_computation=False)

        # L-BFGS-B converges immediately on a flat landscape (zero gradient)
        np.testing.assert_allclose(program.optimize_result.jac, 0.0, atol=1e-10)

    def test_shifted_params_are_mask_plus_input(self, mocker):
        """During gradient computation, shifted param sets equal mask + original params."""
        program = self._create_lbfgsb_program(mocker, n_params_per_layer=3)
        n_params = program.n_layers * program.n_params_per_layer

        captured = []
        initial_params = np.full((1, n_params), 0.25)

        def mock_run(param_sets, **kwargs):
            param_sets = np.atleast_2d(param_sets)
            n_sets = param_sets.shape[0]
            if n_sets > 1:
                captured.append(param_sets.copy())
            return {i: float(i) * 0.1 for i in range(n_sets)}

        mocker.patch.object(program, "_evaluate_cost_param_sets", side_effect=mock_run)

        program.run(
            initial_params=initial_params,
            perform_final_computation=False,
        )

        assert len(captured) >= 1
        # First gradient call uses the initial params
        shifted = captured[0]
        assert shifted.shape == (2 * n_params, n_params)

        # Each row pair should differ from the initial params by exactly ±π/2
        # in exactly one column (the parameter being shifted)
        for i in range(n_params):
            diff_pos = shifted[2 * i] - initial_params.squeeze()
            nonzero = np.nonzero(np.abs(diff_pos) > 1e-10)[0]
            assert len(nonzero) == 1, f"Row {2*i} shifts {len(nonzero)} params"
            assert nonzero[0] == i
            np.testing.assert_allclose(np.abs(diff_pos[nonzero[0]]), np.pi / 2)


class TestLBFGSBGradientIntegration(BaseVariationalQuantumAlgorithmTest):
    """Integration: L-BFGS-B uses the gradient function during optimization."""

    def test_lbfgsb_evaluates_gradient(self, mocker):
        """L-BFGS-B calls the gradient function, producing 2*n_params shifted param sets."""
        optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        backend = mocker.MagicMock()
        backend.shots = 1000
        backend.is_async = False
        backend.supports_expval = False

        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            optimizer=optimizer,
            backend=backend,
            seed=42,
        )
        program.max_iterations = 2
        n_params = program.n_layers * program.n_params_per_layer

        grad_calls = []

        def mock_run(param_sets, **kwargs):
            n_sets = np.atleast_2d(param_sets).shape[0]
            if n_sets > 1:
                grad_calls.append(n_sets)
            return {i: float(i) * 0.1 - 0.5 for i in range(n_sets)}

        mocker.patch.object(program, "_evaluate_cost_param_sets", side_effect=mock_run)
        program.run(perform_final_computation=False)

        # L-BFGS-B must have called the gradient function at least once
        assert len(grad_calls) >= 1
        # Each gradient call should use exactly 2*n_params shifted parameter sets
        for n_sets in grad_calls:
            assert n_sets == 2 * n_params
        # The optimize result should record gradient evaluations
        assert program.optimize_result.njev >= 1
