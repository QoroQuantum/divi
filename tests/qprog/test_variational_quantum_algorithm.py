# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import OptimizeResult

import divi.qprog._program_checkpoint as program_checkpoint_module
from divi.circuits import MetaCircuit
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import CircuitPreprocessor, CostEstimate
from divi.qprog._program_checkpoint import VQACheckpoint
from divi.qprog.checkpointing import (
    PROGRAM_COMPLETION_FILE,
    CheckpointConfig,
    list_checkpoints,
)
from divi.qprog.early_stopping import EarlyStopping, StopReason
from divi.qprog.mixins import SolutionEntry, SolutionSamplingMixin
from divi.qprog.optimizers import (
    GridSearchOptimizer,
    MonteCarloOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    SPSAOptimizer,
)
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.variational_quantum_algorithm import (
    VariationalQuantumAlgorithm,
    _compute_parameter_shift_rule,
)
from divi.reporting._events import ProgressEvent, TerminalStatus
from tests._helpers import exact_match
from tests.pipeline._helpers import meta_from_circuit
from tests.qprog.algorithms._helpers import seed_best_probs


@pytest.fixture
def mock_backend(make_dummy_simulator):
    return make_dummy_simulator(1000)


class SampleVQAProgram(SolutionSamplingMixin, VariationalQuantumAlgorithm):
    def __init__(self, circ_count, run_time, n_params_per_layer=4, **kwargs):
        self.circ_count = circ_count
        self.run_time = run_time
        self._n_params_per_layer = n_params_per_layer

        self.n_layers = 1
        self.current_iteration = 0
        self.max_iterations = 0  # Default value to prevent AttributeError

        super().__init__(backend=kwargs.pop("backend", None), **kwargs)

        self.cost_hamiltonian = SparsePauliOp.from_sparse_list(
            [("X", [0], 1.0), ("Z", [1], 1.0), ("XZ", [0, 1], 1.0)], num_qubits=2
        )
        self.loss_constant = 0.0
        self.checkpointed_value = None

    @property
    def n_params_per_layer(self) -> int:
        return self._n_params_per_layer

    def _create_cost_circuit(self) -> MetaCircuit:
        qc = QuantumCircuit(2)
        qc.rx(Parameter("beta"), 0)
        qc.u(*(Parameter(f"theta_{i}") for i in range(3)), 1)
        qc.cx(0, 1)
        return meta_from_circuit(
            qc,
            parameters=tuple(qc.parameters),
            observable=self.cost_hamiltonian,
            precision=self._precision,
        )

    def _run_solution_measurement_for(self, param_sets, *, backend=None):
        # This double's circuit measures an expectation value, not a sampling
        # distribution, so the real PROBS pipeline would yield malformed
        # ``best_probs``. Sampling-distribution behaviour is exercised by the
        # concrete VQE/QAOA suites; here it is inert.
        return

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
        return {
            **super()._save_subclass_state(),
            "checkpointed_value": self.checkpointed_value,
        }

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load SampleVQAProgram-specific state."""
        super()._load_subclass_state(state)
        self.checkpointed_value = state.get("checkpointed_value")


class _BaseSampler(QuantumProgram):
    """Minimal non-VQA host."""

    def has_results(self) -> bool:
        return bool(self._results)

    def run(self):
        return self


class _NonVQASampler(SolutionSamplingMixin, _BaseSampler):
    """A solution sampler that is NOT a VariationalQuantumAlgorithm.

    Guards the host-agnostic contract: the mixin must not reach for any
    variational-only attribute (``n_layers``, ``_best_params``, ...).
    """

    def __init__(self, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        self._meta = meta_from_circuit(
            qc, measured_wires=(0, 1), precision=self._precision
        )

    @property
    def meta_circuit_factories(self):
        return {"cost_circuit": self._meta}

    def _initial_spec(self):
        return self._meta


def test_solution_sampling_mixin_works_on_non_vqa_host(dummy_simulator):
    """The mixin samples successfully on a plain QuantumProgram host.

    Guards the host-agnostic contract: the mixin must not reach for any
    variational-only attribute (``n_layers``, ``_best_params``, ...).
    """
    host = _NonVQASampler(backend=dummy_simulator)

    # State owned by the mixin's __init__, not inherited from any VQA.
    assert host._results == {}
    assert host._decode_solution_fn("0101") == "0101"
    assert "sample" in [protocol.name for protocol in host._preprocessors()]

    # No trainable parameters: one empty parameter set.
    host.sample_solution(params=np.empty((1, 0), dtype=np.float64))

    assert host._results["best_probs"]  # populated by the real PROBS sample pipeline
    top = host.get_top_solutions(n=2)
    assert top and isinstance(top[0], SolutionEntry)


def test_solution_sampling_mixin_routes_sampling_to_configured_backend(
    dummy_simulator, make_dummy_simulator, mocker
):
    """A configured sampling backend handles PROBS instead of the host backend."""
    sampling_backend = make_dummy_simulator(100, seed=7)
    primary_submit = mocker.spy(dummy_simulator, "submit_circuits")
    sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
    host = _NonVQASampler(
        backend=dummy_simulator,
        sampling_backend=sampling_backend,
    )

    assert host.sampling_backend is sampling_backend
    host.sample_solution(params=np.empty((1, 0), dtype=np.float64))

    primary_submit.assert_not_called()
    sampling_submit.assert_called_once()


def test_sample_solution_backend_override_wins_over_configured_backend(
    dummy_simulator, make_dummy_simulator, mocker
):
    """A one-call override wins without replacing the configured backend."""
    sampling_backend = make_dummy_simulator(100, seed=7)
    override_backend = make_dummy_simulator(100, seed=11)
    configured_submit = mocker.spy(sampling_backend, "submit_circuits")
    override_submit = mocker.spy(override_backend, "submit_circuits")
    host = _NonVQASampler(
        backend=dummy_simulator,
        sampling_backend=sampling_backend,
    )

    host.sample_solution(
        params=np.empty((1, 0), dtype=np.float64),
        backend=override_backend,
    )

    configured_submit.assert_not_called()
    override_submit.assert_called_once()
    assert host.sampling_backend is sampling_backend


class BaseVariationalQuantumAlgorithmTest:
    """Base test class for VariationalQuantumAlgorithm functionality."""

    @pytest.fixture(autouse=True)
    def _dummy_backend_factory(self, make_dummy_simulator):
        self._make_backend = make_dummy_simulator

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
            kwargs["backend"] = self._make_backend(1000)
        return SampleVQAProgram(circ_count=1, run_time=0.1, **kwargs)

    def _setup_program_with_probs(self, mocker, probs_dict: dict[str, float], **kwargs):
        """Helper to create a program with a synthetic probability distribution."""
        program = self._create_program_with_mock_optimizer(mocker, **kwargs)
        seed_best_probs(program, probs_dict)
        return program


class TestProgram(BaseVariationalQuantumAlgorithmTest):
    """Test suite for VariationalQuantumAlgorithm core functionality."""

    def _create_sample_program(self, mocker, **kwargs):
        """Helper to create a SampleVQAProgram with common defaults."""
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._create_mock_optimizer(mocker)
        if "backend" not in kwargs:
            kwargs["backend"] = self._make_backend(100)
        return SampleVQAProgram(10, 5.5, seed=1997, **kwargs)

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

    def test_shot_distribution_threaded_to_measurement_stage(self, mocker):
        """Implementation detail: the program's measurement-stage factory forwards
        shot_distribution to MeasurementStage's constructor."""
        program = self._create_sample_program(
            mocker, shot_distribution="weighted_random"
        )

        assert program._make_measurement_stage()._shot_distribution == "weighted_random"

    def test_shot_distribution_callable_threaded_through(self, mocker):
        """Implementation detail: callable shot_distribution survives threading."""

        def custom(norms, total):
            return [total] + [0] * (len(norms) - 1)

        program = self._create_sample_program(mocker, shot_distribution=custom)

        assert program._make_measurement_stage()._shot_distribution is custom

    def test_program_rng_threaded_into_pipeline_env(self, mocker):
        """Spec: VariationalQuantumAlgorithm._build_pipeline_env populates
        env.rng from self._rng so weighted_random shot allocation is
        reproducible across runs of the same seeded program."""
        program = self._create_sample_program(
            mocker, shot_distribution="weighted_random"
        )
        env = program._build_pipeline_env(param_sets=np.zeros((1, 4)))
        assert env.rng is program._rng

    def test_build_pipeline_env_does_not_consume_rng(self, mocker):
        """Building an env is pure: with no param_sets it uses deterministic zeros,
        never an RNG draw, so it cannot shift the program or optimizer streams."""
        program = self._create_sample_program(mocker)
        rng_before = program._rng.bit_generator.state
        opt_before = program._optimizer_rng.bit_generator.state

        program._build_pipeline_env()  # no param_sets -> zeros placeholder

        assert program._rng.bit_generator.state == rng_before
        assert program._optimizer_rng.bit_generator.state == opt_before

    def test_optimizer_rng_is_isolated_from_program_rng(self, mocker):
        """The optimizer stream is spawned independently from the program RNG."""
        program = self._create_sample_program(mocker)
        assert program._optimizer_rng is not program._rng
        assert (
            program._optimizer_rng.bit_generator.state
            != program._rng.bit_generator.state
        )

    def test_evaluate_cost_param_sets_uses_initial_spec_seed(self, mocker):
        """Cost evaluation seeds from the ``_initial_spec`` hook, threads the param sets,
        adds the loss constant, and returns results sorted by param-set index."""
        program = self._create_sample_program(mocker)
        program.loss_constant = 10.0

        mocker.patch.object(program, "_initial_spec", return_value="hook_spec")
        execute = mocker.patch.object(
            program,
            "_execute",
            return_value={
                (("param_set", 1),): [2.0],
                (("param_set", 0),): [1.0],
            },
        )

        param_sets = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        losses = program._evaluate_cost_param_sets(param_sets)

        # ``_execute(pipeline, initial_spec, **env_overrides)``: the seed comes
        # from the ``_initial_spec`` hook and the param sets ride the env overrides.
        assert execute.call_args.args[1] == "hook_spec"
        np.testing.assert_array_equal(
            execute.call_args.kwargs["param_sets"], param_sets
        )
        assert list(losses.items()) == [(0, 11.0), (1, 12.0)]

    def test_evaluate_cost_param_sets_threads_estimator_samples(self, mocker):
        program = self._create_sample_program(mocker)
        evaluate = mocker.patch.object(
            program,
            "evaluate",
            return_value=({0: [1.0], 1: [2.0]}, {0: 0.1, 1: 0.2}),
        )
        param_sets = np.zeros((2, 4))

        losses = program._evaluate_cost_param_sets(
            param_sets,
            estimator_samples=[2, 8],
            collect_variance=True,
        )

        assert evaluate.call_args.kwargs["estimator_samples"] == [2, 8]
        assert evaluate.call_args.kwargs["return_variance"] is True
        assert losses == {0: 1.0, 1: 2.0}

    def test_evaluate_routes_initial_spec_and_params_to_execute(self, mocker):
        """``evaluate`` itself (not only ``_evaluate_cost_param_sets``) wires the
        ``_initial_spec`` seed and param sets into the pipeline execution."""
        program = self._create_sample_program(mocker)
        mocker.patch.object(program, "_initial_spec", return_value="hook_spec")
        execute = mocker.patch.object(
            program, "_execute", return_value={(("param_set", 0),): [1.0]}
        )

        params = np.array([[0.1, 0.2, 0.3, 0.4]])
        program.evaluate(params, program.cost_preprocessor())

        assert execute.call_args.args[1] == "hook_spec"
        np.testing.assert_array_equal(execute.call_args.kwargs["param_sets"], params)

    def test_evaluate_shots_override_threads_to_execute(self, mocker):
        program = self._create_sample_program(mocker)
        execute = mocker.patch.object(
            program, "_execute", return_value={(("param_set", 0),): [1.0]}
        )

        program.evaluate(np.zeros((1, 4)), program.cost_preprocessor(), shots=512)

        assert execute.call_args.kwargs["shots_override"] == 512

    @pytest.mark.parametrize("estimator_samples", [1, 32])
    def test_evaluate_estimator_samples_threads_scalar_to_execute(
        self, mocker, estimator_samples
    ):
        program = self._create_sample_program(mocker)
        execute = mocker.patch.object(
            program, "_execute", return_value={(("param_set", 0),): [1.0]}
        )

        program.evaluate(
            np.zeros((1, 4)),
            program.cost_preprocessor(),
            estimator_samples=estimator_samples,
        )

        assert execute.call_args.kwargs["estimator_samples"].by_param_set == (
            estimator_samples,
        )

    def test_evaluate_estimator_samples_threads_per_set_budgets(self, mocker):
        program = self._create_sample_program(mocker)
        execute = mocker.patch.object(
            program,
            "_execute",
            return_value={
                (("param_set", 0),): [1.0],
                (("param_set", 1),): [1.0],
            },
        )

        program.evaluate(
            np.zeros((2, 4)),
            program.cost_preprocessor(),
            estimator_samples=[1, 8],
        )

        assert execute.call_args.kwargs["estimator_samples"].by_param_set == (1, 8)

    def test_evaluate_returns_estimator_sufficient_statistics(self, mocker):
        program = self._create_sample_program(mocker)
        mocker.patch.object(
            program,
            "_execute",
            return_value={(("param_set", 0),): [1.25]},
        )
        program._last_cost_variance = {(("param_set", 0),): 0.125}

        estimates = program.evaluate_estimates(
            np.zeros((1, 4)),
            program.cost_preprocessor(),
            estimator_samples=8,
        )

        assert estimates == {
            0: CostEstimate(
                mean=1.25,
                variance_of_mean=0.125,
                single_shot_variance=1.0,
                samples_used=8,
            )
        }

    @pytest.mark.parametrize(
        ("estimator_samples", "message"),
        [
            (0, "estimator_samples must be positive."),
            (-1, "estimator_samples must be positive."),
            (
                1.5,
                "estimator_samples must be a positive integer or a sequence "
                "of positive integers.",
            ),
            (True, "estimator_samples must be a positive integer."),
            ([2, 0], "estimator_samples values must be positive integers."),
            ([2, -1], "estimator_samples values must be positive integers."),
            ([2, True], "estimator_samples values must be positive integers."),
        ],
    )
    def test_evaluate_rejects_non_positive_estimator_samples(
        self, mocker, estimator_samples, message
    ):
        program = self._create_sample_program(mocker)

        with pytest.raises(ValueError, match=exact_match(message)):
            program.evaluate(
                np.zeros((2, 4)),
                program.cost_preprocessor(),
                estimator_samples=estimator_samples,
            )

    def test_evaluate_rejects_wrong_number_of_estimator_samples(self, mocker):
        program = self._create_sample_program(mocker)

        with pytest.raises(
            ValueError,
            match=exact_match(
                "estimator_samples must contain one value per parameter set."
            ),
        ):
            program.evaluate(
                np.zeros((2, 4)),
                program.cost_preprocessor(),
                estimator_samples=[2],
            )

    def test_evaluate_rejects_shots_with_estimator_samples(self, mocker):
        program = self._create_sample_program(mocker)

        with pytest.raises(ValueError, match="mutually exclusive"):
            program.evaluate(
                np.zeros((1, 4)),
                program.cost_preprocessor(),
                shots=100,
                estimator_samples=8,
            )

    def test_evaluate_return_variance_returns_values_and_variances(self, mocker):
        program = self._create_sample_program(mocker)
        mocker.patch.object(
            program, "_execute", return_value={(("param_set", 0),): [0.5]}
        )
        program._last_cost_variance = {(("param_set", 0),): 0.01}

        out = program.evaluate(
            np.zeros((1, 4)), program.cost_preprocessor(), return_variance=True
        )

        assert isinstance(out, tuple) and len(out) == 2
        values, variances = out
        assert values == {0: [0.5]}
        assert variances == {0: 0.01}

    def test_evaluate_preserve_keys_returns_raw_pipeline_result(self, mocker):
        program = self._create_sample_program(mocker)
        raw = {
            (("ham", 0), ("param_set", 0)): [0.5],
            (("ham", 1), ("param_set", 0)): [0.7],
        }
        mocker.patch.object(program, "_execute", return_value=raw)

        out = program.evaluate(
            np.zeros((1, 4)), program.cost_preprocessor(), preserve_keys=True
        )

        assert out is raw

    def test_evaluate_preserve_keys_rejects_variance(self, mocker):
        program = self._create_sample_program(mocker)

        with pytest.raises(ValueError, match="preserve_keys"):
            program.evaluate(
                np.zeros((1, 4)),
                program.cost_preprocessor(),
                preserve_keys=True,
                return_variance=True,
            )

    def test_build_preprocessor_pipeline_memoizes_per_cache_key(self, mocker):
        """A keyed preprocessor returns the cached pipeline (keyed by its
        ``cache_key``, not object identity), so a fresh but equal preprocessor
        hits and its forward cache survives across optimizer iterations."""
        program = self._create_sample_program(mocker)

        pipe_a = program._build_preprocessor_pipeline(program.cost_preprocessor())
        # A freshly-constructed cost preprocessor is a distinct object but shares
        # the "cost" cache key, so it must reuse the same pipeline.
        pipe_b = program._build_preprocessor_pipeline(program.cost_preprocessor())

        assert pipe_a is pipe_b
        assert "cost" in program._preprocessor_pipeline_cache

    def test_build_preprocessor_pipeline_does_not_cache_keyless_preprocessor(
        self, mocker
    ):
        """A preprocessor with ``cache_key=None`` (the metric estimators, whose
        transforms carry per-call state) is rebuilt every call and never retained,
        so a stale closure can never be replayed across iterations."""
        program = self._create_sample_program(mocker)
        preprocessor = CircuitPreprocessor("metric")  # cache_key defaults to None

        pipe_a = program._build_preprocessor_pipeline(preprocessor)
        pipe_b = program._build_preprocessor_pipeline(preprocessor)

        assert pipe_a is not pipe_b
        assert program._initial_spec() is program.cost_circuit
        assert not program._preprocessor_pipeline_cache


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

    def test_optimizer_required_raises_value_error(self, mock_backend):
        """Test that omitting optimizer raises ValueError."""
        with pytest.raises(
            ValueError,
            match="A VariationalQuantumAlgorithm requires an explicit optimizer",
        ):
            SampleVQAProgram(circ_count=1, run_time=0.1, backend=mock_backend)

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


@pytest.mark.parametrize(
    "optimizer, loss, success, message",
    [
        (
            SPSAOptimizer(learning_rate=0.1),
            np.nan,
            False,
            "Optimisation failed: no finite cost value was observed in 2 iterations.",
        ),
        (MonteCarloOptimizer(), -0.5, True, "Optimisation converged."),
    ],
    ids=["optimizer-verdict-kept", "default-verdict"],
)
@pytest.mark.filterwarnings("ignore:SPSAOptimizer appears to be diverging:UserWarning")
def test_run_reports_the_optimizer_verdict(
    mock_backend, mocker, optimizer, loss, success, message
):
    """``run()`` fills in success and message only where the optimizer left them."""
    program = SampleVQAProgram(
        circ_count=1, run_time=0.1, backend=mock_backend, optimizer=optimizer, seed=7
    )
    mocker.patch.object(
        program,
        "_evaluate_cost_param_sets",
        side_effect=lambda params, **_: {
            i: loss for i in range(len(np.atleast_2d(params)))
        },
    )

    program.run(max_iterations=2, perform_final_computation=False)

    assert program.optimize_result.success is success
    assert program.optimize_result.message == message


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

    def test_run_uses_one_direct_session_and_emits_typed_progress(self, mocker):
        """A standalone run owns one session and starts visible iterations at one."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1
        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        params = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.setup_mock_optimizer(program, mocker, [(params, -0.5)])

        emitted = []
        session = mocker.MagicMock()
        session.__enter__.return_value = session
        session.emit.side_effect = emitted.append
        direct = mocker.patch(
            "divi.qprog.quantum_program.ProgressSession.direct",
            return_value=session,
        )

        program.run(perform_final_computation=False)

        direct.assert_called_once()
        session.__enter__.assert_called_once_with()
        session.__exit__.assert_called_once_with(None, None, None)
        assert (
            ProgressEvent.show(
                progress_key=program._progress_key,
                message="Iteration #1: Optimising",
            )
            in emitted
        )
        assert (
            ProgressEvent.advance(
                progress_key=program._progress_key,
                loss=-0.5,
            )
            in emitted
        )
        assert (
            ProgressEvent.finish(
                progress_key=program._progress_key,
                status=TerminalStatus.SUCCESS,
                detail="Finished successfully!",
            )
            in emitted
        )

    def test_run_rejects_a_keyword_no_layer_consumes(self, mocker):
        """``run()`` forwards ``**kwargs`` through several layers, each popping what
        it consumes, so a leftover is claimed by nobody — silently discarded before,
        with the run proceeding anyway."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            program.run(bogus_kwarg=True)

    def test_run_points_a_dry_run_flag_at_the_dry_run_method(self, mocker):
        """A flag on the action is the near-universal convention, so
        ``run(dry_run=True)`` is the reflex to catch: ignoring it would execute the
        very run the caller was avoiding."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        program.max_iterations = 1
        with pytest.raises(TypeError, match=r"program\.dry_run\(\)"):
            program.run(dry_run=True)
        assert program.total_circuit_count == 0

    def test_run_still_accepts_the_keyword_it_pops(self, mocker):
        """``max_iterations`` is consumed out of ``**kwargs`` rather than named, so
        the guard has to see the pop, not the signature."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        program.run(max_iterations=1, perform_final_computation=False)
        assert program.max_iterations == 1

    def test_max_iterations_is_a_total_across_runs(self, mocker):
        """``max_iterations`` caps the program, not the call.

        Only the program knows what it has already spent — the stateless optimizers
        keep the iterate in a local and cannot know at all — so ``run()`` subtracts
        and hands each optimizer the remainder. Before, a repeat ``run()`` spent
        ``max_iterations`` again on SPSA and scipy while doing nothing on the
        population optimizers.
        """
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        spy = mocker.spy(program.optimizer, "optimize")

        program.run(max_iterations=2, perform_final_computation=False)
        assert spy.call_args.kwargs["max_iterations"] == 2

        program.current_iteration = 2
        program.run(max_iterations=5, perform_final_computation=False)
        assert spy.call_args.kwargs["max_iterations"] == 3, "only the remainder"

    def test_a_program_at_its_limit_runs_nothing(self, mocker):
        """Reaching the limit makes ``run()`` a no-op rather than a second batch, so
        a finished program cannot be re-run into more circuits by accident."""
        program = self._create_program_with_mock_optimizer(mocker, seed=42)
        spy = mocker.spy(program.optimizer, "optimize")
        program.max_iterations = 2
        program.current_iteration = 2

        with pytest.warns(UserWarning, match="call sample_solution"):
            program.run(perform_final_computation=False)

        spy.assert_not_called()
        assert program.total_circuit_count == 0

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
        np.testing.assert_allclose(program.final_params, curr_params3.flatten())
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
        """Cancellation now halts the script at ``run()``: ``ExecutionCancelledError``
        re-raises after the partial ``optimize_result`` is recorded, so any
        downstream aggregate that depended on the return value never sees an
        empty/half-set state."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1
        mock_event = mocker.MagicMock()
        mock_event.is_set.return_value = True
        program._cancellation_event = mock_event
        program.optimizer.optimize.side_effect = ExecutionCancelledError(
            "Cancellation requested"
        )

        with pytest.raises(ExecutionCancelledError, match="Cancelled by user"):
            program.run()

        # The bookkeeping side-effects (totals, the recorded partial result)
        # still happen — only the return is replaced by the raise.
        assert program.optimize_result is not None
        assert program.optimize_result.success is False
        assert program.optimize_result.message == "Cancelled by user"

    def test_run_method_keyboard_interrupt_propagates_as_hard_abort(self, mocker):
        """A KeyboardInterrupt escaping the optimizer (e.g. the documented
        second-press hard-abort path, or a host with its own SIGINT handler)
        must propagate unchanged. VQA only translates the cooperative
        ExecutionCancelledError; KeyboardInterrupt is the user's request to
        bypass graceful shutdown."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1
        program.optimizer.optimize.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            program.run()

    def _setup_program_for_final_computation_test(self, mocker):
        """Helper to set up program with mocks for final computation tests."""
        program = self._create_program_with_mock_optimizer(mocker)
        program.max_iterations = 1

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        mock_final_computation = mocker.spy(program, "sample_solution")

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
        """Test that sample_solution is called/not called based on the run() flag."""
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
        original_keys = list(program._results["best_probs"].keys())
        result["new_tag"] = {"11": 1.0}  # Add new key to returned dict

        # Original keys should be unchanged
        assert list(program._results["best_probs"].keys()) == original_keys
        # But modifying nested dicts will affect original (shallow copy)
        # So we test that the outer dict is copied, not the inner dicts
        assert "new_tag" not in program._results["best_probs"]


class TestCheckpointing:
    """Tests for VariationalQuantumAlgorithm checkpointing functionality."""

    def test_vqa_checkpoint_kind_controls_optimizer_state(self, sample_program):
        completed = program_checkpoint_module.VQACheckpoint.from_program(
            sample_program,
            kind="program_completion",
        )
        iterative = program_checkpoint_module.VQACheckpoint.from_program(
            sample_program,
            kind="iteration",
        )

        assert completed.optimizer_config is None
        assert iterative.optimizer_config is not None

    def test_restore_accepts_a_checkpoint_without_best_params(
        self, sample_program, mock_backend, default_optimizer
    ):
        checkpoint = VQACheckpoint.from_program(
            sample_program, kind="program_completion"
        )
        target = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )

        checkpoint.restore(target)

        assert target._best_params.size == 0

    @pytest.fixture
    def sample_program(self, mock_backend, default_optimizer):
        """Create a sample program for testing."""
        program = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
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

    def test_save_state_serializes_populated_best_probs(self, sample_program, mocker):
        # Regression: best_probs is dict[int, dict[str, float]] (a param-set
        # index to its bitstring distribution), nested rather than flat.
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1)
        sample_program._results["best_probs"] = {0: {"01": 0.5, "10": 0.5}}

        state = VQACheckpoint.from_program(sample_program, kind="iteration")

        assert state.subclass_state.data["best_probs"] == {0: {"01": 0.5, "10": 0.5}}

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

    def test_restore_state_applies_checkpoint_to_existing_program(
        self, sample_program, mock_backend, default_optimizer, tmp_path, mocker
    ):
        """Ensembles can restore a program they have already reconstructed."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(
                sample_program,
                n_iterations=3,
                result_x=np.array([[0.1, 0.2, 0.3, 0.4]]),
                result_fun=np.array([0.123]),
            )
        )
        sample_program.run(max_iterations=3, perform_final_computation=False)
        sample_program.save_state(CheckpointConfig(checkpoint_dir=tmp_path))

        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )

        restored = fresh._restore_state(tmp_path)

        assert restored is fresh
        assert fresh.current_iteration == 3
        assert fresh._best_loss == 0.123
        assert isinstance(fresh.optimizer, MonteCarloOptimizer)
        np.testing.assert_allclose(fresh._best_params, [0.1, 0.2, 0.3, 0.4])

    def test_restore_state_rolls_back_when_subclass_restore_fails(
        self, sample_program, mock_backend, default_optimizer, tmp_path, mocker
    ):
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program)
        )
        sample_program.run(max_iterations=1, perform_final_computation=False)
        sample_program.save_state(CheckpointConfig(checkpoint_dir=tmp_path))
        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        original_optimizer = fresh.optimizer
        fresh.current_iteration = 7
        fresh.max_iterations = 11
        fresh.checkpointed_value = "fresh"
        fresh._preprocessor_pipeline_cache["sentinel"] = object()

        def fail_after_mutation(state):
            fresh.checkpointed_value = "mutated"
            fresh._preprocessor_pipeline_cache.clear()
            raise ValueError("stale subclass state")

        mocker.patch.object(fresh, "_load_subclass_state", fail_after_mutation)

        with pytest.raises(ValueError, match="stale subclass state"):
            fresh._restore_state(tmp_path)

        assert fresh.current_iteration == 7
        assert fresh.max_iterations == 11
        assert fresh.checkpointed_value == "fresh"
        assert "sentinel" in fresh._preprocessor_pipeline_cache
        assert fresh.optimizer is original_optimizer

    def test_restore_state_rejects_a_different_program_type(
        self, sample_program, mock_backend, default_optimizer, tmp_path, mocker
    ):
        """A manifest/path mix-up cannot silently mutate the wrong program."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1, perform_final_computation=False)
        checkpoint = sample_program.save_state(
            CheckpointConfig(checkpoint_dir=tmp_path)
        )
        state_file = checkpoint / "program_state.json"
        state = json.loads(state_file.read_text())
        state["program_type"] = "SomeOtherProgram"
        state_file.write_text(json.dumps(state))

        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )

        with pytest.raises(ValueError, match="SomeOtherProgram"):
            fresh._restore_state(tmp_path)

        assert fresh.current_iteration == 0
        assert fresh.optimizer is default_optimizer

    def test_vqa_checkpoint_restore_applies_fields_onto_fresh_program(
        self, sample_program, mock_backend, default_optimizer, mocker
    ):
        """VQACheckpoint.restore() writes every mapped attribute back onto a fresh
        program (in-memory, no disk): scalars, numpy-coerced params, the ndarray
        _param_history, the nested _best_probs, and the RNG bit-generator state."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(
                sample_program,
                n_iterations=3,
                result_x=np.array([[0.1, 0.2, 0.3, 0.4]]),
                result_fun=np.array([0.123]),
            )
        )
        sample_program.run(max_iterations=3)
        sample_program._results["best_probs"] = {0: {"0101": 1.0}}
        state = VQACheckpoint.from_program(sample_program, kind="iteration")

        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        state.restore(fresh)

        assert fresh.current_iteration == 3
        assert fresh._best_loss == 0.123
        assert fresh._total_circuit_count == sample_program._total_circuit_count
        assert fresh._results["best_probs"] == {0: {"0101": 1.0}}
        # params restored as numpy, not lists; _param_history blocks are ndarrays.
        np.testing.assert_allclose(fresh._best_params, [0.1, 0.2, 0.3, 0.4])
        assert isinstance(fresh._best_params, np.ndarray)
        assert all(isinstance(block, np.ndarray) for block in fresh._param_history)
        assert fresh._param_history[0].dtype == np.float64
        # RNG bit-generator state round-trips so resumed runs are reproducible.
        assert fresh._rng.bit_generator.state == sample_program._rng.bit_generator.state

    def test_iterative_vqa_checkpoint_requires_optimizer_config(self, sample_program):
        payload = VQACheckpoint.from_program(
            sample_program, kind="iteration"
        ).model_dump()
        payload.pop("optimizer_config")

        with pytest.raises(ValidationError, match="optimizer_config"):
            VQACheckpoint.model_validate(payload)

    def test_completed_state_round_trips_without_optimizer_checkpointing(
        self, mock_backend, tmp_path
    ):
        program = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            seed=17,
        )
        program.current_iteration = 2
        program.max_iterations = 4
        program._losses_history = [{"0": 0.5}, {"0": 0.25}]
        program._param_history = [
            np.array([[0.1, 0.2, 0.3, 0.4]]),
            np.array([[0.4, 0.3, 0.2, 0.1]]),
        ]
        program._best_loss = 0.25
        program._best_params = np.array([0.4, 0.3, 0.2, 0.1])
        program._final_params = np.array([0.4, 0.3, 0.2, 0.1])
        program._results["best_probs"] = {0: {"01": 0.75, "10": 0.25}}
        program._stop_reason = StopReason.PATIENCE_EXCEEDED
        program._total_circuit_count = 12
        program._total_run_time = 1.5
        program.checkpointed_value = "terminal"
        program._rng.random()

        checkpoint = program._make_checkpoint(tmp_path)
        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            seed=17,
        )
        fresh._restore_checkpoint(checkpoint.model_dump_json(), tmp_path)

        assert isinstance(checkpoint, VQACheckpoint)
        assert checkpoint.kind == "program_completion"
        assert fresh.current_iteration == 2
        assert fresh.max_iterations == 4
        assert fresh._losses_history == program._losses_history
        assert fresh._best_loss == 0.25
        assert fresh._results["best_probs"] == program._results["best_probs"]
        assert fresh._stop_reason is StopReason.PATIENCE_EXCEEDED
        assert fresh._total_circuit_count == 12
        assert fresh._total_run_time == 1.5
        assert fresh.checkpointed_value == "terminal"
        assert fresh._rng.bit_generator.state == program._rng.bit_generator.state
        np.testing.assert_allclose(fresh._best_params, program._best_params)
        np.testing.assert_allclose(fresh._final_params, program._final_params)
        assert all(isinstance(block, np.ndarray) for block in fresh._param_history)

    def test_completed_restore_rejects_an_iterative_vqa_checkpoint(
        self, sample_program, tmp_path
    ):
        sample_program._best_loss = 0.0
        checkpoint = VQACheckpoint.from_program(
            sample_program,
            kind="iteration",
        )

        with pytest.raises(ValueError, match="completed VQA checkpoint"):
            sample_program._restore_checkpoint(checkpoint.model_dump_json(), tmp_path)

    def test_completed_state_restore_rolls_back_when_subclass_restore_fails(
        self, sample_program, mock_backend, default_optimizer, tmp_path, mocker
    ):
        sample_program.checkpointed_value = "checkpoint"
        sample_program._best_loss = 0.0
        checkpoint = sample_program._make_checkpoint(tmp_path)
        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        fresh.current_iteration = 7
        fresh.max_iterations = 11
        fresh.checkpointed_value = "fresh"
        fresh._preprocessor_pipeline_cache["sentinel"] = object()

        def fail_after_mutation(state):
            fresh.checkpointed_value = "mutated"
            fresh._preprocessor_pipeline_cache.clear()
            raise ValueError("stale completed state")

        mocker.patch.object(fresh, "_load_subclass_state", fail_after_mutation)

        with pytest.raises(ValueError, match="stale completed state"):
            fresh._restore_checkpoint(checkpoint.model_dump_json(), tmp_path)

        assert fresh.current_iteration == 7
        assert fresh.max_iterations == 11
        assert fresh.checkpointed_value == "fresh"
        assert "sentinel" in fresh._preprocessor_pipeline_cache

    def test_grid_search_checkpoint_round_trips(self, mock_backend, tmp_path):
        """Every optimizer that reports supports_checkpointing can be loaded back."""
        grid = np.linspace(0.0, 1.0, 12).reshape(3, 4)
        program = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=GridSearchOptimizer(param_grid=grid),
        )
        program.max_iterations = 1
        program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
            perform_final_computation=False,
        )

        loaded = SampleVQAProgram.load_state(
            tmp_path, backend=mock_backend, circ_count=0, run_time=0.0
        )

        assert isinstance(loaded.optimizer, GridSearchOptimizer)
        np.testing.assert_allclose(loaded.optimizer._param_grid, grid)
        assert loaded.current_iteration == program.current_iteration

    def test_load_state_rejects_uncheckpointable_optimizer(
        self, mock_backend, tmp_path
    ):
        """An unknown optimizer type names the ones that can be loaded."""
        checkpoint = tmp_path / "checkpoint_001"
        checkpoint.mkdir()
        (checkpoint / "optimizer_state.json").write_text("{}")
        (checkpoint / "program_state.json").write_text(
            json.dumps(
                {
                    "kind": "iteration",
                    "program_type": "SampleVQAProgram",
                    "current_iteration": 1,
                    "max_iterations": 1,
                    "losses_history": [],
                    "best_loss": 0.0,
                    "total_circuit_count": 0,
                    "total_run_time": 0.0,
                    "seed": None,
                    "grouping_strategy": None,
                    "optimizer_config": {"type": "ScipyOptimizer", "config": {}},
                    "subclass_state": {"data": {}},
                    "rng_state": {},
                    "cost_hamiltonian_fingerprint": "",
                }
            )
        )

        with pytest.raises(ValueError, match="Checkpoints can be loaded for"):
            SampleVQAProgram.load_state(
                tmp_path, backend=mock_backend, circ_count=0, run_time=0.0
            )

    def _save_one_iteration(self, program, checkpoint_dir, mocker):
        program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(program, n_iterations=1)
        )
        program.run(max_iterations=1, perform_final_computation=False)
        program.save_state(CheckpointConfig(checkpoint_dir=checkpoint_dir))

    def test_load_restores_the_rng(
        self, sample_program, tmp_path, mocker, mock_backend
    ):
        self._save_one_iteration(sample_program, tmp_path, mocker)

        loaded = SampleVQAProgram.load_state(
            tmp_path, backend=mock_backend, circ_count=0, run_time=0.0
        )

        assert loaded._rng.random() == sample_program._rng.random()

    @pytest.mark.parametrize("argument", ["optimizer", "max_iterations"])
    def test_load_rejects_arguments_the_checkpoint_restores(
        self, sample_program, tmp_path, mocker, mock_backend, argument
    ):
        self._save_one_iteration(sample_program, tmp_path, mocker)

        with pytest.raises(TypeError, match=f"restores {argument}"):
            SampleVQAProgram.load_state(
                tmp_path,
                backend=mock_backend,
                circ_count=0,
                run_time=0.0,
                **{argument: mocker.sentinel.value},
            )

    @pytest.mark.parametrize(
        "attribute, value, match",
        [
            (
                "_serialized_cost_hamiltonian_fingerprint",
                "another",
                "different cost Hamiltonian",
            ),
            ("_serialized_ansatz_type", "OtherAnsatz", "written with ansatz None"),
        ],
        ids=["cost-hamiltonian", "ansatz"],
    )
    def test_load_rejects_a_differently_constructed_program(
        self, sample_program, tmp_path, mocker, mock_backend, attribute, value, match
    ):
        self._save_one_iteration(sample_program, tmp_path, mocker)
        mocker.patch.object(
            SampleVQAProgram,
            attribute,
            new_callable=mocker.PropertyMock,
            return_value=value,
        )

        with pytest.raises(ValueError, match=match):
            SampleVQAProgram.load_state(
                tmp_path, backend=mock_backend, circ_count=0, run_time=0.0
            )

    def test_fingerprint_ignores_numerical_noise_in_the_coefficients(
        self, sample_program
    ):
        fingerprint = sample_program._serialized_cost_hamiltonian_fingerprint
        hamiltonian = sample_program.cost_hamiltonian
        sample_program.cost_hamiltonian = SparsePauliOp(
            hamiltonian.paulis, hamiltonian.coeffs + 1e-9
        )

        assert sample_program._serialized_cost_hamiltonian_fingerprint == fingerprint

    def test_load_rejects_a_different_parameter_count(
        self, sample_program, tmp_path, mocker, mock_backend
    ):
        self._save_one_iteration(sample_program, tmp_path, mocker)
        mocker.patch.object(
            SampleVQAProgram,
            "n_params",
            new_callable=mocker.PropertyMock,
            return_value=9,
        )

        with pytest.raises(ValueError, match="holds 4 parameters but this program"):
            SampleVQAProgram.load_state(
                tmp_path, backend=mock_backend, circ_count=0, run_time=0.0
            )

    def test_stop_reason_round_trips(
        self, sample_program, mock_backend, default_optimizer, mocker
    ):
        """A program that stopped early comes back reporting why."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1)
        sample_program._stop_reason = StopReason.PATIENCE_EXCEEDED

        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        VQACheckpoint.from_program(sample_program, kind="iteration").restore(fresh)

        assert fresh.stop_reason is StopReason.PATIENCE_EXCEEDED

    def test_stop_reason_absent_restores_as_none(
        self, sample_program, mock_backend, default_optimizer, mocker
    ):
        """A run that never stopped early restores a null stop reason."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(max_iterations=1)

        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        fresh._stop_reason = StopReason.COST_VARIANCE_SETTLED
        VQACheckpoint.from_program(sample_program, kind="iteration").restore(fresh)

        assert fresh.stop_reason is None

    def test_save_state_raises_error_before_optimization(
        self, sample_program, tmp_path
    ):
        """Test that save_state() raises RuntimeError if optimization hasn't been run."""
        checkpoint_dir = tmp_path / "checkpoint"

        with pytest.raises(RuntimeError, match="optimisation has not been run"):
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

        # Over iterations 1, 2, 3 the interval fires only at iteration 2; the
        # final flush then saves iteration 3, so the run ends fully persisted.
        assert sample_program.save_state.call_count == 2
        for iteration in (2, 3):
            assert (
                tmp_path
                / "interval_check"
                / f"checkpoint_{iteration:03d}"
                / "program_state.json"
            ).exists()
        assert not (tmp_path / "interval_check" / "checkpoint_001").exists()

    def _cancel_after_one_iteration(self, program, mocker):
        """Make the run cancel from within the callback, after iteration 1."""
        mock_event = mocker.MagicMock()
        mock_event.is_set.return_value = True
        program._cancellation_event = mock_event
        program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(program, n_iterations=1)
        )

    def _raise_after_one_iteration(self, program, mocker, error):
        """Make the optimizer raise ``error`` after running iteration 1."""
        one_iteration = self._create_mock_optimize(program, n_iterations=1)

        def optimize_then_raise(**kwargs):
            one_iteration(**kwargs)
            raise error

        program.optimizer.optimize = mocker.Mock(side_effect=optimize_then_raise)

    @pytest.mark.parametrize(
        "error", [ExecutionCancelledError, RuntimeError], ids=["cancel", "error"]
    )
    def test_aborted_run_checkpoints_its_last_iteration(
        self, sample_program, tmp_path, mocker, error
    ):
        """An abort mid-run persists the work done before it."""
        self._raise_after_one_iteration(sample_program, mocker, error("abort"))

        # An interval that will not fire at iteration 1, so only the final
        # flush can produce a checkpoint.
        with pytest.raises(error):
            sample_program.run(
                checkpoint_config=CheckpointConfig(
                    checkpoint_dir=tmp_path, checkpoint_interval=5
                )
            )

        assert (tmp_path / "checkpoint_001" / "program_state.json").exists()

    @pytest.mark.parametrize(
        "error", [ExecutionCancelledError, RuntimeError], ids=["cancel", "error"]
    )
    def test_abort_survives_a_failing_last_iteration_checkpoint(
        self, sample_program, tmp_path, mocker, error
    ):
        """A failed checkpoint write is logged, not raised over the abort."""
        self._raise_after_one_iteration(sample_program, mocker, error("abort"))
        sample_program.save_state = mocker.Mock(side_effect=OSError("disk full"))

        with pytest.raises(error):
            sample_program.run(
                checkpoint_config=CheckpointConfig(
                    checkpoint_dir=tmp_path, checkpoint_interval=5
                )
            )

        sample_program.save_state.assert_called_once()

    def test_rejected_run_keeps_the_finished_results(
        self, sample_program, tmp_path, mocker
    ):
        """A run() that fails validation has changed nothing, so it must not
        drop the results or the completion file."""
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        sample_program.max_iterations = 5
        sample_program.optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

        with pytest.raises(ValueError, match="does not support checkpointing"):
            sample_program.run(
                checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
            )

        assert sample_program._results["best_probs"] == {0: {"01": 1.0}}
        assert (tmp_path / PROGRAM_COMPLETION_FILE).is_file()

    def test_later_sample_lands_in_the_completion_file(
        self, sample_program, tmp_path, mocker
    ):
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
            perform_final_computation=False,
        )

        def measure(param_sets, *, backend=None):
            sample_program._results["best_probs"] = {0: {"01": 1.0}}

        mocker.patch.object(
            sample_program, "_run_solution_measurement_for", side_effect=measure
        )
        sample_program.sample_solution()

        completion = json.loads((tmp_path / PROGRAM_COMPLETION_FILE).read_text())
        assert completion["subclass_state"]["data"]["best_probs"] == {"0": {"01": 1.0}}

    def test_later_run_keeps_checkpointing_into_the_same_directory(
        self, sample_program, tmp_path, mocker
    ):
        for _ in range(2):
            sample_program.optimizer.optimize = mocker.Mock(
                side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
            )
            sample_program.run(
                checkpoint_config=(
                    CheckpointConfig(checkpoint_dir=tmp_path)
                    if sample_program.current_iteration == 0
                    else None
                ),
                perform_final_computation=False,
            )

        assert (tmp_path / "checkpoint_002" / "program_state.json").is_file()

    def test_fresh_run_rejects_a_used_directory(
        self, sample_program, tmp_path, mocker, mock_backend, default_optimizer
    ):
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        fresh = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        fresh.max_iterations = 10

        with pytest.raises(ValueError, match="already holds checkpoints"):
            fresh.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))

    def test_resume_from_an_earlier_iteration_rejects_the_later_checkpoints(
        self, sample_program, tmp_path, mocker, mock_backend
    ):
        """A program loaded from a named iteration keeps checkpointing into the
        directory, so it refuses to write beside the original run's later
        checkpoints instead of mixing the two runs."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=3)
        )
        sample_program.run(
            max_iterations=3,
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
            perform_final_computation=False,
        )
        loaded = SampleVQAProgram.load_state(
            tmp_path,
            backend=mock_backend,
            subdirectory="checkpoint_001",
            circ_count=0,
            run_time=0.0,
        )
        loaded.max_iterations = 5

        with pytest.raises(
            ValueError, match=r"after iteration 1 \(iterations \[2, 3\]\)"
        ):
            loaded.run(perform_final_computation=False)

    def test_newer_checkpoint_supersedes_the_completion(
        self, sample_program, tmp_path, mocker
    ):
        """Saving a later iteration removes the older finished results, so a
        load cannot roll the program back to them."""
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        sample_program.current_iteration += 1

        sample_program.save_state(CheckpointConfig(checkpoint_dir=tmp_path))

        assert not (tmp_path / PROGRAM_COMPLETION_FILE).exists()

    def test_loading_a_specific_iteration_skips_the_completion(
        self, sample_program, tmp_path, mocker, mock_backend, default_optimizer
    ):
        """Only the latest state carries the finished results; restoring an
        iteration by name clears whatever results the program held."""
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        target = SampleVQAProgram(
            circ_count=0,
            run_time=0.0,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        target._results = {"best_probs": {0: {"11": 1.0}}}

        target._restore_state(tmp_path, subdirectory="checkpoint_001")
        assert target._results == {}

        target._restore_state(tmp_path)
        assert target._results["best_probs"] == {0: {"01": 1.0}}

    @pytest.mark.parametrize(
        "abort",
        [ExecutionCancelledError("stop"), RuntimeError("backend down")],
        ids=["cancel", "error"],
    )
    def test_aborted_run_final_params_are_the_last_iterate(
        self, sample_program, mocker, abort
    ):
        """The second, worse iterate is where the optimiser stopped."""

        def optimize_then_abort(**kwargs):
            self._setup_optimizer_state(sample_program, iteration=2)
            for x, loss in ((np.zeros((1, 4)), 0.1), (np.ones((1, 4)), 0.9)):
                kwargs["callback_fn"](OptimizeResult(x=x, fun=np.array([loss])))
            raise abort

        sample_program.optimizer.optimize = mocker.Mock(side_effect=optimize_then_abort)

        with pytest.raises(type(abort)):
            sample_program.run()

        np.testing.assert_array_equal(sample_program.best_params, np.zeros(4))
        np.testing.assert_array_equal(sample_program.final_params, np.ones(4))

    def test_failed_finalisation_keeps_the_last_iteration_and_run_stays_idle(
        self, sample_program, tmp_path, mocker
    ):
        """The last iteration is saved before finalising, and a run() with no
        iterations left samples nothing; sample_solution() is the way back."""
        sample_program.max_iterations = 1
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )
        sample_program.sample_solution = mocker.Mock(
            side_effect=RuntimeError("sampler down")
        )
        config = CheckpointConfig(checkpoint_dir=tmp_path, checkpoint_interval=5)

        with pytest.raises(RuntimeError, match="sampler down"):
            sample_program.run(checkpoint_config=config)
        assert (tmp_path / "checkpoint_001" / "program_state.json").is_file()
        assert not (tmp_path / PROGRAM_COMPLETION_FILE).exists()

        sample_program.sample_solution = mocker.Mock(return_value=sample_program)
        with pytest.warns(UserWarning, match="call sample_solution"):
            sample_program.run(checkpoint_config=config)

        sample_program.sample_solution.assert_not_called()

    def test_continued_run_does_not_checkpoint_previous_results(
        self, sample_program, tmp_path, mocker
    ):
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=1)
        )

        sample_program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=tmp_path, checkpoint_interval=1
            ),
            perform_final_computation=False,
        )

        state = json.loads(
            (tmp_path / "checkpoint_002" / "program_state.json").read_text()
        )
        assert "best_probs" not in state["subclass_state"]["data"]

    def test_last_iteration_checkpoint_not_duplicated_on_interval_boundary(
        self, sample_program, tmp_path, mocker
    ):
        """A run ending on an interval boundary is not checkpointed twice."""
        sample_program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(sample_program, n_iterations=2)
        )
        sample_program.save_state = mocker.Mock(wraps=sample_program.save_state)

        sample_program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=tmp_path / "boundary", checkpoint_interval=2
            ),
            perform_final_computation=False,
        )

        assert sample_program.save_state.call_count == 1

    def _run_with_final_sample(self, program, checkpoint_dir, mocker):
        program.optimizer.optimize = mocker.Mock(
            side_effect=self._create_mock_optimize(program, n_iterations=1)
        )

        def sample_solution(**kwargs):
            program._results["best_probs"] = {0: {"01": 1.0}}
            return program

        program.sample_solution = mocker.Mock(side_effect=sample_solution)
        program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=checkpoint_dir, checkpoint_interval=1
            )
        )

    def test_final_sample_lands_in_the_completion_file(
        self, sample_program, tmp_path, mocker
    ):
        """Results go to the completion file; iteration checkpoints stay resume
        state and are not rewritten."""
        self._run_with_final_sample(sample_program, tmp_path, mocker)

        completion = json.loads((tmp_path / PROGRAM_COMPLETION_FILE).read_text())
        iteration = json.loads(
            (tmp_path / "checkpoint_001" / "program_state.json").read_text()
        )
        assert completion["kind"] == "program_completion"
        assert completion["subclass_state"]["data"]["best_probs"] == {"0": {"01": 1.0}}
        assert "best_probs" not in iteration["subclass_state"]["data"]

    def test_new_run_discards_the_previous_completion(
        self, sample_program, tmp_path, mocker
    ):
        """An interrupted rerun must not leave the old results to be loaded over
        its newer iterations."""
        self._run_with_final_sample(sample_program, tmp_path, mocker)
        sample_program.max_iterations = 5
        self._cancel_after_one_iteration(sample_program, mocker)

        with pytest.raises(ExecutionCancelledError):
            sample_program.run(
                checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
            )

        assert not (tmp_path / PROGRAM_COMPLETION_FILE).exists()

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
            match="already run 3 of max_iterations=2",
        ):
            loaded_program.run()

        # Should not run additional iterations since already completed
        assert loaded_program.current_iteration == 3
        assert len(loaded_program.losses_history) == 3


class TestPrecisionFunctionality(BaseVariationalQuantumAlgorithmTest):
    """Test suite for precision functionality in VariationalQuantumAlgorithm."""

    def test_precision_defaults_to_8(self, mock_backend, default_optimizer):
        """Test that precision defaults to 8 when not provided."""
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=default_optimizer,
        )
        assert program._precision == 8

    @pytest.mark.parametrize("precision", [1, 4, 8, 12, 16])
    def test_precision_kwarg_propagates_to_meta_circuit(
        self, mock_backend, default_optimizer, precision
    ):
        """The precision kwarg is stored and propagates to created MetaCircuits."""
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            backend=mock_backend,
            optimizer=default_optimizer,
            precision=precision,
        )
        assert program._precision == precision
        assert program.cost_circuit.precision == precision


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
            result = getattr(program, property_name)
            assert isinstance(result, np.ndarray)
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
            match="best_loss is still infinite after optimisation",
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
        program._results["best_probs"] = {}

        with pytest.raises(
            RuntimeError, match=r"No sampled distribution yet.*sample_solution\(\)"
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

    def test_get_top_solutions_warns_and_uses_first_of_multiple_param_sets(
        self, mocker
    ):
        """With several sampled sets, ranking uses the first and warns."""
        program = self._setup_program_with_probs(mocker, {"00": 0.6, "11": 0.4})
        program._results["best_probs"] = {
            0: {"00": 0.6, "11": 0.4},
            1: {"01": 0.9, "10": 0.1},
        }
        with pytest.warns(UserWarning, match="only the first"):
            result = program.get_top_solutions(n=2)
        assert [s.bitstring for s in result] == ["00", "11"]

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
        probs = {"00": 0.6, "11": 0.4}
        program = self._setup_program_with_probs(mocker, probs)

        result = program.get_top_solutions(n=2)

        assert len(result) == 2
        assert isinstance(result[0], SolutionEntry)
        assert isinstance(result[1], SolutionEntry)
        assert result[0].bitstring == "00"
        assert result[0].prob == 0.6

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


class TestSpinMomentsAPI(BaseVariationalQuantumAlgorithmTest):
    """Test suite for get_correlations() and get_magnetisations()."""

    def test_perfectly_correlated_distribution(self, mocker):
        """An equal mix of "00" and "11" agrees on every pair and has zero bias."""
        program = self._setup_program_with_probs(mocker, {"00": 0.5, "11": 0.5})

        np.testing.assert_allclose(program.get_correlations(), np.ones((2, 2)))
        np.testing.assert_allclose(program.get_magnetisations(), np.zeros(2))

    def test_perfectly_anticorrelated_distribution(self, mocker):
        """An equal mix of "01" and "10" disagrees on every pair."""
        program = self._setup_program_with_probs(mocker, {"01": 0.5, "10": 0.5})

        np.testing.assert_allclose(
            program.get_correlations(), np.array([[1.0, -1.0], [-1.0, 1.0]])
        )
        np.testing.assert_allclose(program.get_magnetisations(), np.zeros(2))

    def test_deterministic_bitstring_saturates_moments(self, mocker):
        """A single certain bitstring pins every spin to +/-1."""
        program = self._setup_program_with_probs(mocker, {"010": 1.0})

        spins = np.array([1.0, -1.0, 1.0])
        np.testing.assert_allclose(program.get_magnetisations(), spins)
        np.testing.assert_allclose(program.get_correlations(), np.outer(spins, spins))

    def test_matches_hand_computed_mixed_distribution(self, mocker):
        """Values match the explicit sum over p(x) * s_i * s_j."""
        probs = {"00": 0.5, "01": 0.2, "10": 0.2, "11": 0.1}
        program = self._setup_program_with_probs(mocker, probs)

        expected_corr = np.zeros((2, 2))
        expected_mag = np.zeros(2)
        for bitstring, prob in probs.items():
            spins = np.array([1 - 2 * int(bit) for bit in bitstring])
            expected_corr += prob * np.outer(spins, spins)
            expected_mag += prob * spins

        np.testing.assert_allclose(program.get_correlations(), expected_corr)
        np.testing.assert_allclose(program.get_magnetisations(), expected_mag)

    def test_correlations_are_symmetric_with_unit_diagonal(self, mocker):
        """Z is symmetric and <Z_i^2> == 1 regardless of the distribution."""
        probs = {"011": 0.4, "101": 0.35, "000": 0.25}
        program = self._setup_program_with_probs(mocker, probs)

        correlations = program.get_correlations()

        np.testing.assert_allclose(correlations, correlations.T)
        np.testing.assert_allclose(np.diag(correlations), np.ones(3))

    def test_index_order_follows_bitstring_position(self, mocker):
        """Index i refers to bitstring position i, as decode_solution_fn does."""
        program = self._setup_program_with_probs(mocker, {"100": 1.0})

        # Only wire 0 is set, so it anticorrelates with the two unset wires.
        np.testing.assert_allclose(
            program.get_correlations()[0], np.array([1.0, -1.0, -1.0])
        )

    def test_partially_correlated_pair_is_between_the_extremes(self, mocker):
        """A 75/25 agree/disagree split gives a correlation of 0.5."""
        program = self._setup_program_with_probs(mocker, {"00": 0.75, "01": 0.25})

        assert program.get_correlations()[0, 1] == pytest.approx(0.5)

    @pytest.mark.parametrize("method", ["get_correlations", "get_magnetisations"])
    def test_raises_when_no_probs(self, mocker, method):
        """Both moments require a sampled distribution."""
        program = self._create_program_with_mock_optimizer(mocker)
        program._results["best_probs"] = {}

        with pytest.raises(
            RuntimeError, match=r"No sampled distribution yet.*sample_solution\(\)"
        ):
            getattr(program, method)()

    @pytest.mark.parametrize("method", ["get_correlations", "get_magnetisations"])
    def test_warns_and_uses_first_of_multiple_param_sets(self, mocker, method):
        """With several sampled sets, the first is used and a warning is emitted."""
        program = self._setup_program_with_probs(mocker, {"00": 1.0})
        program._results["best_probs"] = {0: {"00": 1.0}, 1: {"11": 1.0}}

        with pytest.warns(UserWarning, match="only the first"):
            result = getattr(program, method)()

        # The first set is all-zeros, so every spin sits at +1.
        np.testing.assert_allclose(result, np.ones_like(result))

    @pytest.mark.parametrize("method", ["get_correlations", "get_magnetisations"])
    def test_ragged_bitstrings_raise(self, mocker, method):
        """Bitstrings of differing width have no consistent wire indexing."""
        program = self._setup_program_with_probs(mocker, {"00": 0.5, "111": 0.5})

        with pytest.raises(ValueError, match="same length"):
            getattr(program, method)()


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

    def test_early_stopped_run_checkpoints_its_last_iteration(self, mocker, tmp_path):
        """The iteration early stopping halts on is the one left on disk."""
        program = self._create_program_with_mock_optimizer(
            mocker, seed=42, early_stopping=EarlyStopping(patience=3, min_delta=0.0)
        )
        program.max_iterations = 100

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=100)
        program.optimizer.get_config.side_effect = lambda: {
            "type": "MonteCarloOptimizer"
        }
        program.optimizer.save_state.side_effect = lambda path: (
            Path(path) / "optimizer_state.json"
        ).write_text("{}")

        # An interval that cannot fire before patience runs out at iteration 4.
        program.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=tmp_path, checkpoint_interval=50
            )
        )

        assert program.stop_reason is StopReason.PATIENCE_EXCEEDED
        assert [info.iteration for info in list_checkpoints(tmp_path)] == [
            program.current_iteration
        ]

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

    def _early_stopping_flat_loss_program(self, mocker, patience):
        """A program whose flat losses trip ``patience`` long before its limit."""
        program = self._create_program_with_mock_optimizer(
            mocker,
            seed=42,
            early_stopping=EarlyStopping(patience=patience, min_delta=0.0),
        )
        program.max_iterations = 100
        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=100)
        return program

    def test_early_stopped_final_params_are_the_last_iterate(self, mocker):
        program = self._early_stopping_flat_loss_program(mocker, patience=3)

        program.run(perform_final_computation=False)

        # Flat losses keep iterate 0 as the best; patience stops at iterate 3.
        assert np.all(program.best_params == 0.0)
        assert np.all(program.final_params == 3.0)

    def test_early_stopped_program_runs_nothing_more(self, mocker):
        program = self._early_stopping_flat_loss_program(mocker, patience=1)
        program.run(perform_final_computation=False)
        iterations = program.current_iteration

        with pytest.warns(UserWarning, match="stopped early"):
            program.run(perform_final_computation=False)

        assert program.current_iteration == iterations

    def test_population_early_stop_keeps_final_params_one_dimensional(self, mocker):
        """A population's last iterate reduces to its best member, so the
        program can still be checkpointed."""
        program = self._create_program_with_mock_optimizer(
            mocker,
            seed=42,
            early_stopping=EarlyStopping(patience=1, min_delta=0.0),
            optimizer=self._create_mock_optimizer(mocker, n_param_sets=2),
        )
        program.max_iterations = 100
        n = program.n_params

        def optimize(cost_fn, initial_params, callback_fn, **kwargs):
            for i in range(100):
                x = np.stack([np.full(n, float(i)), np.full(n, i + 10.0)])
                callback_fn(OptimizeResult(x=x, fun=np.array([1.0, 0.5])))

        program.optimizer.optimize.side_effect = optimize

        program.run(perform_final_computation=False)

        np.testing.assert_array_equal(program.final_params, np.full(n, 11.0))
        VQACheckpoint.from_program(program, kind="program_completion")

    def test_final_computation_still_runs_after_early_stop(self, mocker):
        """Verify sample_solution is called after early stopping."""
        es = EarlyStopping(patience=2, min_delta=0.0)
        program = self._create_program_with_mock_optimizer(
            mocker, seed=42, early_stopping=es
        )
        program.max_iterations = 100

        mocker.patch.object(
            program, "_evaluate_cost_param_sets", return_value={0: -0.5}
        )
        self._setup_optimizer_with_flat_losses(program, mocker, n_iterations=100)
        mock_final = mocker.spy(program, "sample_solution")

        program.run()

        assert program.stop_reason == StopReason.PATIENCE_EXCEEDED
        mock_final.assert_called_once()


class TestComputeParameterShiftRule:
    """Spec: _compute_parameter_shift_rule builds shifts and recombination weights."""

    @pytest.mark.parametrize("n_params", [1, 2, 3, 5, 8])
    def test_default_family_shape(self, n_params):
        shifts, weights = _compute_parameter_shift_rule([(1.0, 1)] * n_params)
        assert shifts.shape == (2 * n_params, n_params)
        assert weights.shape == (n_params, 2 * n_params)

    @pytest.mark.parametrize("n_params", [1, 2, 3, 4, 5, 8])
    def test_each_row_shifts_exactly_one_parameter(self, n_params):
        shifts, _ = _compute_parameter_shift_rule([(1.0, 1)] * n_params)
        for index in range(n_params):
            for row in shifts[2 * index : 2 * index + 2]:
                nonzero = np.nonzero(row)[0]
                assert nonzero.tolist() == [index]

    def test_unit_frequency_reproduces_the_two_term_rule(self):
        """The historical +-pi/2 rule with +-1/2 weights is the (1, 1) case."""
        shifts, weights = _compute_parameter_shift_rule([(1.0, 1)] * 2)
        half = np.pi / 2
        np.testing.assert_allclose(
            shifts, [[half, 0], [-half, 0], [0, half], [0, -half]]
        )
        np.testing.assert_allclose(
            weights, [[0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 0.5, -0.5]]
        )

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_order_sets_the_evaluation_count(self, order):
        shifts, weights = _compute_parameter_shift_rule([(1.0, order)])
        assert shifts.shape == (2 * order, 1)
        assert weights.shape == (1, 2 * order)

    def test_families_may_differ_between_parameters(self):
        """A ragged rule is one flat batch, not padded rows per parameter."""
        shifts, weights = _compute_parameter_shift_rule([(1.0, 1), (0.5, 4)])
        assert shifts.shape == (2 + 8, 2)
        # Each parameter's weights touch only its own rows.
        assert np.count_nonzero(weights[0]) == 2
        assert np.count_nonzero(weights[1]) == 8
        np.testing.assert_array_equal(np.nonzero(weights[0])[0], [0, 1])

    def test_folds_shifts_into_the_principal_interval(self):
        """Shifts are reduced modulo the energy's period, keeping them smallest."""
        shifts, _ = _compute_parameter_shift_rule([(0.5, 2)])
        # Unfolded these are (2k-1) * pi/2 for k = 1..4; the period is 4 * pi.
        np.testing.assert_allclose(
            shifts[:, 0],
            [np.pi / 2, 3 * np.pi / 2, -3 * np.pi / 2, -np.pi / 2],
        )

    @pytest.mark.parametrize(
        "frequencies, match",
        [
            ([(0.0, 1)], "omega must be positive"),
            ([(1.0, 0)], "order must be at least"),
        ],
    )
    def test_rejects_invalid_families(self, frequencies, match):
        with pytest.raises(ValueError, match=match):
            _compute_parameter_shift_rule(frequencies)

    @pytest.mark.parametrize("omega, order", [(1.0, 1), (2.0, 1), (1.0, 2), (0.5, 4)])
    def test_differentiates_its_own_frequency_family_exactly(self, omega, order):
        """The rule must be exact on any trigonometric polynomial it declares."""
        rng = np.random.default_rng(0)
        cosines = rng.normal(size=order + 1)
        sines = rng.normal(size=order + 1)

        def energy(theta):
            return sum(
                cosines[r] * np.cos(r * omega * theta)
                + sines[r] * np.sin(r * omega * theta)
                for r in range(order + 1)
            )

        def derivative(theta):
            return sum(
                r
                * omega
                * (
                    -cosines[r] * np.sin(r * omega * theta)
                    + sines[r] * np.cos(r * omega * theta)
                )
                for r in range(order + 1)
            )

        shifts, weights = _compute_parameter_shift_rule([(omega, order)])
        for theta in rng.uniform(-3.0, 3.0, 5):
            values = np.array([energy(theta + row[0]) for row in shifts])
            assert (weights @ values)[0] == pytest.approx(derivative(theta), abs=1e-9)


class TestGradShiftRule(BaseVariationalQuantumAlgorithmTest):
    """Spec: a program's shift rule is built lazily from its declared frequencies."""

    def test_defaults_to_the_two_term_rule(self, mocker):
        program = self._create_program_with_mock_optimizer(mocker)
        shifts, _ = program._grad_shift_rule
        assert shifts.shape == (2 * program.n_params, program.n_params)

    def test_rejects_a_declaration_of_the_wrong_length(self, mocker):
        program = self._create_program_with_mock_optimizer(mocker)
        mocker.patch.object(
            type(program), "_parameter_frequencies", return_value=[(1.0, 1)]
        )
        with pytest.raises(ValueError, match="declared 1 parameter frequencies"):
            program._grad_shift_rule


class TestGradientFunction(BaseVariationalQuantumAlgorithmTest):
    """Spec: grad_fn correctly computes parameter-shift gradients from pipeline results."""

    def _create_lbfgsb_program(self, n_params_per_layer=4, **kwargs):
        """Create a SampleVQAProgram with L-BFGS-B optimizer."""
        program = SampleVQAProgram(
            circ_count=1,
            run_time=0.1,
            n_params_per_layer=n_params_per_layer,
            optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
            backend=self._make_backend(1000),
            seed=42,
            **kwargs,
        )
        program.max_iterations = 2
        return program

    def test_gradient_with_known_return_values(self, mocker):
        """grad_fn produces 0.5 * (positive_shift_values - negative_shift_values)."""
        program = self._create_lbfgsb_program(n_params_per_layer=3)
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
        assert program.optimize_result.njev >= 1

    def test_shifted_params_are_mask_plus_input(self, mocker):
        """During gradient computation, shifted param sets equal mask + original params."""
        program = self._create_lbfgsb_program(n_params_per_layer=3)
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
        assert all(c.shape == (2 * n_params, n_params) for c in captured)
        # First gradient call uses the initial params
        shifted = captured[0]

        # Each row pair should differ from the initial params by exactly ±π/2
        # in exactly one column (the parameter being shifted)
        for i in range(n_params):
            diff_pos = shifted[2 * i] - initial_params.squeeze()
            nonzero = np.nonzero(np.abs(diff_pos) > 1e-10)[0]
            assert len(nonzero) == 1, f"Row {2*i} shifts {len(nonzero)} params"
            assert nonzero[0] == i
            np.testing.assert_allclose(np.abs(diff_pos[nonzero[0]]), np.pi / 2)
