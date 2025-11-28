# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import CheckpointNotFoundError
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    copy_optimizer,
)
from tests.conftest import CHECKPOINTING_OPTIMIZERS
from tests.qprog.qprog_contracts import OPTIMIZERS_TO_TEST


def sphere_cost_fn_population(params: np.ndarray) -> np.ndarray:
    """Sphere cost function (sum of squares) for a population of parameter sets."""
    if params.ndim != 2:
        raise ValueError(
            "Input params for population cost function must be a 2D array."
        )
    return np.sum(params**2, axis=1)


def sphere_cost_fn_single(params: np.ndarray) -> float:
    """Sphere cost function (sum of squares) for a single parameter set."""
    if params.ndim != 1:
        # Allow single-row 2D array for convenience with some optimizers
        if params.ndim == 2 and params.shape[0] == 1:
            params = params.squeeze(0)
        else:
            raise ValueError(
                "Input params for single cost function must be a 1D array."
            )
    return float(np.sum(params**2))


@pytest.fixture(
    params=[
        # Use OPTIMIZERS_TO_TEST as base, converting to (name, factory) tuples
        *[
            (opt_id.lower().replace("_", "-"), factory)
            for factory, opt_id in zip(
                OPTIMIZERS_TO_TEST["argvalues"], OPTIMIZERS_TO_TEST["ids"]
            )
        ],
        # Add extra variant not in OPTIMIZERS_TO_TEST
        (
            "monte-carlo-keep-best",
            lambda: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=True
            ),
        ),
    ],
    ids=lambda param: param[0],
)
def optimizer(request):
    """Provides various optimizer instances to be tested against the contract."""
    _, factory = request.param
    return factory()


class TestOptimizerContract:
    """A contract of tests that every optimizer should pass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    def _get_initial_params(self, optimizer: Optimizer) -> np.ndarray:
        """Generate initial parameters with the correct shape for the optimizer."""
        shape = (optimizer.n_param_sets, self.n_params)
        return self.rng.random(shape) * 2 * np.pi

    def test_final_result_is_consistent(self, optimizer: Optimizer):
        """Verify that the final returned parameters match the final cost."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        result = optimizer.optimize(cost_fn, initial_params, max_iterations=5)

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        recalculated_fun = sphere_cost_fn_single(result.x)
        assert np.isclose(
            result.fun, recalculated_fun
        ), "Final cost value should correspond to the final parameters."

    def test_callback_provides_consistent_results(self, optimizer: Optimizer):
        """Verify that parameters and costs are consistent in each callback call."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        callback_results = []

        def callback(intermediate_result: OptimizeResult):
            callback_results.append(intermediate_result)

        optimizer.optimize(
            cost_fn, initial_params, callback_fn=callback, max_iterations=5
        )

        assert len(callback_results) > 0
        for res in callback_results:
            assert isinstance(res, OptimizeResult)
            assert res.x.shape == (optimizer.n_param_sets, self.n_params)
            assert res.fun.shape == (optimizer.n_param_sets,)

            if optimizer.n_param_sets == 1:
                recalculated_fun = sphere_cost_fn_single(res.x)
            else:
                recalculated_fun = sphere_cost_fn_population(res.x)

            # L-BFGS-B can provide a stale loss value in its callback due to its
            # line-search mechanism. For this optimizer only, we relax the test
            # to confirm the loss is a valid number, but not that it's perfectly
            # in sync with the parameters, as that would require re-computation.
            if (
                isinstance(optimizer, ScipyOptimizer)
                and optimizer.method == ScipyMethod.L_BFGS_B
            ):
                assert np.isfinite(res.fun).all()
            else:
                assert np.allclose(res.fun, recalculated_fun)

    def test_reset_allows_reusing_optimizer(self, optimizer: Optimizer):
        """Verify that reset() allows reusing an optimizer for fresh optimization runs."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        # Run first optimization
        result1 = optimizer.optimize(cost_fn, initial_params, max_iterations=3)
        assert isinstance(result1, OptimizeResult)

        # Reset and run second optimization with different initial params
        optimizer.reset()
        different_initial_params = self._get_initial_params(optimizer)
        result2 = optimizer.optimize(
            cost_fn, different_initial_params, max_iterations=3
        )
        assert isinstance(result2, OptimizeResult)

        # Both results should be valid
        assert result1.x.shape == (self.n_params,)
        assert result2.x.shape == (self.n_params,)
        assert np.isfinite(result1.fun)
        assert np.isfinite(result2.fun)

    def test_reset_preserves_optimizer_configuration(self, optimizer: Optimizer):
        """Verify that reset() does not affect optimizer configuration."""
        # Store original configuration
        if isinstance(optimizer, ScipyOptimizer):
            original_method = optimizer.method
        elif isinstance(optimizer, PymooOptimizer):
            original_method = optimizer.method
            original_pop_size = optimizer.population_size
        elif isinstance(optimizer, MonteCarloOptimizer):
            original_pop_size = optimizer.population_size
            original_n_best = optimizer.n_best_sets
            original_keep_best = optimizer.keep_best_params

        # Run optimization to set state
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )
        optimizer.optimize(cost_fn, initial_params, max_iterations=2)

        # Reset and verify configuration is unchanged
        optimizer.reset()

        if isinstance(optimizer, ScipyOptimizer):
            assert optimizer.method == original_method
        elif isinstance(optimizer, PymooOptimizer):
            assert optimizer.method == original_method
            assert optimizer.population_size == original_pop_size
        elif isinstance(optimizer, MonteCarloOptimizer):
            assert optimizer.population_size == original_pop_size
            assert optimizer.n_best_sets == original_n_best
            assert optimizer.keep_best_params == original_keep_best

    def test_reset_can_be_called_multiple_times(self, optimizer: Optimizer):
        """Verify that reset() can be called multiple times safely."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )

        # Run optimization
        optimizer.optimize(cost_fn, initial_params, max_iterations=2)

        # Call reset multiple times
        optimizer.reset()
        optimizer.reset()
        optimizer.reset()

        # Should still be able to optimize after multiple resets
        result = optimizer.optimize(cost_fn, initial_params, max_iterations=2)
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)


class TestMonteCarloOptimizer:
    """Specific tests for MonteCarloOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)
        self.population_size = 10
        self.n_best_sets = 3

    def _create_optimizer(self, keep_best_params: bool) -> MonteCarloOptimizer:
        """Helper to create a MonteCarloOptimizer with standard test parameters."""
        return MonteCarloOptimizer(
            population_size=self.population_size,
            n_best_sets=self.n_best_sets,
            keep_best_params=keep_best_params,
        )

    def _create_initial_params(self) -> np.ndarray:
        """Helper to create initial parameters with correct shape."""
        return self.rng.random((self.population_size, self.n_params)) * 2 * np.pi

    def _get_best_params(self, population: np.ndarray, n_best: int) -> np.ndarray:
        """Extract the best n_best parameter sets from a population."""
        losses = sphere_cost_fn_population(population)
        best_indices = np.argpartition(losses, n_best - 1)[:n_best]
        return population[best_indices]

    def _run_optimization_with_callback(
        self,
        optimizer: MonteCarloOptimizer,
        initial_params: np.ndarray,
        max_iterations: int = 3,
    ) -> list[np.ndarray]:
        """Run optimization and collect all populations via callback."""
        all_populations = []

        def callback(intermediate_result: OptimizeResult):
            all_populations.append(intermediate_result.x.copy())

        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            callback_fn=callback,
            max_iterations=max_iterations,
            rng=self.rng,
        )
        return all_populations

    def test_keep_best_params_validation_and_property(self):
        """Test validation errors and property access."""
        # Test validation error
        with pytest.raises(
            ValueError,
            match="If keep_best_params is True, n_best_sets must be less than population_size.",
        ):
            MonteCarloOptimizer(
                population_size=10, n_best_sets=10, keep_best_params=True
            )

        # Test no error when keep_best_params=False
        optimizer_no_error = MonteCarloOptimizer(
            population_size=10, n_best_sets=10, keep_best_params=False
        )
        assert optimizer_no_error.keep_best_params is False

        # Test property returns correct values
        optimizer_false = self._create_optimizer(keep_best_params=False)
        optimizer_true = self._create_optimizer(keep_best_params=True)
        assert optimizer_false.keep_best_params is False
        assert optimizer_true.keep_best_params is True

    @pytest.mark.parametrize("keep_best_params", [True, False])
    def test_keep_best_params_behavior(self, keep_best_params: bool):
        """Test that keep_best_params correctly controls whether best parameters are kept."""
        optimizer = self._create_optimizer(keep_best_params=keep_best_params)
        initial_params = self._create_initial_params()

        all_populations = self._run_optimization_with_callback(
            optimizer, initial_params, max_iterations=3
        )

        assert len(all_populations) >= 2

        # Get best params from initial population
        best_params_first = self._get_best_params(
            all_populations[0], optimizer.n_best_sets
        )

        # Check second population
        second_population = all_populations[1]
        assert second_population.shape == (optimizer.population_size, self.n_params)
        first_n_best_in_second = second_population[: optimizer.n_best_sets]

        # Check if each of the best parameters from the previous generation is present
        # in the best parameters of the new generation.
        is_present = [
            np.any(np.all(np.isclose(first_n_best_in_second, p), axis=1))
            for p in best_params_first
        ]

        if keep_best_params:
            # If we keep the best params, all of them should be present.
            assert all(
                is_present
            ), "Not all best parameters were kept when keep_best_params=True"
        else:
            # If we don't, none of them should be present as exact copies.
            assert not any(
                is_present
            ), "A best parameter was kept as an exact copy when keep_best_params=False"

    def test_reset_clears_monte_carlo_state(self):
        """Test that reset() clears all internal state variables for MonteCarloOptimizer."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Verify state is set
        assert optimizer._curr_population is not None
        assert optimizer._curr_evaluated_population is not None
        assert optimizer._curr_losses is not None
        assert optimizer._curr_iteration is not None
        assert optimizer._curr_rng_state is not None

        # Reset and verify state is cleared
        optimizer.reset()
        assert optimizer._curr_population is None
        assert optimizer._curr_evaluated_population is None
        assert optimizer._curr_losses is None
        assert optimizer._curr_iteration is None
        assert optimizer._curr_rng_state is None

    def test_save_state_creates_checkpoint_file(self, tmp_path):
        """Test that save_state() creates the expected checkpoint file."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Verify checkpoint file exists
        state_file = tmp_path / "checkpoint" / "optimizer_state.json"
        assert state_file.exists(), "Checkpoint file should be created"

    def test_save_state_preserves_configuration(self, tmp_path):
        """Test that save_state() saves optimizer configuration correctly."""
        optimizer = self._create_optimizer(keep_best_params=True)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify configuration
        loaded_optimizer = MonteCarloOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.population_size == optimizer.population_size
        assert loaded_optimizer.n_best_sets == optimizer.n_best_sets
        assert loaded_optimizer.keep_best_params == optimizer.keep_best_params

    def test_load_state_restores_optimization_state(self, tmp_path):
        """Test that load_state() correctly restores all optimization state."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Capture state before saving
        original_population = optimizer._curr_population.copy()
        original_evaluated = optimizer._curr_evaluated_population.copy()
        original_losses = optimizer._curr_losses.copy()
        original_iteration = optimizer._curr_iteration

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load state
        loaded_optimizer = MonteCarloOptimizer.load_state(checkpoint_dir)

        # Verify all state is restored
        np.testing.assert_array_equal(
            loaded_optimizer._curr_population, original_population
        )
        np.testing.assert_array_equal(
            loaded_optimizer._curr_evaluated_population, original_evaluated
        )
        np.testing.assert_array_equal(loaded_optimizer._curr_losses, original_losses)
        assert loaded_optimizer._curr_iteration == original_iteration
        assert loaded_optimizer._curr_rng_state is not None

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading state preserves optimizer functionality."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run partial optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and continue optimization
        loaded_optimizer = MonteCarloOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        # Verify optimization completed successfully
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        checkpoint_dir = str(tmp_path / "nonexistent_checkpoint")

        with pytest.raises(CheckpointNotFoundError, match="Checkpoint file not found"):
            MonteCarloOptimizer.load_state(checkpoint_dir)

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        # Save to a non-existent nested directory
        checkpoint_dir = str(tmp_path / "nested" / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Verify directory and file were created
        state_file = tmp_path / "nested" / "checkpoint" / "optimizer_state.json"
        assert state_file.exists(), "Checkpoint directory and file should be created"

    @pytest.mark.parametrize("keep_best_params", [True, False])
    def test_compute_new_parameters_preserves_population_size(
        self, keep_best_params: bool
    ):
        """Low level test to ensure sampling logic preserves shapes and bounds."""
        optimizer = self._create_optimizer(keep_best_params=keep_best_params)
        population = self._create_initial_params()
        losses = sphere_cost_fn_population(population)
        best_indices = np.argsort(losses)[: optimizer.n_best_sets]

        rng = np.random.default_rng(7)
        new_population = optimizer._compute_new_parameters(
            population, curr_iteration=1, best_indices=best_indices, rng=rng
        )

        assert new_population.shape == (optimizer.population_size, self.n_params)
        assert np.all(new_population >= 0.0)
        assert np.all(new_population < 2 * np.pi)

        if keep_best_params:
            np.testing.assert_allclose(
                new_population[: optimizer.n_best_sets], population[best_indices]
            )

    def test_optimize_with_completed_iterations_skips_additional_evaluations(self):
        """Ensure resuming with fewer iterations does not re-evaluate the cost."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        eval_counter = {"calls": 0}

        def counting_cost_fn(population: np.ndarray) -> np.ndarray:
            eval_counter["calls"] += 1
            return sphere_cost_fn_population(population)

        optimizer.optimize(
            counting_cost_fn, initial_params, max_iterations=3, rng=self.rng
        )
        assert eval_counter["calls"] == 3

        # Resume with a smaller total iteration budget, should finish immediately.
        result = optimizer.optimize(
            counting_cost_fn, initial_params, max_iterations=1, rng=self.rng
        )

        assert eval_counter["calls"] == 3, "No additional evaluations should be made"
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)

    def test_save_state_without_prior_run_raises(self, tmp_path):
        """Calling save_state before running optimize should fail clearly."""
        optimizer = self._create_optimizer(keep_best_params=False)
        checkpoint_dir = tmp_path / "checkpoint"

        with pytest.raises(RuntimeError, match="optimization has not been run"):
            optimizer.save_state(str(checkpoint_dir))


class TestPopulationOptimizerCheckpointing:
    """Shared checkpoint/resume behaviours for population-based optimizers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(1337)

    def _initial_params(self, optimizer: Optimizer) -> np.ndarray:
        """Create initial parameters that respect the optimizer contract."""
        shape = (optimizer.n_param_sets, self.n_params)
        return self.rng.random(shape) * 2 * np.pi

    @pytest.mark.parametrize("optimizer_factory", **CHECKPOINTING_OPTIMIZERS)
    def test_checkpoint_dir_without_checkpointing(self, optimizer_factory):
        """Optimization should work normally when no checkpoint_dir is provided."""
        optimizer = optimizer_factory()
        initial_params = self._initial_params(optimizer)

        result = optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize(
        "optimizer_factory",
        **CHECKPOINTING_OPTIMIZERS,
    )
    def test_resume_with_max_iterations_less_than_completed(
        self, tmp_path, optimizer_factory
    ):
        """Resuming with fewer total iterations should exit immediately."""
        optimizer = optimizer_factory()
        load_state_func = type(optimizer).load_state

        initial_params = self._initial_params(optimizer)

        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        checkpoint_dir = tmp_path / "checkpoint"
        optimizer.save_state(str(checkpoint_dir))

        loaded_optimizer = load_state_func(str(checkpoint_dir))
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize(
        "optimizer_factory",
        **CHECKPOINTING_OPTIMIZERS,
    )
    def test_resume_with_different_initial_params(self, tmp_path, optimizer_factory):
        """Checkpoints should ignore newly provided initial parameters when resuming."""
        optimizer = optimizer_factory()
        load_state_func = type(optimizer).load_state
        initial_params = self._initial_params(optimizer)

        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = tmp_path / "checkpoint"
        optimizer.save_state(str(checkpoint_dir))

        loaded_optimizer = load_state_func(str(checkpoint_dir))
        different_initial_params = self._initial_params(optimizer)

        result = loaded_optimizer.optimize(
            sphere_cost_fn_population,
            different_initial_params,
            max_iterations=5,
            rng=self.rng,
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)


class TestPymooOptimizer:
    """Specific tests for PymooOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    def test_reset_clears_pymoo_state(self):
        """Test that reset() clears all internal state variables for PymooOptimizer."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Verify state is set
        assert optimizer._curr_algorithm_obj is not None

        # Reset and verify state is cleared
        optimizer.reset()
        assert optimizer._curr_algorithm_obj is None

    def test_save_state_creates_checkpoint_file(self, tmp_path):
        """Test that save_state() creates the expected checkpoint file."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Verify checkpoint file exists
        state_file = tmp_path / "checkpoint" / "optimizer_state.json"
        assert state_file.exists(), "Checkpoint file should be created"

    def test_save_state_without_prior_run_raises(self, tmp_path):
        """Saving without having run optimize should raise a clear error."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        checkpoint_dir = tmp_path / "checkpoint"

        with pytest.raises(RuntimeError, match="optimization has not been run"):
            optimizer.save_state(str(checkpoint_dir))

    def test_save_state_preserves_configuration(self, tmp_path):
        """Test that save_state() saves optimizer configuration correctly."""
        optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=15)
        initial_params = self.rng.random((15, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify configuration
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.method == optimizer.method
        assert loaded_optimizer.population_size == optimizer.population_size

    def test_load_state_restores_optimization_state(self, tmp_path):
        """Test that load_state() correctly restores optimization state."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Capture algorithm object before saving
        original_algorithm_obj = optimizer._curr_algorithm_obj

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load state
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)

        # Verify algorithm object is restored
        assert loaded_optimizer._curr_algorithm_obj is not None
        # The algorithm object should be the same type and have similar state
        assert type(loaded_optimizer._curr_algorithm_obj) == type(
            original_algorithm_obj
        )

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading state preserves optimizer functionality."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run partial optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and continue optimization
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=5, rng=self.rng
        )

        # Verify optimization completed successfully
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        checkpoint_dir = str(tmp_path / "nonexistent_checkpoint")

        with pytest.raises(CheckpointNotFoundError, match="Checkpoint file not found"):
            PymooOptimizer.load_state(checkpoint_dir)

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run optimization to set state
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        # Save to a non-existent nested directory
        checkpoint_dir = str(tmp_path / "nested" / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Verify directory and file were created
        state_file = tmp_path / "nested" / "checkpoint" / "optimizer_state.json"
        assert state_file.exists(), "Checkpoint directory and file should be created"

    def test_resume_extends_iteration_budget(self, tmp_path):
        """Resuming with a higher max_iterations should continue running iterations."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=6)
        initial_params = (
            self.rng.random((optimizer.n_param_sets, self.n_params)) * 2 * np.pi
        )

        # Run part of the optimization and checkpoint.
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )
        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=4, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.nit == 4, "Resume should complete up to the new iteration target"

    def test_resume_when_algorithm_finished(self, tmp_path):
        """Test resuming when the saved checkpoint was from a completed optimization."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run until completion (2 iterations)
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        # Verify algorithm finished
        if optimizer.method == PymooMethod.CMAES:
            # cma uses countiter
            assert optimizer._curr_algorithm_obj.countiter == 2
        else:
            # n_gen is 1-indexed, so 2 iterations = n_gen 3
            assert optimizer._curr_algorithm_obj.n_gen == 3

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Resume with more iterations
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=5, rng=self.rng
        )

        # Should be able to continue
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_n_param_sets_respects_popsize_kwarg(self):
        """n_param_sets should honor CMAES popsize overrides."""
        optimizer_with_kwarg = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=6, popsize=4
        )
        assert optimizer_with_kwarg.n_param_sets == 4

        optimizer_default_cmaes = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=8
        )
        assert optimizer_default_cmaes.n_param_sets == 8

        optimizer_de = PymooOptimizer(method=PymooMethod.DE, population_size=11)
        assert optimizer_de.n_param_sets == 11

    def test_de_save_load_round_trip(self, tmp_path):
        """Test that DE optimizer can save/load state successfully."""
        optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run partial optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and continue optimization
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=5, rng=self.rng
        )

        # Verify optimization completed successfully
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)
        assert result.nit == 5, "DE should complete up to the new iteration target"

    def test_de_population_preserved_in_checkpoint(self, tmp_path):
        """Test that DE's population is properly preserved during checkpointing."""
        optimizer = PymooOptimizer(method=PymooMethod.DE, population_size=8)
        initial_params = self.rng.random((8, self.n_params)) * 2 * np.pi

        # Run some iterations
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Verify internal state exists
        assert optimizer._curr_algorithm_obj is not None
        assert hasattr(optimizer._curr_algorithm_obj, "pop")
        assert optimizer._curr_algorithm_obj.pop is not None

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify population is restored
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer._curr_algorithm_obj is not None
        assert hasattr(loaded_optimizer._curr_algorithm_obj, "pop")
        assert loaded_optimizer._curr_algorithm_obj.pop is not None

    def test_cmaes_custom_kwargs_preserved_through_save_load(self, tmp_path):
        """Test that custom CMAES kwargs survive save/load cycle."""
        # Create optimizer with custom kwargs
        optimizer = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=10, popsize=8, sigma0=0.3
        )
        initial_params = self.rng.random((8, self.n_params)) * 2 * np.pi

        # Run optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Load and verify kwargs are preserved
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        assert loaded_optimizer.population_size == 10
        assert loaded_optimizer.algorithm_kwargs.get("popsize") == 8
        assert loaded_optimizer.algorithm_kwargs.get("sigma0") == 0.3

        # Verify it can continue optimization
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, max_iterations=4, rng=self.rng
        )
        assert isinstance(result, OptimizeResult)
        assert result.nit == 4

    def test_optimize_with_zero_max_iterations(self):
        """Test that optimize with max_iterations=0 returns immediately."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        initial_params = self.rng.random((5, self.n_params)) * 2 * np.pi

        # First run some iterations
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Now try to resume with max_iterations=0
        result = optimizer.optimize(
            sphere_cost_fn_population, max_iterations=0, rng=self.rng
        )

        # Should return current state without running more iterations
        assert isinstance(result, OptimizeResult)
        assert result.nit == 3  # Should still be at iteration 3

    def test_resume_with_max_iterations_equal_to_completed(self):
        """Test resuming when max_iterations equals already completed iterations."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        initial_params = self.rng.random((5, self.n_params)) * 2 * np.pi

        # Run 5 iterations
        result1 = optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )
        assert result1.nit == 5

        # Resume with same max_iterations should not run additional iterations
        result2 = optimizer.optimize(
            sphere_cost_fn_population, max_iterations=5, rng=self.rng
        )
        assert result2.nit == 5  # Should still be 5, not 10


class TestScipyOptimizer:
    """Specific tests for ScipyOptimizer features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(42)

    def test_reset_is_noop_for_scipy(self):
        """Test that reset() is a no-op for ScipyOptimizer (no state to clear)."""
        optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        initial_params = self.rng.random((1, self.n_params)) * 2 * np.pi

        # Run optimization
        optimizer.optimize(sphere_cost_fn_single, initial_params, max_iterations=3)

        # Reset should not raise an error and should allow reusing
        optimizer.reset()
        result = optimizer.optimize(
            sphere_cost_fn_single, initial_params, max_iterations=3
        )
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)

    def test_save_state_raises_not_implemented(self, tmp_path):
        """Test that ScipyOptimizer.save_state() raises NotImplementedError."""
        optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        checkpoint_dir = str(tmp_path / "checkpoint")

        with pytest.raises(
            NotImplementedError, match="ScipyOptimizer does not support"
        ):
            optimizer.save_state(checkpoint_dir)

    def test_load_state_raises_not_implemented(self, tmp_path):
        """Test that ScipyOptimizer.load_state() raises NotImplementedError."""
        checkpoint_dir = str(tmp_path / "checkpoint")

        with pytest.raises(
            NotImplementedError, match="ScipyOptimizer does not support"
        ):
            ScipyOptimizer.load_state(checkpoint_dir)

    @pytest.mark.parametrize(
        ("method", "expects_jac"),
        [
            (ScipyMethod.L_BFGS_B, True),
            (ScipyMethod.NELDER_MEAD, False),
        ],
    )
    def test_optimize_passes_jac_only_for_l_bfgs_b(
        self, method: ScipyMethod, expects_jac: bool, monkeypatch
    ):
        """Ensure jacobian callbacks are only forwarded when supported."""
        recorded = {}

        def fake_minimize(cost_fn, x0, **kwargs):
            recorded["jac"] = kwargs.get("jac")
            recorded["method"] = kwargs.get("method")
            recorded["callback"] = kwargs.get("callback")
            return OptimizeResult(x=x0, fun=0.0)

        monkeypatch.setattr("divi.qprog.optimizers.minimize", fake_minimize)

        optimizer = ScipyOptimizer(method=method)
        jac_fn = lambda x: x
        initial_params = self.rng.random((1, self.n_params)) * 0.1

        optimizer.optimize(
            sphere_cost_fn_single,
            initial_params,
            jac=jac_fn,
            max_iterations=5,
        )

        if expects_jac:
            assert recorded["jac"] is jac_fn
        else:
            assert recorded["jac"] is None
        assert recorded["method"] == method.value

    def test_callback_shapes_are_normalized(self, monkeypatch):
        """Callbacks should always observe 2D parameters and 1D losses."""
        captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def fake_minimize(cost_fn, x0, **kwargs):
            callback = kwargs.get("callback")
            assert callback is not None
            callback(OptimizeResult(x=np.ones(self.n_params), fun=1.23))
            return OptimizeResult(x=x0, fun=0.0)

        monkeypatch.setattr("divi.qprog.optimizers.minimize", fake_minimize)

        optimizer = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)
        initial_params = self.rng.random((1, self.n_params)) * 0.1

        def tracking_callback(result: OptimizeResult):
            captured_shapes.append((result.x.shape, result.fun.shape))

        optimizer.optimize(
            sphere_cost_fn_single,
            initial_params,
            max_iterations=3,
            callback_fn=tracking_callback,
        )

        assert captured_shapes == [((1, self.n_params), (1,))]


class TestCopyOptimizer:
    """Direct unit tests for the copy_optimizer helper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_params = 4
        self.rng = np.random.default_rng(24)

    def test_copy_monte_carlo_preserves_config_and_resets_state(self):
        optimizer = MonteCarloOptimizer(
            population_size=8, n_best_sets=2, keep_best_params=True
        )
        initial_params = (
            self.rng.random((optimizer.population_size, self.n_params)) * 2 * np.pi
        )
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=1, rng=self.rng
        )
        assert optimizer._curr_population is not None

        copied = copy_optimizer(optimizer)

        assert isinstance(copied, MonteCarloOptimizer)
        assert copied.population_size == optimizer.population_size
        assert copied.keep_best_params == optimizer.keep_best_params
        assert copied.n_best_sets == optimizer.n_best_sets
        assert copied._curr_population is None

    def test_copy_pymoo_preserves_kwargs(self):
        optimizer = PymooOptimizer(
            method=PymooMethod.CMAES, population_size=5, popsize=4, sigma=0.5
        )
        initial_params = (
            self.rng.random((optimizer.n_param_sets, self.n_params)) * 2 * np.pi
        )
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=1, rng=self.rng
        )
        assert optimizer._curr_algorithm_obj is not None

        copied = copy_optimizer(optimizer)

        assert isinstance(copied, PymooOptimizer)
        assert copied.method == optimizer.method
        assert copied.population_size == optimizer.population_size
        assert copied.algorithm_kwargs["popsize"] == 4
        assert copied.algorithm_kwargs["sigma"] == 0.5
        assert copied._curr_algorithm_obj is None

    def test_copy_scipy_returns_fresh_optimizer(self):
        optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)
        copied = copy_optimizer(optimizer)

        assert isinstance(copied, ScipyOptimizer)
        assert copied.method == optimizer.method
