# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
)
from tests.qprog.optimizers._contracts import (
    sphere_cost_fn_population,
    verify_load_state_raises_file_not_found,
    verify_save_creates_checkpoint_file,
    verify_save_creates_directory_if_needed,
    verify_save_load_round_trip,
)


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
        verify_save_creates_checkpoint_file(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            MonteCarloOptimizer.load_state,
        )

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
        verify_save_load_round_trip(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            MonteCarloOptimizer.load_state,
            self.n_params,
        )

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        verify_load_state_raises_file_not_found(
            MonteCarloOptimizer.load_state, tmp_path
        )

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()
        verify_save_creates_directory_if_needed(
            optimizer, initial_params, sphere_cost_fn_population, self.rng, tmp_path
        )

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

    def test_fresh_run_without_initial_params_raises(self):
        """A fresh run must receive initial_params."""
        optimizer = self._create_optimizer(keep_best_params=False)

        with pytest.raises(ValueError, match="initial_params is required"):
            optimizer.optimize(sphere_cost_fn_population, max_iterations=3)

    def test_fresh_run_zero_iterations_raises(self):
        """Fresh run with max_iterations=0 never evaluates the cost, so no state exists to return."""
        optimizer = self._create_optimizer(keep_best_params=False)
        initial_params = self._create_initial_params()

        with pytest.raises(RuntimeError, match="produced no evaluated population"):
            optimizer.optimize(
                sphere_cost_fn_population,
                initial_params,
                max_iterations=0,
                rng=self.rng,
            )

    def test_copy_preserves_config_and_resets_state(self):
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

        copied = optimizer.copy()

        assert isinstance(copied, MonteCarloOptimizer)
        assert copied.population_size == optimizer.population_size
        assert copied.keep_best_params == optimizer.keep_best_params
        assert copied.n_best_sets == optimizer.n_best_sets
        assert copied._curr_population is None


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

    def test_checkpoint_dir_without_checkpointing(self, checkpointing_optimizer):
        """Optimization should work normally when no checkpoint_dir is provided."""
        initial_params = self._initial_params(checkpointing_optimizer)

        result = checkpointing_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_resume_with_max_iterations_less_than_completed(
        self, tmp_path, checkpointing_optimizer
    ):
        """Resuming with fewer total iterations should exit immediately."""
        load_state_func = type(checkpointing_optimizer).load_state

        initial_params = self._initial_params(checkpointing_optimizer)

        checkpointing_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        checkpoint_dir = tmp_path / "checkpoint"
        checkpointing_optimizer.save_state(str(checkpoint_dir))

        loaded_optimizer = load_state_func(str(checkpoint_dir))
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_resume_with_different_initial_params(
        self, tmp_path, checkpointing_optimizer
    ):
        """Checkpoints should ignore newly provided initial parameters when resuming."""
        load_state_func = type(checkpointing_optimizer).load_state
        initial_params = self._initial_params(checkpointing_optimizer)

        checkpointing_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = tmp_path / "checkpoint"
        checkpointing_optimizer.save_state(str(checkpoint_dir))

        loaded_optimizer = load_state_func(str(checkpoint_dir))
        different_initial_params = self._initial_params(checkpointing_optimizer)

        result = loaded_optimizer.optimize(
            sphere_cost_fn_population,
            different_initial_params,
            max_iterations=5,
            rng=self.rng,
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)
