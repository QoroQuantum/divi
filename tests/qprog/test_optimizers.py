# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)


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
        ("scipy-nelder-mead", lambda: ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)),
        ("scipy-cobyla", lambda: ScipyOptimizer(method=ScipyMethod.COBYLA)),
        ("scipy-l-bfgs-b", lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B)),
        ("monte-carlo", lambda: MonteCarloOptimizer(population_size=10, n_best_sets=3)),
        (
            "monte-carlo-keep-best",
            lambda: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=True
            ),
        ),
        (
            "pymoo-cmaes",
            lambda: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        ),
        (
            "pymoo-de",
            lambda: PymooOptimizer(method=PymooMethod.DE, population_size=10),
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

    def test_checkpoint_dir_parameter_accepted(self, optimizer: Optimizer, tmp_path):
        """Verify that all optimizers accept checkpoint_dir parameter (even if not supported)."""
        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )
        checkpoint_dir = str(tmp_path / "test_checkpoint")

        # Should not raise TypeError for unsupported optimizers
        # ScipyOptimizer doesn't support checkpointing but should accept the parameter
        if isinstance(optimizer, ScipyOptimizer):
            # ScipyOptimizer should accept but not use checkpoint_dir
            result = optimizer.optimize(
                cost_fn, initial_params, max_iterations=3, checkpoint_dir=checkpoint_dir
            )
            assert isinstance(result, OptimizeResult)
        else:
            # Other optimizers should use checkpoint_dir
            result = optimizer.optimize(
                cost_fn, initial_params, max_iterations=3, checkpoint_dir=checkpoint_dir
            )
            assert isinstance(result, OptimizeResult)

    def test_checkpoint_dir_with_callback(self, optimizer: Optimizer, tmp_path):
        """Verify that checkpoint_dir works correctly when callback is also provided."""
        # Skip ScipyOptimizer as it doesn't support checkpointing
        if isinstance(optimizer, ScipyOptimizer):
            pytest.skip("ScipyOptimizer does not support checkpointing")

        initial_params = self._get_initial_params(optimizer)
        cost_fn = (
            sphere_cost_fn_single
            if optimizer.n_param_sets == 1
            else sphere_cost_fn_population
        )
        checkpoint_dir = str(tmp_path / "callback_checkpoint")

        callback_calls = []

        def callback(intermediate_result: OptimizeResult):
            callback_calls.append(intermediate_result)

        result = optimizer.optimize(
            cost_fn,
            initial_params,
            callback_fn=callback,
            checkpoint_dir=checkpoint_dir,
            max_iterations=3,
        )

        # Verify both callback and checkpointing worked
        assert isinstance(result, OptimizeResult)
        assert len(callback_calls) > 0
        # Verify checkpoint was created
        if isinstance(optimizer, PymooOptimizer):
            state_file = tmp_path / "callback_checkpoint" / "optimizer_state.pkl"
        elif isinstance(optimizer, MonteCarloOptimizer):
            state_file = tmp_path / "callback_checkpoint" / "optimizer_state.npz"
        assert state_file.exists()


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
        state_file = tmp_path / "checkpoint" / "optimizer_state.npz"
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
        """Test that load_state() raises FileNotFoundError for missing checkpoint."""
        checkpoint_dir = str(tmp_path / "nonexistent_checkpoint")

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
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
        state_file = tmp_path / "nested" / "checkpoint" / "optimizer_state.npz"
        assert state_file.exists(), "Checkpoint directory and file should be created"

    def _get_initial_params_for_optimizer(self, optimizer: Optimizer) -> np.ndarray:
        """Helper to get initial params based on optimizer type."""
        if isinstance(optimizer, MonteCarloOptimizer):
            return self._create_initial_params()
        else:
            return self.rng.random((10, self.n_params)) * 2 * np.pi

    def _get_load_state_func(self, optimizer: Optimizer):
        """Helper to get load_state function based on optimizer type."""
        if isinstance(optimizer, MonteCarloOptimizer):
            return MonteCarloOptimizer.load_state
        else:
            return PymooOptimizer.load_state

    def _get_state_file_path(self, checkpoint_dir: Path, optimizer: Optimizer) -> Path:
        """Helper to get state file path based on optimizer type."""
        if isinstance(optimizer, MonteCarloOptimizer):
            return checkpoint_dir / "optimizer_state.npz"
        else:
            return checkpoint_dir / "optimizer_state.pkl"

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            lambda rng: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=False
            ),
            lambda rng: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_checkpoint_dir_saves_state_each_iteration(
        self, tmp_path, optimizer_factory
    ):
        """Test that providing checkpoint_dir saves state at the end of each iteration."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)
        checkpoint_dir = str(tmp_path / "auto_checkpoint")

        # Run optimization with checkpoint_dir
        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=3,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # Verify checkpoint file exists
        state_file = self._get_state_file_path(tmp_path / "auto_checkpoint", optimizer)
        load_state_func = self._get_load_state_func(optimizer)
        loaded_optimizer = load_state_func(checkpoint_dir)

        if isinstance(optimizer, MonteCarloOptimizer):
            assert loaded_optimizer._curr_population is not None
            assert loaded_optimizer._curr_iteration is not None
        else:
            assert loaded_optimizer._curr_algorithm_obj is not None
            assert loaded_optimizer._curr_pop is not None

        assert state_file.exists(), "Checkpoint file should be created"

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            lambda rng: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=False
            ),
            lambda rng: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_checkpoint_dir_allows_resuming(self, tmp_path, optimizer_factory):
        """Test that checkpoints saved via checkpoint_dir can be used to resume optimization."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)
        checkpoint_dir = str(tmp_path / "resume_checkpoint")

        # Run partial optimization with checkpoint_dir
        result1 = optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=2,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # Load checkpoint and continue
        load_state_func = self._get_load_state_func(optimizer)
        loaded_optimizer = load_state_func(checkpoint_dir)

        result2 = loaded_optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=5,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # Verify optimization completed successfully
        assert isinstance(result2, OptimizeResult)
        assert result2.x.shape == (self.n_params,)
        assert np.isfinite(result2.fun)

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            lambda rng: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=False
            ),
            lambda rng: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_checkpoint_dir_updates_each_iteration(self, tmp_path, optimizer_factory):
        """Test that checkpoint_dir updates the checkpoint file after each iteration."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)
        checkpoint_dir = str(tmp_path / "iter_checkpoint")

        # Run optimization with checkpoint_dir
        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=3,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # Verify checkpoint exists and is loadable
        state_file = self._get_state_file_path(tmp_path / "iter_checkpoint", optimizer)
        load_state_func = self._get_load_state_func(optimizer)
        loaded_optimizer = load_state_func(checkpoint_dir)

        if isinstance(optimizer, MonteCarloOptimizer):
            assert loaded_optimizer._curr_population is not None
            assert loaded_optimizer._curr_iteration is not None
        else:
            assert loaded_optimizer._curr_algorithm_obj is not None
            assert loaded_optimizer._curr_pop is not None

        assert state_file.exists(), "Checkpoint file should exist"

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            lambda rng: MonteCarloOptimizer(
                population_size=10, n_best_sets=3, keep_best_params=False
            ),
            lambda rng: PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_checkpoint_dir_without_checkpointing(self, tmp_path, optimizer_factory):
        """Test that optimization works normally when checkpoint_dir is not provided."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)

        # Run optimization without checkpoint_dir
        result = optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Verify optimization completed successfully
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize(
        "optimizer_factory,load_state_func",
        [
            (
                lambda rng: MonteCarloOptimizer(
                    population_size=10, n_best_sets=3, keep_best_params=False
                ),
                MonteCarloOptimizer.load_state,
            ),
            (
                lambda rng: PymooOptimizer(
                    method=PymooMethod.CMAES, population_size=10
                ),
                PymooOptimizer.load_state,
            ),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_resume_with_max_iterations_less_than_completed(
        self, tmp_path, optimizer_factory, load_state_func
    ):
        """Test resuming when max_iterations is less than already completed iterations."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)

        # Run 5 iterations
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Resume with max_iterations=3 (less than 5 already done)
        loaded_optimizer = load_state_func(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=3, rng=self.rng
        )

        # Should complete immediately since we've already done more iterations
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize(
        "optimizer_factory,load_state_func",
        [
            (
                lambda rng: MonteCarloOptimizer(
                    population_size=10, n_best_sets=3, keep_best_params=False
                ),
                MonteCarloOptimizer.load_state,
            ),
            (
                lambda rng: PymooOptimizer(
                    method=PymooMethod.CMAES, population_size=10
                ),
                PymooOptimizer.load_state,
            ),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_multiple_resume_operations(
        self, tmp_path, optimizer_factory, load_state_func
    ):
        """Test chaining multiple save/load/resume operations."""
        optimizer = optimizer_factory(self.rng)
        initial_params = self._get_initial_params_for_optimizer(optimizer)
        checkpoint_dir = str(tmp_path / "chain_checkpoint")

        # First run: 2 iterations
        optimizer.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=2,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # First resume: continue to 4 iterations
        loaded1 = load_state_func(checkpoint_dir)
        loaded1.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=4,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        # Second resume: continue to 6 iterations
        loaded2 = load_state_func(checkpoint_dir)
        result = loaded2.optimize(
            sphere_cost_fn_population,
            initial_params,
            max_iterations=6,
            checkpoint_dir=checkpoint_dir,
            rng=self.rng,
        )

        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize(
        "optimizer_factory,load_state_func",
        [
            (
                lambda rng: MonteCarloOptimizer(
                    population_size=10, n_best_sets=3, keep_best_params=False
                ),
                MonteCarloOptimizer.load_state,
            ),
            (
                lambda rng: PymooOptimizer(
                    method=PymooMethod.CMAES, population_size=10
                ),
                PymooOptimizer.load_state,
            ),
        ],
        ids=["monte-carlo", "pymoo-cmaes"],
    )
    def test_resume_with_different_initial_params(
        self, tmp_path, optimizer_factory, load_state_func
    ):
        """Test that resuming ignores different initial_params (uses checkpoint state)."""
        optimizer = optimizer_factory(self.rng)
        initial_params1 = self._get_initial_params_for_optimizer(optimizer)

        # Run partial optimization
        optimizer.optimize(
            sphere_cost_fn_population, initial_params1, max_iterations=2, rng=self.rng
        )

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Resume with different initial_params (should be ignored)
        initial_params2 = self._get_initial_params_for_optimizer(optimizer)
        loaded_optimizer = load_state_func(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params2, max_iterations=5, rng=self.rng
        )

        # Should continue from checkpoint, not use new initial_params
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
        assert optimizer._curr_pop is not None

        # Reset and verify state is cleared
        optimizer.reset()
        assert optimizer._curr_algorithm_obj is None
        assert optimizer._curr_pop is None

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
        state_file = tmp_path / "checkpoint" / "optimizer_state.pkl"
        assert state_file.exists(), "Checkpoint file should be created"

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
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        # Verify optimization completed successfully
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises FileNotFoundError for missing checkpoint."""
        checkpoint_dir = str(tmp_path / "nonexistent_checkpoint")

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
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
        state_file = tmp_path / "nested" / "checkpoint" / "optimizer_state.pkl"
        assert state_file.exists(), "Checkpoint directory and file should be created"

    def test_resume_when_algorithm_finished(self, tmp_path):
        """Test resuming when the saved checkpoint was from a completed optimization."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi

        # Run until completion (2 iterations)
        optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=2, rng=self.rng
        )

        # Verify algorithm finished
        assert not optimizer._curr_algorithm_obj.has_next()

        checkpoint_dir = str(tmp_path / "checkpoint")
        optimizer.save_state(checkpoint_dir)

        # Resume with more iterations
        loaded_optimizer = PymooOptimizer.load_state(checkpoint_dir)
        result = loaded_optimizer.optimize(
            sphere_cost_fn_population, initial_params, max_iterations=5, rng=self.rng
        )

        # Should be able to continue
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (self.n_params,)
        assert np.isfinite(result.fun)


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
