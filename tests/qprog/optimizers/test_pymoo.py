# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import (
    PymooMethod,
    PymooOptimizer,
)
from tests.qprog.optimizers._contracts import (
    sphere_cost_fn_population,
    verify_load_state_raises_file_not_found,
    verify_save_creates_checkpoint_file,
    verify_save_creates_directory_if_needed,
    verify_save_load_round_trip,
    verify_save_without_prior_run_raises,
)


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
        verify_save_creates_checkpoint_file(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            PymooOptimizer.load_state,
        )

    def test_save_state_without_prior_run_raises(self, tmp_path):
        """Saving without having run optimize should raise a clear error."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)
        verify_save_without_prior_run_raises(optimizer, tmp_path)

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
        verify_save_load_round_trip(
            optimizer,
            initial_params,
            sphere_cost_fn_population,
            self.rng,
            tmp_path,
            PymooOptimizer.load_state,
            self.n_params,
        )

    def test_load_state_raises_file_not_found(self, tmp_path):
        """Test that load_state() raises CheckpointNotFoundError for missing checkpoint."""
        verify_load_state_raises_file_not_found(PymooOptimizer.load_state, tmp_path)

    def test_save_state_creates_directory_if_needed(self, tmp_path):
        """Test that save_state() creates the checkpoint directory if it doesn't exist."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=10)
        initial_params = self.rng.random((10, self.n_params)) * 2 * np.pi
        verify_save_creates_directory_if_needed(
            optimizer, initial_params, sphere_cost_fn_population, self.rng, tmp_path
        )

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

    def test_fresh_run_without_initial_params_raises(self):
        """A fresh run must receive initial_params."""
        optimizer = PymooOptimizer(method=PymooMethod.CMAES, population_size=5)

        with pytest.raises(ValueError, match="initial_params is required"):
            optimizer.optimize(sphere_cost_fn_population, max_iterations=3)

    def test_copy_preserves_kwargs(self):
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

        copied = optimizer.copy()

        assert isinstance(copied, PymooOptimizer)
        assert copied.method == optimizer.method
        assert copied.population_size == optimizer.population_size
        assert copied.algorithm_kwargs["popsize"] == 4
        assert copied.algorithm_kwargs["sigma"] == 0.5
        assert copied._curr_algorithm_obj is None
