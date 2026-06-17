# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

from divi.qprog.optimizers import (
    GridSearchOptimizer,
)
from tests.qprog.optimizers._contracts import (
    sphere_cost_fn_population,
)


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    # -- Construction --

    def test_param_ranges_creates_cartesian_grid(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1), (0, 2)], grid_points=3)
        assert grid.n_param_sets == 9  # 3 × 3

    def test_param_ranges_1d(self):
        grid = GridSearchOptimizer(param_ranges=[(-5, 5)], grid_points=100)
        assert grid.n_param_sets == 100

    def test_explicit_param_grid(self):
        explicit = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        grid = GridSearchOptimizer(param_grid=explicit)
        assert grid.n_param_sets == 3

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="param_grid or param_ranges"):
            GridSearchOptimizer()

    def test_1d_param_grid_raises(self):
        with pytest.raises(ValueError, match="2D"):
            GridSearchOptimizer(param_grid=np.array([1.0, 2.0, 3.0]))

    # -- Optimization --

    def test_finds_minimum_1d(self):
        grid = GridSearchOptimizer(param_ranges=[(-5, 5)], grid_points=100)
        result = grid.optimize(
            sphere_cost_fn_population, initial_params=np.zeros((100, 1))
        )
        assert abs(result.x[0]) < 0.15
        assert result.fun < 0.03
        assert result.nit == 1

    def test_finds_minimum_2d(self):
        grid = GridSearchOptimizer(param_ranges=[(-3, 3), (-3, 3)], grid_points=50)
        result = grid.optimize(
            sphere_cost_fn_population, initial_params=np.zeros((1, 2))
        )
        assert np.allclose(result.x, [0.0, 0.0], atol=0.15)

    def test_initial_params_ignored(self):
        """Grid is used regardless of initial_params."""
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=10)
        result = grid.optimize(
            sphere_cost_fn_population,
            initial_params=np.array([[999.0]]),  # should be ignored
        )
        assert 0.0 <= result.x[0] <= 1.0

    def test_callback_called(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        callbacks = []
        grid.optimize(
            sphere_cost_fn_population,
            initial_params=np.zeros((5, 1)),
            callback_fn=lambda r: callbacks.append(r),
        )
        assert len(callbacks) == 1

    def test_max_iterations_warning(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        with pytest.warns(UserWarning, match="max_iterations=10 will be ignored"):
            grid.optimize(
                sphere_cost_fn_population,
                initial_params=np.zeros((5, 1)),
                max_iterations=10,
            )

    def test_max_iterations_1_no_warning(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        # Should not warn when max_iterations=1 (the default)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid.optimize(
                sphere_cost_fn_population,
                initial_params=np.zeros((5, 1)),
                max_iterations=1,
            )

    # -- State management --

    def test_reset_clears_state(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        grid.optimize(sphere_cost_fn_population, np.zeros((5, 1)))
        assert grid._best_params is not None

        grid.reset()
        assert grid._best_params is None
        assert grid._best_loss is None
        assert grid._all_losses is None

    def test_get_config(self):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=10)
        config = grid.get_config()
        assert config["type"] == "GridSearchOptimizer"
        assert config["grid_size"] == 10

    # -- Checkpointing --

    def test_save_and_load_state(self, tmp_path):
        grid = GridSearchOptimizer(param_ranges=[(-1, 1)], grid_points=20)
        grid.optimize(sphere_cost_fn_population, np.zeros((20, 1)))

        grid.save_state(tmp_path)
        loaded = GridSearchOptimizer.load_state(tmp_path)

        assert loaded.n_param_sets == 20
        np.testing.assert_array_almost_equal(loaded._best_params, grid._best_params)
        assert loaded._best_loss == pytest.approx(grid._best_loss)

    def test_save_before_optimize_stores_none(self, tmp_path):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        grid.save_state(tmp_path)
        loaded = GridSearchOptimizer.load_state(tmp_path)
        assert loaded._best_params is None
        assert loaded._best_loss is None

    def test_copy_returns_fresh_optimizer(self):
        optimizer = GridSearchOptimizer(param_ranges=[(-1, 1), (-1, 1)], grid_points=5)
        optimizer.optimize(sphere_cost_fn_population, np.zeros((25, 2)))

        copied = optimizer.copy()
        assert isinstance(copied, GridSearchOptimizer)
        assert copied.n_param_sets == 25
        assert copied._best_params is None  # fresh state
