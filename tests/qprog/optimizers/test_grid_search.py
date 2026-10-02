# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

from divi.qprog.checkpointing import OPTIMIZER_STATE_FILE
from divi.qprog.optimizers import (
    GridSearchOptimizer,
)
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    sphere_cost_fn_population,
)


class _CustomGridSearchOptimizer(GridSearchOptimizer):
    pass


def test_copy_preserves_subclass_type():
    optimizer = _CustomGridSearchOptimizer(param_ranges=[(-1, 1)], grid_points=5)
    assert type(optimizer.copy()) is _CustomGridSearchOptimizer


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    # -- Construction --

    @pytest.mark.parametrize(
        ("kwargs", "n_param_sets"),
        [
            ({"param_ranges": [(0, 1), (0, 2)], "grid_points": 3}, 9),
            ({"param_ranges": [(-5, 5)], "grid_points": 100}, 100),
            ({"param_ranges": [(0, 1)]}, 20),
            ({"param_grid": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])}, 3),
        ],
        ids=["cartesian-grid", "1d", "default-grid-points", "explicit-grid"],
    )
    def test_grid_size(self, kwargs, n_param_sets):
        assert GridSearchOptimizer(**kwargs).n_param_sets == n_param_sets

    def test_no_args_raises(self):
        with pytest.raises(
            ValueError,
            match=exact_match("Provide either param_grid or param_ranges."),
        ):
            GridSearchOptimizer()

    def test_1d_param_grid_raises(self):
        with pytest.raises(
            ValueError, match=exact_match("param_grid must be 2D, got 1D.")
        ):
            GridSearchOptimizer(param_grid=np.array([1.0, 2.0, 3.0]))

    # -- Optimization --

    def test_finds_minimum_1d(self):
        grid = GridSearchOptimizer(param_ranges=[(-5, 5)], grid_points=100)
        result = grid.optimize(
            sphere_cost_fn_population, initial_params=np.zeros((100, 1))
        )
        half_step = 5.0 / 99.0
        assert abs(result.x[0]) == pytest.approx(half_step, abs=1e-12)
        assert result.fun == pytest.approx(half_step**2, abs=1e-12)
        assert result.nit == 1

    def test_finds_minimum_2d(self):
        grid = GridSearchOptimizer(param_ranges=[(-3, 3), (-3, 3)], grid_points=50)
        result = grid.optimize(
            sphere_cost_fn_population, initial_params=np.zeros((1, 2))
        )
        np.testing.assert_allclose(np.abs(result.x), [3.0 / 49.0] * 2, atol=1e-12)

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

    @pytest.mark.parametrize("max_iterations", [2, 10])
    def test_max_iterations_warning(self, max_iterations):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        message = (
            "GridSearchOptimizer evaluates all grid points in a single pass. "
            f"max_iterations={max_iterations} will be ignored."
        )
        with pytest.warns(UserWarning, match=exact_match(message)):
            grid.optimize(
                sphere_cost_fn_population,
                initial_params=np.zeros((5, 1)),
                max_iterations=max_iterations,
            )

    @pytest.mark.parametrize(
        "kwargs", [{}, {"max_iterations": 1}], ids=["default", "one"]
    )
    def test_max_iterations_1_no_warning(self, kwargs):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grid.optimize(
                sphere_cost_fn_population,
                initial_params=np.zeros((5, 1)),
                **kwargs,
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
        np.testing.assert_array_equal(loaded._all_losses, grid._all_losses)

    def test_save_before_optimize_stores_none(self, tmp_path):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=5)
        grid.save_state(tmp_path)
        loaded = GridSearchOptimizer.load_state(tmp_path)
        assert loaded._best_params is None
        assert loaded._best_loss is None
        assert loaded._all_losses is None

    def test_save_state_creates_missing_directories(self, tmp_path):
        grid = GridSearchOptimizer(param_ranges=[(0, 1)], grid_points=3)
        grid.save_state(tmp_path / "nested" / "checkpoint")
        assert (tmp_path / "nested" / "checkpoint" / OPTIMIZER_STATE_FILE).exists()

    def test_integer_param_grid_yields_float_parameters(self):
        grid = GridSearchOptimizer(param_grid=[[0, 1], [2, 3]])
        result = grid.optimize(sphere_cost_fn_population)
        assert result.x.dtype == np.float64
        np.testing.assert_array_equal(result.x, [0.0, 1.0])

    def test_param_ranges_grid_varies_the_last_parameter_fastest(self):
        evaluated = []

        def recording_cost(population):
            evaluated.append(population.copy())
            return sphere_cost_fn_population(population)

        GridSearchOptimizer(param_ranges=[(0, 1), (10, 11)], grid_points=2).optimize(
            recording_cost
        )

        (population,) = evaluated
        np.testing.assert_array_equal(population, [[0, 10], [0, 11], [1, 10], [1, 11]])

    def test_copy_returns_fresh_optimizer(self):
        optimizer = GridSearchOptimizer(param_ranges=[(-1, 1), (-1, 1)], grid_points=5)
        optimizer.optimize(sphere_cost_fn_population, np.zeros((25, 2)))

        copied = optimizer.copy()
        assert isinstance(copied, GridSearchOptimizer)
        assert copied.n_param_sets == 25
        assert copied._best_params is None  # fresh state
