# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import OPTIMIZER_STATE_FILE, _atomic_write
from divi.qprog.optimizers._base import Optimizer


class GridSearchOptimizer(Optimizer):
    """Exhaustive grid search optimizer.

    Evaluates all parameter combinations on a user-supplied grid in a
    single iteration and returns the best-performing parameters. Designed
    for low-dimensional parameter spaces like CE-QAOA (gamma, beta).
    """

    def __init__(
        self,
        param_grid: npt.NDArray[np.float64] | None = None,
        *,
        param_ranges: list[tuple[float, float]] | None = None,
        grid_points: int = 20,
    ):
        """Initialize a grid search optimizer.

        Provide either *param_grid* directly or *param_ranges* + *grid_points*
        to auto-generate the grid.

        Args:
            param_grid: Explicit 2D array of shape ``(n_points, n_params)``
                where each row is a parameter combination to evaluate.
            param_ranges: List of ``(low, high)`` tuples, one per parameter.
                Used with *grid_points* to generate a Cartesian-product grid.
            grid_points: Number of grid points per parameter dimension.
                Only used with *param_ranges*. Defaults to 20.

        Raises:
            ValueError: If neither *param_grid* nor *param_ranges* is provided,
                or if *param_grid* is not 2D.
        """
        super().__init__()

        if param_grid is not None:
            param_grid = np.asarray(param_grid, dtype=np.float64)
            if param_grid.ndim != 2:
                raise ValueError(f"param_grid must be 2D, got {param_grid.ndim}D.")
            self._param_grid = param_grid
        elif param_ranges is not None:
            axes = [np.linspace(lo, hi, grid_points) for lo, hi in param_ranges]
            mesh = np.meshgrid(*axes, indexing="ij")
            self._param_grid = np.column_stack([m.ravel() for m in mesh]).astype(
                np.float64
            )
        else:
            raise ValueError("Provide either param_grid or param_ranges.")

        self._best_params: npt.NDArray[np.float64] | None = None
        self._best_loss: float | None = None
        self._all_losses: npt.NDArray[np.float64] | None = None

    @property
    def n_param_sets(self) -> int:
        """Number of parameter sets (grid size)."""
        return len(self._param_grid)

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Evaluate all grid points and return the best parameters.

        The *initial_params* argument is ignored in favor of the grid.
        The optimizer always runs for exactly 1 iteration regardless of
        *max_iterations*.

        Args:
            cost_fn: Cost function accepting a 2D array of parameter sets and
                returning an array of losses.
            initial_params: Ignored (grid is used instead).
            callback_fn: Optional callback with intermediate results.
            **kwargs: Accepts but ignores ``max_iterations`` and ``rng``.

        Returns:
            OptimizeResult with the best parameters.
        """
        max_iterations = kwargs.pop("max_iterations", 1)
        if max_iterations > 1:
            warnings.warn(
                f"GridSearchOptimizer evaluates all grid points in a single pass. "
                f"max_iterations={max_iterations} will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        kwargs.pop("rng", None)

        self._all_losses = np.atleast_1d(cost_fn(self._param_grid)).astype(np.float64)

        best_idx = np.nanargmin(self._all_losses)
        self._best_params = self._param_grid[best_idx].copy()
        self._best_loss = float(self._all_losses[best_idx])

        if callback_fn:
            callback_fn(OptimizeResult(x=self._param_grid, fun=self._all_losses))

        return OptimizeResult(
            x=self._best_params,
            fun=self._best_loss,
            nit=1,
        )

    def get_config(self) -> dict[str, Any]:
        """Get optimizer configuration."""
        return {
            "type": "GridSearchOptimizer",
            "grid_size": len(self._param_grid),
        }

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Save grid search state (grid + results)."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state = {
            "param_grid": self._param_grid.tolist(),
            "all_losses": (
                self._all_losses.tolist() if self._all_losses is not None else None
            ),
            "best_params": (
                self._best_params.tolist() if self._best_params is not None else None
            ),
            "best_loss": self._best_loss,
        }
        state_file = checkpoint_path / OPTIMIZER_STATE_FILE
        _atomic_write(state_file, json.dumps(state, indent=2))

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "GridSearchOptimizer":
        """Load grid search state from checkpoint."""
        state_file = Path(checkpoint_dir) / OPTIMIZER_STATE_FILE
        with open(state_file) as f:
            state = json.load(f)

        opt = cls(param_grid=np.array(state["param_grid"]))
        if state["all_losses"] is not None:
            opt._all_losses = np.array(state["all_losses"])
        if state["best_params"] is not None:
            opt._best_params = np.array(state["best_params"])
        opt._best_loss = state["best_loss"]
        return opt

    def reset(self) -> None:
        """Reset optimizer state."""
        self._best_params = None
        self._best_loss = None
        self._all_losses = None

    def copy(self) -> "GridSearchOptimizer":
        """Fresh copy, rebuilt from the parameter grid (drops evaluated losses)."""
        return GridSearchOptimizer(param_grid=self._param_grid.copy())
