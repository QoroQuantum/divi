# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import (
    OPTIMIZER_STATE_FILE,
    _atomic_write,
    _load_and_validate_pydantic_model,
)
from divi.qprog.optimizers._base import Optimizer


class MonteCarloState(BaseModel):
    """Pydantic model for Monte Carlo optimizer state."""

    population_size: int
    n_best_sets: int
    keep_best_params: bool
    curr_iteration: int
    # Store arrays as lists for JSON serialization
    # Population arrays are always 2D: (population_size, n_params)
    population: list[list[float]]
    evaluated_population: list[list[float]]
    losses: list[float]
    # RNG state is a dict/tuple complex structure, simplified storage as dict or bytes
    # Stored as base64 encoded string for JSON compatibility
    rng_state_b64: str
    # Best-ever parameters across all generations
    best_params: list[float] | None = None
    best_loss: float | None = None


class MonteCarloOptimizer(Optimizer):
    """
    Monte Carlo-based parameter search optimizer.

    This optimizer samples parameter space randomly, selects the best-performing
    samples, and uses them as centers for the next generation of samples with
    decreasing variance. This implements a simple but effective evolutionary strategy.
    """

    def __init__(
        self,
        population_size: int = 10,
        n_best_sets: int = 3,
        keep_best_params: bool = False,
    ):
        """
        Initialize a Monte Carlo optimizer.

        Args:
            population_size (int, optional): Size of the population for the algorithm.
                Defaults to 10.
            n_best_sets (int, optional): Number of top-performing parameter sets to
                use as seeds for the next generation. Defaults to 3.
            keep_best_params (bool, optional): If True, includes the best parameter sets
                directly in the new population. If False, generates all new parameters
                by sampling around the best ones. Defaults to False.

        Raises:
            ValueError: If n_best_sets is greater than population_size.
            ValueError: If keep_best_params is True and n_best_sets equals population_size.
        """
        super().__init__()

        if n_best_sets > population_size:
            raise ValueError(
                "n_best_sets must be less than or equal to population_size."
            )

        if keep_best_params and n_best_sets == population_size:
            raise ValueError(
                "If keep_best_params is True, n_best_sets must be less than population_size."
            )

        self._population_size = population_size
        self._n_best_sets = n_best_sets
        self._keep_best_params = keep_best_params

        # Optimization state (updated during optimize(), used for checkpointing)
        self._curr_population: npt.NDArray[np.float64] | None = None
        self._curr_evaluated_population: npt.NDArray[np.float64] | None = None
        self._curr_losses: npt.NDArray[np.float64] | None = None
        self._curr_iteration: int | None = None
        self._curr_rng_state: dict | None = None
        self._curr_best_params: npt.NDArray[np.float64] | None = None
        self._curr_best_loss: float | None = None

    @property
    def _has_checkpoint(self) -> bool:
        return self._curr_population is not None and self._curr_iteration is not None

    @property
    def population_size(self) -> int:
        """
        Get the size of the population.

        Returns:
            int: Size of the population.
        """
        return self._population_size

    @property
    def n_param_sets(self) -> int:
        """Number of parameter sets (population size), per the Optimizer interface.

        Returns:
            int: The population size.
        """
        return self._population_size

    @property
    def n_best_sets(self) -> int:
        """
        Get the number of best parameter sets used for seeding the next generation.

        Returns:
            int: Number of best-performing sets kept.
        """
        return self._n_best_sets

    @property
    def keep_best_params(self) -> bool:
        """
        Get whether the best parameters are kept in the new population.

        Returns:
            bool: True if best parameters are included in new population, False otherwise.
        """
        return self._keep_best_params

    def get_config(self) -> dict[str, Any]:
        """Get optimizer configuration for checkpoint reconstruction.

        Returns:
            dict[str, Any]: Dictionary containing optimizer type and configuration parameters.
        """
        return {
            "type": "MonteCarloOptimizer",
            "population_size": self._population_size,
            "n_best_sets": self._n_best_sets,
            "keep_best_params": self._keep_best_params,
        }

    def _compute_new_parameters(
        self,
        params: npt.NDArray[np.float64],
        curr_iteration: int,
        best_indices: npt.NDArray[np.intp],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """
        Generates a new population of parameters based on the best-performing ones.
        """

        # 1. Select the best parameter sets from the current population
        best_params = params[best_indices]

        # 2. Determine how many new samples to generate and calculate repeat counts
        if self._keep_best_params:
            n_new_samples = self._population_size - self._n_best_sets
            # Calculate repeat counts for new samples only
            samples_per_best = n_new_samples // self._n_best_sets
            remainder = n_new_samples % self._n_best_sets
        else:
            # Calculate repeat counts for the entire population
            samples_per_best = self._population_size // self._n_best_sets
            remainder = self._population_size % self._n_best_sets

        repeat_counts = np.full(self._n_best_sets, samples_per_best)
        repeat_counts[:remainder] += 1

        # 3. Prepare the means for sampling by repeating each best parameter set
        new_means = np.repeat(best_params, repeat_counts, axis=0)

        # 4. Define the standard deviation (scale), which shrinks over iterations
        scale = 1.0 / (2.0 * (curr_iteration + 1.0))

        # 5. Generate new parameters by sampling around the best ones
        new_params = rng.normal(loc=new_means, scale=scale)

        # 6. Apply periodic boundary conditions
        new_params = new_params % (2 * np.pi)

        # 7. Conditionally combine with best params if keeping them
        if self._keep_best_params:
            return np.vstack([best_params, new_params])
        else:
            return new_params

    def _update_best_ever(
        self,
        losses: npt.NDArray[np.float64],
        population: npt.NDArray[np.float64],
    ) -> None:
        """Track the best finite loss (and its params) seen across all generations."""
        safe = np.where(np.isfinite(losses), losses, np.inf)
        idx = int(np.argmin(safe))
        if not np.isfinite(safe[idx]):
            return
        loss = float(losses[idx])
        if self._curr_best_loss is None or loss < self._curr_best_loss:
            self._curr_best_loss = loss
            self._curr_best_params = population[idx].copy()

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Perform Monte Carlo optimization on the cost function.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Optional callback function to monitor progress.
            **kwargs: Additional keyword arguments:

                - max_iterations (int, optional): Total desired number of iterations.
                  When resuming from a checkpoint, this represents the total iterations
                  desired across all runs. The optimizer will automatically calculate
                  and run only the remaining iterations needed. Defaults to 5.
                - rng (np.random.Generator, optional): Random number generator for
                  parameter sampling. Defaults to a new generator if not provided.

        Returns:
            Optimized parameters.
        """
        rng = kwargs.pop("rng", np.random.default_rng())
        max_iterations = kwargs.pop("max_iterations", 5)

        # Resume from checkpoint or initialize fresh
        if self._curr_population is not None and self._curr_iteration is not None:
            start_iter = self._curr_iteration + 1
            rng.bit_generator.state = self._curr_rng_state
            # Calculate remaining iterations to reach total desired
            iterations_completed = self._curr_iteration + 1
            iterations_remaining = max_iterations - iterations_completed
            end_iter = start_iter + max(0, iterations_remaining)
        else:
            if initial_params is None:
                raise ValueError(
                    "initial_params is required for a fresh MonteCarloOptimizer run."
                )
            self._curr_population = np.copy(initial_params)
            start_iter = 0
            end_iter = max_iterations

        # Seed best-ever from a resumed population predating best-ever tracking.
        if (
            self._curr_best_loss is None
            and self._curr_losses is not None
            and self._curr_evaluated_population is not None
        ):
            self._update_best_ever(self._curr_losses, self._curr_evaluated_population)

        for curr_iter in range(start_iter, end_iter):
            # Evaluate the entire population once
            self._curr_losses = np.atleast_1d(cost_fn(self._curr_population)).astype(
                np.float64
            )
            self._curr_evaluated_population = np.copy(self._curr_population)
            self._update_best_ever(self._curr_losses, self._curr_evaluated_population)

            # Non-finite losses sort as worst so a NaN candidate is never selected.
            safe_losses = np.where(
                np.isfinite(self._curr_losses), self._curr_losses, np.inf
            )
            # Find the indices of the best-performing parameter sets
            best_indices = np.argpartition(safe_losses, self.n_best_sets - 1)[
                : self.n_best_sets
            ]

            # Generate the next generation of parameters (uses RNG, so capture state after)
            self._curr_population = self._compute_new_parameters(
                self._curr_evaluated_population, curr_iter, best_indices, rng
            )
            self._curr_iteration = curr_iter
            self._curr_rng_state = rng.bit_generator.state

            if callback_fn:
                callback_fn(
                    OptimizeResult(
                        x=self._curr_evaluated_population, fun=self._curr_losses
                    )
                )

        # Note: 'losses' here are from the last successfully evaluated population
        # (either from the loop above, or from checkpoint state if loop didn't run)
        if self._curr_losses is None or self._curr_evaluated_population is None:
            raise RuntimeError(
                "MonteCarloOptimizer.optimize produced no evaluated population; "
                "nothing to return."
            )
        # Return the best result seen across ALL generations, not just the last
        # population, which under shot noise can regress below an earlier best.
        # nit should be the total number of iterations completed
        total_iterations_completed = (
            self._curr_iteration + 1 if self._curr_iteration is not None else 0
        )
        return OptimizeResult(
            x=self._curr_best_params,
            fun=self._curr_best_loss,
            nit=total_iterations_completed,
        )

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir (Path | str): Directory path where the optimizer state will be saved.

        Raises:
            RuntimeError: If optimization has not been run (no state to save).
        """
        if (
            self._curr_population is None
            or self._curr_evaluated_population is None
            or self._curr_losses is None
            or self._curr_iteration is None
        ):
            raise RuntimeError(
                "Cannot save checkpoint: optimization has not been run. "
                "At least one iteration must complete before saving optimizer state."
            )

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / OPTIMIZER_STATE_FILE

        # RNG state is a dict/tuple structure, pickle it for bytes storage
        # Then encode to base64 string for JSON serialization
        rng_state_bytes = pickle.dumps(self._curr_rng_state)
        rng_state_b64 = base64.b64encode(rng_state_bytes).decode("ascii")

        state = MonteCarloState(
            population_size=self._population_size,
            n_best_sets=self._n_best_sets,
            keep_best_params=self._keep_best_params,
            curr_iteration=self._curr_iteration,
            population=self._curr_population.tolist(),
            evaluated_population=self._curr_evaluated_population.tolist(),
            losses=self._curr_losses.tolist(),
            rng_state_b64=rng_state_b64,
            best_params=(
                self._curr_best_params.tolist()
                if self._curr_best_params is not None
                else None
            ),
            best_loss=self._curr_best_loss,
        )

        _atomic_write(state_file, state.model_dump_json(indent=2))

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "MonteCarloOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new MonteCarloOptimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir (Path | str): Directory path where the optimizer state is saved.

        Returns:
            MonteCarloOptimizer: A new optimizer instance with restored state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(checkpoint_dir)
        state_file = checkpoint_path / OPTIMIZER_STATE_FILE

        state = _load_and_validate_pydantic_model(
            state_file,
            MonteCarloState,
            required_fields=["population_size", "curr_iteration", "rng_state_b64"],
            error_context="Monte Carlo optimizer",
        )

        # Create new instance with saved configuration
        optimizer = cls(
            population_size=state.population_size,
            n_best_sets=state.n_best_sets,
            keep_best_params=state.keep_best_params,
        )

        # Restore state
        optimizer._curr_population = (
            np.array(state.population) if state.population else None
        )
        optimizer._curr_evaluated_population = (
            np.array(state.evaluated_population) if state.evaluated_population else None
        )
        optimizer._curr_losses = np.array(state.losses) if state.losses else None
        optimizer._curr_iteration = (
            state.curr_iteration if state.curr_iteration != -1 else None
        )
        optimizer._curr_best_params = (
            np.array(state.best_params) if state.best_params else None
        )
        optimizer._curr_best_loss = state.best_loss

        # Restore RNG state from base64 string -> bytes -> pickle
        rng_state_bytes = base64.b64decode(state.rng_state_b64)
        optimizer._curr_rng_state = pickle.loads(rng_state_bytes)

        return optimizer

    def reset(self) -> None:
        """Reset the optimizer's internal state.

        Clears all current optimization state (population, losses, iteration, RNG state),
        allowing the optimizer to be reused for fresh optimization runs.
        """
        self._curr_population = None
        self._curr_evaluated_population = None
        self._curr_losses = None
        self._curr_iteration = None
        self._curr_rng_state = None
        self._curr_best_params = None
        self._curr_best_loss = None

    def copy(self) -> "MonteCarloOptimizer":
        """Fresh copy, rebuilt from configuration (drops population/RNG state)."""
        return MonteCarloOptimizer(
            population_size=self._population_size,
            n_best_sets=self._n_best_sets,
            keep_best_params=self._keep_best_params,
        )
