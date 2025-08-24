# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from divi.exp.scipy._cobyla import _minimize_cobyla as cobyla_fn


class ScipyMethod(Enum):
    NELDER_MEAD = "Nelder-Mead"
    COBYLA = "COBYLA"
    L_BFGS_B = "L-BFGS-B"


class Optimizer(ABC):
    @property
    @abstractmethod
    def n_param_sets(self):
        """ """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """
        Optimize the given cost function starting from initial parameters.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            Optimized parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class ScipyOptimizer(Optimizer):
    def __init__(self, method: ScipyMethod):
        self.method = method

    @property
    def n_param_sets(self):
        return 1

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable | None = None,
        **kwargs,
    ):
        max_iterations = kwargs.pop("maxiter", None)

        if max_iterations is None or self.method == ScipyMethod.COBYLA:
            # COBYLA perceive maxiter as maxfev so we need
            # to use the callback fn for counting instead.
            maxiter = None
        else:
            # Need to add one more iteration for Nelder-Mead's simplex initialization step
            maxiter = (
                max_iterations + 1
                if self.method == ScipyMethod.NELDER_MEAD
                else max_iterations
            )

        return minimize(
            cost_fn,
            initial_params.squeeze(),
            method=(
                cobyla_fn if self.method == ScipyMethod.COBYLA else self.method.value
            ),
            jac=(
                kwargs.pop("jac", None) if self.method == ScipyMethod.L_BFGS_B else None
            ),
            callback=callback_fn,
            options={"maxiter": maxiter},
        )


class MonteCarloOptimizer(Optimizer):
    def __init__(self, n_param_sets: int = 10, n_best_sets: int = 10):
        super().__init__()

        self._n_param_sets = n_param_sets
        self._n_best_sets = n_best_sets

    @property
    def n_param_sets(self):
        return self._n_param_sets

    @property
    def n_best_sets(self):
        return self._n_best_sets

    def _compute_new_parameters(
        self,
        params: np.ndarray,
        curr_iteration: int,
        best_indices: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates a new population of parameters based on the best-performing ones.

        """
        # 1. Select the best parameter sets from the current population
        best_params = params[best_indices]

        # 2. Prepare the means for sampling by repeating each best parameter set
        # This creates the 'loc' argument for rng.normal
        # Shape: (n_best_sets * n_param_sets, param_dims...)
        new_means = np.repeat(best_params, self.n_param_sets, axis=0)

        # 3. Define the standard deviation (scale), which shrinks over iterations
        scale = 1.0 / (2.0 * (curr_iteration + 1.0))

        # 4. Generate all new parameters in a single vectorized call
        new_params = rng.normal(loc=new_means, scale=scale)

        # Apply periodic boundary conditions
        return new_params % (2 * np.pi)

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """
        Perform Monte Carlo optimization on the cost function.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Optional callback function to monitor progress.
            **kwargs: Additional keyword arguments for the optimizer.
        Returns:
            Optimized parameters.
        """
        rng = kwargs.pop("rng", np.random.default_rng())
        max_iterations = kwargs.pop("maxiter", 5)

        population = np.copy(initial_params)

        final_params = None
        final_losses = None

        for curr_iter in range(max_iterations):
            # Evaluate the entire population once
            losses = cost_fn(population)

            # Find the indices of the best-performing parameter sets (only once)
            best_indices = np.argpartition(losses, self.n_best_sets - 1)[
                : self.n_best_sets
            ]

            # Store the current best results
            final_params = population[best_indices]
            final_losses = losses[best_indices]

            if callback_fn:
                callback_fn(OptimizeResult(x=final_params, fun=final_losses))

            # Generate the next generation of parameters
            population = self._compute_new_parameters(
                population, curr_iter, best_indices, rng
            )

        # Return the best results from the LAST EVALUATED population
        return OptimizeResult(
            x=final_params,
            fun=final_losses,
            nit=max_iterations,
        )
