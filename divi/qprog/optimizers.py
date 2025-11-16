# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem
from pymoo.termination import get_termination
from scipy.optimize import OptimizeResult, minimize


class Optimizer(ABC):
    @property
    @abstractmethod
    def n_param_sets(self):
        """
        Returns the number of parameter sets the optimizer can handle per optimization run.
        Returns:
            int: Number of parameter sets.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Optimize the given cost function starting from initial parameters.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            **kwargs: Additional keyword arguments for the optimizer:

                - maxiter (int, optional): Maximum number of iterations.
                  Defaults vary by optimizer (e.g., 5 for population-based optimizers,
                  None for some scipy methods).
                - rng (np.random.Generator, optional): Random number generator for
                  stochastic optimizers (PymooOptimizer, MonteCarloOptimizer).
                  Defaults to a new generator if not provided.
                - jac (Callable, optional): Gradient/Jacobian function for
                  gradient-based optimizers (only used by ScipyOptimizer with
                  L_BFGS_B method). Defaults to None.

        Returns:
            Optimized parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class PymooMethod(Enum):
    """Supported optimization methods from the pymoo library."""

    CMAES = "CMAES"
    DE = "DE"


class PymooOptimizer(Optimizer):
    """
    Optimizer wrapper for pymoo optimization algorithms.

    Supports population-based optimization methods from the pymoo library,
    including CMAES (Covariance Matrix Adaptation Evolution Strategy) and
    DE (Differential Evolution).
    """

    def __init__(self, method: PymooMethod, population_size: int = 50, **kwargs):
        """
        Initialize a pymoo-based optimizer.

        Args:
            method (PymooMethod): The optimization algorithm to use (CMAES or DE).
            population_size (int, optional): Size of the population for the algorithm.
                Defaults to 50.
            **kwargs: Additional algorithm-specific parameters passed to pymoo.
        """
        super().__init__()

        self.method = method
        self.population_size = population_size
        self.algorithm_kwargs = kwargs

    @property
    def n_param_sets(self):
        """
        Get the number of parameter sets (population size) used by this optimizer.

        Returns:
            int: Population size for the optimization algorithm.
        """
        # Determine population size from stored parameters
        if self.method.value == "DE":
            return self.population_size
        elif self.method.value == "CMAES":
            # CMAES uses 'popsize' in options dict
            return self.algorithm_kwargs.get("popsize", self.population_size)
        return self.population_size

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable | None = None,
        **kwargs,
    ):
        """
        Run the pymoo optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 2D array of
                parameter sets and return an array of cost values.
            initial_params (np.ndarray): Initial parameter values as a 2D array
                of shape (n_param_sets, n_params).
            callback_fn (Callable, optional): Function called after each iteration
                with an OptimizeResult object. Defaults to None.
            **kwargs: Additional keyword arguments:
                - maxiter (int): Maximum number of iterations
                - rng (np.random.Generator): Random number generator

        Returns:
            OptimizeResult: Optimization result with final parameters and cost value.
        """

        # Create fresh algorithm instance for this optimization run
        # since pymoo has no reset()-like functionality
        optimizer_obj = globals()[self.method.value](
            pop_size=self.population_size, parallelize=False, **self.algorithm_kwargs
        )

        max_iterations = kwargs.pop("maxiter", 5)
        rng = kwargs.pop("rng", np.random.default_rng())
        seed = rng.bit_generator.seed_seq.spawn(1)[0].generate_state(1)[0]

        n_var = initial_params.shape[-1]

        xl = np.zeros(n_var)
        xu = np.ones(n_var) * 2 * np.pi

        problem = Problem(n_var=n_var, n_obj=1, xl=xl, xu=xu)

        optimizer_obj.setup(
            problem,
            termination=get_termination("n_gen", max_iterations),
            seed=int(seed),
            verbose=False,
        )
        optimizer_obj.start_time = time.time()

        pop = Population.create(
            *[Individual(X=initial_params[i]) for i in range(self.n_param_sets)]
        )

        while optimizer_obj.has_next():
            evaluated_X = pop.get("X")

            curr_losses = cost_fn(evaluated_X)
            static = StaticProblem(problem, F=curr_losses)
            Evaluator().eval(static, pop)

            optimizer_obj.tell(infills=pop)

            pop = optimizer_obj.ask()

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_X, fun=curr_losses))

        result = optimizer_obj.result()

        return OptimizeResult(
            x=result.X,
            fun=result.F,
            nit=optimizer_obj.n_gen - 1,
        )


class ScipyMethod(Enum):
    """Supported optimization methods from scipy.optimize."""

    NELDER_MEAD = "Nelder-Mead"
    COBYLA = "COBYLA"
    L_BFGS_B = "L-BFGS-B"


class ScipyOptimizer(Optimizer):
    """
    Optimizer wrapper for scipy.optimize methods.

    Supports gradient-free and gradient-based optimization algorithms from scipy,
    including Nelder-Mead simplex, COBYLA, and L-BFGS-B.
    """

    def __init__(self, method: ScipyMethod):
        """
        Initialize a scipy-based optimizer.

        Args:
            method (ScipyMethod): The optimization algorithm to use.
        """
        super().__init__()

        self.method = method

    @property
    def n_param_sets(self) -> int:
        """
        Get the number of parameter sets used by this optimizer.

        Returns:
            int: Always returns 1, as scipy optimizers use single-point optimization.
        """
        return 1

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """
        Run the scipy optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 1D array of
                parameters and return a scalar cost value.
            initial_params (np.ndarray): Initial parameter values as a 1D or 2D array.
                If 2D with shape (1, n_params), it will be squeezed to 1D.
            callback_fn (Callable, optional): Function called after each iteration
                with an `OptimizeResult` object. Defaults to None.
            **kwargs: Additional keyword arguments:
                - maxiter (int): Maximum number of iterations
                - jac (Callable): Gradient function (only used for L-BFGS-B)

        Returns:
            OptimizeResult: Optimization result with final parameters and cost value.
        """
        max_iterations = kwargs.pop("maxiter", None)

        # If a callback is provided, we wrap the cost function and callback
        # to ensure the data passed to the callback has a consistent shape.
        if callback_fn:

            def callback_wrapper(intermediate_result: OptimizeResult):
                # Create a dictionary from the intermediate result to preserve all of its keys.
                result_dict = dict(intermediate_result)

                # Overwrite 'x' and 'fun' to ensure they have consistent dimensions.
                result_dict["x"] = np.atleast_2d(intermediate_result.x)
                result_dict["fun"] = np.atleast_1d(intermediate_result.fun)

                # Create a new OptimizeResult and pass it to the user's callback.
                return callback_fn(OptimizeResult(**result_dict))

        else:
            callback_wrapper = None

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
            method=self.method.value,
            jac=(
                kwargs.pop("jac", None) if self.method == ScipyMethod.L_BFGS_B else None
            ),
            callback=callback_wrapper,
            options={"maxiter": maxiter},
        )


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

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Perform Monte Carlo optimization on the cost function.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Optional callback function to monitor progress.
            **kwargs: Additional keyword arguments:

                - maxiter (int, optional): Maximum number of iterations. Defaults to 5.
                - rng (np.random.Generator, optional): Random number generator for
                  parameter sampling. Defaults to a new generator if not provided.

        Returns:
            Optimized parameters.
        """
        rng = kwargs.pop("rng", np.random.default_rng())
        max_iterations = kwargs.pop("maxiter", 5)

        population = np.copy(initial_params)
        evaluated_population = population

        for curr_iter in range(max_iterations):
            # Evaluate the entire population once
            losses = cost_fn(population)
            evaluated_population = population

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_population, fun=losses))

            # Find the indices of the best-performing parameter sets
            best_indices = np.argpartition(losses, self.n_best_sets - 1)[
                : self.n_best_sets
            ]

            # Generate the next generation of parameters
            population = self._compute_new_parameters(
                evaluated_population, curr_iter, best_indices, rng
            )

        # Note: 'losses' here are from the last successfully evaluated population
        best_idx = np.argmin(losses)

        # Return the best results from the LAST EVALUATED population
        return OptimizeResult(
            x=evaluated_population[best_idx],
            fun=losses[best_idx],
            nit=max_iterations,
        )
