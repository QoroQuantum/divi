# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum

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

from divi.extern.scipy._cobyla import _minimize_cobyla as cobyla_fn


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
            X = pop.get("X")

            curr_losses = cost_fn(X)
            static = StaticProblem(problem, F=curr_losses)
            Evaluator().eval(static, pop)

            optimizer_obj.tell(infills=pop)

            pop = optimizer_obj.ask()

            if callback_fn:
                callback_fn(OptimizeResult(x=pop.get("X"), fun=curr_losses))

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
    def n_param_sets(self):
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
        callback_fn: Callable | None = None,
        **kwargs,
    ):
        """
        Run the scipy optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 1D array of
                parameters and return a scalar cost value.
            initial_params (np.ndarray): Initial parameter values as a 1D or 2D array.
                If 2D with shape (1, n_params), it will be squeezed to 1D.
            callback_fn (Callable, optional): Function called after each iteration.
                Defaults to None.
            **kwargs: Additional keyword arguments:
                - maxiter (int): Maximum number of iterations
                - jac (Callable): Gradient function (only used for L-BFGS-B)

        Returns:
            OptimizeResult: Optimization result with final parameters and cost value.
        """
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
    """
    Monte Carlo-based parameter search optimizer.

    This optimizer samples parameter space randomly, selects the best-performing
    samples, and uses them as centers for the next generation of samples with
    decreasing variance. This implements a simple but effective evolutionary strategy.
    """

    def __init__(self, n_param_sets: int = 10, n_best_sets: int = 3):
        """
        Initialize a Monte Carlo optimizer.

        Args:
            n_param_sets (int, optional): Total number of parameter sets to evaluate
                per iteration. Defaults to 10.
            n_best_sets (int, optional): Number of top-performing parameter sets to
                use as seeds for the next generation. Defaults to 3.

        Raises:
            ValueError: If n_best_sets is greater than n_param_sets.
        """
        super().__init__()

        if n_best_sets > n_param_sets:
            raise ValueError("n_best_sets must be less than or equal to n_param_sets.")

        self._n_param_sets = n_param_sets
        self._n_best_sets = n_best_sets

        # Calculate how many times each of the best sets should be repeated
        samples_per_best = self.n_param_sets // self.n_best_sets
        remainder = self.n_param_sets % self.n_best_sets
        self._repeat_counts = np.full(self.n_best_sets, samples_per_best)
        self._repeat_counts[:remainder] += 1

    @property
    def n_param_sets(self):
        """
        Get the number of parameter sets evaluated per iteration.

        Returns:
            int: Total number of parameter sets.
        """
        return self._n_param_sets

    @property
    def n_best_sets(self):
        """
        Get the number of best parameter sets used for seeding the next generation.

        Returns:
            int: Number of best-performing sets kept.
        """
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
        # according to its assigned count
        new_means = np.repeat(best_params, self._repeat_counts, axis=0)

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
        callback_fn: Callable[[OptimizeResult], float | np.ndarray] | None = None,
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

        for curr_iter in range(max_iterations):
            # Evaluate the entire population once
            losses = cost_fn(population)

            # Find the indices of the best-performing parameter sets (only once)
            best_indices = np.argpartition(losses, self.n_best_sets - 1)[
                : self.n_best_sets
            ]

            if callback_fn:
                callback_fn(
                    OptimizeResult(x=population[best_indices], fun=losses[best_indices])
                )

            # Generate the next generation of parameters
            population = self._compute_new_parameters(
                population, curr_iter, best_indices, rng
            )

        best_idx = np.argmin(losses)
        # Return the best results from the LAST EVALUATED population
        return OptimizeResult(
            x=population[best_idx],
            fun=losses[best_idx],
            nit=max_iterations,
        )
