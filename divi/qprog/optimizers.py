# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import dill
import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES  # type: ignore
from pymoo.algorithms.soo.nonconvex.de import DE  # type: ignore
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
        cost_fn: Callable[[npt.NDArray[np.float64]], float],
        initial_params: npt.NDArray[np.float64],
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        checkpoint_dir: str | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Optimize the given cost function starting from initial parameters.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Function called after each iteration with an OptimizeResult object.
            checkpoint_dir: Directory path where optimizer state will be saved at the end
                of each iteration. If provided, state is automatically saved after each
                iteration completes. Defaults to None.
            **kwargs: Additional keyword arguments for the optimizer:

                - max_iterations (int, optional): Total desired number of iterations.
                  When resuming from a checkpoint, this represents the total iterations
                  desired across all runs. The optimizer will automatically calculate
                  and run only the remaining iterations needed.
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

    @abstractmethod
    def save_state(self, checkpoint_dir: str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state will be saved.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    @abstractmethod
    def load_state(cls, checkpoint_dir: str) -> "Optimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new optimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state is saved.

        Returns:
            Optimizer: A new optimizer instance with restored state.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def reset(self) -> None:
        """Reset the optimizer's internal state to allow fresh optimization runs.

        Clears any state accumulated during previous optimization runs, allowing
        the optimizer to be reused for new optimization problems without creating
        a new instance.
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

        # Optimization state (updated during optimize(), used for checkpointing)
        self._curr_algorithm_obj: Any | None = None
        self._curr_pop: Population | None = None

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

    def _initialize_optimizer(
        self,
        initial_params: npt.NDArray[np.float64],
        max_iterations: int,
        rng: np.random.Generator,
    ) -> tuple[Any, Problem, Population]:
        """Initialize a fresh pymoo optimizer instance.

        Args:
            initial_params: Initial parameter values.
            max_iterations: Maximum number of iterations.
            rng: Random number generator.

        Returns:
            Tuple of (optimizer_obj, problem, population).
        """
        optimizer_obj = globals()[self.method.value](
            pop_size=self.population_size,
            parallelize=False,
            **self.algorithm_kwargs,
        )

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

        init_pop = Population.create(
            *[Individual(X=initial_params[i]) for i in range(self.n_param_sets)]
        )

        return optimizer_obj, problem, init_pop

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float],
        initial_params: npt.NDArray[np.float64],
        callback_fn: Callable | None = None,
        checkpoint_dir: str | None = None,
        **kwargs,
    ):
        """
        Run the pymoo optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 2D array of
                parameter sets and return an array of cost values.
            initial_params (npt.NDArray[np.float64]): Initial parameter values as a 2D array
                of shape (n_param_sets, n_params).
            callback_fn (Callable, optional): Function called after each iteration
                with an OptimizeResult object. Defaults to None.
            checkpoint_dir (str, optional): Directory path where optimizer state will be
                saved at the end of each iteration. If provided, state is automatically
                saved after each iteration completes. Defaults to None.
            **kwargs: Additional keyword arguments:

                - max_iterations (int): Total desired number of iterations.
                  When resuming from a checkpoint, this represents the total iterations
                  desired across all runs. The optimizer will automatically calculate
                  and run only the remaining iterations needed. Defaults to 5.
                - rng (np.random.Generator): Random number generator.

        Returns:
            OptimizeResult: Optimization result with final parameters and cost value.
        """

        max_iterations = kwargs.pop("max_iterations", 5)

        # Resume from checkpoint or initialize fresh
        if self._curr_algorithm_obj is not None:
            problem = self._curr_algorithm_obj.problem
            self._curr_algorithm_obj.termination.n_max_gen = max_iterations
        else:
            rng = kwargs.pop("rng", np.random.default_rng())
            self._curr_algorithm_obj, problem, self._curr_pop = (
                self._initialize_optimizer(initial_params, max_iterations, rng)
            )

        while self._curr_algorithm_obj.has_next():
            evaluated_X = self._curr_pop.get("X")

            curr_losses = cost_fn(evaluated_X)
            Evaluator().eval(StaticProblem(problem, F=curr_losses), self._curr_pop)

            self._curr_algorithm_obj.tell(infills=self._curr_pop)

            # Ask for next population to evaluate
            self._curr_pop = self._curr_algorithm_obj.ask()

            # Save checkpoint at end of iteration if checkpoint_dir is provided
            if checkpoint_dir is not None:
                self.save_state(checkpoint_dir)

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_X, fun=curr_losses))

        result = self._curr_algorithm_obj.result()

        return OptimizeResult(
            x=result.X,
            fun=result.F,
            nit=self._curr_algorithm_obj.n_gen - 1,
        )

    def save_state(self, checkpoint_dir: str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state will be saved.
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / "optimizer_state.pkl"

        # Save configuration, algorithm object, and current population
        # The population is the next one to evaluate, avoiding need to call ask() when resuming
        state_data = {
            "method": self.method.value,
            "population_size": self.population_size,
            "algorithm_kwargs": self.algorithm_kwargs,
            "algorithm_obj": self._curr_algorithm_obj,
            "curr_pop": self._curr_pop,
        }

        with open(state_file, "wb") as f:
            dill.dump(state_data, f)

    @classmethod
    def load_state(cls, checkpoint_dir: str) -> "PymooOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new PymooOptimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state is saved.

        Returns:
            PymooOptimizer: A new optimizer instance with restored state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(checkpoint_dir)
        state_file = checkpoint_path / "optimizer_state.pkl"

        if not state_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {state_file}")

        with open(state_file, "rb") as f:
            state_data = dill.load(f)

        # Create new instance with saved configuration
        optimizer = cls(
            method=PymooMethod(state_data["method"]),
            population_size=state_data["population_size"],
            **state_data["algorithm_kwargs"],
        )

        # Restore algorithm object and population
        optimizer._curr_algorithm_obj = state_data["algorithm_obj"]
        optimizer._curr_pop = state_data.get("curr_pop", None)

        return optimizer

    def reset(self) -> None:
        """Reset the optimizer's internal state.

        Clears the current algorithm object and population, allowing the optimizer
        to be reused for fresh optimization runs.
        """
        self._curr_algorithm_obj = None
        self._curr_pop = None


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
        cost_fn: Callable[[npt.NDArray[np.float64]], float],
        initial_params: npt.NDArray[np.float64],
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        checkpoint_dir: str | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """
        Run the scipy optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 1D array of
                parameters and return a scalar cost value.
            initial_params (npt.NDArray[np.float64]): Initial parameter values as a 1D or 2D array.
                If 2D with shape (1, n_params), it will be squeezed to 1D.
            callback_fn (Callable, optional): Function called after each iteration
                with an `OptimizeResult` object. Defaults to None.
            **kwargs: Additional keyword arguments:

                - max_iterations (int, optional): Total desired number of iterations.
                  Defaults to None (no limit for some methods).
                - jac (Callable): Gradient function (only used for L-BFGS-B).

        Returns:
            OptimizeResult: Optimization result with final parameters and cost value.
        """
        max_iterations = kwargs.pop("max_iterations", None)

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

    def save_state(self, checkpoint_dir: str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Scipy optimizers do not support saving state mid-minimization as scipy.optimize
        does not provide access to the internal optimizer state.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state would be saved.

        Raises:
            NotImplementedError: Always raised, as scipy optimizers cannot save state.
        """
        raise NotImplementedError(
            "ScipyOptimizer does not support state saving. Scipy's optimization methods "
            "do not provide access to internal optimizer state during minimization. "
            "Please use MonteCarloOptimizer or PymooOptimizer for checkpointing support."
        )

    @classmethod
    def load_state(cls, checkpoint_dir: str) -> "ScipyOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Scipy optimizers do not support loading state as they cannot save state.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state would be loaded from.

        Raises:
            NotImplementedError: Always raised, as scipy optimizers cannot load state.
        """
        raise NotImplementedError(
            "ScipyOptimizer does not support state loading. Scipy's optimization methods "
            "do not provide access to internal optimizer state during minimization. "
            "Please use MonteCarloOptimizer or PymooOptimizer for checkpointing support."
        )

    def reset(self) -> None:
        """Reset the optimizer's internal state.

        ScipyOptimizer does not maintain internal state between optimization runs,
        so this method is a no-op.
        """
        pass


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

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float],
        initial_params: npt.NDArray[np.float64],
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        checkpoint_dir: str | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Perform Monte Carlo optimization on the cost function.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Optional callback function to monitor progress.
            checkpoint_dir: Directory path where optimizer state will be saved at the end
                of each iteration. If provided, state is automatically saved after each
                iteration completes. Defaults to None.
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
        if self._curr_population is not None:
            start_iter = self._curr_iteration + 1
            rng.bit_generator.state = self._curr_rng_state
            # Calculate remaining iterations to reach total desired
            iterations_completed = self._curr_iteration + 1
            iterations_remaining = max_iterations - iterations_completed
            end_iter = start_iter + max(0, iterations_remaining)
        else:
            self._curr_population = np.copy(initial_params)
            start_iter = 0
            end_iter = max_iterations

        for curr_iter in range(start_iter, end_iter):
            # Evaluate the entire population once
            self._curr_losses = cost_fn(self._curr_population)
            self._curr_evaluated_population = np.copy(self._curr_population)

            if callback_fn:
                callback_fn(
                    OptimizeResult(
                        x=self._curr_evaluated_population, fun=self._curr_losses
                    )
                )

            # Find the indices of the best-performing parameter sets
            best_indices = np.argpartition(self._curr_losses, self.n_best_sets - 1)[
                : self.n_best_sets
            ]

            # Generate the next generation of parameters
            self._curr_population = self._compute_new_parameters(
                self._curr_evaluated_population, curr_iter, best_indices, rng
            )
            self._curr_iteration = curr_iter
            self._curr_rng_state = rng.bit_generator.state

            # Save checkpoint at end of iteration if checkpoint_dir is provided
            if checkpoint_dir is not None:
                self.save_state(checkpoint_dir)

        # Note: 'losses' here are from the last successfully evaluated population
        # (either from the loop above, or from checkpoint state if loop didn't run)
        best_idx = np.argmin(self._curr_losses)

        # Return the best results from the LAST EVALUATED population
        # nit should be the total number of iterations completed
        total_iterations_completed = (
            self._curr_iteration + 1 if self._curr_iteration is not None else 0
        )
        return OptimizeResult(
            x=self._curr_evaluated_population[best_idx],
            fun=self._curr_losses[best_idx],
            nit=total_iterations_completed,
        )

    def save_state(self, checkpoint_dir: str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state will be saved.
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / "optimizer_state.npz"

        # RNG state is a dict, need to convert to a format npz can handle
        # We'll save it as a pickle within the npz
        rng_state_bytes = pickle.dumps(self._curr_rng_state)

        save_dict = {
            "population_size": self._population_size,
            "n_best_sets": self._n_best_sets,
            "keep_best_params": self._keep_best_params,
            "population": self._curr_population,
            "evaluated_population": self._curr_evaluated_population,
            "losses": self._curr_losses,
            "curr_iter": self._curr_iteration,
            "rng_state": np.frombuffer(rng_state_bytes, dtype=np.uint8),
        }

        np.savez(state_file, **save_dict)

    @classmethod
    def load_state(cls, checkpoint_dir: str) -> "MonteCarloOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new MonteCarloOptimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir (str): Directory path where the optimizer state is saved.

        Returns:
            MonteCarloOptimizer: A new optimizer instance with restored state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(checkpoint_dir)
        state_file = checkpoint_path / "optimizer_state.npz"

        if not state_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {state_file}")

        loaded = np.load(state_file, allow_pickle=True)

        # Create new instance with saved configuration
        optimizer = cls(
            population_size=int(loaded["population_size"]),
            n_best_sets=int(loaded["n_best_sets"]),
            keep_best_params=bool(loaded["keep_best_params"]),
        )

        # Restore state
        optimizer._curr_population = loaded["population"]
        optimizer._curr_evaluated_population = loaded["evaluated_population"]
        optimizer._curr_losses = loaded["losses"]
        optimizer._curr_iteration = int(loaded["curr_iter"])

        # Restore RNG state from pickle bytes
        rng_state_bytes = loaded["rng_state"].tobytes()
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


def copy_optimizer(optimizer: Optimizer) -> Optimizer:
    """Create a new optimizer instance with the same configuration as the given optimizer.

    This function creates a fresh copy of an optimizer with identical configuration
    parameters but with reset internal state. This is useful when multiple programs
    need their own optimizer instances to avoid state contamination.

    Args:
        optimizer: The optimizer to copy.

    Returns:
        A new optimizer instance with the same configuration but fresh state.

    Raises:
        ValueError: If the optimizer type is not recognized.
    """
    if isinstance(optimizer, MonteCarloOptimizer):
        return MonteCarloOptimizer(
            population_size=optimizer.population_size,
            n_best_sets=optimizer.n_best_sets,
            keep_best_params=optimizer.keep_best_params,
        )
    elif isinstance(optimizer, PymooOptimizer):
        return PymooOptimizer(
            method=optimizer.method,
            population_size=optimizer.population_size,
            **optimizer.algorithm_kwargs,
        )
    elif isinstance(optimizer, ScipyOptimizer):
        return ScipyOptimizer(method=optimizer.method)
    else:
        raise ValueError(f"Unknown optimizer type: {type(optimizer)}")
