# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, cast

import cma
import dill
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from pymoo.algorithms.soo.nonconvex.de import DE  # type: ignore
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import (
    OPTIMIZER_STATE_FILE,
    _atomic_write,
    _load_and_validate_pydantic_model,
)
from divi.qprog.optimizers._base import Optimizer


class PymooState(BaseModel):
    """Pydantic model for Pymoo optimizer state."""

    method_value: str
    population_size: int
    algorithm_kwargs: dict[str, Any]
    # We store the pickled algorithm object as base64 encoded string
    algorithm_obj_b64: str


class PymooMethod(Enum):
    """Supported optimization methods from the pymoo library."""

    CMAES = "CMAES"
    DE = "DE"


class PymooOptimizer(Optimizer):
    """
    Optimizer wrapper for pymoo optimization algorithms and CMA-ES.

    Supports population-based optimization methods from the pymoo library (DE)
    and the cma library (CMAES).
    """

    def __init__(self, method: PymooMethod, population_size: int = 50, **kwargs):
        """
        Initialize a pymoo-based optimizer.

        Args:
            method (PymooMethod): The optimization algorithm to use (CMAES or DE).
            population_size (int, optional): Size of the population for the algorithm.
                Defaults to 50.
            **kwargs: Additional algorithm-specific parameters passed to pymoo/cma.
        """
        super().__init__()

        self.method = method
        self.population_size = population_size
        self.algorithm_kwargs = kwargs

        # Optimization state (updated during optimize(), used for checkpointing)
        self._curr_algorithm_obj: Any | None = None

    @property
    def n_param_sets(self) -> int:
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

    def get_config(self) -> dict[str, Any]:
        """Get optimizer configuration for checkpoint reconstruction.

        Returns:
            dict[str, Any]: Dictionary containing optimizer type and configuration parameters.
        """
        return {
            "type": "PymooOptimizer",
            "method": self.method.value,
            "population_size": self.population_size,
            **self.algorithm_kwargs,
        }

    def _initialize_cmaes(
        self,
        initial_params: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> Any:
        """Initialize CMA-ES strategy."""
        # Initialize CMA-ES using cma library
        # cma expects a single initial solution (mean) and initial sigma
        x0 = initial_params[0]  # Use first parameter set as mean

        # Handle sigma/sigma0
        sigma0 = self.algorithm_kwargs.get(
            "sigma0", self.algorithm_kwargs.get("sigma", 0.1)
        )

        # Filter kwargs for CMAEvolutionStrategy
        cma_kwargs = {
            k: v
            for k, v in self.algorithm_kwargs.items()
            if k not in ["sigma0", "sigma", "popsize"]
        }
        cma_kwargs["popsize"] = self.population_size
        cma_kwargs["seed"] = rng.integers(0, 2**32)
        cma_kwargs.setdefault("verbose", -9)

        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_kwargs)
        return es

    def _initialize_pymoo(
        self,
        initial_params: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> Any:
        """Initialize Pymoo strategy (DE)."""
        # Initialize DE using pymoo
        optimizer_obj = globals()[self.method.value](
            pop_size=self.population_size,
            parallelize=False,
            **self.algorithm_kwargs,
        )

        # numpy's stub types seed_seq as ISeedSequence which lacks `spawn`;
        # at runtime it's always the concrete SeedSequence.
        seed_seq = cast(np.random.SeedSequence, rng.bit_generator.seed_seq)
        seed = seed_seq.spawn(1)[0].generate_state(1)[0]
        n_var = initial_params.shape[-1]

        xl = np.zeros(n_var)
        xu = np.ones(n_var) * 2 * np.pi
        problem = Problem(n_var=n_var, n_obj=1, xl=xl, xu=xu)

        optimizer_obj.setup(
            problem,
            termination=NoTermination(),
            seed=int(seed),
            verbose=False,
        )
        optimizer_obj.start_time = time.time()

        init_pop = Population.create(
            *[Individual(X=initial_params[i]) for i in range(self.n_param_sets)]
        )
        optimizer_obj.pop = init_pop

        return optimizer_obj

    def _initialize_optimizer(
        self,
        initial_params: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> Any:
        """Initialize a fresh optimizer instance.

        Args:
            initial_params: Initial parameter values.
            rng: Random number generator.

        Returns:
            Optimizer object (cma.CMAEvolutionStrategy or pymoo.DE).
        """
        if self.method == PymooMethod.CMAES:
            return self._initialize_cmaes(initial_params, rng)
        else:
            return self._initialize_pymoo(initial_params, rng)

    def _optimize_cmaes(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        iterations_to_run: int,
        callback_fn: Callable | None,
    ) -> OptimizeResult:
        """Run CMA-ES optimization loop."""
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "_curr_algorithm_obj is not initialized; call optimize() first "
                "so _initialize_optimizer runs."
            )
        es = self._curr_algorithm_obj
        for _ in range(iterations_to_run):
            # Ask
            X = es.ask()
            evaluated_X = np.array(X)

            # Evaluate
            curr_losses = cost_fn(evaluated_X)
            # Non-finite losses become worst so a NaN individual is never kept as best.
            safe_losses = np.where(np.isfinite(curr_losses), curr_losses, np.inf)

            # Tell
            es.tell(X, safe_losses)

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_X, fun=curr_losses))

        # Return result
        return OptimizeResult(
            x=es.result.xbest,
            fun=es.result.fbest,
            nit=es.countiter,
        )

    def _optimize_pymoo(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        iterations_to_run: int,
        callback_fn: Callable | None,
    ) -> OptimizeResult:
        """Run Pymoo (DE) optimization loop."""
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "_curr_algorithm_obj is not initialized; call optimize() first "
                "so _initialize_optimizer runs."
            )
        algo = self._curr_algorithm_obj
        problem = algo.problem

        for _ in range(iterations_to_run):
            pop = algo.pop
            evaluated_X = pop.get("X")

            curr_losses = cost_fn(evaluated_X)
            # Non-finite losses become worst so a NaN individual is never kept as best.
            safe_losses = np.where(np.isfinite(curr_losses), curr_losses, np.inf)
            Evaluator().eval(StaticProblem(problem, F=safe_losses), pop)

            algo.tell(infills=pop)

            # Ask for next population to evaluate
            algo.pop = algo.ask()

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_X, fun=curr_losses))

        result = algo.result()

        # nit should represent total iterations completed (n_gen is 1-indexed)
        return OptimizeResult(
            x=result.X,
            fun=result.F,
            nit=algo.n_gen - 1,
        )

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable | None = None,
        **kwargs,
    ):
        """
        Run the optimization algorithm.

        Args:
            cost_fn (Callable): Function to minimize. Should accept a 2D array of
                parameter sets and return an array of cost values.
            initial_params (npt.NDArray[np.float64], optional): Initial parameter values as a 2D array
                of shape (n_param_sets, n_params). Should be None when resuming from a checkpoint.
            callback_fn (Callable, optional): Function called after each iteration
                with an OptimizeResult object. Defaults to None.
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
            if self.method == PymooMethod.CMAES:
                es = self._curr_algorithm_obj
                # cma uses counteigen as generation counter roughly
                # strictly speaking es.countiter is the iteration counter
                iterations_completed = es.countiter
            else:
                # Pymoo DE
                # n_gen is 1-indexed (includes initialization), so actual iterations = n_gen - 1
                iterations_completed = self._curr_algorithm_obj.n_gen - 1

            iterations_remaining = max_iterations - iterations_completed
            iterations_to_run = max(0, iterations_remaining)
        else:
            if initial_params is None:
                raise ValueError(
                    "initial_params is required for a fresh PymooOptimizer run."
                )
            rng = kwargs.pop("rng", np.random.default_rng())
            self._curr_algorithm_obj = self._initialize_optimizer(initial_params, rng)
            iterations_to_run = max_iterations

        if self.method == PymooMethod.CMAES:
            return self._optimize_cmaes(cost_fn, iterations_to_run, callback_fn)
        else:
            return self._optimize_pymoo(cost_fn, iterations_to_run, callback_fn)

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir (Path | str): Directory path where the optimizer state will be saved.

        Raises:
            RuntimeError: If optimization has not been run (no state to save).
        """
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "Cannot save checkpoint: optimization has not been run. "
                "At least one iteration must complete before saving optimizer state."
            )

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / OPTIMIZER_STATE_FILE

        # Serialize algorithm object using dill, then base64 encode
        # For CMAES (cma lib), algorithm object is picklable.
        # For DE (pymoo), algorithm object is picklable and includes pop and problem.

        algorithm_obj_bytes = dill.dumps(self._curr_algorithm_obj)
        algorithm_obj_b64 = base64.b64encode(algorithm_obj_bytes).decode("ascii")

        state = PymooState(
            method_value=self.method.value,
            population_size=self.population_size,
            algorithm_kwargs=self.algorithm_kwargs,
            algorithm_obj_b64=algorithm_obj_b64,
        )

        _atomic_write(state_file, state.model_dump_json(indent=2))

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "PymooOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new PymooOptimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir (Path | str): Directory path where the optimizer state is saved.

        Returns:
            PymooOptimizer: A new optimizer instance with restored state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(checkpoint_dir)
        state_file = checkpoint_path / OPTIMIZER_STATE_FILE

        state = _load_and_validate_pydantic_model(
            state_file,
            PymooState,
            required_fields=["method_value", "algorithm_obj_b64"],
            error_context="Pymoo optimizer",
        )

        # Create new instance with saved configuration
        optimizer = cls(
            method=PymooMethod(state.method_value),
            population_size=state.population_size,
            **state.algorithm_kwargs,
        )

        # Restore algorithm object from base64 string
        # For DE, this includes the population and problem
        optimizer._curr_algorithm_obj = dill.loads(
            base64.b64decode(state.algorithm_obj_b64)
        )

        return optimizer

    def reset(self) -> None:
        """Reset the optimizer's internal state.

        Clears the current algorithm object, allowing the optimizer
        to be reused for fresh optimization runs.
        """
        self._curr_algorithm_obj = None

    def copy(self) -> "PymooOptimizer":
        """Fresh copy, rebuilt from configuration (drops the pymoo algorithm state)."""
        return PymooOptimizer(
            method=self.method,
            population_size=self.population_size,
            **self.algorithm_kwargs,
        )
