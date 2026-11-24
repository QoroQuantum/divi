# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import time
from collections.abc import Callable
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast
from warnings import warn

import cma
import dill
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from pymoo.algorithms.soo.nonconvex.de import DE  # type: ignore  # noqa: F401
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from scipy.optimize import OptimizeResult

from divi.qprog._parameter_space import ParameterSpace
from divi.qprog.checkpointing.names import OPTIMIZER_STATE_FILE
from divi.qprog.checkpointing.read import load_and_validate_pydantic_model
from divi.qprog.checkpointing.write import atomic_write

from ._base import Optimizer

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class PymooState(BaseModel):
    """Pydantic model for Pymoo optimizer state."""

    method_value: str
    population_size: int
    algorithm_kwargs: dict[str, Any]
    # We store the pickled algorithm object as base64 encoded string
    algorithm_obj_b64: str


class PymooMethod(Enum):
    """Supported optimisation methods from the pymoo library."""

    CMAES = "CMAES"
    DE = "DE"


def _plain_python(value: Any) -> Any:
    """A numpy scalar or array as its plain-Python equivalent; others unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class PymooOptimizer(Optimizer):
    """
    Optimizer wrapper for pymoo optimisation algorithms and CMA-ES.

    Supports population-based optimisation methods from the pymoo library (DE)
    and the cma library (CMAES).
    """

    def __init__(self, method: PymooMethod, population_size: int = 50, **kwargs):
        """
        Initialise a pymoo-based optimizer.

        Args:
            method (PymooMethod): The optimisation algorithm to use (CMAES or DE).
            population_size (int, optional): Size of the population for the algorithm.
                Defaults to 50.
            **kwargs: Additional algorithm-specific parameters passed to pymoo/cma.
        """
        super().__init__()

        if isinstance(population_size, bool) or not isinstance(
            population_size, Integral
        ):
            raise TypeError("population_size must be an integer.")

        self.method = method
        self.population_size = int(population_size)
        self.algorithm_kwargs = {
            key: _plain_python(value) for key, value in kwargs.items()
        }

        # Optimisation state (updated during optimize(), used for checkpointing)
        self._curr_algorithm_obj: Any | None = None

    @property
    def n_param_sets(self) -> int:
        """
        Get the number of parameter sets (population size) used by this optimizer.

        Returns:
            int: Population size for the optimisation algorithm.
        """
        if self.method == PymooMethod.CMAES:
            # CMA-ES accepts a ``popsize`` override.
            return self.algorithm_kwargs.get("popsize", self.population_size)
        return self.population_size

    @property
    def has_resume_state(self) -> bool:
        """Whether an algorithm from an earlier run is held."""
        return self._curr_algorithm_obj is not None

    def build_evaluators(
        self, program: "VariationalQuantumAlgorithm"
    ) -> dict[str, Callable[..., Any]]:
        """Bind the program's parameter space, which sizes the DE box."""
        return {"parameter_space": lambda: program.parameter_space}

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
        """Initialise CMA-ES strategy."""
        # cma takes a single mean plus an initial sigma, not a population.
        x0 = initial_params[0]

        sigma0 = self.algorithm_kwargs.get(
            "sigma0", self.algorithm_kwargs.get("sigma", 0.1)
        )

        cma_kwargs = {
            k: v
            for k, v in self.algorithm_kwargs.items()
            if k not in ["sigma0", "sigma", "popsize"]
        }
        cma_kwargs["popsize"] = self.n_param_sets
        cma_kwargs["seed"] = rng.integers(0, 2**32)
        cma_kwargs.setdefault("verbose", -9)

        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_kwargs)
        return es

    def _initialize_pymoo(
        self,
        initial_params: npt.NDArray[np.float64],
        rng: np.random.Generator,
        space: ParameterSpace,
    ) -> Any:
        """Initialise Pymoo strategy (DE) in a box centred on the initial sets'
        midpoint."""
        optimizer_obj = globals()[self.method.value](
            pop_size=self.population_size,
            parallelize=False,
            **self.algorithm_kwargs,
        )

        # numpy's stub types seed_seq as ISeedSequence which lacks `spawn`;
        # at runtime it's always the concrete SeedSequence.
        seed_seq = cast(np.random.SeedSequence, rng.bit_generator.seed_seq)
        seed = seed_seq.spawn(1)[0].generate_state(1)[0]
        initial = initial_params[: self.n_param_sets]
        center = (initial.min(axis=0) + initial.max(axis=0)) / 2
        xl = center - space.search_widths / 2
        xu = center + space.search_widths / 2
        problem = Problem(n_var=initial_params.shape[-1], n_obj=1, xl=xl, xu=xu)

        optimizer_obj.setup(
            problem,
            termination=NoTermination(),
            seed=int(seed),
            verbose=False,
        )
        optimizer_obj.start_time = time.time()

        # Wrap known periods into the box, then clip what wrapping cannot reach.
        wrapped = space.wrap(initial, center)
        rows = np.clip(wrapped, xl, xu)
        clipped = np.flatnonzero(np.any(rows != wrapped, axis=0))
        if clipped.size:
            warn(
                f"The initial parameter sets span more than the DE search box in "
                f"parameters {clipped.tolist()}, so they were clipped to its edges.",
                UserWarning,
            )
        optimizer_obj.pop = Population.create(*[Individual(X=row) for row in rows])

        return optimizer_obj

    def _optimize_cmaes(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        iterations_to_run: int,
        callback_fn: Callable | None,
    ) -> OptimizeResult:
        """Run CMA-ES optimisation loop."""
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "_curr_algorithm_obj is not initialised; call optimize() first."
            )
        es = self._curr_algorithm_obj
        for _ in range(iterations_to_run):
            X = es.ask()
            evaluated_X = np.array(X)

            curr_losses = cost_fn(evaluated_X)
            # Non-finite losses become worst so a NaN individual is never kept as best.
            safe_losses = np.where(np.isfinite(curr_losses), curr_losses, np.inf)

            es.tell(X, safe_losses)

            if callback_fn:
                callback_fn(OptimizeResult(x=evaluated_X, fun=curr_losses))

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
        """Run Pymoo (DE) optimisation loop."""
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "_curr_algorithm_obj is not initialised; call optimize() first."
            )
        algo = self._curr_algorithm_obj

        for _ in range(iterations_to_run):
            # Evaluate the initial population first, then each generation's offspring.
            infills = algo.ask() if algo.is_initialized else algo.pop
            evaluated_X = infills.get("X")

            curr_losses = cost_fn(evaluated_X)
            # Non-finite losses become worst so a NaN individual is never kept as best.
            safe_losses = np.where(np.isfinite(curr_losses), curr_losses, np.inf)
            Evaluator().eval(StaticProblem(algo.problem, F=safe_losses), infills)

            algo.tell(infills=infills)

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
        Run the optimisation algorithm.

        Args:
            cost_fn (Callable): Function to minimise. Should accept a 2D array of
                parameter sets and return an array of cost values.
            initial_params (npt.NDArray[np.float64], optional): Initial parameter values as a 2D array
                of shape (n_param_sets, n_params). Should be None when resuming from a checkpoint.
            callback_fn (Callable, optional): Function called after each iteration
                with an OptimizeResult object. Defaults to None.
            **kwargs: Additional keyword arguments:

                - max_iterations (int, optional): Iterations to run in this call.
                  The caller subtracts whatever a previous run already spent, so a
                  resumed optimisation receives only what is left. Defaults to 5.
                - rng (np.random.Generator): Random number generator.
                - parameter_space (Callable[[], ParameterSpace], optional):
                  Returns the space whose search widths size the DE box, centred
                  on the midpoint of the initial sets. Defaults to ``2 * pi``
                  widths.

        Returns:
            OptimizeResult: Optimisation result with final parameters and cost value.
        """
        max_iterations = kwargs.pop("max_iterations", 5)
        parameter_space = kwargs.pop("parameter_space", None)

        # Resume from checkpoint or initialise fresh
        if self._curr_algorithm_obj is not None:
            iterations_to_run = max_iterations
        else:
            if initial_params is None:
                raise ValueError(
                    "initial_params is required for a fresh PymooOptimizer run."
                )
            if max_iterations < 1:
                raise ValueError(
                    "A fresh PymooOptimizer run needs max_iterations >= 1, got "
                    f"{max_iterations}."
                )
            rng = kwargs.pop("rng", np.random.default_rng())
            if self.method == PymooMethod.CMAES:
                self._curr_algorithm_obj = self._initialize_cmaes(initial_params, rng)
            else:
                space = (
                    ParameterSpace(((1.0, 1),) * initial_params.shape[-1])
                    if parameter_space is None
                    else parameter_space()
                )
                self._curr_algorithm_obj = self._initialize_pymoo(
                    initial_params, rng, space
                )
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
            RuntimeError: If optimisation has not been run (no state to save).
        """
        if self._curr_algorithm_obj is None:
            raise RuntimeError(
                "Cannot save checkpoint: optimisation has not been run. "
                "At least one iteration must complete before saving optimizer state."
            )

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / OPTIMIZER_STATE_FILE

        # dill, not pickle: the DE object carries its population and problem.
        algorithm_obj_bytes = dill.dumps(self._curr_algorithm_obj)
        algorithm_obj_b64 = base64.b64encode(algorithm_obj_bytes).decode("ascii")

        state = PymooState(
            method_value=self.method.value,
            population_size=self.population_size,
            algorithm_kwargs=self.algorithm_kwargs,
            algorithm_obj_b64=algorithm_obj_b64,
        )

        atomic_write(state_file, state.model_dump_json(indent=2))

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> Self:
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

        state = load_and_validate_pydantic_model(
            state_file,
            PymooState,
            required_fields=["method_value", "algorithm_obj_b64"],
            error_context="Pymoo optimizer",
        )

        optimizer = cls(
            method=PymooMethod(state.method_value),
            population_size=state.population_size,
            **state.algorithm_kwargs,
        )

        optimizer._curr_algorithm_obj = dill.loads(
            base64.b64decode(state.algorithm_obj_b64)
        )

        return optimizer

    def reset(self) -> None:
        """Reset the optimizer's internal state.

        Clears the current algorithm object, allowing the optimizer
        to be reused for fresh optimisation runs.
        """
        self._curr_algorithm_obj = None

    def copy(self) -> Self:
        """Fresh copy, rebuilt from configuration (drops the pymoo algorithm state)."""
        return type(self)(
            method=self.method,
            population_size=self.population_size,
            **self.algorithm_kwargs,
        )
