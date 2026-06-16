# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import json
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
from scipy.linalg import solve as _solve_linear_system
from scipy.optimize import OptimizeResult, minimize

from divi.qprog._metrics import (
    FubiniStudyMetricEstimator,
    MetricEstimator,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
)
from divi.qprog.checkpointing import (
    OPTIMIZER_STATE_FILE,
    _atomic_write,
    _load_and_validate_pydantic_model,
)

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

__all__ = [
    "FubiniStudyMetricEstimator",
    "GridSearchOptimizer",
    "MetricEstimator",
    "MonteCarloOptimizer",
    "MonteCarloState",
    "Optimizer",
    "PullbackMetricEstimator",
    "PymooMethod",
    "PymooOptimizer",
    "PymooState",
    "QNGOptimizer",
    "QNSPSAOptimizer",
    "SPSAOptimizer",
    "ScipyMethod",
    "ScipyOptimizer",
    "StochasticFidelityMetricEstimator",
]


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


class PymooState(BaseModel):
    """Pydantic model for Pymoo optimizer state."""

    method_value: str
    population_size: int
    algorithm_kwargs: dict[str, Any]
    # We store the pickled algorithm object as base64 encoded string
    algorithm_obj_b64: str


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.

    .. warning::
        **Thread Safety**: Optimizer instances are **not thread-safe**. They maintain
        internal state (e.g., current population, iteration count, RNG state) that changes
        during optimization.

        Do **not** share a single `Optimizer` instance across multiple `QuantumProgram`
        instances or threads running in parallel. Doing so will lead to race conditions,
        corrupted state, and potential crashes.

        If you need to use the same optimizer configuration for multiple programs,
        create a separate instance for each program. You can call
        :meth:`copy` to create a fresh copy with the same configuration.
    """

    @property
    @abstractmethod
    def n_param_sets(self) -> int:
        """
        Returns the number of parameter sets the optimizer can handle per optimization run.
        Returns:
            int: Number of parameter sets.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Optimize the given cost function starting from initial parameters.

        Parameters:
            cost_fn: The cost function to minimize.
            initial_params: Initial parameters for the optimization.
            callback_fn: Function called after each iteration with an OptimizeResult object.
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
    def get_config(self) -> dict[str, Any]:
        """Get optimizer configuration for checkpoint reconstruction.

        Returns:
            dict[str, Any]: Dictionary containing optimizer type and configuration parameters.

        Raises:
            NotImplementedError: If the optimizer does not support checkpointing.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Args:
            checkpoint_dir: Directory path where the optimizer state will be saved.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    @abstractmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "Optimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Creates a new optimizer instance with the state restored from the checkpoint.

        Args:
            checkpoint_dir: Directory path where the optimizer state is saved.

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

    @property
    def supports_checkpointing(self) -> bool:
        """Whether this optimizer can persist and restore its state mid-run.

        Programs guard on this before checkpointing so an optimizer that cannot
        save state fails upfront rather than mid-optimization. Optimizers whose
        :meth:`save_state` raises :class:`NotImplementedError` override this to
        return ``False``.
        """
        return True

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        """Check that this optimizer can be applied to ``program``.

        Called at the start of
        :meth:`~divi.qprog.VariationalQuantumAlgorithm.run`, before
        any optimization, so an incompatible optimizer/program pairing fails
        loudly and early. The base implementation accepts any program; override
        to raise when the optimizer's requirements are not met.
        """

    def build_evaluators(
        self, program: "VariationalQuantumAlgorithm"
    ) -> dict[str, Callable[[npt.NDArray[np.float64]], Any]]:
        """Extra per-run evaluators this optimizer needs from the program.

        Called once by the variational algorithm before optimization. The
        returned mapping may override ``"jac"`` and/or add ``"metric_fn"``;
        keys absent from the mapping fall back to the algorithm's defaults.
        The base implementation needs nothing extra and returns ``{}``.
        """
        return {}

    def copy(self) -> "Optimizer":
        """Return a fresh instance with the same configuration and no accumulated
        run state.

        The default deep-copies the optimizer, which is correct for the stateless
        optimizers whose only attributes are configuration; optimizers that
        accumulate per-run state override this to rebuild from configuration alone.

        .. tip::
            Use this when preparing a batch of programs that will run in parallel.
            Optimizer instances are not thread-safe (see the class warning); give
            each program its own ``optimizer.copy()`` to avoid state contamination.
        """
        return copy.deepcopy(self)


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

            # Tell
            es.tell(X, curr_losses)

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
            Evaluator().eval(StaticProblem(problem, F=curr_losses), pop)

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
    def supports_checkpointing(self) -> bool:
        """``False`` — scipy.optimize exposes no mid-minimization state to save."""
        return False

    @property
    def n_param_sets(self) -> int:
        """
        Get the number of parameter sets used by this optimizer.

        Returns:
            int: Always returns 1, as scipy optimizers use single-point optimization.
        """
        return 1

    # pyrefly: ignore[bad-override]
    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
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

        if initial_params is None:
            raise ValueError("ScipyOptimizer requires initial_params.")

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

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Save the optimizer's internal state to a checkpoint directory.

        Scipy optimizers do not support saving state mid-minimization as scipy.optimize
        does not provide access to the internal optimizer state.

        Args:
            checkpoint_dir: Directory path where the optimizer state would be saved.

        Raises:
            NotImplementedError: Always raised, as scipy optimizers cannot save state.
        """
        raise NotImplementedError(
            "ScipyOptimizer does not support state saving. Scipy's optimization methods "
            "do not provide access to internal optimizer state during minimization. "
            "Please use MonteCarloOptimizer or PymooOptimizer for checkpointing support."
        )

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "ScipyOptimizer":
        """Load the optimizer's internal state from a checkpoint directory.

        Scipy optimizers do not support loading state as they cannot save state.

        Args:
            checkpoint_dir: Directory path where the optimizer state would be loaded from.

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

    def get_config(self) -> dict[str, Any]:
        """Get optimizer configuration for checkpoint reconstruction.

        Raises:
            NotImplementedError: ScipyOptimizer does not support checkpointing.
        """
        raise NotImplementedError(
            "ScipyOptimizer does not support checkpointing. Please use "
            "MonteCarloOptimizer or PymooOptimizer for checkpointing support."
        )


def _spsa_gain_a(k: int, a: float, A: float, alpha: float) -> float:
    """SPSA learning-rate gain ``a_k = a / (A + k + 1)**alpha``."""
    return a / (A + k + 1.0) ** alpha


def _spsa_gain_c(k: int, c: float, gamma: float) -> float:
    """SPSA perturbation gain ``c_k = c / (k + 1)**gamma``."""
    return c / (k + 1.0) ** gamma


def _spsa_gradient(
    cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
    theta: npt.NDArray[np.float64],
    c_k: float,
    rng: np.random.Generator,
    *,
    direction: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """One simultaneous-perturbation gradient sample at ``theta``.

    Draws a Bernoulli ±1 direction ``h`` (unless ``direction`` is supplied) and
    evaluates ``theta ± c_k·h`` as a single two-row batch, so a stochastic cost
    (e.g. QDrift) scores both perturbations against the same sampled Hamiltonian.
    Returns ``(ghat, h, f_plus, f_minus)``; the caller can reuse the perturbed
    values as a loss proxy without a third evaluation.
    """
    h = (
        direction
        if direction is not None
        else rng.choice([-1.0, 1.0], size=theta.shape[0])
    )
    batch = np.vstack([theta + c_k * h, theta - c_k * h])
    values = np.asarray(cost_fn(batch), dtype=np.float64).reshape(-1)
    f_plus, f_minus = float(values[0]), float(values[1])
    ghat = (f_plus - f_minus) / (2.0 * c_k) * h
    return ghat, h, f_plus, f_minus


def _fidelity_metric_sample(
    fidelity_fn: Callable,
    theta: npt.NDArray[np.float64],
    h1: npt.NDArray[np.float64],
    c_k: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """One stochastic Fubini–Study sample from four state-fidelity overlaps.

    Draws a second Bernoulli ±1 direction ``h2`` and forms the mixed second
    difference of the fidelity ``F(theta, theta + perturbation)`` along ``h1``
    and ``h2``, giving ``-(δF / 8 c_k²)(h1 h2ᵀ + h2 h1ᵀ)``.
    """
    h2 = rng.choice([-1.0, 1.0], size=theta.shape[0])
    fidelities = fidelity_fn(
        theta,
        [
            c_k * h1 + c_k * h2,
            c_k * h1,
            -c_k * h1 + c_k * h2,
            -c_k * h1,
        ],
    )
    delta_f = fidelities[0] - fidelities[1] - fidelities[2] + fidelities[3]
    return -(delta_f / (8.0 * c_k * c_k)) * (np.outer(h1, h2) + np.outer(h2, h1))


def _regularized_solve(
    grad: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    solver: Literal["tikhonov", "pinv"],
    regularization: float,
    scale_regularization: bool,
    rcond: float,
) -> npt.NDArray[np.float64]:
    """Solve ``metric @ delta = grad`` with PSD-aware regularization.

    Symmetrizes ``metric`` defensively against round-off. ``"tikhonov"`` solves
    ``(G + λI) delta = grad`` via a Cholesky-based symmetric solve (``λ`` scaled
    by ``max(1, mean(diag(G)))`` when ``scale_regularization``); ``"pinv"`` applies
    the Moore-Penrose pseudo-inverse with cutoff ``rcond``.
    """
    grad = np.asarray(grad, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    metric = 0.5 * (metric + metric.T)

    if solver == "pinv":
        return np.linalg.pinv(metric, rcond=rcond) @ grad

    lam = regularization
    if scale_regularization:
        lam *= max(1.0, float(np.mean(np.diag(metric))))
    damped = metric + lam * np.eye(metric.shape[0])
    try:
        return _solve_linear_system(damped, grad, assume_a="pos")
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Regularized natural-gradient solve failed: the damped metric "
            "(G + λI) is not positive-definite — the metric is rank-deficient "
            "and regularization is too small to lift it (λ=0 leaves it "
            "singular). Raise `regularization` or use solver='pinv'."
        ) from exc


def _matrix_abs_psd(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Matrix absolute value ``V |Λ| Vᵀ`` of a symmetric matrix.

    Eigendecomposes the symmetrized matrix and takes the absolute value of its
    eigenvalues, yielding a real positive-semidefinite result — QN-SPSA's
    conditioning of a noisy, possibly-indefinite metric estimate before the
    identity shift. Decomposing the matrix directly (rather than ``GᵀG``) avoids
    squaring the condition number.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    return (eigvecs * np.abs(eigvals)) @ eigvecs.T


class QNGOptimizer(Optimizer):
    """Quantum Natural Gradient optimizer.

    Performs regularized natural-gradient descent

    .. math::
        \\theta \\leftarrow \\theta - \\eta \\, (G + \\lambda I)^{-1} \\nabla L,

    where :math:`\\nabla L` is the parameter-shift gradient and :math:`G` is a
    positive-semidefinite metric tensor. The optimizer is **metric-agnostic**:
    the metric is produced by an injected :class:`MetricEstimator` strategy
    (default :class:`PullbackMetricEstimator`), bound to the program's
    capabilities via :meth:`build_evaluators`. Swapping the estimator changes
    the metric without changing the optimizer.

    Because the default pullback metric is only PSD — and singular whenever the
    number of parameters exceeds the number of Hamiltonian terms — Tikhonov
    damping is applied before solving the linear system.

    This is a single-point optimizer (``n_param_sets == 1``). The variational
    algorithm wires the estimator's gradient and metric in via
    :meth:`build_evaluators`; calling ``optimize`` directly without ``jac`` and
    ``metric_fn`` raises ``ValueError``.

    Args:
        step_size: Learning rate :math:`\\eta` for the parameter update.
        regularization: Tikhonov damping :math:`\\lambda` added to the metric
            diagonal before solving. Must be positive when ``solver`` is
            ``"tikhonov"`` so the damped system is positive-definite.
        scale_regularization: When ``True``, scale :math:`\\lambda` by
            ``max(1, mean(diag(G)))`` so the damping tracks the metric's
            magnitude instead of being fixed in absolute terms.
        solver: ``"tikhonov"`` solves ``(G + lambda I) delta = grad`` via a
            Cholesky-based symmetric solve (exploits PSD structure).
            ``"pinv"`` instead applies the Moore-Penrose pseudo-inverse of the
            raw (undamped) metric with cutoff ``rcond``.
        rcond: Singular-value cutoff for the ``"pinv"`` solver.
        max_step_norm: If set, clip the L2 norm of the per-iteration parameter
            update ``step_size * (G + lambda I)^-1 grad`` to this value. ``None``
            (default) applies no clip.
        metric_estimator: The strategy that builds the metric ``G`` from the
            program's capabilities. Defaults to
            :class:`~divi.qprog._metrics.PullbackMetricEstimator` (the
            Hamiltonian-aware pullback metric). Inject a different estimator to
            change the metric without touching the optimizer.

    .. note::
        Because the pullback metric is only PSD (and singular whenever the
        parameter count exceeds the number of Hamiltonian terms), the
        preconditioned step can grow large along weakly-curved directions when
        ``regularization`` is small relative to the metric scale. If the
        optimizer oscillates (visible in the loss history), raise
        ``regularization``, lower ``step_size``, or set ``max_step_norm``. A
        non-finite update raises ``FloatingPointError``.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        regularization: float = 1e-3,
        scale_regularization: bool = True,
        solver: Literal["tikhonov", "pinv"] = "tikhonov",
        rcond: float = 1e-6,
        max_step_norm: float | None = None,
        metric_estimator: MetricEstimator | None = None,
    ):
        super().__init__()

        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if regularization < 0:
            raise ValueError(
                f"regularization must be non-negative, got {regularization}."
            )
        if solver not in ("tikhonov", "pinv"):
            raise ValueError(f"solver must be 'tikhonov' or 'pinv', got {solver!r}.")
        if max_step_norm is not None and max_step_norm <= 0:
            raise ValueError(
                f"max_step_norm must be positive or None, got {max_step_norm}."
            )

        self.step_size = step_size
        self.regularization = regularization
        self.scale_regularization = scale_regularization
        self.solver: Literal["tikhonov", "pinv"] = solver
        self.rcond = rcond
        self.max_step_norm = max_step_norm
        self.metric_estimator = metric_estimator or PullbackMetricEstimator()

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        """Reject a program whose loss the chosen metric estimator cannot model."""
        self.metric_estimator.check_compatible(program)

    def build_evaluators(
        self, program: "VariationalQuantumAlgorithm"
    ) -> dict[str, Callable[[npt.NDArray[np.float64]], Any]]:
        """Bind the metric estimator to the program's metric pipeline.

        The pullback estimator returns a fused ``jac`` + ``metric_fn`` (one
        memoized measurement serves both); the Fubini–Study estimator returns
        only ``metric_fn`` and lets the gradient fall back to the program's
        parameter-shift rule.
        """
        return self.metric_estimator.bind(program)

    @property
    def supports_checkpointing(self) -> bool:
        """``False`` — QNG's only state is the parameter vector, already persisted
        by the variational algorithm's program state."""
        return False

    @property
    def n_param_sets(self) -> int:
        """Number of parameter sets per step — always ``1`` (single-point)."""
        return 1

    def _natural_gradient(
        self,
        grad: npt.NDArray[np.float64],
        metric: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Precondition ``grad`` with the (regularized) inverse metric."""
        delta = _regularized_solve(
            grad,
            metric,
            solver=self.solver,
            regularization=self.regularization,
            scale_regularization=self.scale_regularization,
            rcond=self.rcond,
        )

        update = self.step_size * delta
        if not np.all(np.isfinite(update)):
            raise FloatingPointError(
                "QNGOptimizer produced a non-finite parameter update; the "
                "pullback metric is severely ill-conditioned. Raise "
                "`regularization` or lower `step_size`."
            )

        if self.max_step_norm is not None:
            update_norm = float(np.linalg.norm(update))
            if update_norm > self.max_step_norm:
                delta = delta * (self.max_step_norm / update_norm)
        return delta

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run natural-gradient descent for ``max_iterations`` steps.

        Args:
            cost_fn: Scalar cost function; called once per iteration at the
                current parameters to record the loss.
            initial_params: Starting parameters (1D, or 2D with a single row).
            callback_fn: Called after each step with an ``OptimizeResult`` whose
                ``x`` is 2D and ``fun`` is 1D, matching the variational-algorithm
                callback contract. May raise ``StopIteration`` for early stopping.
            **kwargs: ``max_iterations`` (default 50), ``jac`` (required gradient
                function), ``metric_fn`` (required metric function). ``rng`` is
                accepted and ignored — QNG is deterministic.

        Returns:
            OptimizeResult: Best evaluated iterate over the run.
        """
        max_iterations = kwargs.pop("max_iterations", None) or 50
        jac = kwargs.pop("jac", None)
        metric_fn = kwargs.pop("metric_fn", None)
        kwargs.pop("rng", None)  # QNG is deterministic; ignore any provided RNG.

        if jac is None or metric_fn is None:
            raise ValueError(
                "QNGOptimizer requires both a gradient function (`jac`) and a "
                "metric function (`metric_fn`). It is driven by "
                "VariationalQuantumAlgorithm.run(), which supplies both."
            )
        if initial_params is None:
            raise ValueError("QNGOptimizer requires initial_params.")

        theta = np.atleast_1d(np.asarray(initial_params, dtype=np.float64).squeeze())

        best_x = theta.copy()
        best_fun = np.inf

        for it in range(max_iterations):
            fun = float(np.asarray(cost_fn(theta)).reshape(-1)[0])
            grad = jac(theta)
            metric = metric_fn(theta)

            if fun < best_fun:
                best_fun = fun
                best_x = theta.copy()

            if callback_fn is not None:
                # StopIteration (early stopping) is intentionally NOT caught —
                # VariationalQuantumAlgorithm.run() handles it.
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(theta.copy()),
                        fun=np.atleast_1d(fun),
                        nit=it + 1,
                        success=True,
                        message="Optimization in progress.",
                    )
                )

            delta = self._natural_gradient(grad, metric)
            theta = theta - self.step_size * delta

        return OptimizeResult(
            x=np.atleast_2d(best_x),
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
        )

    def get_config(self) -> dict[str, Any]:
        """QNGOptimizer does not support checkpointing.

        Raises:
            NotImplementedError: Always. The only optimizer state is the current
                parameter vector, which the variational algorithm already
                checkpoints via its parameter history.
        """
        raise NotImplementedError(
            "QNGOptimizer does not support checkpointing. Its only state is the "
            "current parameter vector, which is already persisted by the "
            "variational algorithm's program state."
        )

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """QNGOptimizer does not support state saving.

        Raises:
            NotImplementedError: Always; see :meth:`get_config`.
        """
        raise NotImplementedError(
            "QNGOptimizer does not support state saving. Its only state is the "
            "current parameter vector, which is already persisted by the "
            "variational algorithm's program state."
        )

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> "QNGOptimizer":
        """QNGOptimizer does not support state loading.

        Raises:
            NotImplementedError: Always; see :meth:`get_config`.
        """
        raise NotImplementedError(
            "QNGOptimizer does not support state loading. Its only state is the "
            "current parameter vector, which is already persisted by the "
            "variational algorithm's program state."
        )

    def reset(self) -> None:
        """No-op: QNGOptimizer maintains no internal state between runs."""


class _SPSAConfigMixin:
    """Shared SPSA gain-schedule config + validation for SPSA and QN-SPSA.

    Holds Spall's gain-sequence hyperparameters and an optional look-ahead
    blocking guard. Neither optimizer keeps mid-run state on the instance — the
    per-run iterate, blocking history, and (for QN-SPSA) the running-average
    metric live as locals inside ``optimize``.
    """

    def __init__(
        self,
        learning_rate: float,
        c: float,
        alpha: float,
        gamma: float,
        A: float | None,
        resamplings: int,
        blocking: bool,
        blocking_history: int,
        blocking_tol: float,
        exact_loss: bool,
    ):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}.")
        if resamplings < 1:
            raise ValueError(f"resamplings must be >= 1, got {resamplings}.")
        if blocking_history < 1:
            raise ValueError(f"blocking_history must be >= 1, got {blocking_history}.")

        self.learning_rate = learning_rate
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.resamplings = resamplings
        self.blocking = blocking
        self.blocking_history = blocking_history
        self.blocking_tol = blocking_tol
        self.exact_loss = exact_loss

    @property
    def n_param_sets(self) -> int:
        """Number of parameter sets per step — always ``1`` (single-point)."""
        return 1

    def _step_loss(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
        proxy: float,
    ) -> float:
        """Loss recorded for the callback and best-iterate tracking.

        By default the perturbation-average ``proxy`` (no extra evaluation, but
        biased by ``O(c_k²)``). When ``exact_loss`` is set, one additional
        unperturbed evaluation ``f(theta)`` is spent for an unbiased value.
        """
        if self.exact_loss:
            return float(np.asarray(cost_fn(theta)).reshape(-1)[0])
        return proxy

    def _block_or_step(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
        proposed: npt.NDArray[np.float64],
        current_loss: float,
        recent: "deque[float]",
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Look-ahead blocking (Spall/Gacon): move to ``proposed`` only if its loss
        does not exceed ``current_loss`` by more than ``blocking_tol``·std(``recent``);
        otherwise hold ``theta``. Returns ``(next_theta, loss_at_next_theta)``.

        Costs one extra cost evaluation per step (the candidate's loss); the
        accepted value carries over as the next ``current_loss``, so it is not
        re-measured. The std band needs at least two prior losses; before that the
        candidate is accepted (matching the start-up behavior of Spall's rule).

        A non-finite candidate loss is treated as a rejection (hold ``theta``)
        rather than accepted — ``nan > x`` is ``False``, so without this guard a
        NaN candidate would slip through and poison the ``recent`` window,
        permanently disabling blocking. Holding keeps the run bounded and the next
        gradient is taken at the finite held point.
        """
        f_proposed = float(np.asarray(cost_fn(proposed)).reshape(-1)[0])
        band = self.blocking_tol * float(np.std(recent)) if len(recent) >= 2 else np.inf
        if not np.isfinite(f_proposed) or f_proposed > current_loss + band:
            return theta, current_loss
        return proposed, f_proposed

    def _warn_if_diverging(
        self, fun: float, reference: float | None, already_warned: bool
    ) -> bool:
        """Emit a one-time warning if the loss has blown up relative to its start.

        Without ``blocking`` a divergent run (e.g. a noisy QN-SPSA metric driving
        huge steps) is otherwise silent: best-iterate tracking returns an early,
        finite iterate while the trajectory has exploded. Returns the updated
        "already warned" flag — call as
        ``warned = self._warn_if_diverging(fun, reference, warned)``.
        """
        if already_warned or reference is None:
            return already_warned
        if not np.isfinite(fun) or abs(fun) > 1e3 * max(abs(reference), 1.0):
            warnings.warn(
                f"{type(self).__name__} appears to be diverging (loss "
                f"{reference:.3e} -> {fun:.3e}); best_loss/best_params may reflect "
                "an early iterate. Enable blocking, raise regularization, or lower "
                "learning_rate.",
                stacklevel=3,
            )
            return True
        return already_warned

    @property
    def supports_checkpointing(self) -> bool:
        """``False`` — like QNG, the only persistent state is the parameter vector,
        already persisted by the variational algorithm's program state."""
        return False

    def get_config(self) -> dict[str, Any]:
        """Not supported; see :attr:`supports_checkpointing`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support checkpointing. Its only "
            "persistent state is the current parameter vector, already persisted "
            "by the variational algorithm's program state."
        )

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Not supported; see :meth:`get_config`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support state saving. Its only "
            "persistent state is the current parameter vector, already persisted "
            "by the variational algorithm's program state."
        )

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str):
        """Not supported; see :meth:`get_config`."""
        raise NotImplementedError(
            f"{cls.__name__} does not support state loading. Its only persistent "
            "state is the current parameter vector, already persisted by the "
            "variational algorithm's program state."
        )

    def reset(self) -> None:
        """No-op: no internal state is kept between runs."""


class SPSAOptimizer(_SPSAConfigMixin, Optimizer):
    r"""Simultaneous Perturbation Stochastic Approximation (Spall).

    Estimates the gradient from just **two** cost evaluations per step,
    independent of the parameter count, by perturbing all parameters at once
    along a random Bernoulli ±1 direction :math:`h`:

    .. math::
        \hat g_k = \frac{f(\theta + c_k h) - f(\theta - c_k h)}{2 c_k}\, h,
        \qquad \theta \leftarrow \theta - a_k \hat g_k,

    with decaying gains :math:`a_k = a/(A+k+1)^\alpha` and
    :math:`c_k = c/(k+1)^\gamma`. This makes it attractive for many-parameter,
    shot-noisy circuits where parameter-shift gradients are prohibitively
    expensive. Single-point optimizer (``n_param_sets == 1``); gradient-free, so
    any ``jac``/``metric_fn`` supplied by the variational algorithm is ignored.

    Args:
        learning_rate: Spall's :math:`a` — the learning-rate gain numerator.
        c: Perturbation-size gain numerator :math:`c` (≈ the std of the cost
            noise is a good starting scale).
        alpha: Decay exponent for the learning-rate gain (Spall default 0.602).
        gamma: Decay exponent for the perturbation gain (Spall default 0.101).
        A: Learning-rate stability constant; defaults to ``0.1 * max_iterations``.
        resamplings: Average this many independent SPSA gradient samples per step
            to reduce variance (each costs two more evaluations).
        blocking: Enable look-ahead blocking — evaluate the candidate's loss and
            reject the step if it exceeds the current loss by more than
            ``blocking_tol``·std of the recent window, otherwise accept. Prevents
            runaway divergence on noisy/high-curvature landscapes. Costs one extra
            evaluation per step, plus one at the start to seed the baseline. Off by
            default.
        blocking_history: Window length for the std band used by ``blocking``.
        blocking_tol: Reject a candidate whose loss exceeds the current loss by
            more than ``blocking_tol``·std of the recent window. This is the knob
            that absorbs cost noise in the accept/reject decision (``resamplings``
            de-noises the gradient, not this single-evaluation comparison).
        exact_loss: When ``True``, spend one extra unperturbed evaluation per step
            to record the exact ``f(theta)`` for the callback and best-iterate
            tracking, instead of the (biased but free) perturbation-average proxy.
            Has no effect when ``blocking`` is set — blocking already records the
            exact loss.
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float | None = None,
        resamplings: int = 1,
        blocking: bool = False,
        blocking_history: int = 5,
        blocking_tol: float = 2.0,
        exact_loss: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=resamplings,
            blocking=blocking,
            blocking_history=blocking_history,
            blocking_tol=blocking_tol,
            exact_loss=exact_loss,
        )

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run SPSA for ``max_iterations`` steps.

        Args:
            cost_fn: Cost function; called with a two-row batch per gradient
                sample so both perturbations share one stochastic-cost draw.
            initial_params: Starting parameters (1D, or 2D with a single row).
            callback_fn: Called after each step with an ``OptimizeResult`` whose
                ``x`` is 2D and ``fun`` is 1D. May raise ``StopIteration``.
            **kwargs: ``max_iterations`` (default 50) and ``rng`` (the source of
                the perturbation directions — pass it for reproducible runs).
                ``jac`` and ``metric_fn`` are accepted and ignored (SPSA is
                gradient-free).
        """
        max_iterations = kwargs.pop("max_iterations", None) or 50
        rng = kwargs.pop("rng", None) or np.random.default_rng()
        kwargs.pop("jac", None)
        kwargs.pop("metric_fn", None)

        if initial_params is None:
            raise ValueError("SPSAOptimizer requires initial_params.")

        theta = np.atleast_1d(np.asarray(initial_params, dtype=np.float64).squeeze())
        A = self.A if self.A is not None else 0.1 * max_iterations

        best_x = theta.copy()
        best_fun = np.inf
        recent: deque[float] = deque(maxlen=self.blocking_history)
        # Seeded only for the blocking path; off it the value is never read (``fun``
        # routes through ``_step_loss`` instead).
        current_loss: float = (
            float(np.asarray(cost_fn(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False

        for k in range(max_iterations):
            c_k = _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, self.learning_rate, A, self.alpha)

            ghats = []
            losses = []
            for _ in range(self.resamplings):
                ghat, _, f_plus, f_minus = _spsa_gradient(cost_fn, theta, c_k, rng)
                ghats.append(ghat)
                losses.append(0.5 * (f_plus + f_minus))
            ghat = np.mean(ghats, axis=0)
            fun = (
                current_loss
                if self.blocking
                else self._step_loss(cost_fn, theta, float(np.mean(losses)))
            )
            if reference_loss is None:
                reference_loss = fun
            diverged_warned = self._warn_if_diverging(
                fun, reference_loss, diverged_warned
            )

            recent.append(fun)
            if fun < best_fun:
                best_fun = fun
                best_x = theta.copy()
            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(theta.copy()),
                        fun=np.atleast_1d(fun),
                        nit=k + 1,
                        success=True,
                        message="Optimization in progress.",
                    )
                )

            proposed = theta - a_k * ghat
            if self.blocking:
                theta, current_loss = self._block_or_step(
                    cost_fn, theta, proposed, current_loss, recent
                )
            else:
                theta = proposed

        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
        )


class QNSPSAOptimizer(_SPSAConfigMixin, Optimizer):
    r"""Quantum Natural SPSA (Gacon et al.).

    Combines the cheap SPSA gradient with a *stochastic* Fubini–Study metric, so
    both the gradient and the geometry cost a constant number of circuit
    evaluations per step regardless of the parameter count. The default metric is
    estimated from state-fidelity overlaps via two random directions
    :math:`h_1, h_2`:

    .. math::
        \delta F = F(\theta,\theta + c_k h_1 + c_k h_2) - F(\theta,\theta + c_k h_1)
                 - F(\theta,\theta - c_k h_1 + c_k h_2) + F(\theta,\theta - c_k h_1),
        \quad
        \hat g = -\frac{\delta F}{8 c_k^2}\,(h_1 h_2^\top + h_2 h_1^\top),

    accumulated into a running average :math:`\bar g_k=(k\,\bar g_{k-1}+\hat g)/(k+1)`
    seeded at the identity, conditioned as :math:`|\bar g_k| + \beta I` (matrix
    absolute value plus an identity shift), and used to precondition the SPSA
    gradient: :math:`\theta \leftarrow \theta - a_k (|\bar g_k|+\beta I)^{-1}\hat g`.

    The metric backend is pluggable, exactly as for
    :class:`~divi.qprog.optimizers.QNGOptimizer`. The
    default :class:`~divi.qprog._metrics.StochasticFidelityMetricEstimator` is the
    faithful QN-SPSA metric; passing
    :class:`~divi.qprog._metrics.FubiniStudyMetricEstimator` (or
    :class:`~divi.qprog._metrics.PullbackMetricEstimator`) instead uses that
    estimator's exact metric while keeping the SPSA gradient.

    Single-point optimizer (``n_param_sets == 1``); the variational algorithm
    supplies the metric evaluator via :meth:`build_evaluators`.

    Args:
        learning_rate: Spall's :math:`a` — the learning-rate gain numerator.
        c: Perturbation-size gain numerator :math:`c`.
        alpha: Decay exponent for the learning-rate gain (Spall default 0.602).
        gamma: Decay exponent for the perturbation gain (Spall default 0.101).
        A: Learning-rate stability constant; defaults to ``0.1 * max_iterations``.
        regularization: Identity-shift :math:`\beta` added to the conditioned
            metric so the linear solve stays positive-definite.
        resamplings: Average this many independent gradient/metric samples per
            step to reduce variance.
        blocking: Enable look-ahead blocking (reject a step whose candidate loss
            exceeds the current loss by more than ``blocking_tol``·std of the
            recent window). Recommended for high-dimensional or noisy runs where
            the stochastic metric can otherwise drive a divergent step. Costs one
            extra evaluation per step, plus one at the start to seed the baseline.
            Off by default.
        blocking_history: Window length for the std band used by ``blocking``.
        blocking_tol: Reject a candidate whose loss exceeds the current loss by
            more than ``blocking_tol``·std of the recent window. This is the knob
            that absorbs cost noise in the accept/reject decision (``resamplings``
            de-noises the gradient/metric, not this single-evaluation comparison).
        exact_loss: When ``True``, spend one extra unperturbed evaluation per step
            to record the exact ``f(theta)`` for the callback and best-iterate
            tracking, instead of the (biased but free) perturbation-average proxy.
            Has no effect when ``blocking`` is set — blocking already records the
            exact loss.
        metric_estimator: Strategy supplying the metric. Defaults to the
            stochastic-fidelity estimator (the faithful QN-SPSA metric).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float | None = None,
        regularization: float = 1e-3,
        resamplings: int = 1,
        blocking: bool = False,
        blocking_history: int = 5,
        blocking_tol: float = 2.0,
        exact_loss: bool = False,
        metric_estimator: MetricEstimator | None = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=resamplings,
            blocking=blocking,
            blocking_history=blocking_history,
            blocking_tol=blocking_tol,
            exact_loss=exact_loss,
        )
        if regularization < 0:
            raise ValueError(
                f"regularization must be non-negative, got {regularization}."
            )
        self.regularization = regularization
        self.metric_estimator = metric_estimator or StochasticFidelityMetricEstimator()

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        """Reject a program whose ansatz the chosen metric estimator cannot model."""
        self.metric_estimator.check_compatible(program)

    def build_evaluators(
        self, program: "VariationalQuantumAlgorithm"
    ) -> dict[str, Callable[[npt.NDArray[np.float64]], Any]]:
        """Bind the metric estimator (its ``fidelity_fn`` or ``metric_fn``)."""
        return self.metric_estimator.bind(program)

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run QN-SPSA for ``max_iterations`` steps.

        Args:
            cost_fn: Cost function; called with a two-row batch per gradient
                sample so both perturbations share one stochastic-cost draw.
            initial_params: Starting parameters (1D, or 2D with a single row).
            callback_fn: Called after each step with an ``OptimizeResult`` whose
                ``x`` is 2D and ``fun`` is 1D. May raise ``StopIteration``.
            **kwargs: ``max_iterations`` (default 50), ``rng`` (the source of the
                perturbation directions — pass it for reproducible runs), and
                exactly one metric evaluator — ``fidelity_fn`` (stochastic, the
                default) or ``metric_fn`` (an exact estimator). ``jac`` is accepted
                and ignored (QN-SPSA's gradient is the SPSA estimate).
        """
        max_iterations = kwargs.pop("max_iterations", None) or 50
        rng = kwargs.pop("rng", None) or np.random.default_rng()
        kwargs.pop("jac", None)
        fidelity_fn = kwargs.pop("fidelity_fn", None)
        metric_fn = kwargs.pop("metric_fn", None)

        if fidelity_fn is None and metric_fn is None:
            raise ValueError(
                "QNSPSAOptimizer requires a metric evaluator (`fidelity_fn` or "
                "`metric_fn`). It is driven by VariationalQuantumAlgorithm.run(), "
                "which supplies one via the metric estimator."
            )
        if initial_params is None:
            raise ValueError("QNSPSAOptimizer requires initial_params.")

        theta = np.atleast_1d(np.asarray(initial_params, dtype=np.float64).squeeze())
        n_params = theta.shape[0]
        A = self.A if self.A is not None else 0.1 * max_iterations

        g_bar = np.eye(n_params)
        best_x = theta.copy()
        best_fun = np.inf
        recent: deque[float] = deque(maxlen=self.blocking_history)
        # Seeded only for the blocking path; off it the value is never read (``fun``
        # routes through ``_step_loss`` instead).
        current_loss: float = (
            float(np.asarray(cost_fn(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False

        for k in range(max_iterations):
            c_k = _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, self.learning_rate, A, self.alpha)

            ghats = []
            losses = []
            raws = []
            for _ in range(self.resamplings):
                ghat, h1, f_plus, f_minus = _spsa_gradient(cost_fn, theta, c_k, rng)
                ghats.append(ghat)
                losses.append(0.5 * (f_plus + f_minus))
                if fidelity_fn is not None:
                    raws.append(
                        _fidelity_metric_sample(fidelity_fn, theta, h1, c_k, rng)
                    )
            ghat = np.mean(ghats, axis=0)
            fun = (
                current_loss
                if self.blocking
                else self._step_loss(cost_fn, theta, float(np.mean(losses)))
            )
            if reference_loss is None:
                reference_loss = fun
            diverged_warned = self._warn_if_diverging(
                fun, reference_loss, diverged_warned
            )

            if fidelity_fn is not None:
                g_bar = (k * g_bar + np.mean(raws, axis=0)) / (k + 1.0)
            else:
                g_bar = np.asarray(metric_fn(theta), dtype=np.float64)

            g_reg = _matrix_abs_psd(g_bar) + self.regularization * np.eye(n_params)
            delta = _regularized_solve(
                ghat,
                g_reg,
                solver="tikhonov",
                regularization=0.0,
                scale_regularization=False,
                rcond=1e-6,
            )

            recent.append(fun)
            if fun < best_fun:
                best_fun = fun
                best_x = theta.copy()
            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(theta.copy()),
                        fun=np.atleast_1d(fun),
                        nit=k + 1,
                        success=True,
                        message="Optimization in progress.",
                    )
                )

            proposed = theta - a_k * delta
            if self.blocking:
                theta, current_loss = self._block_or_step(
                    cost_fn, theta, proposed, current_loss, recent
                )
            else:
                theta = proposed

        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
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

        # Optimization state (updated during optimize(), used for checkpointing)
        self._curr_population: npt.NDArray[np.float64] | None = None
        self._curr_evaluated_population: npt.NDArray[np.float64] | None = None
        self._curr_losses: npt.NDArray[np.float64] | None = None
        self._curr_iteration: int | None = None
        self._curr_rng_state: dict | None = None

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

        for curr_iter in range(start_iter, end_iter):
            # Evaluate the entire population once
            self._curr_losses = np.atleast_1d(cost_fn(self._curr_population)).astype(
                np.float64
            )
            self._curr_evaluated_population = np.copy(self._curr_population)

            # Find the indices of the best-performing parameter sets
            best_indices = np.argpartition(self._curr_losses, self.n_best_sets - 1)[
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

    def copy(self) -> "MonteCarloOptimizer":
        """Fresh copy, rebuilt from configuration (drops population/RNG state)."""
        return MonteCarloOptimizer(
            population_size=self._population_size,
            n_best_sets=self._n_best_sets,
            keep_best_params=self._keep_best_params,
        )


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

        best_idx = np.argmin(self._all_losses)
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
