# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import pickle
import time
import warnings
from abc import ABC, abstractmethod
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
    "ScipyMethod",
    "ScipyOptimizer",
    "copy_optimizer",
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
        create a separate instance for each program. You can use the helper function
        :func:`copy_optimizer` to create a fresh copy with the same configuration.
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
        self.solver = solver
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
    def n_param_sets(self) -> int:
        """Number of parameter sets per step — always ``1`` (single-point)."""
        return 1

    def _natural_gradient(
        self,
        grad: npt.NDArray[np.float64],
        metric: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Precondition ``grad`` with the (regularized) inverse metric."""
        grad = np.asarray(grad, dtype=np.float64)
        metric = np.asarray(metric, dtype=np.float64)
        # The pullback metric is symmetric by construction; symmetrize defensively
        # against float round-off so the Cholesky-based solve stays valid.
        metric = 0.5 * (metric + metric.T)

        if self.solver == "pinv":
            delta = np.linalg.pinv(metric, rcond=self.rcond) @ grad
        else:
            lam = self.regularization
            if self.scale_regularization:
                lam *= max(1.0, float(np.mean(np.diag(metric))))
            damped = metric + lam * np.eye(metric.shape[0])
            delta = _solve_linear_system(damped, grad, assume_a="pos")

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


def copy_optimizer(optimizer: Optimizer) -> Optimizer:
    """Create a new optimizer instance with the same configuration as the given optimizer.

    This function creates a fresh copy of an optimizer with identical configuration
    parameters but with reset internal state. This is useful when multiple programs
    need their own optimizer instances to avoid state contamination.

    .. tip::
        Use this function when preparing a batch of programs that will run in parallel.
        Pass a fresh copy of the optimizer to each program instance to ensure thread safety.

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
    elif isinstance(optimizer, GridSearchOptimizer):
        return GridSearchOptimizer(param_grid=optimizer._param_grid.copy())
    else:
        raise ValueError(f"Unknown optimizer type: {type(optimizer)}")


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
