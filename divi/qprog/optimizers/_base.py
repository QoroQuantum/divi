# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


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
            OptimizeResult whose ``x`` is the single best parameter set as a 1-D
            array of shape ``(n_params,)``. (The per-iteration ``callback_fn``
            receives a 2-D ``x`` of shape ``(n_param_sets, n_params)``; only the
            final result is 1-D.)
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

    def _resolve_max_iterations(self, kwargs: dict[str, Any]) -> int:
        """Pop ``max_iterations`` (default 50); reject values < 1.

        For step-based optimizers with no resume: zero steps run the loop never,
        which would otherwise return a successful result with an infinite loss.
        """
        max_iterations = kwargs.pop("max_iterations", None)
        if max_iterations is None:
            return 50
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}; "
                "the optimization loop performs no evaluation with zero steps."
            )
        return max_iterations
