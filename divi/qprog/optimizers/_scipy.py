# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from divi.qprog.optimizers._base import Optimizer


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
            initial_params.reshape(-1),
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
