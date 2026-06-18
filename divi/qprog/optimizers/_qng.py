# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog._metrics import MetricEstimator, PullbackMetricEstimator
from divi.qprog.optimizers._base import Optimizer
from divi.qprog.optimizers._linalg import _regularized_solve

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


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
        max_iterations = self._resolve_max_iterations(kwargs)
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
            # 1-D best iterate, consistent with every other optimizer (the callback
            # x is 2-D as the iteration contract requires; the final result is not).
            x=best_x,
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
