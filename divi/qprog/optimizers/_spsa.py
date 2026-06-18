# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog._metrics import MetricEstimator, StochasticFidelityMetricEstimator
from divi.qprog.optimizers._base import Optimizer
from divi.qprog.optimizers._linalg import (
    _matrix_abs_psd,
    _regularized_solve,
)

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

#: Maps ``(theta, [perturbations])`` to one squared overlap per perturbation.
FidelityFn = Callable[
    [npt.NDArray[np.float64], list[npt.NDArray[np.float64]]],
    npt.NDArray[np.float64],
]


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
    if values.size < 2:
        raise ValueError(
            "cost_fn must return one value per batch row; the two-row "
            f"perturbation batch produced {values.size} value(s). Ensure the cost "
            "function is batch-aware (returns a 1-D array for a 2-D input)."
        )
    f_plus, f_minus = float(values[0]), float(values[1])
    ghat = (f_plus - f_minus) / (2.0 * c_k) * h
    return ghat, h, f_plus, f_minus


def _fidelity_metric_sample(
    fidelity_fn: FidelityFn,
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
        recent: deque[float],
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
            **kwargs: ``max_iterations`` (default 50, must be >= 1) and ``rng``
                (the perturbation directions — pass it for reproducible runs).
                ``jac`` and ``metric_fn`` are accepted and ignored (SPSA is
                gradient-free).
        """
        max_iterations = self._resolve_max_iterations(kwargs)
        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()
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

        # A step accepted on the final iteration carries its measured loss in
        # current_loss but was not yet best-tracked (that runs at the top of the
        # loop), so fold it in — otherwise a one-step accepted run returns the
        # stale starting point.
        if self.blocking and current_loss < best_fun:
            best_fun = current_loss
            best_x = theta.copy()

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
            **kwargs: ``max_iterations`` (default 50, must be >= 1), ``rng`` (the
                perturbation directions — pass it for reproducible runs), and
                exactly one metric evaluator — ``fidelity_fn`` (stochastic, the
                default) or ``metric_fn`` (an exact estimator). ``jac`` is accepted
                and ignored (QN-SPSA's gradient is the SPSA estimate).
        """
        max_iterations = self._resolve_max_iterations(kwargs)
        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()
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
        metric_samples = 1  # the identity seed counts as the first metric sample
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
                # Fold the raw sample into the running average, keeping the
                # identity seed as the first sample so it conditions the early
                # (noisy, low-rank) solves instead of being discarded at k=0.
                g_bar = (metric_samples * g_bar + np.mean(raws, axis=0)) / (
                    metric_samples + 1.0
                )
                metric_samples += 1
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

        # A step accepted on the final iteration carries its measured loss in
        # current_loss but was not yet best-tracked (that runs at the top of the
        # loop), so fold it in — otherwise a one-step accepted run returns the
        # stale starting point.
        if self.blocking and current_loss < best_fun:
            best_fun = current_loss
            best_x = theta.copy()

        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
        )


def _cost_fn_supports_variance(cost_fn: Callable) -> bool:
    """Whether ``cost_fn`` exposes the shot-variance channel.

    Capability is *declared by the producer*, not inferred from the signature: a
    producer that supports the variance channel sets ``supports_variance = True``
    on its callable (the variational algorithm's cost closure does). A plain
    callable from a direct ``optimize`` call carries no such flag, so QUIVER uses
    the variance path only against a declaring producer and degrades gracefully
    otherwise — no fragile signature sniffing.
    """
    return bool(getattr(cost_fn, "supports_variance", False))


class QUIVEROptimizer(_SPSAConfigMixin, Optimizer):
    r"""Adaptive directional (forward) gradients — QUIVER (arXiv 2606.09734).

    Reconstructs the full gradient from ``V`` random Rademacher directional
    derivatives, independent of the parameter count ``N``:

    .. math::
        \tilde\nabla^{\mathsf F} f = \frac{1}{V}\sum_{\ell=1}^{V}
        \Big(\frac{f(\theta+\varepsilon v_\ell) - f(\theta-\varepsilon v_\ell)}
        {2\varepsilon}\Big)\, v_\ell ,
        \qquad \theta \leftarrow \theta - a_k\,\tilde\nabla^{\mathsf F} f ,

    costing ``2V`` evaluations per step. This unifies SPSA (``V=1``,
    finite-difference), random coordinate descent (``V=1``, parameter-shift
    directional derivative) and the full parameter-shift rule (``V=N``) under one
    tunable ``V``.

    QUIVER additionally adapts ``V`` and the per-direction shot count ``M`` each
    step (iCANS/gCANS-style), maximising expected progress per measurement shot:

    * **``V`` from the sample spread (no backend variance needed).** The ``V``
      i.i.d. directional samples already estimate the forward-gradient variance
      ``S²``; more directions are spent when the relative gradient variance is
      high and fewer as the estimate concentrates. This encodes the paper's
      assumption that measurement noise concentrates uniformly across random
      directions, so allocation is by *number of directions*, not per-parameter.
    * **``M`` from the injected measurement variance.** When the variational
      algorithm's cost closure exposes a shot-noise variance (shot-based
      backends), QUIVER reads the single-shot cost variance and sets ``M`` to
      balance derivative noise against the gradient signal. On native-expval
      backends or a plain ``cost_fn`` (no variance channel) it falls back to a
      fixed ``M`` and ``V``-from-spread only — still a valid forward-gradient
      optimizer.

    .. note::
        A per-evaluation shot budget is delivered to the backend as explicit
        per-circuit ``shot_groups``, which disables circuit-template batching for
        that submission. On template-capable backends (e.g. the Qoro cloud) an
        adapting ``M`` therefore trades template reuse for shot adaptivity; if
        submission overhead dominates, prefer ``adapt_M=False`` there and keep
        ``adapt_M`` for local shot-based simulators.

    Single-point optimizer (``n_param_sets == 1``); gradient-free, so any
    ``jac``/``metric_fn`` supplied by the variational algorithm is ignored.

    Args:
        learning_rate: Step-size numerator ``a`` (constant by default; the gain
            schedule reuses Spall's ``a/(A+k+1)**alpha`` with ``alpha=0``).
        epsilon: Finite-difference step ``ε`` (paper default ``0.1``); for
            ``derivative_mode='parameter_shift'`` the shift ``π/2`` is used
            instead.
        V_init/V_min/V_max: Initial / minimum / maximum number of random
            directions per step.
        M_init/M_min/M_max: Initial / minimum / maximum shots per directional
            evaluation (only adapted on shot-based backends).
        adapt_V: Adapt the number of directions from the sample spread.
        adapt_M: Adapt the shot budget from the injected measurement variance.
        derivative_mode: ``'finite_diff'`` (default, central difference with step
            ``ε``) or ``'parameter_shift'`` (directional shift ``π/2``; exact only
            for equal-eigenvalue generators along basis directions, otherwise an
            approximation).
        lipschitz: Smoothness constant ``L`` for the gain bound; when ``None`` the
            optimal step ``a = 1/L`` is taken to be ``learning_rate``.
        mu: EMA decay for the running gradient / variance estimates.
        b: Small floor guarding divisions by a vanishing gradient norm.
        alpha/gamma/A: Spall gain-schedule knobs (default to constant gains).
        blocking/blocking_history/blocking_tol/exact_loss: Inherited look-ahead
            blocking and loss-recording behaviour (see :class:`SPSAOptimizer`).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        V_init: int = 1,
        V_min: int = 1,
        V_max: int = 50,
        M_init: int = 100,
        M_min: int = 10,
        M_max: int = 10000,
        adapt_V: bool = True,
        adapt_M: bool = True,
        derivative_mode: Literal["finite_diff", "parameter_shift"] = "finite_diff",
        lipschitz: float | None = None,
        mu: float = 0.99,
        b: float = 1e-6,
        alpha: float = 0.0,
        gamma: float = 0.0,
        A: float | None = None,
        blocking: bool = False,
        blocking_history: int = 5,
        blocking_tol: float = 2.0,
        exact_loss: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=epsilon,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=V_init,
            blocking=blocking,
            blocking_history=blocking_history,
            blocking_tol=blocking_tol,
            exact_loss=exact_loss,
        )
        if not (1 <= V_min <= V_init <= V_max):
            raise ValueError(
                "Require 1 <= V_min <= V_init <= V_max, got "
                f"V_min={V_min}, V_init={V_init}, V_max={V_max}."
            )
        if not (1 <= M_min <= M_init <= M_max):
            raise ValueError(
                "Require 1 <= M_min <= M_init <= M_max, got "
                f"M_min={M_min}, M_init={M_init}, M_max={M_max}."
            )
        if not (0.0 < mu < 1.0):
            raise ValueError(f"mu must be in (0, 1), got {mu}.")
        if lipschitz is not None and lipschitz <= 0:
            raise ValueError(f"lipschitz must be positive, got {lipschitz}.")
        if derivative_mode not in ("finite_diff", "parameter_shift"):
            raise ValueError(
                "derivative_mode must be 'finite_diff' or 'parameter_shift', "
                f"got {derivative_mode!r}."
            )

        self.epsilon = epsilon
        self.V_init = V_init
        self.V_min = V_min
        self.V_max = V_max
        self.M_init = M_init
        self.M_min = M_min
        self.M_max = M_max
        self.adapt_V = adapt_V
        self.adapt_M = adapt_M
        self.derivative_mode = derivative_mode
        self.lipschitz = lipschitz
        self.mu = mu
        self.b = b

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        """Warn when ``adapt_M`` is combined with a configured shot distribution.

        ``M``-adaptivity recovers the single-shot cost variance as
        ``Var(<H>)·M``, which assumes every measurement group received the same
        ``M`` shots. A shot distribution splits the budget unevenly across
        groups, so that recovery is miscalibrated — the gradient and loss stay
        correct, but the adapted ``M`` may be off. Disable ``adapt_M`` or drop
        the shot distribution to silence this.
        """
        if self.adapt_M and getattr(program, "_shot_distribution", None) is not None:
            warnings.warn(
                f"{type(self).__name__}: adapt_M=True with a configured "
                "shot_distribution — the per-direction shot-budget adaptation "
                "assumes uniform per-group shots and may be miscalibrated. "
                "Disable adapt_M or remove the shot distribution.",
                stacklevel=2,
            )

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Run QUIVER for ``max_iterations`` steps.

        Args:
            cost_fn: Cost function; called with a two-row batch per directional
                sample. When it accepts ``shots``/``return_variance`` (the
                variational algorithm's closure), QUIVER drives the adaptive shot
                budget and reads the measurement variance for ``M``-adaptivity.
            initial_params: Starting parameters (1D, or 2D with a single row).
            callback_fn: Called after each step with an ``OptimizeResult`` whose
                ``x`` is 2D and ``fun`` is 1D. May raise ``StopIteration``.
            **kwargs: ``max_iterations`` (default 50, must be >= 1) and ``rng``.
                ``jac`` and ``metric_fn`` are accepted and ignored (QUIVER is
                gradient-free).
        """
        max_iterations = self._resolve_max_iterations(kwargs)
        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()
        kwargs.pop("jac", None)
        kwargs.pop("metric_fn", None)

        if initial_params is None:
            raise ValueError("QUIVEROptimizer requires initial_params.")

        # Single-point optimizer: accept 1-D or a single-row (1, n) array, but
        # reject any other 2-D shape rather than silently flattening a multi-start
        # array into one long vector (or crashing later on a broadcast mismatch).
        theta = np.asarray(initial_params, dtype=np.float64)
        if theta.ndim == 2 and theta.shape[0] == 1:
            theta = theta[0]
        if theta.ndim != 1:
            raise ValueError(
                "QUIVEROptimizer is a single-point optimizer; initial_params must "
                f"be 1-D or shape (1, n_params), got shape {theta.shape}."
            )
        A = self.A if self.A is not None else 0.1 * max_iterations
        shift = (0.5 * np.pi) if self.derivative_mode == "parameter_shift" else None

        supports_variance = _cost_fn_supports_variance(cost_fn)
        M_k: int | None = self.M_init  # adapted shot budget; read live by cost_only
        last_variance: npt.NDArray[np.float64] | None = None

        def cost_only(
            batch: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Loss-only adapter; stashes the latest measurement variance."""
            nonlocal last_variance
            if supports_variance:
                # The variational algorithm's cost closure accepts these kwargs
                # and returns (losses, variances); the base ``cost_fn`` type
                # cannot express that optional contract, so call through Any.
                losses, variances = cast(Any, cost_fn)(
                    batch, shots=M_k, return_variance=True
                )
                last_variance = np.asarray(variances, dtype=np.float64).reshape(-1)
                return np.asarray(losses, dtype=np.float64).reshape(-1)
            return np.asarray(cost_fn(batch), dtype=np.float64).reshape(-1)

        # gCANS optimal step a* = 1/L; default L so that a* == learning_rate.
        L = self.lipschitz if self.lipschitz is not None else 1.0 / self.learning_rate

        V_k = self.V_init
        chi = np.zeros_like(theta)  # EMA of the gradient estimate
        xi = 0.0  # EMA of the scalar gradient-sample variance

        best_x = theta.copy()
        best_fun = np.inf
        recent: deque[float] = deque(maxlen=self.blocking_history)
        current_loss: float = (
            float(np.asarray(cost_only(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False
        step_size_warned = False

        for k in range(max_iterations):
            eps_k = shift if shift is not None else _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, self.learning_rate, A, self.alpha)

            ghats = []
            losses = []
            var_samples: list[float] = []
            for _ in range(V_k):
                ghat_l, _, f_plus, f_minus = _spsa_gradient(
                    cost_only, theta, eps_k, rng
                )
                if shift is not None:
                    # ``_spsa_gradient`` divides by ``2·eps_k``; the parameter-shift
                    # rule for ±π/2 evaluations of an equal-eigenvalue (±½)
                    # generator uses a ½ prefactor, i.e. divides by 2. Rescale by
                    # ``eps_k`` so the estimate is the true parameter-shift
                    # gradient (½(f₊−f₋)·v), not the ``2/π``-scaled value the
                    # finite-difference normalization would give.
                    ghat_l = ghat_l * eps_k
                ghats.append(ghat_l)
                losses.append(0.5 * (f_plus + f_minus))
                v = last_variance
                if v is not None and len(v) and np.all(np.isfinite(v)):
                    # Reported variance is Var(<H>) at M_k shots; recover the
                    # single-shot cost variance as Var·M.
                    m_now = M_k if M_k is not None else 1
                    var_samples.append(float(np.mean(v)) * float(m_now))

            ghat = np.mean(ghats, axis=0)

            # Variance of the single-direction estimator across the V samples
            # (sum of per-component variances); folds in both direction and
            # measurement noise. Needs >= 2 samples, else carry the prior EMA.
            if V_k >= 2:
                spread = np.stack(ghats) - ghat
                S2 = float(np.sum(spread * spread) / (V_k - 1))
            else:
                S2 = xi  # reuse last estimate when a single direction was drawn

            chi = self.mu * chi + (1.0 - self.mu) * ghat
            xi = self.mu * xi + (1.0 - self.mu) * S2
            bias_corr = 1.0 - self.mu ** (k + 1)
            chi_hat = chi / bias_corr
            xi_hat = xi / bias_corr
            g2 = float(chi_hat @ chi_hat) + self.b

            fun = (
                current_loss
                if self.blocking
                else self._step_loss(cost_only, theta, float(np.mean(losses)))
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
                    cost_only, theta, proposed, current_loss, recent
                )
            else:
                theta = proposed

            # --- Adapt (V, M) for the next step ---
            if not step_size_warned and L * a_k >= 2.0:
                warnings.warn(
                    f"{type(self).__name__}: L*a_k = {L * a_k:.3g} >= 2 leaves the "
                    "gCANS stability regime (requires a < 2/L); the (V, M) "
                    "allocation will saturate at its bounds. Lower learning_rate "
                    "or raise lipschitz.",
                    stacklevel=2,
                )
                step_size_warned = True
            kappa = (2.0 * L * a_k) / max(2.0 - L * a_k, self.b)
            if self.adapt_V:
                V_k = int(np.clip(np.ceil(kappa * xi_hat / g2), self.V_min, self.V_max))
            if self.adapt_M and supports_variance and var_samples:
                sigma2 = float(np.mean(var_samples))
                M_next = np.ceil(kappa * sigma2 / (2.0 * eps_k * eps_k * g2))
                M_k = int(np.clip(M_next, self.M_min, self.M_max))

        # Best-iterate tracking runs at the top of the loop, so the iterate
        # reached by the final step is never tracked. Fold it in:
        #   * blocking: the accepted step's loss is already measured in
        #     current_loss, so no extra evaluation is needed.
        #   * exact_loss (no blocking): the per-step loss is the true f(theta),
        #     so spend one more exact evaluation on the final iterate to match
        #     that contract. (The free-proxy path is left as-is: its recorded
        #     loss is the perturbation average, not f(theta).)
        if self.blocking:
            if current_loss < best_fun:
                best_fun = current_loss
                best_x = theta.copy()
        elif self.exact_loss:
            final_loss = float(np.asarray(cost_only(theta)).reshape(-1)[0])
            if final_loss < best_fun:
                best_fun = final_loss
                best_x = theta.copy()

        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
        )
