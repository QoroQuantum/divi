# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    def _resolve_max_iterations(self, kwargs: dict[str, Any]) -> int:
        """Pop ``max_iterations`` (default 50); reject values < 1."""
        max_iterations = kwargs.pop("max_iterations", None)
        if max_iterations is None:
            return 50
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}; "
                "SPSA performs no evaluation with zero steps."
            )
        return max_iterations

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

        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimization terminated: reached max_iterations.",
        )
