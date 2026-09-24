# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog._metrics import (
    MetricEstimator,
    StochasticFidelityMetricEstimator,
    _MetricOptimizerMixin,
)
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


#: Target size of the first update in parameter space when the learning rate is
#: calibrated rather than supplied (Spall's guidance, as used by Qiskit's SPSA).
_TARGET_STEP_MAGNITUDE = 2 * np.pi / 10

#: Below this, the measured directional derivative is treated as no signal.
_CALIBRATION_FLOOR = 1e-10


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
    """Shared SPSA gain-schedule config, calibration and validation.

    Holds Spall's gain-sequence hyperparameters and an optional look-ahead
    blocking guard. No mid-run state lives on the instance — the per-run iterate,
    calibrated gains and running-average metric are locals inside ``optimize``.
    """

    def __init__(
        self,
        learning_rate: float | None,
        c: float,
        alpha: float,
        gamma: float,
        A: float | None,
        resamplings: int,
        blocking: bool,
        allowed_increase: float | None,
        exact_loss: bool,
        calibration_steps: int,
    ):
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}.")
        if resamplings < 1:
            raise ValueError(f"resamplings must be >= 1, got {resamplings}.")
        if allowed_increase is not None and allowed_increase < 0:
            raise ValueError(
                f"allowed_increase must be non-negative, got {allowed_increase}."
            )
        if calibration_steps < 1:
            raise ValueError(
                f"calibration_steps must be >= 1, got {calibration_steps}."
            )

        self.learning_rate = learning_rate
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.resamplings = resamplings
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.exact_loss = exact_loss
        self.calibration_steps = calibration_steps

    @property
    def n_param_sets(self) -> int:
        """Number of parameter sets per step — always ``1`` (single-point)."""
        return 1

    def _calibrated_learning_rate(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> float:
        """``learning_rate``, measured from the loss when it was not supplied.

        Averages the magnitude of the SPSA directional derivative over
        ``calibration_steps`` random directions and returns the gain that makes
        the first update about :data:`_TARGET_STEP_MAGNITUDE` in parameter space.
        A fixed gain has to match the loss's own scale to be stable, which the
        caller cannot know in advance; this measures it in ``2 *
        calibration_steps`` evaluations.
        """
        if self.learning_rate is not None:
            return self.learning_rate

        magnitudes = []
        for _ in range(self.calibration_steps):
            _, _, f_plus, f_minus = _spsa_gradient(cost_fn, theta, self.c, rng)
            magnitudes.append(abs((f_plus - f_minus) / (2.0 * self.c)))

        finite = [m for m in magnitudes if np.isfinite(m)]
        average = float(np.mean(finite)) if finite else 0.0
        if average < _CALIBRATION_FLOOR:
            return _TARGET_STEP_MAGNITUDE
        return _TARGET_STEP_MAGNITUDE / average

    def _calibrated_allowed_increase(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
    ) -> float:
        """``allowed_increase``, measured from the loss when it was not supplied.

        Twice the standard deviation of the loss at the starting point, so the
        band admits ordinary shot noise and rejects a genuine regression. Fixed
        for the run: a band recomputed from accepted losses collapses to zero
        once a few steps in a row are rejected, and then nothing can be accepted
        again.
        """
        if self.allowed_increase is not None:
            return self.allowed_increase

        repeats = np.tile(theta, (self.calibration_steps, 1))
        values = np.asarray(cost_fn(repeats), dtype=np.float64).reshape(-1)
        finite_values = values[np.isfinite(values)]
        spread = float(np.std(finite_values)) if finite_values.size else 0.0
        return 2.0 * spread

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
        allowed_increase: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Look-ahead blocking (Spall/Gacon): move to ``proposed`` only if its loss
        does not exceed ``current_loss`` by more than ``allowed_increase``;
        otherwise hold ``theta``.

        Costs one extra cost evaluation per step (the candidate's loss); the
        accepted value carries over as the next ``current_loss``, so it is not
        re-measured.

        A non-finite candidate loss is treated as a rejection rather than accepted
        — ``nan > x`` is ``False``, so without this guard a NaN candidate would
        slip through into the iterate.

        Returns ``(next_theta, loss_at_next_theta)``.
        """
        f_proposed = float(np.asarray(cost_fn(proposed)).reshape(-1)[0])
        if not np.isfinite(f_proposed) or f_proposed > current_loss + allowed_increase:
            return theta, current_loss
        return proposed, f_proposed

    def _final_result(
        self,
        best_x: npt.NDArray[np.float64],
        best_fun: float,
        max_iterations: int,
    ) -> OptimizeResult:
        """The run's result, reporting ``success=False`` when no finite cost was
        ever observed and ``best_fun`` is therefore still its ``inf`` seed."""
        if not np.isfinite(best_fun):
            return OptimizeResult(
                x=best_x,
                fun=np.atleast_1d(best_fun),
                nit=max_iterations,
                success=False,
                message=(
                    "Optimisation failed: no finite cost value was observed in "
                    f"{max_iterations} iterations."
                ),
            )
        return OptimizeResult(
            x=best_x,
            fun=np.atleast_1d(best_fun),
            nit=max_iterations,
            success=True,
            message="Optimisation terminated: reached max_iterations.",
        )

    def _fold_final_iterate(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
        current_loss: float,
        best_x: npt.NDArray[np.float64],
        best_fun: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Best-track the iterate the final step reached.

        Tracking runs at the top of the loop, so the last step's destination is
        never seen there and a one-step accepted run would return its starting
        point. ``blocking`` has already measured that iterate's loss;
        ``exact_loss`` spends one more evaluation to match its own contract. The
        proxy path is left alone, since its recorded loss is the perturbation
        average rather than ``f(theta)``.
        """
        if self.blocking:
            final_loss = current_loss
        elif self.exact_loss:
            final_loss = float(np.asarray(cost_fn(theta)).reshape(-1)[0])
        else:
            return best_x, best_fun
        if final_loss < best_fun:
            return theta.copy(), final_loss
        return best_x, best_fun

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
            ``None`` (the default) calibrates it against the loss at the starting
            point, which costs ``2 * calibration_steps`` evaluations and is what
            keeps the run stable on a landscape whose scale the caller does not
            know. Pass a float to skip that and fix the gain.
        c: Perturbation-size gain numerator :math:`c` (≈ the std of the cost
            noise is a good starting scale).
        alpha: Decay exponent for the learning-rate gain (Spall default 0.602).
        gamma: Decay exponent for the perturbation gain (Spall default 0.101).
        A: Learning-rate stability constant; defaults to ``0.1 * max_iterations``.
        resamplings: Average this many independent SPSA gradient samples per step
            to reduce variance (each costs two more evaluations).
        blocking: Enable look-ahead blocking — evaluate the candidate's loss and
            reject the step if it exceeds the current loss by more than
            ``allowed_increase``, otherwise accept. Prevents runaway divergence on
            noisy/high-curvature landscapes. Costs one extra evaluation per step,
            plus one at the start to seed the baseline. Off by default.
        allowed_increase: How much the loss may rise and the step still be
            accepted. ``None`` (the default) calibrates it to twice the standard
            deviation of the loss at the starting point, so the band admits
            ordinary shot noise. Only read when ``blocking`` is set.
        exact_loss: When ``True``, spend one extra unperturbed evaluation per step
            to record the exact ``f(theta)`` for the callback and best-iterate
            tracking, instead of the (biased but free) perturbation-average proxy.
            Has no effect when ``blocking`` is set — blocking already records the
            exact loss.
        calibration_steps: Random directions averaged when calibrating
            ``learning_rate``, and repeats used for ``allowed_increase``.
    """

    def __init__(
        self,
        learning_rate: float | None = None,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float | None = None,
        resamplings: int = 1,
        blocking: bool = False,
        allowed_increase: float | None = None,
        exact_loss: bool = False,
        calibration_steps: int = 25,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=resamplings,
            blocking=blocking,
            allowed_increase=allowed_increase,
            exact_loss=exact_loss,
            calibration_steps=calibration_steps,
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
                ``x`` and ``jac`` are 2D and ``fun`` is 1D; ``jac`` contains the
                reconstructed directional gradient. May raise ``StopIteration``.
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
        learning_rate = self._calibrated_learning_rate(cost_fn, theta, rng)
        allowed_increase = (
            self._calibrated_allowed_increase(cost_fn, theta) if self.blocking else 0.0
        )

        best_x = theta.copy()
        best_fun = np.inf
        # Seeded only for the blocking path; off it the value is never read (``fun``
        # routes through ``_step_loss`` instead).
        current_loss: float = (
            float(np.asarray(cost_fn(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False

        for k in range(max_iterations):
            c_k = _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, learning_rate, A, self.alpha)

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
                        message="Optimisation in progress.",
                    )
                )

            # Holding a non-finite step keeps it out of the iterate for good.
            proposed = theta - a_k * ghat
            if np.all(np.isfinite(proposed)):
                if self.blocking:
                    theta, current_loss = self._block_or_step(
                        cost_fn, theta, proposed, current_loss, allowed_increase
                    )
                else:
                    theta = proposed

        best_x, best_fun = self._fold_final_iterate(
            cost_fn, theta, current_loss, best_x, best_fun
        )

        return self._final_result(best_x, best_fun, max_iterations)


class QNSPSAOptimizer(_SPSAConfigMixin, _MetricOptimizerMixin, Optimizer):
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
    supplies the metric evaluator via
    :meth:`~divi.qprog.optimizers.Optimizer.build_evaluators`.

    Args:
        learning_rate: Spall's :math:`a` — the learning-rate gain numerator.
            ``None`` (the default) calibrates it against the loss at the starting
            point; see :class:`SPSAOptimizer`.
        c: Perturbation-size gain numerator :math:`c`.
        alpha: Decay exponent for the learning-rate gain (Spall default 0.602).
        gamma: Decay exponent for the perturbation gain (Spall default 0.101).
        A: Learning-rate stability constant; defaults to ``0.1 * max_iterations``.
        regularization: Identity-shift :math:`\beta` added to the conditioned
            metric so the linear solve stays positive-definite.
        resamplings: Average this many independent gradient/metric samples per
            step to reduce variance.
        blocking: Enable look-ahead blocking (reject a step whose candidate loss
            exceeds the current loss by more than ``allowed_increase``).
            Recommended for high-dimensional or noisy runs where the stochastic
            metric can otherwise drive a divergent step. Costs one extra
            evaluation per step, plus one at the start to seed the baseline. Off
            by default.
        allowed_increase: How much the loss may rise and the step still be
            accepted; ``None`` calibrates it. See :class:`SPSAOptimizer`.
        exact_loss: When ``True``, spend one extra unperturbed evaluation per step
            to record the exact ``f(theta)`` for the callback and best-iterate
            tracking, instead of the (biased but free) perturbation-average proxy.
            Has no effect when ``blocking`` is set — blocking already records the
            exact loss.
        calibration_steps: Random directions averaged when calibrating.
        metric_estimator: Strategy supplying the metric. Defaults to the
            stochastic-fidelity estimator (the faithful QN-SPSA metric).
        max_step_norm: Trust-region radius for the full natural-gradient update.
            The default bounds metric amplification to the same ``2π/10`` scale
            used by automatic SPSA gain calibration. Set to ``None`` to disable.
    """

    def __init__(
        self,
        learning_rate: float | None = None,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float | None = None,
        regularization: float = 1e-3,
        resamplings: int = 1,
        blocking: bool = False,
        allowed_increase: float | None = None,
        exact_loss: bool = False,
        calibration_steps: int = 25,
        metric_estimator: MetricEstimator | None = None,
        max_step_norm: float | None = 2.0 * np.pi / 10.0,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=resamplings,
            blocking=blocking,
            allowed_increase=allowed_increase,
            exact_loss=exact_loss,
            calibration_steps=calibration_steps,
        )
        if regularization < 0:
            raise ValueError(
                f"regularization must be non-negative, got {regularization}."
            )
        if max_step_norm is not None and max_step_norm <= 0:
            raise ValueError(
                f"max_step_norm must be positive or None, got {max_step_norm}."
            )
        self.regularization = regularization
        self.metric_estimator = metric_estimator or StochasticFidelityMetricEstimator()
        self.max_step_norm = max_step_norm

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
        learning_rate = self._calibrated_learning_rate(cost_fn, theta, rng)
        allowed_increase = (
            self._calibrated_allowed_increase(cost_fn, theta) if self.blocking else 0.0
        )

        g_bar = np.eye(n_params)
        metric_samples = 1  # the identity seed counts as the first metric sample
        best_x = theta.copy()
        best_fun = np.inf
        # Seeded only for the blocking path; off it the value is never read (``fun``
        # routes through ``_step_loss`` instead).
        current_loss: float = (
            float(np.asarray(cost_fn(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False

        for k in range(max_iterations):
            c_k = _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, learning_rate, A, self.alpha)

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
                # A non-finite sample would poison the average permanently.
                raw = np.mean(raws, axis=0)
                if np.all(np.isfinite(raw)):
                    g_bar = (metric_samples * g_bar + raw) / (metric_samples + 1.0)
                    metric_samples += 1
            else:
                g_bar = np.asarray(metric_fn(theta), dtype=np.float64)

            # The eigendecomposition and solve below both reject a nan.
            step_is_usable = bool(
                np.all(np.isfinite(ghat)) and np.all(np.isfinite(g_bar))
            )

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
                        message="Optimisation in progress.",
                    )
                )

            if step_is_usable:
                g_reg = _matrix_abs_psd(g_bar) + self.regularization * np.eye(n_params)
                delta = _regularized_solve(
                    ghat,
                    g_reg,
                    solver="tikhonov",
                    regularization=0.0,
                    scale_regularization=False,
                    rcond=1e-6,
                )
                update = a_k * delta
                update_norm = float(np.linalg.norm(update))
                if (
                    self.max_step_norm is not None
                    and np.isfinite(update_norm)
                    and update_norm > self.max_step_norm
                ):
                    update *= self.max_step_norm / update_norm
                proposed = theta - update
                if np.all(np.isfinite(proposed)):
                    if self.blocking:
                        theta, current_loss = self._block_or_step(
                            cost_fn, theta, proposed, current_loss, allowed_increase
                        )
                    else:
                        theta = proposed

        best_x, best_fun = self._fold_final_iterate(
            cost_fn, theta, current_loss, best_x, best_fun
        )

        return self._final_result(best_x, best_fun, max_iterations)


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

    costing ``2V`` evaluations per step. With the Rademacher directions used by
    this implementation, ``V=1`` is SPSA and larger ``V`` averages independent
    directional estimates. The wider framework in the paper also recovers random
    coordinate descent and the full parameter-shift rule with basis directions;
    this class does not implement that alternative direction distribution.

    QUIVER additionally adapts ``V`` and the per-direction shot count ``M`` with
    the paper's joint minimum-cost allocation rule:

    .. math::
        V^* = \frac{(N - 1 + \alpha_a)\widehat g^2}{\tau^2},
        \qquad
        M^* = \frac{N\widehat\sigma^2}{\alpha_a\widehat g^2}.

    Here ``gradient_norm_squared`` and ``measurement_variance`` are exponential
    moving averages of the reconstructed gradient norm and the per-direction
    measurement variance. The allocations are held fixed during ``warmup`` and then
    rate-limited before their integer clamps are applied. Parameter updates use
    Adam, as in the published algorithm.

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
        learning_rate: Adam step-size numerator ``a`` in
            ``a/(A+k+1)**alpha``. ``None`` chooses
            ``2(2π/10)/n_params``. The inverse-dimension scaling is conservative
            for sparse gradients, whose forward estimate has nonzero entries in
            every parameter.
        epsilon: Finite-difference step ``ε`` (paper default ``0.1``); for
            ``derivative_mode='parameter_shift'`` the shift ``π/2`` is used
            instead.
        V_init/V_min/V_max: Initial / minimum / maximum number of random
            directions per step. ``V_max=None`` applies a practical automatic
            cap of ``min(n_params, 8)`` (never below ``V_init`` or ``V_min``),
            avoiding parameter-shift-scale work by default on large models.
        M_init/M_min/M_max: Initial / minimum / maximum shots per directional
            evaluation (only adapted on shot-based backends). ``M_init=None``
            inherits the backend's configured shot count instead of silently
            replacing it. ``M_max=None`` caps adaptation at that effective
            initial count; pass a larger value to opt into late-run shot growth.
        adapt_V: Adapt the number of directions from the sample spread.
        adapt_M: Adapt the shot budget from the injected measurement variance.
        derivative_mode: ``'finite_diff'`` (default, central difference with step
            ``ε``) or ``'parameter_shift'`` (directional shift ``π/2``). Because
            this class samples Rademacher rather than basis directions, the latter
            is generally an approximation for multi-parameter circuits.
        allocation_alpha: Positive dimensionless allocation ratio
            :math:`\alpha_a` from the joint rule.
        target_gradient_variance: Positive target absolute reconstructed-gradient
            variance :math:`\tau^2`. ``None`` anchors the first allocation target
            to ``V_init``, avoiding an arbitrary loss-scale-dependent default.
        warmup: Number of initial steps for which ``V`` and ``M`` stay fixed.
        rate_down/rate_up: Multiplicative bounds on an allocation change.
        mu: EMA decay for the gradient-norm and directional-variance estimates.
        adam_beta1/adam_beta2/adam_epsilon: Adam moment parameters.
        b: Small floor guarding divisions by a vanishing gradient norm.
        alpha/gamma/A: Spall gain-schedule knobs; both exponents default to 0, the
            paper's constant step.
        blocking/allowed_increase/exact_loss/calibration_steps: Inherited
            look-ahead blocking and calibration (see :class:`SPSAOptimizer`).
    """

    def __init__(
        self,
        learning_rate: float | None = None,
        epsilon: float = 0.1,
        V_init: int = 2,
        V_min: int = 1,
        V_max: int | None = None,
        M_init: int | None = None,
        M_min: int = 10,
        M_max: int | None = None,
        adapt_V: bool = True,
        adapt_M: bool = True,
        derivative_mode: Literal["finite_diff", "parameter_shift"] = "finite_diff",
        allocation_alpha: float = 1.0,
        target_gradient_variance: float | None = None,
        warmup: int = 5,
        rate_down: float = 0.7,
        rate_up: float = 1.5,
        mu: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        b: float = 1e-6,
        alpha: float = 0.0,
        gamma: float = 0.0,
        A: float | None = None,
        blocking: bool = False,
        allowed_increase: float | None = None,
        exact_loss: bool = False,
        calibration_steps: int = 25,
    ):
        super().__init__(
            learning_rate=learning_rate,
            c=epsilon,
            alpha=alpha,
            gamma=gamma,
            A=A,
            resamplings=V_init,
            blocking=blocking,
            allowed_increase=allowed_increase,
            exact_loss=exact_loss,
            calibration_steps=calibration_steps,
        )
        if not (1 <= V_min <= V_init) or (V_max is not None and V_init > V_max):
            raise ValueError(
                "Require 1 <= V_min <= V_init <= V_max when V_max is set, got "
                f"V_min={V_min}, V_init={V_init}, V_max={V_max}."
            )
        if M_min < 1 or (M_init is not None and M_init < M_min):
            raise ValueError(
                "Require 1 <= M_min and, when set, M_min <= M_init, got "
                f"M_min={M_min}, M_init={M_init}, M_max={M_max}."
            )
        if M_max is not None and (
            M_max < M_min or (M_init is not None and M_init > M_max)
        ):
            raise ValueError(
                "When M_max is set, require M_min <= M_max and "
                "M_init <= M_max, got "
                f"M_min={M_min}, M_init={M_init}, M_max={M_max}."
            )
        if not (0.0 < mu < 1.0):
            raise ValueError(f"mu must be in (0, 1), got {mu}.")
        if allocation_alpha <= 0:
            raise ValueError(
                f"allocation_alpha must be positive, got {allocation_alpha}."
            )
        if target_gradient_variance is not None and target_gradient_variance <= 0:
            raise ValueError(
                "target_gradient_variance must be positive, got "
                f"{target_gradient_variance}."
            )
        if warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {warmup}.")
        if not (0.0 < rate_down <= 1.0 <= rate_up):
            raise ValueError(
                "Require 0 < rate_down <= 1 <= rate_up, got "
                f"rate_down={rate_down}, rate_up={rate_up}."
            )
        if not (0.0 < adam_beta1 < 1.0 and 0.0 < adam_beta2 < 1.0):
            raise ValueError("adam_beta1 and adam_beta2 must both be in (0, 1).")
        if adam_epsilon <= 0:
            raise ValueError(f"adam_epsilon must be positive, got {adam_epsilon}.")
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
        self.allocation_alpha = allocation_alpha
        self.target_gradient_variance = target_gradient_variance
        self.warmup = warmup
        self.rate_down = rate_down
        self.rate_up = rate_up
        self.mu = mu
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.b = b

    def _allocation_targets(
        self,
        n_params: int,
        gradient_norm_squared: float,
        measurement_variance: float,
        target_gradient_variance: float | None = None,
    ) -> tuple[float, float]:
        """Return the published joint ``(V*, M*)`` allocation."""
        g2 = max(float(gradient_norm_squared), self.b)
        tau2 = (
            self.target_gradient_variance
            if target_gradient_variance is None
            else target_gradient_variance
        )
        if tau2 is None:
            raise ValueError(
                "target_gradient_variance must be supplied before allocation."
            )
        V_star = (n_params - 1.0 + self.allocation_alpha) * g2 / tau2
        M_star = (
            n_params
            * max(float(measurement_variance), 0.0)
            / (self.allocation_alpha * g2)
        )
        return V_star, M_star

    def _rate_limited_integer(
        self,
        target: float,
        current: int,
        lower: int,
        upper: int,
    ) -> int:
        """Clamp and round one allocation without exceeding its rate limits."""
        rate_lower = max(float(lower), self.rate_down * current)
        rate_upper = min(float(upper), self.rate_up * current)
        if rate_lower > rate_upper:
            # A current allocation can be outside newly changed hard bounds.
            # No value then satisfies both intervals, so honour the hard bound.
            return int(np.clip(current, lower, upper))
        clipped = float(np.clip(target, rate_lower, rate_upper))
        # Apply the paper's multiplicative clamp in the continuous domain before
        # integerising. Ceil-ing the lower rate bound first pins V=2 forever:
        # ceil(0.7 * 2) == 2, so the configured V_min=1 is unreachable.
        return int(np.clip(np.floor(clipped + 0.5), lower, upper))

    def _calibrated_learning_rate(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        theta: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> float:
        """Return the explicit rate or conservatively dimension-scale Adam.

        Adam normalizes the gradient magnitude, so applying SPSA's loss-scale
        calibration as well would double-normalize it. A Rademacher estimate has
        a nonzero entry in every parameter when its directional derivative is
        nonzero. Inverse-dimension scaling keeps even a sparse-gradient first
        step inside the forward-estimator descent regime.
        """
        if self.learning_rate is not None:
            return self.learning_rate
        return 2.0 * _TARGET_STEP_MAGNITUDE / theta.shape[0]

    def validate_program(self, program: "VariationalQuantumAlgorithm") -> None:
        """Warn when ``adapt_M`` is combined with a configured shot distribution.

        ``M``-adaptivity recovers the single-shot cost variance as
        ``Var(<H>)·M``, which assumes every measurement group received the same
        ``M`` shots. A shot distribution splits the budget unevenly across
        groups, so that recovery is miscalibrated — the gradient and loss stay
        correct, but the adapted ``M`` may be off. Disable ``adapt_M`` or drop
        the shot distribution to silence this.
        """
        if self.adapt_M and program._shot_distribution is not None:
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
        inherited_shots = int(getattr(cost_fn, "default_shots", self.M_min))
        initial_shots = inherited_shots if self.M_init is None else self.M_init
        M_max = max(initial_shots, self.M_min) if self.M_max is None else self.M_max
        M_k = int(
            np.clip(
                initial_shots,
                self.M_min,
                M_max,
            )
        )
        last_variance: npt.NDArray[np.float64] | None = None

        def cost_only(
            batch: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Loss-only adapter that forwards the live shot allocation."""
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

        learning_rate = self._calibrated_learning_rate(cost_only, theta, rng)
        allowed_increase = (
            self._calibrated_allowed_increase(cost_only, theta)
            if self.blocking
            else 0.0
        )
        n_params = theta.shape[0]
        V_max = (
            max(self.V_init, self.V_min, min(n_params, 8))
            if self.V_max is None
            else self.V_max
        )
        if self.V_init > V_max or self.V_min > V_max:
            raise ValueError(
                "V_init and V_min cannot exceed the effective V_max "
                f"({V_max}) for {n_params} parameters."
            )
        V_k = self.V_init
        gradient_norm_squared_ema: float | None = None
        measurement_variance_ema: float | None = None
        target_gradient_variance = self.target_gradient_variance
        adam_first_moment = np.zeros_like(theta)
        adam_second_moment = np.zeros_like(theta)

        best_x = theta.copy()
        best_fun = np.inf
        current_loss: float = (
            float(np.asarray(cost_only(theta)).reshape(-1)[0]) if self.blocking else 0.0
        )
        reference_loss: float | None = None
        diverged_warned = False

        for k in range(max_iterations):
            eps_k = shift if shift is not None else _spsa_gain_c(k, self.c, self.gamma)
            a_k = _spsa_gain_a(k, learning_rate, A, self.alpha)

            ghats = []
            losses = []
            measurement_variances: list[float] = []
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
                    # finite-difference normalisation would give.
                    ghat_l = ghat_l * eps_k
                ghats.append(ghat_l)
                losses.append(0.5 * (f_plus + f_minus))
                variances = last_variance
                if (
                    variances is not None
                    and variances.size >= 2
                    and np.all(np.isfinite(variances[:2]))
                ):
                    derivative_scale = 0.5 if shift is not None else 1.0 / (2.0 * eps_k)
                    measurement_variances.append(
                        cast(int, M_k)
                        * derivative_scale**2
                        * float(np.sum(variances[:2]))
                    )

            ghat = np.mean(ghats, axis=0)
            step_is_usable = bool(np.all(np.isfinite(ghat)))
            if step_is_usable:
                # Algorithm 1 of Coyle et al. plugs the raw reconstructed-
                # gradient norm into the allocation EMA. It is upward-biased by
                # the directional estimator's own V-dependent variance, but
                # debiasing it here would no longer implement the published rule.
                gradient_norm_squared = float(ghat @ ghat)
                if gradient_norm_squared_ema is None:
                    gradient_norm_squared_ema = gradient_norm_squared
                    if target_gradient_variance is None:
                        target_gradient_variance = (
                            (n_params - 1.0 + self.allocation_alpha)
                            * max(gradient_norm_squared, self.b)
                            / self.V_init
                        )
                else:
                    gradient_norm_squared_ema = (
                        self.mu * gradient_norm_squared_ema
                        + (1.0 - self.mu) * gradient_norm_squared
                    )
                if measurement_variances:
                    measurement_variance = float(np.mean(measurement_variances))
                    if measurement_variance_ema is None:
                        measurement_variance_ema = measurement_variance
                    else:
                        measurement_variance_ema = (
                            self.mu * measurement_variance_ema
                            + (1.0 - self.mu) * measurement_variance
                        )

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

            if fun < best_fun:
                best_fun = fun
                best_x = theta.copy()
            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(theta.copy()),
                        fun=np.atleast_1d(fun),
                        jac=np.atleast_2d(ghat.copy()),
                        nit=k + 1,
                        success=True,
                        message="Optimisation in progress.",
                    )
                )

            if step_is_usable:
                adam_first_moment = (
                    self.adam_beta1 * adam_first_moment + (1.0 - self.adam_beta1) * ghat
                )
                adam_second_moment = (
                    self.adam_beta2 * adam_second_moment
                    + (1.0 - self.adam_beta2) * ghat * ghat
                )
                adam_first_hat = adam_first_moment / (1.0 - self.adam_beta1 ** (k + 1))
                adam_second_hat = adam_second_moment / (
                    1.0 - self.adam_beta2 ** (k + 1)
                )
                proposed = theta - a_k * adam_first_hat / (
                    np.sqrt(adam_second_hat) + self.adam_epsilon
                )
                if np.all(np.isfinite(proposed)):
                    if self.blocking:
                        theta, current_loss = self._block_or_step(
                            cost_only, theta, proposed, current_loss, allowed_increase
                        )
                    else:
                        theta = proposed

            # --- Adapt (V, M) for the next step ---
            if k >= self.warmup and gradient_norm_squared_ema is not None:
                V_star, M_star = self._allocation_targets(
                    n_params,
                    gradient_norm_squared_ema,
                    measurement_variance_ema or 0.0,
                    target_gradient_variance,
                )
                if self.adapt_V and np.isfinite(V_star):
                    V_k = self._rate_limited_integer(V_star, V_k, self.V_min, V_max)
                if (
                    self.adapt_M
                    and measurement_variance_ema is not None
                    and np.isfinite(M_star)
                ):
                    M_k = self._rate_limited_integer(
                        M_star, cast(int, M_k), self.M_min, M_max
                    )

        best_x, best_fun = self._fold_final_iterate(
            cost_only, theta, current_loss, best_x, best_fun
        )

        return self._final_result(best_x, best_fun, max_iterations)
