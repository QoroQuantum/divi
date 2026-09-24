# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shot-adaptive ROSALIN optimization."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import OPTIMIZER_STATE_FILE, _atomic_write
from divi.qprog.optimizers._base import Optimizer

_PARAMETER_SHIFT = np.pi / 2


def _icans_shots_and_gains(
    gradient_ema: npt.NDArray[np.float64],
    variance_ema: npt.NDArray[np.float64],
    *,
    learning_rate: float,
    lipschitz: float,
    bias_term: float,
    min_shots: int,
    evaluation_counts: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Compute the iCANS allocation and expected gain from paper equations 10–11."""
    factor = 2 * lipschitz * learning_rate / (2 - lipschitz * learning_rate)
    denominator = np.square(gradient_ema) + bias_term
    raw_shots = factor * variance_ema / denominator
    finite_shots = np.nan_to_num(
        raw_shots,
        nan=float(min_shots),
        posinf=float(np.iinfo(np.int32).max),
        neginf=float(min_shots),
    )
    shots = np.maximum(np.ceil(finite_shots), min_shots).astype(np.int64)

    gains = (
        (learning_rate - lipschitz * learning_rate**2 / 2) * np.square(gradient_ema)
        - lipschitz * learning_rate**2 * variance_ema / (2 * shots)
    ) / shots

    # ``smax`` is the shot count at the best-gain coordinate, not that
    # coordinate's integer index.
    gains_per_evaluation = (
        gains if evaluation_counts is None else gains / evaluation_counts
    )
    smax = int(shots[int(np.nanargmax(gains_per_evaluation))])
    return np.clip(shots, min_shots, smax), gains


def _gcans_shots(
    gradient_ema: npt.NDArray[np.float64],
    variance_ema: npt.NDArray[np.float64],
    *,
    learning_rate: float,
    lipschitz: float,
    bias_term: float,
    min_shots: int,
    evaluation_counts: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.int64]:
    """Compute paper equation 13, generalised to unequal shift-rule costs."""
    factor = 2 * lipschitz * learning_rate / (2 - lipschitz * learning_rate)
    standard_deviations = np.sqrt(np.maximum(variance_ema, 0.0))
    costs = (
        np.ones_like(standard_deviations)
        if evaluation_counts is None
        else np.asarray(evaluation_counts, dtype=np.float64)
    )
    if np.any(costs <= 0):
        raise ValueError("evaluation_counts must be positive.")

    weighted_deviation = np.sum(standard_deviations * np.sqrt(costs))
    denominator = float(gradient_ema @ gradient_ema) + bias_term
    raw_shots = (
        factor
        * standard_deviations
        * weighted_deviation
        / (np.sqrt(costs) * denominator)
    )
    finite_shots = np.nan_to_num(
        raw_shots,
        nan=float(min_shots),
        posinf=float(np.iinfo(np.int32).max),
        neginf=float(min_shots),
    )
    return np.maximum(np.ceil(finite_shots), min_shots).astype(np.int64)


def _fit_shots_to_budget(
    shots: npt.NDArray[np.int64],
    *,
    evaluation_counts: npt.NDArray[np.int64],
    max_cost: int,
    min_shots: int,
) -> npt.NDArray[np.int64]:
    """Scale an iCANS allocation down while preserving its relative excess."""
    if int(shots @ evaluation_counts) <= max_cost:
        return shots

    minimum = np.full_like(shots, min_shots)
    excess = shots - minimum
    capacity = max_cost - int(minimum @ evaluation_counts)
    scaled = excess * (capacity / int(excess @ evaluation_counts))
    allocated = np.floor(scaled).astype(np.int64)
    remainder = capacity - int(allocated @ evaluation_counts)
    if remainder > 0:
        fractional_order = np.argsort(-(scaled - allocated), kind="stable")
        for index in fractional_order:
            evaluation_cost = int(evaluation_counts[index])
            if evaluation_cost <= remainder:
                allocated[index] += 1
                remainder -= evaluation_cost
    return minimum + allocated


def _default_shift_rule(
    n_params: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the conventional two-term rule for every parameter."""
    shifts = np.repeat(np.eye(n_params), 2, axis=0)
    shifts[0::2] *= _PARAMETER_SHIFT
    shifts[1::2] *= -_PARAMETER_SHIFT
    weights = np.zeros((n_params, 2 * n_params), dtype=np.float64)
    indices = np.arange(n_params)
    weights[indices, 2 * indices] = 0.5
    weights[indices, 2 * indices + 1] = -0.5
    return shifts, weights


def _shift_rule_layout(
    shifts: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    n_params: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Validate a shift rule and identify each evaluation's parameter."""
    shifts = np.asarray(shifts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if shifts.ndim != 2 or shifts.shape[1] != n_params:
        raise ValueError("shift_rule shifts must have shape (n_evaluations, n_params).")
    if weights.shape != (n_params, shifts.shape[0]):
        raise ValueError(
            "shift_rule weights must have shape (n_params, n_evaluations)."
        )

    active = weights != 0
    active_per_evaluation = np.sum(active, axis=0)
    if np.any(active_per_evaluation != 1):
        raise ValueError(
            "RosalinOptimizer requires each shift-rule evaluation to contribute "
            "to exactly one gradient coordinate."
        )
    evaluation_counts = np.sum(active, axis=1).astype(np.int64)
    if np.any(evaluation_counts == 0):
        raise ValueError("shift_rule must estimate every gradient coordinate.")
    owners = np.argmax(active, axis=0).astype(np.int64)
    return shifts, weights, owners, evaluation_counts


class RosalinOptimizer(Optimizer):
    r"""ROSALIN with weighted-random sampling and adaptive shot allocation.

    Each iteration evaluates the current loss and the program's complete
    parameter-shift recipe in one batch. The gradient coordinates receive
    independent adaptive sample counts; generalised rules may use more than two
    evaluations per coordinate. The current loss uses ``min_shots`` as
    iteration telemetry; noisy current-point estimates are never compared to
    select a best iterate. All samples count toward ``total_shots``.

    See :ref:`rosalin-optimizer` for algorithm background, primary references,
    and selection guidance.

    Args:
        learning_rate: Constant gradient-descent step size :math:`\alpha`.
        total_shots: Maximum estimator-sample budget for the whole run. This is
            distinct from the backend's per-circuit ``shots`` setting. It is a
            hard ceiling, not a target: ``max_iterations`` may terminate the
            run with budget remaining.
        lipschitz: Gradient Lipschitz bound :math:`L`. The ROSALIN paper
            recommends the Hamiltonian coefficient L1 norm.
        min_shots: Persistent minimum samples allocated to every expectation
            estimate. Must be at least two to estimate variance.
        ema_decay: Exponential-moving-average decay :math:`\mu`.
        bias: Positive stabiliser :math:`b` in the allocation denominator.
        allocation: Adaptive allocation rule. ``"icans"`` optimises each
            gradient coordinate independently; ``"gcans"`` optimises expected
            improvement per shot across the complete gradient.
    """

    requires_variance = True

    def __init__(
        self,
        learning_rate: float,
        total_shots: int,
        lipschitz: float,
        min_shots: int = 2,
        ema_decay: float = 0.99,
        bias: float = 1e-6,
        allocation: Literal["icans", "gcans"] = "icans",
    ):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if isinstance(total_shots, bool) or not isinstance(total_shots, int):
            raise TypeError("total_shots must be an integer.")
        if total_shots < 1:
            raise ValueError(f"total_shots must be positive, got {total_shots}.")
        if lipschitz <= 0:
            raise ValueError(f"lipschitz must be positive, got {lipschitz}.")
        if learning_rate >= 2 / lipschitz:
            raise ValueError(
                "learning_rate must be strictly smaller than 2 / lipschitz."
            )
        if isinstance(min_shots, bool) or not isinstance(min_shots, int):
            raise TypeError("min_shots must be an integer.")
        if min_shots < 2:
            raise ValueError(f"min_shots must be >= 2, got {min_shots}.")
        if not 0 <= ema_decay < 1:
            raise ValueError(
                f"ema_decay must satisfy 0 <= ema_decay < 1, got {ema_decay}."
            )
        if bias <= 0:
            raise ValueError(f"bias must be positive, got {bias}.")
        if allocation not in ("icans", "gcans"):
            raise ValueError(
                "allocation must be either 'icans' or 'gcans', " f"got {allocation!r}."
            )

        self.learning_rate = float(learning_rate)
        self.total_shots = total_shots
        self.lipschitz = float(lipschitz)
        self.min_shots = min_shots
        self.ema_decay = float(ema_decay)
        self.bias = float(bias)
        self.allocation = allocation
        self._theta: npt.NDArray[np.float64] | None = None
        self._shots: npt.NDArray[np.int64] | None = None
        self._gradient_ema: npt.NDArray[np.float64] | None = None
        self._variance_ema: npt.NDArray[np.float64] | None = None
        self._last_x: npt.NDArray[np.float64] | None = None
        self._last_fun = np.inf
        self._shots_used = 0
        self._nit = 0

    @property
    def n_param_sets(self) -> int:
        """Number of independent iterates optimised at once."""
        return 1

    def validate_program(self, program) -> None:
        """Require the weighted-random estimator used by ROSALIN."""
        if program._shot_distribution != "weighted_random":
            raise ValueError(
                "RosalinOptimizer requires shot_distribution='weighted_random'."
            )

    def build_evaluators(self, program) -> dict[str, Callable[..., Any]]:
        """Bind the program's generalised parameter-shift recipe."""
        return {"shift_rule": lambda: program._grad_shift_rule}

    def optimize(
        self,
        cost_fn: Callable[..., Any],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """Optimise until the shot budget or ``max_iterations`` is exhausted."""
        max_iterations = self._resolve_max_iterations(kwargs)
        shift_rule = kwargs.pop("shift_rule", None)
        kwargs.pop("rng", None)
        kwargs.pop("jac", None)
        kwargs.pop("metric_fn", None)
        if not bool(getattr(cost_fn, "supports_variance", False)):
            raise TypeError(
                "RosalinOptimizer requires a variance-aware cost function that "
                "declares supports_variance=True."
            )

        if self._theta is None:
            if initial_params is None:
                raise ValueError("RosalinOptimizer requires initial_params.")
            theta = np.atleast_1d(
                np.asarray(initial_params, dtype=np.float64).squeeze()
            )
            if theta.ndim != 1:
                raise ValueError(
                    "initial_params must contain exactly one parameter set."
                )
        else:
            theta = self._theta.copy()
        n_params = theta.size
        if n_params == 0:
            raise ValueError("initial_params must contain at least one parameter.")
        rule = _default_shift_rule(n_params) if shift_rule is None else shift_rule()
        rule_shifts, rule_weights = rule
        shifts, weights, owners, evaluation_counts = _shift_rule_layout(
            rule_shifts, rule_weights, n_params
        )
        minimum_iteration_cost = self.min_shots * (1 + int(np.sum(evaluation_counts)))
        if self.total_shots < minimum_iteration_cost:
            raise ValueError(
                "total_shots is too small for one ROSALIN iteration: "
                f"need at least {minimum_iteration_cost}, got {self.total_shots}."
            )

        if self._shots is None:
            shots = np.full(n_params, self.min_shots, dtype=np.int64)
            gradient_ema = np.zeros(n_params, dtype=np.float64)
            variance_ema = np.zeros(n_params, dtype=np.float64)
            last_x = theta.copy()
            last_fun = np.inf
            shots_used = 0
            nit = 0
        else:
            if (
                self._gradient_ema is None
                or self._variance_ema is None
                or self._last_x is None
            ):
                raise RuntimeError("ROSALIN checkpoint state is incomplete.")
            if self._shots.shape != (n_params,):
                raise ValueError(
                    "The resumed ROSALIN state does not match the parameter count."
                )
            shots = self._shots.copy()
            gradient_ema = self._gradient_ema.copy()
            variance_ema = self._variance_ema.copy()
            last_x = self._last_x.copy()
            last_fun = self._last_fun
            shots_used = self._shots_used
            nit = self._nit

        end_iteration = nit + max_iterations
        while nit < end_iteration:
            remaining_shots = self.total_shots - shots_used
            max_gradient_cost = remaining_shots - self.min_shots
            if max_gradient_cost < self.min_shots * int(np.sum(evaluation_counts)):
                break
            shots = _fit_shots_to_budget(
                shots,
                evaluation_counts=evaluation_counts,
                max_cost=max_gradient_cost,
                min_shots=self.min_shots,
            )
            evaluated_shots = shots.copy()
            evaluation_samples = shots[owners]
            iteration_cost = self.min_shots + int(np.sum(evaluation_samples))

            shifted = np.vstack((theta, shifts + theta))
            estimator_samples = (self.min_shots, *evaluation_samples.tolist())

            values, variances = cost_fn(
                shifted,
                estimator_samples=estimator_samples,
                return_variance=True,
            )

            values = np.asarray(values, dtype=np.float64).reshape(-1)
            variances = np.asarray(variances, dtype=np.float64).reshape(-1)
            if values.size != shifts.shape[0] + 1 or variances.size != values.size:
                raise ValueError(
                    "The ROSALIN cost function returned an unexpected number of "
                    "values or variances."
                )

            fun = float(values[0])
            gradient = weights @ values[1:]
            single_shot_variance = np.square(weights) @ (
                evaluation_samples * variances[1:]
            )
            finite_gradient = np.where(np.isfinite(gradient), gradient, 0.0)
            finite_variance = np.where(
                np.isfinite(single_shot_variance),
                np.maximum(single_shot_variance, 0.0),
                variance_ema,
            )

            shots_used += iteration_cost
            nit += 1
            last_fun = fun
            last_x = theta.copy()
            theta = theta - self.learning_rate * finite_gradient

            gradient_ema = (
                self.ema_decay * gradient_ema + (1 - self.ema_decay) * finite_gradient
            )
            variance_ema = (
                self.ema_decay * variance_ema + (1 - self.ema_decay) * finite_variance
            )
            correction = 1 - self.ema_decay**nit
            corrected_gradient = gradient_ema / correction
            corrected_variance = variance_ema / correction
            allocation_kwargs = {
                "learning_rate": self.learning_rate,
                "lipschitz": self.lipschitz,
                "bias_term": self.bias * self.ema_decay ** (nit - 1),
                "min_shots": self.min_shots,
                "evaluation_counts": evaluation_counts,
            }
            if self.allocation == "icans":
                shots, _ = _icans_shots_and_gains(
                    corrected_gradient,
                    corrected_variance,
                    **allocation_kwargs,
                )
            else:
                shots = _gcans_shots(
                    corrected_gradient,
                    corrected_variance,
                    **allocation_kwargs,
                )

            self._theta = theta.copy()
            self._shots = shots.copy()
            self._gradient_ema = gradient_ema.copy()
            self._variance_ema = variance_ema.copy()
            self._last_x = last_x.copy()
            self._last_fun = last_fun
            self._shots_used = shots_used
            self._nit = nit

            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(
                        x=np.atleast_2d(last_x.copy()),
                        fun=np.atleast_1d(fun),
                        jac=np.atleast_2d(finite_gradient.copy()),
                        gradient_variance=finite_variance.copy(),
                        shots_per_parameter=evaluated_shots,
                        shots_used=shots_used,
                        nit=nit,
                        track_best=False,
                        success=True,
                        message="Optimisation in progress.",
                    )
                )

        budget_exhausted = self.total_shots - shots_used < minimum_iteration_cost
        message = (
            "Optimisation terminated: total_shots budget exhausted."
            if budget_exhausted
            else "Optimisation terminated: reached max_iterations."
        )
        return OptimizeResult(
            x=last_x,
            fun=np.atleast_1d(last_fun),
            nit=nit,
            shots_used=shots_used,
            success=np.isfinite(last_fun),
            message=message,
        )

    @property
    def supports_checkpointing(self) -> bool:
        """ROSALIN persists its iterate, budget, and adaptive allocation state."""
        return True

    def get_config(self) -> dict[str, Any]:
        """Return constructor configuration for checkpoint reconstruction."""
        return {
            "type": type(self).__name__,
            "learning_rate": self.learning_rate,
            "total_shots": self.total_shots,
            "lipschitz": self.lipschitz,
            "min_shots": self.min_shots,
            "ema_decay": self.ema_decay,
            "bias": self.bias,
            "allocation": self.allocation,
        }

    def save_state(self, checkpoint_dir: Path | str) -> None:
        """Persist configuration and adaptive state after a completed iteration."""
        if (
            self._theta is None
            or self._shots is None
            or self._gradient_ema is None
            or self._variance_ema is None
            or self._last_x is None
        ):
            raise RuntimeError(
                "Cannot save checkpoint: ROSALIN optimization has not been run."
            )

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state = {
            "config": {
                key: value for key, value in self.get_config().items() if key != "type"
            },
            "theta": self._theta.tolist(),
            "shots": self._shots.tolist(),
            "gradient_ema": self._gradient_ema.tolist(),
            "variance_ema": self._variance_ema.tolist(),
            "last_x": self._last_x.tolist(),
            "last_fun": self._last_fun,
            "shots_used": self._shots_used,
            "nit": self._nit,
        }
        _atomic_write(
            checkpoint_path / OPTIMIZER_STATE_FILE,
            json.dumps(state, indent=2),
        )

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> Self:
        """Restore a ROSALIN optimizer from a checkpoint directory."""
        state_file = Path(checkpoint_dir) / OPTIMIZER_STATE_FILE
        with open(state_file) as file:
            state = json.load(file)

        optimizer = cls(**state["config"])
        optimizer._theta = np.asarray(state["theta"], dtype=np.float64)
        optimizer._shots = np.asarray(state["shots"], dtype=np.int64)
        optimizer._gradient_ema = np.asarray(state["gradient_ema"], dtype=np.float64)
        optimizer._variance_ema = np.asarray(state["variance_ema"], dtype=np.float64)
        optimizer._last_x = np.asarray(state["last_x"], dtype=np.float64)
        optimizer._last_fun = float(state["last_fun"])
        optimizer._shots_used = int(state["shots_used"])
        optimizer._nit = int(state["nit"])
        return optimizer

    def reset(self) -> None:
        """Clear the iterate, budget usage, and adaptive allocation state."""
        self._theta = None
        self._shots = None
        self._gradient_ema = None
        self._variance_ema = None
        self._last_x = None
        self._last_fun = np.inf
        self._shots_used = 0
        self._nit = 0

    def copy(self) -> Self:
        """Return a fresh optimizer with the same constructor configuration."""
        config = self.get_config()
        config.pop("type")
        return type(self)(**config)
