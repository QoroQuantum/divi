# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Early stopping utilities for variational quantum algorithms."""

from collections import deque
from enum import Enum

import numpy as np


class StopReason(str, Enum):
    """Reason why early stopping was triggered.

    Inherits from ``str`` so that values serialize naturally to JSON
    and can be compared directly with plain strings.
    """

    PATIENCE_EXCEEDED = "patience_exceeded"
    """Cost did not improve by at least ``min_delta`` for ``patience`` consecutive iterations."""

    GRADIENT_BELOW_THRESHOLD = "gradient_below_threshold"
    """L2 norm of the gradient fell below ``grad_norm_threshold``."""

    COST_VARIANCE_SETTLED = "cost_variance_settled"
    """Variance of recent cost values dropped below ``variance_threshold``."""


class EarlyStopping:
    """Early stopping controller for variational quantum algorithm optimisation.

    Tracks optimisation progress and signals when to stop based on
    configurable criteria.  A single instance is created before the
    optimisation loop and :meth:`check` is called once per iteration.

    Args:
        patience: Number of consecutive iterations with no improvement
            (by at least ``min_delta``) before stopping. Must be ≥ 1.
        min_delta: Minimum absolute decrease in ``best_loss`` that counts
            as an improvement.  Must be ≥ 0.
        grad_norm_threshold: If not ``None``, stop when the L2 norm of the
            gradient drops below this value.  Only effective when the
            optimiser exposes gradient information (e.g.
            ``ScipyOptimizer`` with ``L_BFGS_B``).
        variance_window: Number of recent cost values used to compute the
            rolling variance.  Must be ≥ 2.
        variance_threshold: If not ``None``, stop when the variance of
            the last ``variance_window`` cost values drops below this value.

    Raises:
        ValueError: If any parameter violates its constraints.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        grad_norm_threshold: float | None = None,
        variance_window: int = 20,
        variance_threshold: float | None = None,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if variance_window < 2:
            raise ValueError(f"variance_window must be >= 2, got {variance_window}")

        self.patience = patience
        self.min_delta = min_delta
        self.grad_norm_threshold = grad_norm_threshold
        self.variance_window = variance_window
        self.variance_threshold = variance_threshold

        # --- Internal state ---
        self._stale_count: int = 0
        self._tracked_best: float = float("inf")
        self._loss_history: deque[float] = deque(maxlen=variance_window)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check(
        self,
        current_loss: float,
        *,
        grad_norm: float | None = None,
    ) -> StopReason | None:
        """Evaluate all enabled stopping criteria.

        Must be called **once per iteration**, after loss (and optionally
        gradient) computation.

        Args:
            current_loss: The minimum loss value observed at this iteration.
            grad_norm: L2 norm of the current gradient vector, or ``None``
                if gradient information is not available.

        Returns:
            A :class:`StopReason` if any criterion triggered, otherwise
            ``None`` (meaning optimisation should continue).
        """
        # 1. Patience --------------------------------------------------
        if current_loss < self._tracked_best - self.min_delta:
            self._tracked_best = current_loss
            self._stale_count = 0
        else:
            self._stale_count += 1

        if self._stale_count >= self.patience:
            return StopReason.PATIENCE_EXCEEDED

        # 2. Gradient norm ---------------------------------------------
        if (
            self.grad_norm_threshold is not None
            and grad_norm is not None
            and grad_norm < self.grad_norm_threshold
        ):
            return StopReason.GRADIENT_BELOW_THRESHOLD

        # 3. Cost variance ---------------------------------------------
        if self.variance_threshold is not None:
            self._loss_history.append(current_loss)
            if len(self._loss_history) >= self.variance_window:
                variance = float(np.var(self._loss_history))
                if variance < self.variance_threshold:
                    return StopReason.COST_VARIANCE_SETTLED

        return None

    def reset(self) -> None:
        """Reset internal state so the instance can be reused."""
        self._stale_count = 0
        self._tracked_best = float("inf")
        self._loss_history.clear()
