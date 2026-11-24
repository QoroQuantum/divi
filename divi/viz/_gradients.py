# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient strategies for the Hessian and NEB tools."""

from collections.abc import Callable
from enum import Enum

import numpy as np
import numpy.typing as npt


class GradientMethod(str, Enum):
    """Strategy for computing gradients in viz analysis functions.

    ``PROGRAM`` uses the program's own gradients and Hessian.
    ``FINITE_DIFFERENCE`` uses centred finite differences of the loss, shifting
    each parameter by ``+-eps``; the Hessian applies them twice, so its diagonal
    steps by ``2 * eps``.
    """

    PROGRAM = "program"
    FINITE_DIFFERENCE = "finite_difference"


def _compute_finite_difference_gradients(
    evaluate_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    pivots: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    """Centred-difference gradients at every pivot, in one batch."""
    m, d = pivots.shape
    shifts = np.vstack([eps * np.eye(d), -eps * np.eye(d)])
    losses = evaluate_fn((pivots[:, None, :] + shifts).reshape(-1, d)).reshape(m, 2, d)
    return (losses[:, 0] - losses[:, 1]) / (2 * eps)
