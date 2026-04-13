# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient computation strategies for :mod:`divi.viz`.

Provides finite-difference and parameter-shift gradient methods, used by
:func:`~divi.viz.compute_hessian` and :func:`~divi.viz.run_neb`.
"""

from enum import Enum

import numpy as np
import numpy.typing as npt


class GradientMethod(str, Enum):
    """Strategy for computing gradients in viz analysis functions.

    ``PARAMETER_SHIFT`` uses the parameter-shift rule (shift = π/2,
    exact for standard quantum gates).  ``FINITE_DIFFERENCE`` uses centered
    finite differences with a configurable step size ``eps``.
    """

    PARAMETER_SHIFT = "parameter_shift"
    FINITE_DIFFERENCE = "finite_difference"


_PARAM_SHIFT = 0.5 * np.pi


def _finite_difference_gradients(
    evaluate_fn,
    pivots: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    """Compute gradients via centered finite differences (shift = *eps*)."""
    m, d = pivots.shape
    eye = eps * np.eye(d, dtype=np.float64)

    pivots_exp = pivots[:, np.newaxis, :]  # (m, 1, d)
    plus = (pivots_exp + eye).reshape(m * d, d)
    minus = (pivots_exp - eye).reshape(m * d, d)

    probes = np.empty((2 * m * d, d), dtype=np.float64)
    probes[0::2] = plus
    probes[1::2] = minus

    losses = evaluate_fn(probes)
    return ((losses[0::2] - losses[1::2]) / (2.0 * eps)).reshape(m, d)


def _parameter_shift_gradients(
    evaluate_fn,
    pivots: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute gradients via the parameter-shift rule (shift = π/2)."""
    m, d = pivots.shape
    eye = _PARAM_SHIFT * np.eye(d, dtype=np.float64)

    pivots_exp = pivots[:, np.newaxis, :]
    plus = (pivots_exp + eye).reshape(m * d, d)
    minus = (pivots_exp - eye).reshape(m * d, d)

    probes = np.empty((2 * m * d, d), dtype=np.float64)
    probes[0::2] = plus
    probes[1::2] = minus

    losses = evaluate_fn(probes)
    return (0.5 * (losses[0::2] - losses[1::2])).reshape(m, d)


def _compute_gradients(
    evaluate_fn,
    pivots: npt.NDArray[np.float64],
    method: GradientMethod,
    eps: float,
) -> npt.NDArray[np.float64]:
    """Dispatch gradient computation to the chosen method."""
    if method is GradientMethod.PARAMETER_SHIFT:
        return _parameter_shift_gradients(evaluate_fn, pivots)
    return _finite_difference_gradients(evaluate_fn, pivots, eps)
