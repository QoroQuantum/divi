# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Attribution: Periodic wrapping follows orqviz (Zapata Engineering, Apache-2.0).
# See LICENSES/ORQViz-Apache-2.0-acknowledgement.txt.

"""Periodic parameter wrapping for variational-program trajectories.

Quantum gate parameters are typically :math:`2\\pi`-periodic.  When an
optimization trajectory crosses the period boundary, PCA can see an
artificial jump and produce distorted landscapes.  The functions here
re-center each parameter vector to its closest periodic copy relative to a
reference point, ensuring continuity along the trajectory.
"""

import numpy as np
import numpy.typing as npt


def periodic_wrap(
    point: npt.ArrayLike,
    reference: npt.ArrayLike,
    period: float = 2 * np.pi,
) -> npt.NDArray[np.float64]:
    """Wrap *point* to the closest periodic copy relative to *reference*.

    For each element, the returned value satisfies
    ``|wrapped[i] - reference[i]| <= period / 2`` (up to floating-point
    rounding).  This is the element-wise equivalent of ``orqviz``'s
    ``relative_periodic_wrap``.

    Args:
        point: Parameter vector to wrap.
        reference: Reference parameter vector.
        period: Periodicity of each parameter.  Defaults to :math:`2\\pi`.

    Returns:
        Wrapped copy of *point*.
    """
    p = np.asarray(point, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    # Shift so that reference is at zero, wrap into [-period/2, period/2), shift back.
    diff = (p - r + period / 2) % period - period / 2
    return r + diff


def periodic_trajectory_wrap(
    trajectory: npt.ArrayLike,
    period: float = 2 * np.pi,
) -> npt.NDArray[np.float64]:
    """Unwrap a trajectory so consecutive rows are within half a period.

    Iterates forward through the rows, wrapping each row relative to its
    predecessor.  The result is a continuous trajectory suitable for PCA.

    Args:
        trajectory: Parameter vectors of shape ``(n_steps, n_params)``.
        period: Periodicity of each parameter.  Defaults to :math:`2\\pi`.

    Returns:
        Unwrapped copy of *trajectory* with the same shape.

    Raises:
        ValueError: If *trajectory* is not 2-D or has fewer than two rows.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError("trajectory must be a 2-D array of shape (n_steps, n_params).")
    if traj.shape[0] < 2:
        raise ValueError("trajectory must contain at least two rows.")

    out = np.empty_like(traj)
    out[0] = traj[0]
    for k in range(1, traj.shape[0]):
        out[k] = periodic_wrap(traj[k], out[k - 1], period)
    return out
