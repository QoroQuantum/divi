# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve as _solve_linear_system


def _regularized_solve(
    grad: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
    *,
    solver: Literal["tikhonov", "pinv"],
    regularization: float,
    scale_regularization: bool,
    rcond: float,
) -> npt.NDArray[np.float64]:
    """Solve ``metric @ delta = grad`` with PSD-aware regularization.

    Symmetrizes ``metric`` defensively against round-off. ``"tikhonov"`` solves
    ``(G + λI) delta = grad`` via a Cholesky-based symmetric solve (``λ`` scaled
    by ``max(1, mean(diag(G)))`` when ``scale_regularization``); ``"pinv"`` applies
    the Moore-Penrose pseudo-inverse with cutoff ``rcond``.
    """
    grad = np.asarray(grad, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    metric = 0.5 * (metric + metric.T)

    if solver == "pinv":
        return np.linalg.pinv(metric, rcond=rcond) @ grad

    lam = regularization
    if scale_regularization:
        lam *= max(1.0, float(np.mean(np.diag(metric))))
    damped = metric + lam * np.eye(metric.shape[0])
    try:
        return _solve_linear_system(damped, grad, assume_a="pos")
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Regularized natural-gradient solve failed: the damped metric "
            "(G + λI) is not positive-definite — the metric is rank-deficient "
            "and regularization is too small to lift it (λ=0 leaves it "
            "singular). Raise `regularization` or use solver='pinv'."
        ) from exc


def _matrix_abs_psd(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Matrix absolute value ``V |Λ| Vᵀ`` of a symmetric matrix.

    Eigendecomposes the symmetrized matrix and takes the absolute value of its
    eigenvalues, yielding a real positive-semidefinite result — QN-SPSA's
    conditioning of a noisy, possibly-indefinite metric estimate before the
    identity shift. Decomposing the matrix directly (rather than ``GᵀG``) avoids
    squaring the condition number.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    return (eigvecs * np.abs(eigvals)) @ eigvecs.T
