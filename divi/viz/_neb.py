# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Attribution: NEB algorithm follows patterns from orqviz (Zapata Engineering,
# Apache-2.0) and arXiv:1803.00885. See LICENSES/ORQViz-Apache-2.0-acknowledgement.txt.

"""NEB helper functions (chain redistribution, tangent projection).

The public :func:`divi.viz.run_neb` entry point and :class:`divi.viz.NEBResult`
live in ``_api.py`` and ``_results.py`` respectively.  Gradient computation
lives in ``_gradients.py``.
"""

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def _cumulative_distances(chain: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalised cumulative Euclidean distance along a chain."""
    diffs = np.diff(chain, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total > 0:
        cum /= total
    cum[-1] = 1.0  # numerical stability
    return cum.astype(np.float64)


def _redistribute_uniform(
    chain: npt.NDArray[np.float64], n_pivots: int
) -> npt.NDArray[np.float64]:
    """Redistribute *n_pivots* images uniformly along the piece-wise linear path."""
    cum = _cumulative_distances(chain)
    interp = interp1d(cum, chain, kind="linear", axis=0)
    weights = np.linspace(0.0, 1.0, n_pivots)
    return interp(weights).astype(np.float64)


def _neb_perpendicular_gradients(
    chain: npt.NDArray[np.float64],
    gradients: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Remove the tangential component from each interior gradient.

    *chain* has shape ``(n_pivots, d)`` (full chain including endpoints).
    *gradients* has shape ``(n_pivots - 2, d)`` (interior pivots only).
    """
    # Backward-difference tangents for interior pivots: chain[1:-1] - chain[:-2]
    tangents = chain[1:-1] - chain[:-2]  # (n_interior, d)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)  # (n_interior, 1)

    # Normalise; where norm ~ 0, keep tangent as zero (projection becomes identity).
    safe = norms > 1e-30
    tangents = np.where(safe, tangents / np.where(safe, norms, 1.0), 0.0)

    # Project: perp = grad - (grad . tan) * tan
    dots = np.sum(gradients * tangents, axis=1, keepdims=True)  # (n_interior, 1)
    return gradients - dots * tangents
