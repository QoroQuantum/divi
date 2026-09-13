# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared builders, constants, and cost functions for optimizer tests."""

import numpy as np

#: Eigenvalues of the bowl the noisy contracts descend; condition number 10. Its
#: length fixes the parameter count for every ``noisy_bowl`` run.
BOWL_CURVATURE = np.logspace(0.0, 1.0, 4)


def sphere_cost_fn_population(params: np.ndarray) -> np.ndarray:
    """Sphere cost function for a population of parameter sets."""
    if params.ndim != 2:
        raise ValueError(
            "Input params for population cost function must be a 2D array."
        )
    return np.sum(params**2, axis=1)


def sphere_cost_fn_single(params: np.ndarray) -> float:
    """Sphere cost function for a single parameter set."""
    if params.ndim != 1:
        if params.ndim == 2 and params.shape[0] == 1:
            params = params.squeeze(0)
        else:
            raise ValueError(
                "Input params for single cost function must be a 1D array."
            )
    return float(np.sum(params**2))


def sphere_cost_fn_batch_aware(params: np.ndarray) -> float | np.ndarray:
    """Sphere cost supporting both one parameter set and a 2-D batch."""
    params = np.atleast_2d(params)
    values = np.sum(params**2, axis=1)
    return values if params.shape[0] > 1 else float(values[0])


def noisy_bowl(rng, sigma=0.0, dropout=0.0):
    """Return a noisy quadratic cost and its noise-free scoring function."""

    def true_cost(params):
        rows = np.atleast_2d(np.asarray(params, dtype=np.float64))
        values = 0.5 * np.einsum("ij,j,ij->i", rows, BOWL_CURVATURE, rows)
        return values if rows.shape[0] > 1 else float(values[0])

    def cost_fn(params):
        rows = np.atleast_2d(np.asarray(params, dtype=np.float64))
        values = 0.5 * np.einsum("ij,j,ij->i", rows, BOWL_CURVATURE, rows)
        if sigma:
            values = values + rng.normal(0.0, sigma, size=values.shape)
        if dropout:
            values = np.where(rng.random(values.shape) < dropout, np.nan, values)
        return values if rows.shape[0] > 1 else float(values[0])

    return cost_fn, true_cost


def bowl_jac(params: np.ndarray) -> np.ndarray:
    """Exact gradient of :func:`noisy_bowl`'s noise-free quadratic."""
    return BOWL_CURVATURE * np.asarray(params, dtype=np.float64)


def bowl_metric(_params: np.ndarray) -> np.ndarray:
    """Positive-definite metric matching the bowl curvature."""
    return np.diag(BOWL_CURVATURE)
