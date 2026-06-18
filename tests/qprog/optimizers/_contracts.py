# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared optimizer-test infrastructure: cost functions, the single source of
optimizer variants, and behavioural save/load (checkpointing) contracts.

The variant lists are the *single source of truth* consumed both by the
cross-suite ``optimizer`` fixture (``tests/qprog/conftest.py``) and by the
optimizer-contract suite, so the two can no longer drift apart.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import CheckpointNotFoundError
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    Optimizer,
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)

# --------------------------------------------------------------------------- #
# Cost functions
# --------------------------------------------------------------------------- #


def sphere_cost_fn_population(params: np.ndarray) -> np.ndarray:
    """Sphere cost function (sum of squares) for a population of parameter sets."""
    if params.ndim != 2:
        raise ValueError(
            "Input params for population cost function must be a 2D array."
        )
    return np.sum(params**2, axis=1)


def sphere_cost_fn_single(params: np.ndarray) -> float:
    """Sphere cost function (sum of squares) for a single parameter set."""
    if params.ndim != 1:
        # Allow single-row 2D array for convenience with some optimizers
        if params.ndim == 2 and params.shape[0] == 1:
            params = params.squeeze(0)
        else:
            raise ValueError(
                "Input params for single cost function must be a 1D array."
            )
    return float(np.sum(params**2))


def sphere_cost_fn_batch_aware(params: np.ndarray) -> float | np.ndarray:
    """Sphere cost returning a scalar for a single param set and a vector for a
    2-D batch — for the gradient/metric optimizers (SPSA, QN-SPSA, QNG) that
    evaluate perturbations as one batch but report a single loss."""
    params = np.atleast_2d(params)
    values = np.sum(params**2, axis=1)
    return values if params.shape[0] > 1 else float(values[0])


# --------------------------------------------------------------------------- #
# Optimizer variants — single source of truth
# --------------------------------------------------------------------------- #

#: Every supported optimizer variant (drives the cross-suite ``optimizer`` fixture).
OPTIMIZER_VARIANTS = [
    ("monte-carlo", lambda: MonteCarloOptimizer(population_size=5, n_best_sets=2)),
    ("l-bfgs-b", lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B)),
    ("cobyla", lambda: ScipyOptimizer(method=ScipyMethod.COBYLA)),
    ("nelder-mead", lambda: ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)),
    ("cmaes", lambda: PymooOptimizer(method=PymooMethod.CMAES, population_size=10)),
    ("de", lambda: PymooOptimizer(method=PymooMethod.DE, population_size=5)),
]

#: Variants that support save/load checkpointing (subset of OPTIMIZER_VARIANTS).
CHECKPOINTING_VARIANT_IDS = {"monte-carlo", "cmaes", "de"}

#: Contract suite extends the base set with a MonteCarlo "keep best" mode.
CONTRACT_VARIANTS = OPTIMIZER_VARIANTS + [
    (
        "monte-carlo-keep-best",
        lambda: MonteCarloOptimizer(
            population_size=10, n_best_sets=3, keep_best_params=True
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Save/load (checkpointing) contracts
# --------------------------------------------------------------------------- #


def verify_save_creates_checkpoint_file(
    optimizer: Optimizer,
    initial_params: np.ndarray,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    tmp_path: Path,
    load_state: Callable[[str], Optimizer],
) -> None:
    optimizer.optimize(cost_fn, initial_params, max_iterations=2, rng=rng)
    checkpoint_dir = str(tmp_path / "checkpoint")
    optimizer.save_state(checkpoint_dir)
    state_file = tmp_path / "checkpoint" / "optimizer_state.json"
    assert state_file.exists(), "Checkpoint file should be created"


def verify_load_state_raises_file_not_found(
    load_state: Callable[[str], Optimizer],
    tmp_path: Path,
) -> None:
    checkpoint_dir = str(tmp_path / "nonexistent_checkpoint")
    with pytest.raises(CheckpointNotFoundError, match="Checkpoint file not found"):
        load_state(checkpoint_dir)


def verify_save_creates_directory_if_needed(
    optimizer: Optimizer,
    initial_params: np.ndarray,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    tmp_path: Path,
) -> None:
    optimizer.optimize(cost_fn, initial_params, max_iterations=2, rng=rng)
    checkpoint_dir = str(tmp_path / "nested" / "checkpoint")
    optimizer.save_state(checkpoint_dir)
    state_file = tmp_path / "nested" / "checkpoint" / "optimizer_state.json"
    assert state_file.exists(), "Checkpoint directory and file should be created"


def verify_save_load_round_trip(
    optimizer: Optimizer,
    initial_params: np.ndarray,
    cost_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    tmp_path: Path,
    load_state: Callable[[str], Optimizer],
    n_params: int,
) -> None:
    optimizer.optimize(cost_fn, initial_params, max_iterations=2, rng=rng)
    checkpoint_dir = str(tmp_path / "checkpoint")
    optimizer.save_state(checkpoint_dir)
    loaded_optimizer = load_state(checkpoint_dir)
    result = loaded_optimizer.optimize(
        cost_fn, initial_params, max_iterations=5, rng=rng
    )
    assert isinstance(result, OptimizeResult)
    assert result.x.shape == (n_params,)
    assert np.isfinite(result.fun)


def verify_save_without_prior_run_raises(
    optimizer: Optimizer,
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="optimization has not been run"):
        optimizer.save_state(str(tmp_path / "checkpoint"))
