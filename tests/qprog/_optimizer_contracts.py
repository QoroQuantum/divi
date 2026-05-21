# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural contracts for optimizer save/load (checkpointing)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from divi.qprog.checkpointing import CheckpointNotFoundError
from divi.qprog.optimizers import Optimizer


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
