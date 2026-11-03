# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Names of the files and directories that make up a checkpoint."""

from pathlib import Path

PROGRAM_STATE_FILE = "program_state.json"
PROGRAM_COMPLETION_FILE = "program_completion.json"
OPTIMIZER_STATE_FILE = "optimizer_state.json"
SUBDIR_PREFIX = "checkpoint_"

# Maximum reasonable iteration number (prevents parsing errors from corrupted names)
_MAX_ITERATION_NUMBER = 1_000_000


def _get_checkpoint_subdir_name(iteration: int) -> str:
    """Generate checkpoint subdirectory name for a given iteration.

    Args:
        iteration (int): Iteration number.

    Returns:
        str: Subdirectory name (e.g., "checkpoint_001").
    """
    return f"{SUBDIR_PREFIX}{iteration:03d}"


def _extract_iteration_from_subdir(subdir_name: str) -> int | None:
    """Extract iteration number from checkpoint subdirectory name.

    Args:
        subdir_name (str): Subdirectory name (e.g., "checkpoint_001").

    Returns:
        int | None: Iteration number if valid and reasonable, None otherwise.
    """
    if not subdir_name.startswith(SUBDIR_PREFIX):
        return None
    suffix = subdir_name[len(SUBDIR_PREFIX) :]
    if not suffix.isdigit():
        return None
    iteration = int(suffix)
    if iteration > _MAX_ITERATION_NUMBER:
        return None
    return iteration


def get_checkpoint_subdir_path(main_dir: Path, iteration: int) -> Path:
    """Get the path to a checkpoint subdirectory for a given iteration.

    Args:
        main_dir (Path): Main checkpoint directory.
        iteration (int): Iteration number.

    Returns:
        Path: Path to the checkpoint subdirectory.
    """
    subdir_name = _get_checkpoint_subdir_name(iteration)
    return main_dir / subdir_name
