# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Writing and removing checkpoint files."""

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .names import PROGRAM_COMPLETION_FILE
from .read import list_checkpoints

if TYPE_CHECKING:
    from divi.qprog.quantum_program import QuantumProgram


def ensure_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Ensure checkpoint directory exists.

    Args:
        checkpoint_dir (Path): Directory path.

    Returns:
        Path: The checkpoint directory path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform supports it."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically using a temporary file and rename.

    This ensures that if the write is interrupted, the original file is not corrupted.

    Args:
        path (Path): Target file path.
        content (str): Content to write.

    Raises:
        OSError: If the file cannot be written.
    """
    # Create temporary file in the same directory to ensure atomic rename works
    temp_file = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_file, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(path)
        fsync_directory(path.parent)
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        raise OSError(f"Failed to write checkpoint file {path}: {e}") from e


def write_program_completion(
    program: "QuantumProgram", directory: Path | str | None
) -> None:
    """Write ``program``'s state as the directory's completion file.

    The completion file holds the program as the last ``run()`` or
    ``sample_solution()`` left it, including what it computed at the end (e.g.
    the final sample); the iteration checkpoints
    beside it keep the optimizer state. No-op without a directory, or for a
    program that cannot produce a completed checkpoint.
    """
    if directory is None:
        return
    directory = Path(directory)
    checkpoint = program._make_checkpoint(directory)
    if checkpoint is None:
        return
    atomic_write(
        ensure_checkpoint_dir(directory) / PROGRAM_COMPLETION_FILE,
        checkpoint.model_dump_json(indent=2),
    )


def cleanup_old_checkpoints(main_dir: Path, keep_last_n: int) -> None:
    """Remove old checkpoints, keeping only the most recent N.

    Args:
        main_dir (Path): Main checkpoint directory.
        keep_last_n (int): Number of most recent checkpoints to keep.

    Raises:
        ValueError: If keep_last_n is less than 1.
        CheckpointNotFoundError: If the main directory does not exist.
    """
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least 1")

    checkpoints = list_checkpoints(main_dir)

    if len(checkpoints) <= keep_last_n:
        return

    checkpoints.sort(key=lambda x: x.iteration, reverse=True)
    to_remove = checkpoints[keep_last_n:]

    for checkpoint_info in to_remove:
        shutil.rmtree(checkpoint_info.path)
