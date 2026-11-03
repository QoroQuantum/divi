# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Locating, describing and loading checkpoints."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import CheckpointCorruptedError, CheckpointNotFoundError
from .names import (
    OPTIMIZER_STATE_FILE,
    PROGRAM_STATE_FILE,
    SUBDIR_PREFIX,
    _extract_iteration_from_subdir,
)


def _validate_checkpoint_json(
    path: Path, required_fields: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validate that a checkpoint JSON file exists and is valid.

    Args:
        path (Path): Path to the JSON file.
        required_fields (list[str] | None): List of required top-level fields.

    Returns:
        dict[str, Any]: Parsed JSON data.

    Raises:
        CheckpointNotFoundError: If the file does not exist.
        CheckpointCorruptedError: If the file is invalid JSON or missing required fields.
    """
    if not path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint file not found: {path}",
            main_dir=path.parent,
        )

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CheckpointCorruptedError(
            f"Checkpoint file is not valid JSON: {path}",
            file_path=path,
            details=f"JSON decode error: {e}",
        ) from e
    except Exception as e:
        raise CheckpointCorruptedError(
            f"Failed to read checkpoint file: {path}",
            file_path=path,
            details=str(e),
        ) from e

    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise CheckpointCorruptedError(
                f"Checkpoint file is missing required fields: {path}",
                file_path=path,
                details=f"Missing fields: {', '.join(missing_fields)}",
            )

    return data


def load_and_validate_pydantic_model(
    path: Path,
    model_class: type,
    required_fields: Sequence[str] | None = None,
    error_context: str | None = None,
) -> Any:
    """Load and validate a checkpoint JSON file with a Pydantic model.

    This function combines JSON validation, conversion to string, and Pydantic
    model validation into a single operation.

    Args:
        path (Path): Path to the JSON file.
        model_class (type): Pydantic model class to validate against.
        required_fields (list[str] | None): List of required top-level JSON fields.
        error_context (str | None): Additional context for error messages (e.g., "Program state" or "Pymoo optimizer state").

    Returns:
        Any: Validated Pydantic model instance.

    Raises:
        CheckpointNotFoundError: If the file does not exist.
        CheckpointCorruptedError: If the file is invalid JSON, missing required fields, or fails Pydantic validation.
    """
    json_data = json.dumps(
        _validate_checkpoint_json(path, required_fields=required_fields)
    )

    try:
        return model_class.model_validate_json(json_data)
    except Exception as e:
        context = f"{error_context} " if error_context else ""
        raise CheckpointCorruptedError(
            f"Failed to validate {context}checkpoint state: {path}",
            file_path=path,
            details=str(e),
        ) from e


def _is_checkpoint_valid(checkpoint_path: Path) -> bool:
    """Check if a checkpoint directory contains required files.

    Args:
        checkpoint_path (Path): Path to checkpoint subdirectory.

    Returns:
        bool: True if checkpoint has required files, False otherwise.
    """
    program_state = checkpoint_path / PROGRAM_STATE_FILE
    optimizer_state = checkpoint_path / OPTIMIZER_STATE_FILE
    return program_state.exists() and optimizer_state.exists()


def find_latest_checkpoint_subdir(main_dir: Path) -> Path:
    """Find the latest complete checkpoint subdirectory by iteration number.

    Incomplete checkpoints — a directory missing either state file, as left
    behind by a write interrupted midway — are skipped, so an aborted save does
    not hide the last good checkpoint.

    Args:
        main_dir (Path): Main checkpoint directory.

    Returns:
        Path: Path to the latest complete checkpoint subdirectory.

    Raises:
        CheckpointNotFoundError: If no complete checkpoint subdirectory is found.
    """
    iteration_by_dir = {
        d: iteration
        for d in main_dir.iterdir()
        if d.is_dir()
        and (iteration := _extract_iteration_from_subdir(d.name)) is not None
    }
    if not iteration_by_dir:
        raise CheckpointNotFoundError(
            f"No checkpoint subdirectories found in {main_dir}",
            main_dir=main_dir,
            available_directories=[d.name for d in main_dir.iterdir() if d.is_dir()],
        )
    checkpoint_dirs = sorted(iteration_by_dir, key=iteration_by_dir.__getitem__)

    complete_dirs = [d for d in checkpoint_dirs if _is_checkpoint_valid(d)]
    if not complete_dirs:
        raise CheckpointNotFoundError(
            f"No complete checkpoint found in {main_dir}: every checkpoint "
            f"subdirectory is missing {PROGRAM_STATE_FILE} or "
            f"{OPTIMIZER_STATE_FILE}.",
            main_dir=main_dir,
            available_directories=[d.name for d in checkpoint_dirs],
        )

    return complete_dirs[-1]


def resolve_checkpoint_path(
    main_dir: Path | str, subdirectory: str | None = None
) -> Path:
    """Resolve the path to a checkpoint subdirectory.

    Args:
        main_dir (Path | str): Main checkpoint directory.
        subdirectory (str | None): Specific checkpoint subdirectory to load
            (e.g., "checkpoint_001"). If None, loads the latest checkpoint
            based on iteration number.

    Returns:
        Path: Path to the checkpoint subdirectory.

    Raises:
        CheckpointNotFoundError: If the main directory or specified subdirectory
            does not exist.
    """
    main_path = Path(main_dir)
    if not main_path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint directory not found: {main_path}",
            main_dir=main_path,
        )

    if subdirectory is None:
        checkpoint_path = find_latest_checkpoint_subdir(main_path)
    else:
        checkpoint_path = main_path / subdirectory
        if not checkpoint_path.exists():
            # Provide helpful error with available checkpoints
            available = [
                d.name
                for d in main_path.iterdir()
                if d.is_dir() and d.name.startswith(SUBDIR_PREFIX)
            ]
            raise CheckpointNotFoundError(
                f"Checkpoint subdirectory not found: {checkpoint_path}",
                main_dir=main_path,
                available_directories=available,
            )

    return checkpoint_path


@dataclass(frozen=True)
class CheckpointInfo:
    """Information about a checkpoint.

    Attributes:
        path: Path to the checkpoint subdirectory.
        iteration: Iteration number of this checkpoint.
        timestamp: Modification time of the checkpoint directory.
        size_bytes: Total size of the checkpoint in bytes.
        is_valid: Whether the checkpoint is valid (has required files).
    """

    path: Path
    iteration: int
    timestamp: datetime
    size_bytes: int
    is_valid: bool


def _calculate_checkpoint_size(checkpoint_path: Path) -> int:
    """Calculate total size of a checkpoint directory in bytes.

    Args:
        checkpoint_path (Path): Path to checkpoint subdirectory.

    Returns:
        int: Total size in bytes.
    """
    if not checkpoint_path.exists():
        return 0
    return sum(
        file_path.stat().st_size
        for file_path in checkpoint_path.rglob("*")
        if file_path.is_file()
    )


def get_checkpoint_info(checkpoint_path: Path) -> CheckpointInfo:
    """Get information about a checkpoint.

    Args:
        checkpoint_path (Path): Path to the checkpoint subdirectory.

    Returns:
        CheckpointInfo: Information about the checkpoint.

    Raises:
        CheckpointNotFoundError: If the checkpoint directory does not exist.
    """
    if not checkpoint_path.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}",
            main_dir=checkpoint_path.parent,
        )

    if not checkpoint_path.is_dir():
        raise CheckpointNotFoundError(
            f"Checkpoint path is not a directory: {checkpoint_path}",
            main_dir=checkpoint_path.parent,
        )

    iteration = _extract_iteration_from_subdir(checkpoint_path.name)
    if iteration is None:
        raise ValueError(
            f"Invalid checkpoint directory name: {checkpoint_path.name}. "
            f"Expected format: {SUBDIR_PREFIX}XXX"
        )

    mtime = checkpoint_path.stat().st_mtime
    timestamp = datetime.fromtimestamp(mtime)
    size_bytes = _calculate_checkpoint_size(checkpoint_path)
    is_valid = _is_checkpoint_valid(checkpoint_path)

    return CheckpointInfo(
        path=checkpoint_path,
        iteration=iteration,
        timestamp=timestamp,
        size_bytes=size_bytes,
        is_valid=is_valid,
    )


def list_checkpoints(main_dir: Path) -> list[CheckpointInfo]:
    """List all checkpoints in a main checkpoint directory.

    Args:
        main_dir (Path): Main checkpoint directory.

    Returns:
        list[CheckpointInfo]: List of checkpoint information, sorted by iteration number.

    Raises:
        CheckpointNotFoundError: If the main directory does not exist.
    """
    if not main_dir.exists():
        raise CheckpointNotFoundError(
            f"Checkpoint directory not found: {main_dir}",
            main_dir=main_dir,
        )

    if not main_dir.is_dir():
        raise CheckpointNotFoundError(
            f"Path is not a directory: {main_dir}",
            main_dir=main_dir,
        )

    checkpoints = [
        get_checkpoint_info(subdir)
        for subdir in main_dir.iterdir()
        if subdir.is_dir() and _extract_iteration_from_subdir(subdir.name) is not None
    ]
    checkpoints.sort(key=lambda x: x.iteration)
    return checkpoints


def get_latest_checkpoint(main_dir: Path) -> Path | None:
    """Get the path to the latest checkpoint.

    Args:
        main_dir (Path): Main checkpoint directory.

    Returns:
        Path | None: Path to the latest checkpoint, or None if no checkpoints exist.
    """
    try:
        return find_latest_checkpoint_subdir(main_dir)
    except CheckpointNotFoundError:
        return None
