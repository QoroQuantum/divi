# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpointing utilities for variational quantum algorithms."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Constants for checkpoint file and directory naming
PROGRAM_STATE_FILE = "program_state.json"
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
    # Validate that iteration number is reasonable
    if iteration < 0 or iteration > _MAX_ITERATION_NUMBER:
        return None
    return iteration


def _ensure_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Ensure checkpoint directory exists.

    Args:
        checkpoint_dir (Path): Directory path.

    Returns:
        Path: The checkpoint directory path.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _get_checkpoint_subdir_path(main_dir: Path, iteration: int) -> Path:
    """Get the path to a checkpoint subdirectory for a given iteration.

    Args:
        main_dir (Path): Main checkpoint directory.
        iteration (int): Iteration number.

    Returns:
        Path: Path to the checkpoint subdirectory.
    """
    subdir_name = _get_checkpoint_subdir_name(iteration)
    return main_dir / subdir_name


def _find_latest_checkpoint_subdir(main_dir: Path) -> Path:
    """Find the latest checkpoint subdirectory by iteration number.

    Args:
        main_dir (Path): Main checkpoint directory.

    Returns:
        Path: Path to the latest checkpoint subdirectory.

    Raises:
        CheckpointNotFoundError: If no checkpoint subdirectories are found.
    """
    checkpoint_dirs = [
        d
        for d in main_dir.iterdir()
        if d.is_dir() and _extract_iteration_from_subdir(d.name) is not None
    ]
    if not checkpoint_dirs:
        # Provide helpful error message with available directories
        available_dirs = [d.name for d in main_dir.iterdir() if d.is_dir()]
        available_str = ", ".join(available_dirs[:5])  # Show first 5
        if len(available_dirs) > 5:
            available_str += f", ... ({len(available_dirs) - 5} more)"
        raise CheckpointNotFoundError(
            f"No checkpoint subdirectories found in {main_dir}",
            main_dir=main_dir,
            available_directories=available_dirs,
        )
    checkpoint_dirs.sort(key=lambda d: _extract_iteration_from_subdir(d.name) or -1)
    return checkpoint_dirs[-1]


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

    # Determine which subdirectory to load
    if subdirectory is None:
        checkpoint_path = _find_latest_checkpoint_subdir(main_path)
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


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint directory or file is not found."""

    def __init__(
        self,
        message: str,
        main_dir: Path | None = None,
        available_directories: list[str] | None = None,
    ):
        super().__init__(message)
        self.main_dir = main_dir
        self.available_directories = available_directories or []


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint file is corrupted or invalid."""

    def __init__(
        self, message: str, file_path: Path | None = None, details: str | None = None
    ):
        super().__init__(message)
        self.file_path = file_path
        self.details = details


def _atomic_write(path: Path, content: str) -> None:
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
        # Atomic rename on POSIX systems
        temp_file.replace(path)
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        raise OSError(f"Failed to write checkpoint file {path}: {e}") from e


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


def _load_and_validate_pydantic_model(
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
    try:
        json_data_dict = _validate_checkpoint_json(
            path, required_fields=required_fields
        )
        # Convert dict back to JSON string for Pydantic
        json_data = json.dumps(json_data_dict)
    except CheckpointNotFoundError:
        raise CheckpointNotFoundError(
            f"Checkpoint file not found: {path}",
            main_dir=path.parent,
        )
    except CheckpointCorruptedError:
        # Re-raise JSON validation errors as-is
        raise

    try:
        return model_class.model_validate_json(json_data)
    except Exception as e:
        context = f"{error_context} " if error_context else ""
        raise CheckpointCorruptedError(
            f"Failed to validate {context}checkpoint state: {path}",
            file_path=path,
            details=str(e),
        ) from e


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpointing during optimization.

    Attributes:
        checkpoint_dir (Path | None): Directory path for saving checkpoints.
            - If None: No checkpointing.
            - If Path: Uses that directory.
        checkpoint_interval (int | None): Save checkpoint every N iterations.
            If None, saves every iteration (if checkpoint_dir is set).
    """

    checkpoint_dir: Path | None = None
    checkpoint_interval: int | None = None

    @classmethod
    def with_timestamped_dir(
        cls, checkpoint_interval: int | None = None
    ) -> "CheckpointConfig":
        """Create CheckpointConfig with auto-generated directory name.

        Args:
            checkpoint_interval (int | None): Save checkpoint every N iterations.
                If None, saves every iteration (default).

        Returns:
            CheckpointConfig: A new CheckpointConfig with auto-generated directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_dir = Path(f"checkpoint_{timestamp}")
        return cls(
            checkpoint_dir=generated_dir, checkpoint_interval=checkpoint_interval
        )

    def _should_checkpoint(self, iteration: int) -> bool:
        """Determine if a checkpoint should be saved at the given iteration.

        Args:
            iteration (int): Current iteration number.

        Returns:
            bool: True if checkpointing is enabled and should occur at this iteration.
        """
        if self.checkpoint_dir is None:
            return False

        if self.checkpoint_interval is None:
            return True

        return iteration % self.checkpoint_interval == 0


@dataclass(frozen=True)
class CheckpointInfo:
    """Information about a checkpoint.

    Attributes:
        path (Path): Path to the checkpoint subdirectory.
        iteration (int): Iteration number of this checkpoint.
        timestamp (datetime): Modification time of the checkpoint directory.
        size_bytes (int): Total size of the checkpoint in bytes.
        is_valid (bool): Whether the checkpoint is valid (has required files).
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
    total_size = 0
    if checkpoint_path.exists():
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size


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

    # Get modification time
    mtime = checkpoint_path.stat().st_mtime
    timestamp = datetime.fromtimestamp(mtime)

    # Calculate size
    size_bytes = _calculate_checkpoint_size(checkpoint_path)

    # Check validity
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

    checkpoints = []
    for subdir in main_dir.iterdir():
        if not subdir.is_dir():
            continue

        iteration = _extract_iteration_from_subdir(subdir.name)
        if iteration is None:
            continue

        try:
            info = get_checkpoint_info(subdir)
            checkpoints.append(info)
        except (CheckpointNotFoundError, ValueError):
            # Skip invalid checkpoints
            continue

    # Sort by iteration number
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
        return _find_latest_checkpoint_subdir(main_dir)
    except CheckpointNotFoundError:
        return None


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

    # Sort by iteration (descending) and remove oldest
    checkpoints.sort(key=lambda x: x.iteration, reverse=True)
    to_remove = checkpoints[keep_last_n:]

    for checkpoint_info in to_remove:
        # Remove directory and all contents
        import shutil

        shutil.rmtree(checkpoint_info.path)
