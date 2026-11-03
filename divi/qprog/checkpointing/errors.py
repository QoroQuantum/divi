# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised while locating and reading checkpoints."""

from pathlib import Path


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""


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
