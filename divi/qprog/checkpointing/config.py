# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""User-facing checkpointing configuration."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpointing during optimisation.

    Attributes:
        checkpoint_dir: Directory path for saving checkpoints.
            - If None: No checkpointing.
            - If a path (``Path`` or ``str``): Uses that directory.
        checkpoint_interval: Save checkpoint every N iterations (N ≥ 1).
            If None, saves every iteration (if checkpoint_dir is set).

    Raises:
        ValueError: If ``checkpoint_interval`` is below 1.
    """

    checkpoint_dir: Path | None = None
    checkpoint_interval: int | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is not None:
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be positive; got {self.checkpoint_interval}."
            )

    @classmethod
    def with_timestamped_dir(cls, checkpoint_interval: int | None = None) -> Self:
        """Create CheckpointConfig with auto-generated directory name.

        Args:
            checkpoint_interval (int | None): Save checkpoint every N iterations.
                If None, saves every iteration (default).

        Returns:
            CheckpointConfig: A new CheckpointConfig with auto-generated directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
