# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for checkpointing utilities."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from divi.qprog.checkpointing import (
    PROGRAM_STATE_FILE,
    CheckpointConfig,
    CheckpointInfo,
    CheckpointNotFoundError,
    _ensure_checkpoint_dir,
    _extract_iteration_from_subdir,
    _find_latest_checkpoint_subdir,
    _get_checkpoint_subdir_name,
    _get_checkpoint_subdir_path,
    cleanup_old_checkpoints,
    get_checkpoint_info,
    get_latest_checkpoint,
    list_checkpoints,
)


class TestNamingUtilities:
    """Tests for checkpoint naming utility functions."""

    def test_get_checkpoint_subdir_name(self):
        """Test subdirectory name generation."""
        assert _get_checkpoint_subdir_name(0) == "checkpoint_000"
        assert _get_checkpoint_subdir_name(1) == "checkpoint_001"
        assert _get_checkpoint_subdir_name(42) == "checkpoint_042"
        assert _get_checkpoint_subdir_name(999) == "checkpoint_999"
        assert _get_checkpoint_subdir_name(1000) == "checkpoint_1000"

    def test_extract_iteration_from_subdir(self):
        """Test iteration extraction from subdirectory names."""
        assert _extract_iteration_from_subdir("checkpoint_001") == 1
        assert _extract_iteration_from_subdir("checkpoint_042") == 42
        assert _extract_iteration_from_subdir("checkpoint_000") == 0
        assert _extract_iteration_from_subdir("checkpoint_999") == 999

    def test_extract_iteration_from_subdir_invalid(self):
        """Test iteration extraction with invalid names."""
        assert _extract_iteration_from_subdir("invalid") is None
        assert _extract_iteration_from_subdir("checkpoint_abc") is None
        assert _extract_iteration_from_subdir("checkpoint_") is None
        assert _extract_iteration_from_subdir("") is None
        assert _extract_iteration_from_subdir("checkpoint_001_extra") is None


class TestDirectoryUtilities:
    """Tests for checkpoint directory management utility functions."""

    def test_ensure_checkpoint_dir_creates_directory(self, tmp_path):
        """Test that ensure_checkpoint_dir creates the directory."""
        checkpoint_dir = tmp_path / "new_checkpoint"
        assert not checkpoint_dir.exists()

        result = _ensure_checkpoint_dir(checkpoint_dir)
        assert result == checkpoint_dir
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_ensure_checkpoint_dir_creates_parents(self, tmp_path):
        """Test that ensure_checkpoint_dir creates parent directories."""
        checkpoint_dir = tmp_path / "nested" / "deep" / "checkpoint"
        assert not checkpoint_dir.exists()

        result = _ensure_checkpoint_dir(checkpoint_dir)
        assert result == checkpoint_dir
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_ensure_checkpoint_dir_existing_directory(self, tmp_path):
        """Test that ensure_checkpoint_dir handles existing directories."""
        checkpoint_dir = tmp_path / "existing"
        checkpoint_dir.mkdir()

        result = _ensure_checkpoint_dir(checkpoint_dir)
        assert result == checkpoint_dir
        assert checkpoint_dir.exists()

    def test_get_checkpoint_subdir_path(self, tmp_path):
        """Test getting checkpoint subdirectory path."""
        main_dir = tmp_path / "checkpoints"
        path = _get_checkpoint_subdir_path(main_dir, 5)

        assert path == main_dir / "checkpoint_005"
        assert path.name == "checkpoint_005"

    def test_find_latest_checkpoint_subdir(self, tmp_path):
        """Test finding the latest checkpoint subdirectory."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Create multiple checkpoint subdirectories
        (main_dir / "checkpoint_001").mkdir()
        (main_dir / "checkpoint_003").mkdir()
        (main_dir / "checkpoint_002").mkdir()
        (main_dir / "other_dir").mkdir()  # Should be ignored

        latest = _find_latest_checkpoint_subdir(main_dir)
        assert latest.name == "checkpoint_003"

    def test_find_latest_checkpoint_subdir_no_checkpoints(self, tmp_path):
        """Test finding latest checkpoint when none exist."""
        main_dir = tmp_path / "empty"
        main_dir.mkdir()

        with pytest.raises(
            CheckpointNotFoundError, match="No checkpoint subdirectories found"
        ):
            _find_latest_checkpoint_subdir(main_dir)

    def test_find_latest_checkpoint_subdir_ignores_invalid_names(self, tmp_path):
        """Test that invalid subdirectory names are ignored."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        (main_dir / "checkpoint_001").mkdir()
        (main_dir / "checkpoint_invalid").mkdir()  # Should be ignored
        (main_dir / "checkpoint_002").mkdir()

        latest = _find_latest_checkpoint_subdir(main_dir)
        assert latest.name == "checkpoint_002"


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_checkpoint_config_default(self):
        """Test default CheckpointConfig."""
        config = CheckpointConfig()
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval is None

    def test_checkpoint_config_with_dir(self, tmp_path):
        """Test CheckpointConfig with directory."""
        checkpoint_dir = tmp_path / "my_checkpoint"
        config = CheckpointConfig(checkpoint_dir=checkpoint_dir)
        assert config.checkpoint_dir == checkpoint_dir
        assert config.checkpoint_interval is None

    def test_checkpoint_config_with_interval(self, tmp_path):
        """Test CheckpointConfig with interval."""
        checkpoint_dir = tmp_path / "my_checkpoint"
        config = CheckpointConfig(checkpoint_dir=checkpoint_dir, checkpoint_interval=5)
        assert config.checkpoint_dir == checkpoint_dir
        assert config.checkpoint_interval == 5

    def test_checkpoint_config_with_timestamped_dir(self):
        """Test auto-generation of checkpoint directory."""
        config = CheckpointConfig.with_timestamped_dir()
        assert config.checkpoint_dir is not None
        assert isinstance(config.checkpoint_dir, Path)
        assert config.checkpoint_dir.name.startswith("checkpoint_")
        assert len(config.checkpoint_dir.name) > len("checkpoint_")
        assert config.checkpoint_interval is None  # Default is None

        # Test with checkpoint interval
        config_with_interval = CheckpointConfig.with_timestamped_dir(
            checkpoint_interval=5
        )
        assert config_with_interval.checkpoint_dir is not None
        assert config_with_interval.checkpoint_interval == 5

    def test_should_checkpoint_disabled(self):
        """Test should_checkpoint when checkpointing is disabled."""
        config = CheckpointConfig()
        assert config._should_checkpoint(0) is False
        assert config._should_checkpoint(1) is False
        assert config._should_checkpoint(100) is False

    def test_should_checkpoint_every_iteration(self, tmp_path):
        """Test should_checkpoint with no interval (every iteration)."""
        config = CheckpointConfig(checkpoint_dir=tmp_path / "checkpoint")
        assert config._should_checkpoint(0) is True
        assert config._should_checkpoint(1) is True
        assert config._should_checkpoint(2) is True
        assert config._should_checkpoint(100) is True

    def test_should_checkpoint_with_interval(self, tmp_path):
        """Test should_checkpoint with interval."""
        config = CheckpointConfig(
            checkpoint_dir=tmp_path / "checkpoint", checkpoint_interval=3
        )
        assert config._should_checkpoint(0) is True  # 0 % 3 == 0
        assert config._should_checkpoint(1) is False
        assert config._should_checkpoint(2) is False
        assert config._should_checkpoint(3) is True  # 3 % 3 == 0
        assert config._should_checkpoint(4) is False
        assert config._should_checkpoint(5) is False
        assert config._should_checkpoint(6) is True  # 6 % 3 == 0

    def test_checkpoint_config_frozen(self, tmp_path):
        """Test that CheckpointConfig is frozen."""
        config = CheckpointConfig(checkpoint_dir=tmp_path / "checkpoint")
        with pytest.raises(FrozenInstanceError):
            config.checkpoint_dir = tmp_path / "other"


class TestManagementUtilities:
    """Tests for checkpoint management utilities."""

    def test_get_checkpoint_info(self, tmp_path):
        """Test getting checkpoint information."""
        checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_001"
        checkpoint_dir.mkdir(parents=True)

        # Create required files
        (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
        (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        info = get_checkpoint_info(checkpoint_dir)

        assert info.path == checkpoint_dir
        assert info.iteration == 1
        assert info.is_valid is True
        assert info.size_bytes > 0
        assert isinstance(info.timestamp, datetime)

    def test_get_checkpoint_info_invalid_directory(self, tmp_path):
        """Test get_checkpoint_info with invalid directory name."""
        invalid_dir = tmp_path / "invalid_name"
        invalid_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid checkpoint directory name"):
            get_checkpoint_info(invalid_dir)

    def test_get_checkpoint_info_nonexistent(self, tmp_path):
        """Test get_checkpoint_info with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(CheckpointNotFoundError):
            get_checkpoint_info(nonexistent)

    def test_get_checkpoint_info_missing_files(self, tmp_path):
        """Test get_checkpoint_info with missing required files."""
        checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_001"
        checkpoint_dir.mkdir(parents=True)

        # Only create one file
        (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')

        info = get_checkpoint_info(checkpoint_dir)
        assert info.is_valid is False

    def test_list_checkpoints(self, tmp_path):
        """Test listing all checkpoints."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Create multiple checkpoints
        for i in [1, 3, 2]:
            checkpoint_dir = main_dir / f"checkpoint_{i:03d}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
            (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        checkpoints = list_checkpoints(main_dir)

        assert len(checkpoints) == 3
        # Should be sorted by iteration
        assert [c.iteration for c in checkpoints] == [1, 2, 3]
        assert all(c.is_valid for c in checkpoints)

    def test_list_checkpoints_ignores_invalid(self, tmp_path):
        """Test that list_checkpoints ignores invalid directories."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Valid checkpoint
        valid_dir = main_dir / "checkpoint_001"
        valid_dir.mkdir()
        (valid_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
        (valid_dir / "optimizer_state.json").write_text('{"test": "data"}')

        # Invalid directories (should be ignored)
        (main_dir / "other_dir").mkdir()
        (main_dir / "checkpoint_invalid").mkdir()

        checkpoints = list_checkpoints(main_dir)

        assert len(checkpoints) == 1
        assert checkpoints[0].iteration == 1

    def test_list_checkpoints_empty_directory(self, tmp_path):
        """Test listing checkpoints in empty directory."""
        main_dir = tmp_path / "empty"
        main_dir.mkdir()

        checkpoints = list_checkpoints(main_dir)
        assert checkpoints == []

    def test_list_checkpoints_nonexistent(self, tmp_path):
        """Test list_checkpoints with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(CheckpointNotFoundError):
            list_checkpoints(nonexistent)

    def test_get_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Create checkpoints
        for i in [1, 3, 2]:
            checkpoint_dir = main_dir / f"checkpoint_{i:03d}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
            (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        latest = get_latest_checkpoint(main_dir)

        assert latest is not None
        assert latest.name == "checkpoint_003"

    def test_get_latest_checkpoint_empty(self, tmp_path):
        """Test getting latest checkpoint when none exist."""
        main_dir = tmp_path / "empty"
        main_dir.mkdir()

        latest = get_latest_checkpoint(main_dir)
        assert latest is None

    def test_cleanup_old_checkpoints(self, tmp_path):
        """Test cleaning up old checkpoints."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Create 5 checkpoints
        for i in range(1, 6):
            checkpoint_dir = main_dir / f"checkpoint_{i:03d}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
            (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        # Keep only last 2
        cleanup_old_checkpoints(main_dir, keep_last_n=2)

        remaining = list_checkpoints(main_dir)
        assert len(remaining) == 2
        assert [c.iteration for c in remaining] == [4, 5]

    def test_cleanup_old_checkpoints_keep_all(self, tmp_path):
        """Test cleanup when keeping all checkpoints."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        # Create 3 checkpoints
        for i in range(1, 4):
            checkpoint_dir = main_dir / f"checkpoint_{i:03d}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
            (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        # Keep all (keep_last_n >= number of checkpoints)
        cleanup_old_checkpoints(main_dir, keep_last_n=5)

        remaining = list_checkpoints(main_dir)
        assert len(remaining) == 3

    def test_cleanup_old_checkpoints_invalid_keep_n(self, tmp_path):
        """Test cleanup with invalid keep_last_n."""
        main_dir = tmp_path / "checkpoints"
        main_dir.mkdir()

        with pytest.raises(ValueError, match="keep_last_n must be at least 1"):
            cleanup_old_checkpoints(main_dir, keep_last_n=0)

    def test_checkpoint_info_size_calculation(self, tmp_path):
        """Test that checkpoint size is calculated correctly."""
        checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_001"
        checkpoint_dir.mkdir(parents=True)

        # Create files with known sizes
        large_content = "x" * 1000
        (checkpoint_dir / PROGRAM_STATE_FILE).write_text(large_content)
        (checkpoint_dir / "optimizer_state.json").write_text(large_content)

        info = get_checkpoint_info(checkpoint_dir)

        # Should be at least 2000 bytes (2 files * 1000 bytes)
        assert info.size_bytes >= 2000

    def test_checkpoint_info_dataclass(self, tmp_path):
        """Test CheckpointInfo dataclass properties."""
        checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_001"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / PROGRAM_STATE_FILE).write_text('{"test": "data"}')
        (checkpoint_dir / "optimizer_state.json").write_text('{"test": "data"}')

        info = get_checkpoint_info(checkpoint_dir)

        # Test that it's a CheckpointInfo instance
        assert isinstance(info, CheckpointInfo)

        # Test that it's frozen (immutable)
        with pytest.raises(FrozenInstanceError):
            info.iteration = 999

        # Test all fields are accessible
        assert isinstance(info.path, Path)
        assert isinstance(info.iteration, int)
        assert isinstance(info.timestamp, datetime)
        assert isinstance(info.size_bytes, int)
        assert isinstance(info.is_valid, bool)
