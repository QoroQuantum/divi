# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Private state models and path helpers for ensemble checkpoints."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpointing import (
    CheckpointCorruptedError,
    CheckpointNotFoundError,
    _load_and_validate_pydantic_model,
)

ROUND_PREFIX = "round_"
ROUND_START_FILE = "round_start.json"
ROUND_COMPLETION_FILE = "round_completion.json"
PROGRAM_COMPLETION_FILE = "program_completion.json"
_ROUND_PATTERN = re.compile(r"^round_(\d+)$")


class ProgramRoundRecord(BaseModel):
    """State needed to recover one child after an interrupted round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    program_id: dict[str, Any]
    program_type: str = Field(min_length=1)
    circuit_count_at_round_start: int = Field(ge=0)
    run_time_at_round_start: float = Field(ge=0, allow_inf_nan=False)


class RoundCheckpoint(BaseModel):
    """State saved at an interrupted or completed ensemble round boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["1.0"] = "1.0"
    kind: Literal["round_start", "round_completion"]
    ensemble_type: str = Field(min_length=1)
    round_index: int = Field(ge=1)
    ensemble_state: dict[str, Any]
    programs: list[ProgramRoundRecord] | None = None
    round_history: list[dict[str, Any]] | None = None
    total_circuit_count: int | None = Field(default=None, ge=0)
    total_run_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_kind(self) -> Self:
        interrupted_fields = (self.programs,)
        completed_fields = (
            self.round_history,
            self.total_circuit_count,
            self.total_run_time,
        )
        if self.kind == "round_start":
            valid = all(value is not None for value in interrupted_fields) and all(
                value is None for value in completed_fields
            )
        else:
            valid = all(value is None for value in interrupted_fields) and all(
                value is not None for value in completed_fields
            )
        if not valid:
            raise ValueError(f"round checkpoint fields do not match kind {self.kind!r}")
        return self

    def round_start_data(self) -> list[ProgramRoundRecord]:
        """Return fields guaranteed by the interrupted kind."""
        if self.kind != "round_start" or self.programs is None:
            raise ValueError("round checkpoint is not interrupted")
        return self.programs

    def round_completion_data(self) -> tuple[list[dict[str, Any]], int, float]:
        """Return fields guaranteed by the completed kind."""
        if (
            self.kind != "round_completion"
            or self.round_history is None
            or self.total_circuit_count is None
            or self.total_run_time is None
        ):
            raise ValueError("round checkpoint is not completed")
        return (
            self.round_history,
            self.total_circuit_count,
            self.total_run_time,
        )


def _round_dir(root: Path, round_index: int) -> Path:
    """Return the zero-padded directory for ``round_index``."""
    return root / f"{ROUND_PREFIX}{round_index:03d}"


def _program_checkpoint_path(round_path: Path, slot: int) -> Path:
    return round_path / f"program_{slot:03d}"


def _encode_program_id(value: Any) -> dict[str, Any]:
    """Encode supported hashable program IDs without losing their Python type."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("checkpoint program ID floats must be finite")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_program_id(item) for item in value],
        }
    raise TypeError(
        "checkpoint program ID must contain only stable scalar or tuple values; "
        f"got {type(value).__name__}"
    )


def _round_index(path: Path) -> int | None:
    match = _ROUND_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _state_artifacts_exist(payload: Any, round_dir: Path) -> bool:
    """Validate artifact references embedded in a workflow-state payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "artifact":
                if not isinstance(value, str):
                    return False
                artifact = Path(value)
                if artifact.is_absolute() or ".." in artifact.parts:
                    return False
                if not (round_dir / artifact).is_file():
                    return False
            elif not _state_artifacts_exist(value, round_dir):
                return False
    elif isinstance(payload, list):
        return all(_state_artifacts_exist(value, round_dir) for value in payload)
    return True


def _load_round(path: Path) -> RoundCheckpoint | None:
    round_index = _round_index(path)
    if round_index is None or not path.is_dir():
        return None

    for filename, kind, error_context in (
        (ROUND_COMPLETION_FILE, "round_completion", "Ensemble state"),
        (ROUND_START_FILE, "round_start", "Interrupted ensemble round"),
    ):
        checkpoint_path = path / filename
        if not checkpoint_path.is_file():
            continue
        try:
            state = _load_and_validate_pydantic_model(
                checkpoint_path,
                RoundCheckpoint,
                required_fields=["kind", "ensemble_type", "round_index"],
                error_context=error_context,
            )
        except (CheckpointNotFoundError, CheckpointCorruptedError):
            continue
        if (
            state.kind == kind
            and state.round_index == round_index
            and _state_artifacts_exist(state.ensemble_state, path)
        ):
            return state

    return None


def _resolve_ensemble_checkpoint(
    root: Path | str, subdirectory: str | None = None
) -> RoundCheckpoint:
    """Resolve the latest valid ensemble round or an explicit subdirectory."""
    main_dir = Path(root)
    if subdirectory is not None:
        candidates = [main_dir / subdirectory]
    elif main_dir.is_dir():
        candidates = sorted(
            (
                child
                for child in main_dir.iterdir()
                if child.is_dir() and _round_index(child) is not None
            ),
            key=lambda child: _round_index(child) or -1,
            reverse=True,
        )
    else:
        candidates = []

    for candidate in candidates:
        resolved = _load_round(candidate)
        if resolved is not None:
            return resolved

    raise CheckpointNotFoundError(
        f"No valid ensemble checkpoint found in {main_dir}.",
        main_dir=main_dir,
        available_directories=[path.name for path in candidates],
    )
