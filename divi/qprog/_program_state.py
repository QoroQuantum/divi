# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for :class:`~divi.qprog.VariationalQuantumAlgorithm` checkpointing.

The three models here own the serialization schema; the save/load logic lives on
the algorithm class itself (which holds the program instance and can coordinate
the optimizer).
"""

import pickle
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

if TYPE_CHECKING:
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class SubclassState(BaseModel):
    """Container for subclass-specific state."""

    data: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for reconstructing an optimizer."""

    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class ProgramState(BaseModel):
    """Pydantic model for VariationalQuantumAlgorithm state."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # Metadata
    program_type: str = Field(validation_alias="_serialized_program_type")
    version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Core Algorithm State (mapped to private attributes)
    current_iteration: int
    max_iterations: int
    losses_history: list[dict[str, float]] = Field(validation_alias="_losses_history")
    param_history: list[list[list[float]]] = Field(
        default_factory=list, validation_alias="_param_history"
    )
    best_loss: float = Field(validation_alias="_best_loss")
    # Only solution-sampling programs (SolutionSamplingMixin) carry _best_probs;
    # it maps a parameter-set index to that set's {bitstring: probability} dict.
    best_probs: dict[int, dict[str, float]] = Field(
        default_factory=dict, validation_alias="_best_probs"
    )
    total_circuit_count: int = Field(validation_alias="_total_circuit_count")
    total_run_time: float = Field(validation_alias="_total_run_time")
    seed: int | None = Field(validation_alias="_seed")
    stop_reason: str | None = Field(
        default=None, validation_alias="_serialized_stop_reason"
    )
    grouping_strategy: str | None = Field(validation_alias="_grouping_strategy")

    # Arrays
    best_params: list[float] | None = Field(
        default=None, validation_alias="_best_params"
    )
    final_params: list[float] | None = Field(
        default=None, validation_alias="_final_params"
    )

    # Complex State (mapped to adapter properties on the program)
    rng_state_bytes: bytes | None = Field(
        default=None, validation_alias="_serialized_rng_state"
    )
    optimizer_config: OptimizerConfig = Field(
        validation_alias="_serialized_optimizer_config"
    )
    subclass_state: SubclassState = Field(validation_alias="_serialized_subclass_state")

    @field_serializer("rng_state_bytes")
    def serialize_bytes(self, v: bytes | None, _info):
        return v.hex() if v is not None else None

    @field_validator("rng_state_bytes", mode="before")
    @classmethod
    def validate_bytes(cls, v):
        return bytes.fromhex(v) if isinstance(v, str) else v

    @field_validator("param_history", mode="before")
    @classmethod
    def normalize_param_history(cls, v):
        """Accept nested lists or per-iteration ndarray snapshots from disk or program."""
        if not v:
            return []
        rows: list[list[list[float]]] = []
        for item in v:
            arr = np.asarray(item, dtype=np.float64)
            rows.append(arr.tolist())
        return rows

    @field_serializer("best_params", "final_params")
    def serialize_arrays(self, v: npt.NDArray | list | None, _info):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def restore(self, program: "VariationalQuantumAlgorithm") -> None:
        """Apply this state object back to a program instance."""
        # 1. Bulk restore standard attributes
        for name, field in self.__class__.model_fields.items():
            alias = field.validation_alias
            target_attr = alias if isinstance(alias, str) else name

            # Skip adapter properties (they are read-only / calculated)
            if target_attr.startswith("_serialized_"):
                continue

            val = getattr(self, name)

            if target_attr == "_param_history" and val is not None:
                val = [np.asarray(block, dtype=np.float64) for block in val]
            # Handle numpy conversion
            elif "params" in target_attr and val is not None:
                val = np.array(val)

            if hasattr(program, target_attr):
                setattr(program, target_attr, val)

        # 2. Restore complex state
        if self.rng_state_bytes:
            program._rng.bit_generator.state = pickle.loads(self.rng_state_bytes)

        program._load_subclass_state(self.subclass_state.data)
