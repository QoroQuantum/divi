# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for :class:`~divi.qprog.VariationalQuantumAlgorithm` checkpointing.

The three models here own the serialization schema; the save/load logic lives on
the algorithm class itself (which holds the program instance and can coordinate
the optimizer).
"""

import pickle
from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from divi.qprog.early_stopping import StopReason

if TYPE_CHECKING:
    from divi.qprog.quantum_program import QuantumProgram
    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class _ProgramAttributeView:
    """Expose program attributes while treating exclusions as absent fields."""

    def __init__(
        self,
        program: "QuantumProgram",
        excluded_attributes: Collection[str] = (),
        values: dict[str, Any] | None = None,
    ):
        self._program = program
        self._excluded_attributes = frozenset(excluded_attributes)
        self._values = values or {}

    def __getattr__(self, name: str) -> Any:
        if name in self._excluded_attributes:
            raise AttributeError(name)
        if name in self._values:
            return self._values[name]
        if name == "_serialized_program_type":
            return type(self._program).__name__
        return getattr(self._program, name)


class ProgramCheckpoint(BaseModel):
    """Metadata shared by program checkpoints."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )

    version: Literal["1.0"] = "1.0"
    program_type: str = Field(min_length=1, validation_alias="_serialized_program_type")
    total_circuit_count: int = Field(ge=0, validation_alias="_total_circuit_count")
    total_run_time: float = Field(
        ge=0, allow_inf_nan=False, validation_alias="_total_run_time"
    )

    @classmethod
    def _from_program(
        cls,
        program: "QuantumProgram",
        *,
        excluded_attributes: Collection[str] = (),
        values: dict[str, Any] | None = None,
    ) -> Self:
        return cls.model_validate(
            _ProgramAttributeView(program, excluded_attributes, values)
        )


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays to their plain-Python equivalents.

    :attr:`SubclassState.data` is untyped, so a subclass storing numpy — a
    problem's ``decode_fn`` may return an array of bits, or a
    ``{variable: numpy scalar}`` mapping — otherwise reaches the JSON
    serialiser as a type it cannot write. Restores as plain Python.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class SubclassState(BaseModel):
    """Container for subclass-specific state."""

    data: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for reconstructing an optimizer."""

    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class VQACheckpoint(ProgramCheckpoint):
    """Iterative or completed state for a variational quantum algorithm."""

    kind: Literal["iteration", "program_completion"]
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
    subclass_state: SubclassState = Field(validation_alias="_serialized_subclass_state")
    optimizer_config: OptimizerConfig | None = Field(
        default=None, validation_alias="_serialized_optimizer_config"
    )

    @model_validator(mode="after")
    def _validate_kind(self) -> Self:
        if (self.kind == "iteration") != (self.optimizer_config is not None):
            raise ValueError(
                "optimizer_config is required for iterative VQA checkpoints"
            )
        return self

    @classmethod
    def from_program(
        cls,
        program: "QuantumProgram",
        *,
        kind: Literal["iteration", "program_completion"],
    ) -> Self:
        excluded = (
            ("_serialized_optimizer_config",) if kind == "program_completion" else ()
        )
        return cls._from_program(
            program,
            excluded_attributes=excluded,
            values={"kind": kind},
        )

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
        return [np.asarray(item, dtype=np.float64).tolist() for item in v]

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
        program._stop_reason = (
            StopReason(self.stop_reason) if self.stop_reason is not None else None
        )

        if self.rng_state_bytes:
            program._rng.bit_generator.state = pickle.loads(self.rng_state_bytes)

        program._load_subclass_state(self.subclass_state.data)
