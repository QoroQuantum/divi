# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# isort: skip_file

from typing import TYPE_CHECKING, Any

from ._circuit_spec_stage import CircuitSpecStage
from ._pauli_twirl_stage import PauliTwirlStage
from ._qem_stage import QEMStage
from ._parameter_binding_stage import ParameterBindingStage
from ._data_binding_stage import (
    DataBindingStage,
    LossReductionFn,
    SampleLossFn,
    resolve_loss_reduction,
    resolve_sample_loss,
)
from ._measurement_stage import MeasurementStage
from ._pce_cost_stage import PCECostStage
from ._preprocess_stage import PreprocessStage
from ._qiskit_spec_stage import QiskitSpecStage
from ._trotter_spec_stage import TrotterSpecStage

from divi._optional import import_optional

if TYPE_CHECKING:
    from ._pennylane_spec_stage import PennyLaneSpecStage

__all__ = [
    "CircuitSpecStage",
    "DataBindingStage",
    "LossReductionFn",
    "MeasurementStage",
    "ParameterBindingStage",
    "PauliTwirlStage",
    "PCECostStage",
    "PennyLaneSpecStage",
    "PreprocessStage",
    "QEMStage",
    "QiskitSpecStage",
    "SampleLossFn",
    "TrotterSpecStage",
    "resolve_loss_reduction",
    "resolve_sample_loss",
]


def __getattr__(name: str) -> Any:
    """Resolve :class:`PennyLaneSpecStage` on first access."""
    if name == "PennyLaneSpecStage":
        import_optional("pennylane", extra="pennylane", capability="PennyLaneSpecStage")
        from ._pennylane_spec_stage import PennyLaneSpecStage

        globals()[name] = PennyLaneSpecStage
        return PennyLaneSpecStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
