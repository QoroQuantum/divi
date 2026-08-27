# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

# isort: off
from ._types import AxisLabel, QASMTag
from ._core import DEFAULT_PRECISION, MetaCircuit
from ._conversions import (
    dag_to_qasm_body,
    measurement_qasms_from_groups,
)
from ._fidelity import build_overlap_meta
from ._payloads import CircuitPayload, bound_circuits
from ._qasm_template import (
    QASMTemplate,
    build_template,
    render_template,
)

# isort: on

from divi._optional import import_optional

if TYPE_CHECKING:
    from ._pennylane import qnode_to_meta, qscript_to_meta

__all__ = [
    "DEFAULT_PRECISION",
    "AxisLabel",
    "CircuitPayload",
    "MetaCircuit",
    "QASMTag",
    "QASMTemplate",
    "bound_circuits",
    "build_overlap_meta",
    "build_template",
    "dag_to_qasm_body",
    "measurement_qasms_from_groups",
    "qnode_to_meta",
    "qscript_to_meta",
    "render_template",
]


def __getattr__(name: str) -> Any:
    """Resolve PennyLane circuit adapters on first access."""
    if name in {"qnode_to_meta", "qscript_to_meta"}:
        import_optional("pennylane", extra="pennylane", capability=name)
        from . import _pennylane

        value = getattr(_pennylane, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
