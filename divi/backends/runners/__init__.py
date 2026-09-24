# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The concrete :class:`~divi.backends.CircuitRunner` implementations."""

from typing import Any

from divi._optional import import_optional

from .._job_status import JobStatus
from ._maestro import MaestroConfig, MaestroSimulator
from ._qoro import JobType, QoroService

__all__ = [
    "JobStatus",
    "JobType",
    "MaestroConfig",
    "MaestroSimulator",
    "QiskitSimulator",
    "QoroService",
]


def __getattr__(name: str) -> Any:
    """Resolve :class:`QiskitSimulator` on first access."""
    if name == "QiskitSimulator":
        import_optional(
            "qiskit_aer",
            extra="aer",
            capability="QiskitSimulator",
            hint=(
                "Divi's default simulator, MaestroSimulator, is included in the "
                "core install."
            ),
        )
        from ._qiskit import QiskitSimulator

        globals()[name] = QiskitSimulator
        return QiskitSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
