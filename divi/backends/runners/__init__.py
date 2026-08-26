# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The concrete :class:`~divi.backends.CircuitRunner` implementations."""

from ._maestro import MaestroConfig, MaestroSimulator
from ._qoro import JobStatus, JobType, QoroService

__all__ = [
    "JobStatus",
    "JobType",
    "MaestroConfig",
    "MaestroSimulator",
    "QiskitSimulator",
    "QoroService",
]


def __getattr__(name: str):
    """Resolve :class:`QiskitSimulator` on first access."""
    if name == "QiskitSimulator":
        try:
            from ._qiskit import QiskitSimulator
        except ImportError as exc:
            raise ImportError(
                "QiskitSimulator requires the 'aer' extra; install it with "
                "`pip install qoro-divi[aer]`. Divi's default simulator, "
                "MaestroSimulator, is included in the core install."
            ) from exc
        return QiskitSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
