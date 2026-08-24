# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from ._backend_properties_conversion import create_backend_from_properties
from ._base import (
    AsyncJobBackend,
    CircuitBatch,
    CircuitRunner,
    ExecutionResult,
    normalise_circuit_batch,
)
from ._config import ExecutionConfig, JobConfig, SimulationMethod, Simulator
from ._job_status import (
    InsufficientCreditsError,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimedOutError,
    QoroJobError,
)
from ._results_processing import convert_counts_to_probs, reverse_dict_endianness
from ._systems import QPU, QPUSystem, SimulatorCluster
from .runners import (
    JobType,
    MaestroConfig,
    MaestroSimulator,
    QoroService,
    QRMIBackend,
)

__all__ = [
    "AsyncJobBackend",
    "CircuitBatch",
    "CircuitRunner",
    "ExecutionConfig",
    "ExecutionResult",
    "InsufficientCreditsError",
    "JobConfig",
    "JobCancelledError",
    "JobFailedError",
    "JobStatus",
    "JobTimedOutError",
    "JobType",
    "MaestroConfig",
    "MaestroSimulator",
    "QPU",
    "QPUSystem",
    "QRMIBackend",
    "QiskitSimulator",
    "QoroService",
    "QoroJobError",
    "SimulationMethod",
    "Simulator",
    "SimulatorCluster",
    "convert_counts_to_probs",
    "create_backend_from_properties",
    "normalise_circuit_batch",
    "reverse_dict_endianness",
]


def __getattr__(name: str):
    """Resolve :class:`QiskitSimulator` on first access."""
    if name == "QiskitSimulator":
        from .runners import QiskitSimulator

        return QiskitSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
