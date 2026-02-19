# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import IntEnum


class Simulator(IntEnum):
    """Simulator backend type for execution configuration."""

    QiskitAer = 0
    """IBM Qiskit Aer simulator."""

    QCSim = 1
    """QCSim simulator."""

    CompositeQiskitAer = 2
    """Composite Qiskit Aer simulator."""

    CompositeQCSim = 3
    """Composite QCSim simulator."""

    GpuSim = 4
    """GPU-accelerated simulator."""


class SimulationMethod(IntEnum):
    """Simulation method for execution configuration."""

    Statevector = 0
    """Full statevector simulation."""

    MatrixProductState = 1
    """Matrix product state (MPS) simulation."""

    Stabilizer = 2
    """Stabilizer (Clifford) simulation."""

    TensorNetwork = 3
    """Tensor network simulation."""

    PauliPropagator = 4
    """Pauli propagator simulation."""

    ExtendedStabilizer = 5
    """Extended stabilizer simulation."""


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution configuration for a Qoro Service job.

    All fields are optional. When set on a job via
    :meth:`QoroService.set_execution_config`, unset (``None``) fields are
    omitted from the request so the server keeps its own defaults.

    Attributes:
        bond_dimension: MPS bond dimension. Subject to tier-based caps.
        truncation_threshold: MPS truncation threshold
            (only meaningful for ``SimulationMethod.MatrixProductState``).
        simulator: Simulator backend to use.
        simulation_method: Simulation method to use.
        api_meta: Runtime pass-through metadata (e.g. ``optimization_level``).
    """

    bond_dimension: int | None = None
    """MPS bond dimension."""

    truncation_threshold: float | None = None
    """MPS truncation threshold."""

    simulator: Simulator | None = None
    """Simulator backend."""

    simulation_method: SimulationMethod | None = None
    """Simulation method."""

    api_meta: dict | None = field(default=None)
    """Runtime pass-through metadata."""

    def to_payload(self) -> dict:
        """Serialise to the JSON body expected by the API.

        ``None`` fields are omitted; enum values are converted to their
        integer representation.

        Returns:
            dict: JSON-serialisable payload for
                ``POST /api/job/<job_id>/execution_config/``.
        """
        payload: dict = {}

        if self.bond_dimension is not None:
            payload["bond_dimension"] = self.bond_dimension
        if self.truncation_threshold is not None:
            payload["truncation_threshold"] = self.truncation_threshold
        if self.simulator is not None:
            payload["simulator_type"] = int(self.simulator)
        if self.simulation_method is not None:
            payload["simulation_type"] = int(self.simulation_method)
        if self.api_meta is not None:
            payload["api_meta"] = self.api_meta

        return payload

    @staticmethod
    def from_response(data: dict) -> "ExecutionConfig":
        """Construct an ``ExecutionConfig`` from an API response dictionary.

        Args:
            data: The ``execution_configuration`` dict from the API response.

        Returns:
            ExecutionConfig: A new instance populated from the response.
        """
        raw_simulator = data.get("simulator_type")
        raw_simulation_method = data.get("simulation_type")

        return ExecutionConfig(
            bond_dimension=data.get("bond_dimension"),
            truncation_threshold=data.get("truncation_threshold"),
            simulator=(Simulator(raw_simulator) if raw_simulator is not None else None),
            simulation_method=(
                SimulationMethod(raw_simulation_method)
                if raw_simulation_method is not None
                else None
            ),
            api_meta=data.get("api_meta"),
        )
