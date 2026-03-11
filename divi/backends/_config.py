# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields
from enum import IntEnum

from divi.backends._systems import QPUSystem, SimulatorCluster


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

    def override(self, other: "ExecutionConfig") -> "ExecutionConfig":
        """Creates a new config by overriding attributes with non-None values.

        This method ensures immutability by always returning a new
        ``ExecutionConfig`` object and leaving the original instance unmodified.

        Args:
            other: Another ExecutionConfig instance to take values from. Only
                non-None attributes from this instance will be used for the
                override.

        Returns:
            A new ExecutionConfig instance with the merged configurations.
        """
        current_attrs = {f.name: getattr(self, f.name) for f in fields(self)}

        for f in fields(other):
            other_value = getattr(other, f.name)
            if other_value is not None:
                current_attrs[f.name] = other_value

        return ExecutionConfig(**current_attrs)

    def to_payload(self) -> dict:
        """Serialize to the JSON body expected by the API.

        ``None`` fields are omitted; enum values are converted to their
        integer representation.

        Returns:
            dict: JSON-serializable payload for
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


@dataclass(frozen=True)
class JobConfig:
    """Configuration for a Qoro Service job.

    Exactly one of ``simulator_cluster`` or ``qpu_system`` should be set to
    target the job. If neither is provided, the service defaults to the
    ``qoro_maestro`` simulator cluster.
    """

    shots: int | None = None
    """Number of shots for the job."""

    simulator_cluster: SimulatorCluster | str | None = None
    """The simulator cluster to target, can be a string name or a SimulatorCluster object."""

    qpu_system: QPUSystem | str | None = None
    """The QPU system to target, can be a string name or a QPUSystem object."""

    use_circuit_packing: bool | None = None
    """Whether to use circuit packing optimization."""

    tag: str = "default"
    """Tag to associate with the job for identification."""

    force_sampling: bool = False
    """Whether to force sampling instead of expectation value measurements."""

    def override(self, other: "JobConfig") -> "JobConfig":
        """Creates a new config by overriding attributes with non-None values.

        This method ensures immutability by always returning a new `JobConfig` object
        and leaving the original instance unmodified.

        If the override sets ``simulator_cluster``, any existing ``qpu_system``
        is cleared (and vice versa), so the mutual-exclusivity constraint is
        preserved.

        Args:
            other: Another JobConfig instance to take values from. Only non-None
                   attributes from this instance will be used for the override.

        Returns:
            A new JobConfig instance with the merged configurations.
        """
        current_attrs = {f.name: getattr(self, f.name) for f in fields(self)}

        for f in fields(other):
            other_value = getattr(other, f.name)
            if other_value is not None:
                current_attrs[f.name] = other_value

        # Ensure mutual exclusivity: if override sets one target, clear the other
        if other.simulator_cluster is not None:
            current_attrs["qpu_system"] = None
        elif other.qpu_system is not None:
            current_attrs["simulator_cluster"] = None

        return JobConfig(**current_attrs)

    def __post_init__(self):
        """Sanitizes and validates the configuration."""
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {self.shots}.")

        if self.simulator_cluster is not None and self.qpu_system is not None:
            raise ValueError(
                "Provide either 'simulator_cluster' or 'qpu_system', not both."
            )

        if isinstance(self.simulator_cluster, str):
            pass  # Deferred resolution in QoroService
        elif self.simulator_cluster is not None and not isinstance(
            self.simulator_cluster, SimulatorCluster
        ):
            raise TypeError(
                f"Expected a SimulatorCluster instance or str, got {type(self.simulator_cluster)}"
            )

        if isinstance(self.qpu_system, str):
            pass  # Deferred resolution in QoroService
        elif self.qpu_system is not None and not isinstance(self.qpu_system, QPUSystem):
            raise TypeError(
                f"Expected a QPUSystem instance or str, got {type(self.qpu_system)}"
            )

        if self.use_circuit_packing is not None and not isinstance(
            self.use_circuit_packing, bool
        ):
            raise TypeError(f"Expected a bool, got {type(self.use_circuit_packing)}")
