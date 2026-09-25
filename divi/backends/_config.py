# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from ._systems import QPUSystem, SimulatorCluster


class DeviceConfig(BaseModel):
    """Per-job hardware options for a Qoro Service job on a QPU.

    Pass one to :meth:`~divi.backends.QoroService.submit_circuits`; the service
    rejects options it does not recognise. Unset (``None``) options are not
    sent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimization_level: StrictInt | None = None
    """Transpiler optimisation level."""

    resilience_level: StrictInt | None = None
    """Error-resilience level."""

    max_execution_time: StrictInt | None = None
    """Upper bound on the job's execution time, in seconds."""

    transpilation_seed: StrictInt | None = None
    """Seed for the transpiler's stochastic passes."""

    layout_method: StrictStr | None = Field(default=None, max_length=64)
    """Transpiler layout method, e.g. ``"sabre"``."""

    routing_method: StrictStr | None = Field(default=None, max_length=64)
    """Transpiler routing method, e.g. ``"sabre"``."""

    approximation_degree: StrictInt | StrictFloat | None = None
    """Transpiler approximation degree."""

    def to_payload(self) -> dict:
        """Serialise to the ``device_config`` object, dropping unset options."""
        return self.model_dump(exclude_none=True)


class JobConfig(BaseModel):
    """Configuration for a Qoro Service job.

    Exactly one of ``simulator_cluster`` or ``qpu_system`` should be set to
    target the job. If neither is provided, the service defaults to the
    ``qoro_maestro`` simulator cluster.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shots: int | None = Field(default=None, gt=0)
    """Number of shots for the job."""

    simulator_cluster: SimulatorCluster | str | None = None
    """The simulator cluster to target, can be a string name or a SimulatorCluster object."""

    qpu_system: QPUSystem | str | None = None
    """The QPU system to target, can be a string name or a QPUSystem object."""

    use_circuit_packing: bool | None = Field(default=None, strict=True)
    """Whether to use circuit packing optimisation."""

    tag: str | None = "default"
    """Tag to associate with the job for identification. ``None`` in an
    override means "keep the base tag"."""

    force_sampling: bool = Field(default=False, strict=True)
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
        current_attrs = dict(self)

        for name in type(other).model_fields:
            other_value = getattr(other, name)
            if other_value is not None:
                current_attrs[name] = other_value

        # Ensure mutual exclusivity: if override sets one target, clear the other
        if other.simulator_cluster is not None:
            current_attrs["qpu_system"] = None
        elif other.qpu_system is not None:
            current_attrs["simulator_cluster"] = None

        return JobConfig(**current_attrs)

    @model_validator(mode="after")
    def _check_single_target(self):
        """A job targets one place; string names are resolved later in QoroService."""
        if self.simulator_cluster is not None and self.qpu_system is not None:
            raise ValueError(
                "Provide either 'simulator_cluster' or 'qpu_system', not both."
            )
        return self
