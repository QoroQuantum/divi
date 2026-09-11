# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._systems import QPUSystem, SimulatorCluster


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


class DeviceConfig(BaseModel):
    """Per-job overrides for the device settings of the target QPU.

    Each QPU carries these settings already, configured on the Qoro dashboard.
    Setting one here overrides it for a single job; anything left unset keeps
    the target's stored value.

    Written in Python spelling and translated to the service's own on the way
    out, so ``use_twirling=True`` is sent as ``"USE_TWIRLING": "true"`` — the
    exact string the vendor workers compare against.

    Which settings a given QPU accepts depends on its vendor, and a setting the
    target's vendor does not recognise is ignored rather than rejected. Use
    :meth:`~divi.backends.QoroService.fetch_vendor_blueprints` to see what each
    vendor takes. ``extra`` carries anything the service has added since this
    version of Divi, written in the service's own spelling.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", alias_generator=str.upper, populate_by_name=True
    )

    ibm_device: str | None = None
    """Name of the IBM backend to run on, e.g. ``"ibm_fez"``."""

    transpile_level: int | None = Field(default=None, ge=0, le=3)
    """Qiskit optimisation level, 0 to 3."""

    use_twirling: bool | None = None
    """Enable Pauli twirling."""

    use_mitigation: bool | None = None
    """Enable error mitigation."""

    use_error_suppression: bool | None = None
    """Enable error suppression."""

    aggressive_compiling: bool | None = None
    """Try several transpilation seeds and keep the lowest-noise circuit."""

    aggressive_compiling_seeds: int | None = Field(default=None, gt=0)
    """How many seeds aggressive compiling tries."""

    iqm_device_url: str | None = None
    """URL of the IQM device; its last segment names the device."""

    device_max_shots_per_batch: int | None = Field(default=None, gt=0)
    """Largest shot count the device accepts in one batch."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """Settings this version of Divi does not model, keyed as the service
    spells them. Values are sent unchanged."""

    @model_validator(mode="after")
    def _reject_shadowed_extras(self):
        """An ``extra`` key that duplicates a field would silently win."""
        modelled = {
            spec.alias or name
            for name, spec in type(self).model_fields.items()
            if name != "extra"
        }
        clashes = sorted(modelled & set(self.extra))
        if clashes:
            raise ValueError(
                f"{clashes} are already fields on DeviceConfig; set them directly "
                "instead of through 'extra'."
            )
        return self

    def to_api_meta(self) -> dict:
        """Flatten to the keys the service expects, dropping unset settings."""
        payload = self.model_dump(by_alias=True, exclude_none=True, exclude={"extra"})
        canonical = {
            key: str(value).lower() if isinstance(value, bool) else value
            for key, value in payload.items()
        }
        return canonical | self.extra


class ExecutionConfig(BaseModel):
    """Execution configuration for a Qoro Service job.

    All fields are optional. When set on a job via
    :meth:`QoroService.set_execution_config`, unset (``None``) fields are
    omitted from the request so the server keeps its own defaults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bond_dimension: int | None = Field(default=None, gt=0)
    """MPS bond dimension."""

    truncation_threshold: float | None = Field(default=None, ge=0)
    """MPS truncation threshold."""

    simulator: Simulator | None = None
    """Simulator backend."""

    simulation_method: SimulationMethod | None = None
    """Simulation method."""

    noisy_device: str | None = None
    """Name of a noisy device backend to emulate (e.g. ``"ibm_fake_fez"``)."""

    noise_realizations: int | None = Field(default=None, gt=0)
    """Number of noise realizations to average over."""

    noise_scaling_factor: float | None = Field(default=None, ge=0, le=1)
    """Scaling factor applied to the device noise, between 0 and 1."""

    device_config: DeviceConfig | None = None
    """Per-job overrides for the target QPU's device settings."""

    extra_kwargs: dict | None = None
    """Escape hatch for runtime settings Divi does not model, passed to the
    service unchanged; unknown keys are rejected server-side.

    Prefer :attr:`~divi.backends.ExecutionConfig.device_config` for device
    settings — it is typed, and it spells them the way the service expects.
    """

    @model_validator(mode="after")
    def _warn_on_inert_combinations(self):
        """Flags settings that are accepted but will not do what they look like."""
        if self.noisy_device is not None and self.noise_scaling_factor == 0:
            warnings.warn(
                f"noise_scaling_factor=0 cancels all noise from noisy_device "
                f"'{self.noisy_device}', producing a noiseless run.",
                stacklevel=3,
            )

        mps_only_set = (
            self.bond_dimension is not None or self.truncation_threshold is not None
        )
        if (
            mps_only_set
            and self.simulation_method is not None
            and self.simulation_method != SimulationMethod.MatrixProductState
        ):
            warnings.warn(
                "bond_dimension and truncation_threshold only apply to "
                f"MatrixProductState simulations; they will be ignored with "
                f"simulation_method={self.simulation_method.name}.",
                stacklevel=3,
            )

        return self

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
        current_attrs = dict(self)

        for name in type(other).model_fields:
            other_value = getattr(other, name)
            if other_value is not None:
                current_attrs[name] = other_value

        return ExecutionConfig.model_construct(**current_attrs)

    def to_payload(self) -> dict:
        """Serialise to the JSON body expected by the API.

        ``None`` fields are omitted; enum values are converted to their
        integer representation. :attr:`~divi.backends.ExecutionConfig.device_config`
        is flattened into the same settings field the service reads both from.

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
        if self.noisy_device is not None:
            payload["noisy_device"] = self.noisy_device
        if self.noise_realizations is not None:
            payload["noise_realizations"] = self.noise_realizations
        if self.noise_scaling_factor is not None:
            payload["noise_scaling_factor"] = self.noise_scaling_factor
        api_meta = dict(self.extra_kwargs) if self.extra_kwargs is not None else {}
        if self.device_config is not None:
            device = self.device_config.to_api_meta()
            clashes = sorted(set(api_meta) & set(device))
            if clashes:
                raise ValueError(
                    f"{clashes} set on both device_config and extra_kwargs. Keep "
                    "device settings on device_config."
                )
            api_meta |= device
        if api_meta:
            payload["api_meta"] = api_meta

        return payload

    @staticmethod
    def from_response(data: dict) -> "ExecutionConfig":
        """Construct an ``ExecutionConfig`` from an API response dictionary.

        Values are taken as the service reported them: this reflects a job's
        stored configuration, so a value outside the range this class accepts
        on input is still what that job will run with. The service reports one
        settings field and does not say which half a key came from, so it lands
        whole on :attr:`~divi.backends.ExecutionConfig.extra_kwargs` rather than
        split across :attr:`~divi.backends.ExecutionConfig.device_config`.

        Args:
            data: The ``execution_configuration`` dict from the API response.

        Returns:
            ExecutionConfig: A new instance populated from the response.
        """
        raw_simulator = data.get("simulator_type")
        raw_simulation_method = data.get("simulation_type")

        return ExecutionConfig.model_construct(
            bond_dimension=data.get("bond_dimension"),
            truncation_threshold=data.get("truncation_threshold"),
            simulator=(Simulator(raw_simulator) if raw_simulator is not None else None),
            simulation_method=(
                SimulationMethod(raw_simulation_method)
                if raw_simulation_method is not None
                else None
            ),
            noisy_device=data.get("noisy_device"),
            noise_realizations=data.get("noise_realizations"),
            noise_scaling_factor=data.get("noise_scaling_factor"),
            device_config=None,
            extra_kwargs=data.get("api_meta"),
        )


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
