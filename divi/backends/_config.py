# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._systems import QPUSystem, SimulatorCluster

_TOGGLES_BY_WIRE_VALUE = {"true": True, "false": False}

# The job's QPU system picks the device, so a job may not redirect it.
_DEVICE_SELECTION_KEYS = frozenset({"IBM_DEVICE", "IQM_DEVICE_URL"})


class DeviceConfig(BaseModel):
    """Per-job execution options for a Qoro Service job on a QPU.

    Options are the QPU vendors' own device keys, the ones
    :meth:`~divi.backends.QoroService.fetch_vendor_blueprints` lists, given as
    keyword arguments in any case::

        DeviceConfig(transpile_level=3, use_mitigation=True)

    Toggles take ``True`` or ``False``. The options apply to every QPU in the
    target system, and each QPU reads only its own vendor's keys. Options left
    out, or set to ``None``, keep the QPU's own settings.
    :meth:`~divi.backends.QoroService.submit_circuits` rejects a key no vendor
    accepts.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _lowercase_keys(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return {str(key).lower(): value for key, value in data.items()}
        return data

    @model_validator(mode="after")
    def _reject_device_selection(self):
        selected = sorted(self.to_payload().keys() & _DEVICE_SELECTION_KEYS)
        if selected:
            raise ValueError(
                f"{selected} would choose the device, which the job's QPU system "
                "already does."
            )
        return self

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the ``device_config`` object, dropping unset options."""
        return {
            name.upper(): value
            for name, value in (self.model_extra or {}).items()
            if value is not None
        }

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "DeviceConfig":
        """Rebuild a :class:`DeviceConfig` from the Qoro Service's ``device_config``.

        The inverse of :meth:`to_payload`: the service stores keys uppercase and
        toggles as ``"true"`` / ``"false"``.
        """
        return cls.model_validate(
            {
                key: (
                    _TOGGLES_BY_WIRE_VALUE.get(value, value)
                    if isinstance(value, str)
                    else value
                )
                for key, value in data.items()
            }
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
