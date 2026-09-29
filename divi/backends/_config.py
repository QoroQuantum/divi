# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import difflib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._systems import QPUSystem, SimulatorCluster

_TOGGLES_BY_WIRE_VALUE = {"true": True, "false": False}

# The job's QPU system picks the device, so a job may not redirect it.
_DEVICE_SELECTION_KEYS = frozenset({"IBM_DEVICE", "IQM_DEVICE_URL"})


def _lowercased(options: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in options.items()}


def describe_unknown(names: Iterable[str], known: Iterable[str]) -> str:
    """The ``names`` not in ``known``, each with its closest match; empty if none."""
    known = sorted(known)
    return ", ".join(
        (
            f"{name!r} (did you mean {match[0]!r}?)"
            if (match := difflib.get_close_matches(name, known, n=1))
            else repr(name)
        )
        for name in sorted(set(names) - set(known))
    )


def reject_unknown_reset(names: Sequence[str], known: Iterable[str]) -> None:
    """Raise unless ``names`` is a non-empty list of fields ``reset`` can restore."""
    if not names:
        raise ValueError("reset() needs at least one field name.")
    if unknown := describe_unknown(names, known):
        raise ValueError(f"Cannot reset unknown fields {unknown}.")


class DeviceConfig(BaseModel):
    """Per-job execution options for a Qoro Service job on a QPU.

    Options are the QPU vendors' own device keys, the ones
    :meth:`~divi.backends.QoroService.fetch_vendor_blueprints` lists, given as
    keyword arguments in any case::

        DeviceConfig(transpile_level=3, use_mitigation=True)

    Toggles take ``True`` or ``False``. The options apply to every QPU in the
    target system, and each QPU reads only its own vendor's keys. Options left
    out, or set to ``None``, keep the QPU's own settings. Configurations are frozen;
    :meth:`override` and :meth:`reset` return changed copies.
    :meth:`~divi.backends.QoroService.submit_circuits` rejects a key no vendor
    accepts.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _lowercase_keys(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return _lowercased(data)
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

    def override(self, **options: Any) -> "DeviceConfig":
        """Return a copy with ``options`` set, e.g.
        ``config.override(transpile_level=3)``."""
        return DeviceConfig(**(self.model_extra or {}) | _lowercased(options))

    def reset(self, *names: str) -> "DeviceConfig":
        """Return a copy without the options ``names``, so the QPU's own
        settings apply to them."""
        names = tuple(name.lower() for name in names)
        options = self.model_extra or {}
        reject_unknown_reset(names, options)
        return DeviceConfig(
            **{name: value for name, value in options.items() if name not in names}
        )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> "DeviceConfig":
        """Copy, validating ``update`` as :meth:`override` does."""
        copied = super().model_copy(deep=deep)
        return copied.override(**update) if update else copied

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

    shots: int = Field(default=1000, gt=0)
    """Number of shots for the job."""

    simulator_cluster: SimulatorCluster | str | None = None
    """The simulator cluster to target, can be a string name or a SimulatorCluster object."""

    qpu_system: QPUSystem | str | None = None
    """The QPU system to target, can be a string name or a QPUSystem object."""

    use_circuit_packing: bool | None = Field(default=None, strict=True)
    """Whether to use circuit packing optimisation."""

    tag: str | None = "default"
    """Tag to associate with the job for identification."""

    force_sampling: bool = Field(default=False, strict=True)
    """Whether to force sampling instead of expectation value measurements."""

    def override(self, **fields: Any) -> "JobConfig":
        """Return a copy with ``fields`` set.

        Setting one target clears the other, e.g.
        ``config.override(qpu_system="ibm_torino")`` drops ``simulator_cluster``.
        """
        current = dict(self)
        if fields.get("simulator_cluster") is not None:
            current["qpu_system"] = None
        if fields.get("qpu_system") is not None:
            current["simulator_cluster"] = None
        return JobConfig(**current | fields)

    def reset(self, *names: str) -> "JobConfig":
        """Return a copy with the fields ``names`` back at their defaults."""
        reject_unknown_reset(names, type(self).model_fields)
        return JobConfig(**{name: value for name, value in self if name not in names})

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> "JobConfig":
        """Copy, validating ``update`` as :meth:`override` does."""
        copied = super().model_copy(deep=deep)
        return copied.override(**update) if update else copied

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and (
            unknown := describe_unknown(data, cls.model_fields)
        ):
            raise ValueError(f"JobConfig got unknown fields {unknown}.")
        return data

    @model_validator(mode="after")
    def _check_single_target(self):
        """A job targets one place; string names are resolved later in QoroService."""
        if self.simulator_cluster is not None and self.qpu_system is not None:
            raise ValueError(
                "Provide either 'simulator_cluster' or 'qpu_system', not both."
            )
        return self
