# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The Qoro Service's ``maestro_config`` wire format for a :class:`MaestroConfig`."""

import json
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import maestro
import numpy as np

from ._maestro import MaestroConfig

_SERVICE_FIELD_NAMES = {
    "max_bond_dimension": "bond_dimension",
    "singular_value_threshold": "truncation_threshold",
}
_FIELD_NAMES_BY_SERVICE_NAME = {
    service: field for field, service in _SERVICE_FIELD_NAMES.items()
}
_ENUMS = {
    "simulator_type": maestro.SimulatorType,
    "simulation_type": maestro.SimulationType,
}


def _to_json(value: Any) -> Any:
    """``json`` fallback for noise-model arguments; complex numbers are tagged."""
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def _from_json(obj: dict) -> Any:
    if "__complex__" in obj:
        real, imag = obj["__complex__"]
        return complex(real, imag)
    return obj


def _noise_model_to_payload(noise_model: Any) -> list[dict]:
    """The model's recorded ``set_*`` calls, checked to rebuild the same model."""
    calls = [
        {"method": method, "args": list(args), "kwargs": dict(kwargs)}
        for method, args, kwargs in noise_model._call_log
    ]
    calls = json.loads(json.dumps(calls, default=_to_json))
    try:
        rebuilt = _noise_model_from_payload(calls)
    except Exception as exc:
        raise ValueError(
            "noise_model's recorded calls do not replay, so it cannot be sent to "
            f"the Qoro Service: {exc}"
        ) from exc
    lost = [
        name
        for name in dir(noise_model)
        if name.startswith("has_")
        and getattr(noise_model, name)() != getattr(rebuilt, name)()
    ]
    if lost:
        raise ValueError(
            "noise_model was changed by calls that are not recorded (only set_* "
            f"methods are), so the Qoro Service would not see them; differs in {lost}."
        )
    return calls


def _noise_model_from_payload(calls: Sequence[Mapping[str, Any]]) -> Any:
    noise_model = maestro.NoiseModel()
    for call in json.loads(json.dumps(calls), object_hook=_from_json):
        method = call["method"]
        if not method.startswith("set_"):
            raise ValueError(f"Noise-model call {method!r} is not a set_* method.")
        getattr(noise_model, method)(*call["args"], **call["kwargs"])
    return noise_model


def maestro_config_to_payload(config: MaestroConfig) -> dict:
    """Serialise ``config`` to the Qoro Service's ``maestro_config`` object.

    Every field that is not ``None`` is sent, defaults included, so a cloud run
    sees the values a local one would. ``max_bond_dimension`` and
    ``singular_value_threshold`` travel as ``bond_dimension`` and
    ``truncation_threshold``, the enum names as maestro's integer codes, and
    the noise model as the list of ``set_*`` calls that built it.

    Raises:
        ValueError: If the noise model cannot be rebuilt from its recorded
            calls.
    """
    payload = config.model_dump(exclude_none=True, exclude={"noise_model"})
    for name, enum in _ENUMS.items():
        if name in payload:
            payload[name] = enum[payload[name]].value
    if config.noise_model is not None:
        payload["noise_model"] = _noise_model_to_payload(config.noise_model)
    return {_SERVICE_FIELD_NAMES.get(key, key): value for key, value in payload.items()}


def maestro_config_from_payload(data: Mapping[str, Any]) -> MaestroConfig:
    """Rebuild a :class:`MaestroConfig` from the Qoro Service's ``maestro_config``.

    The inverse of :func:`maestro_config_to_payload`; the noise model is rebuilt
    by replaying its recorded ``set_*`` calls. Keys that are not
    :class:`MaestroConfig` fields are dropped with a warning.

    Raises:
        ValueError: If an enum code is not one of maestro's, or a recorded
            noise-model call is not a ``set_*`` method.
    """
    fields: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in data.items():
        name = _FIELD_NAMES_BY_SERVICE_NAME.get(key, key)
        if name not in MaestroConfig.model_fields:
            unknown.append(key)
            continue
        if name in _ENUMS:
            value = _ENUMS[name](value).name
        elif name == "noise_model":
            value = _noise_model_from_payload(value)
        fields[name] = value
    if unknown:
        warnings.warn(
            "Ignoring stored maestro_config keys this version of divi does not "
            f"support: {sorted(unknown)}. Please report this to the divi "
            "maintainers.",
            stacklevel=2,
        )
    return MaestroConfig(**fields)
