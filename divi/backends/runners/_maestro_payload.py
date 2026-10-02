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

from ._maestro import _ENUMS, _SIMULATOR_OPTIONS, MaestroConfig

_PAYLOAD_KEYS = _SIMULATOR_OPTIONS | {"noise_model", "noise_seed", "noise_realizations"}


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
    """The model's recorded ``set_*`` calls, checked to replay."""
    calls = [
        {"method": method, "args": list(args), "kwargs": dict(kwargs)}
        for method, args, kwargs in noise_model._call_log
    ]
    try:
        calls = json.loads(json.dumps(calls, default=_to_json))
        _noise_model_from_payload(calls)
    except Exception as exc:
        raise ValueError(
            "noise_model's recorded calls do not replay, so it cannot be sent to "
            f"the Qoro Service: {exc}"
        ) from exc
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

    The simulator options ``config`` sets travel under maestro's own names,
    with the enums as maestro's integer codes; the noise model travels as the
    list of ``set_*`` calls that built it, with ``noise_realizations`` resolved
    to the value a local run would use.

    Raises:
        ValueError: If the noise model cannot be rebuilt from its recorded
            calls.
    """
    payload = {
        name: value.value if name in _ENUMS else value
        for name, value in config._simulator_options().items()
        if value is not None
    }
    if config.noise_seed is not None:
        payload["noise_seed"] = config.noise_seed
    if config.noise_model is not None:
        payload["noise_model"] = _noise_model_to_payload(config.noise_model)
        payload |= config._noise_realization_kwargs()
    elif config.noise_realizations is not None:
        payload["noise_realizations"] = config.noise_realizations
    return payload


def maestro_config_from_payload(data: Mapping[str, Any]) -> MaestroConfig:
    """Rebuild a :class:`MaestroConfig` from the Qoro Service's ``maestro_config``.

    The inverse of :func:`maestro_config_to_payload`; the noise model is rebuilt
    by replaying its recorded ``set_*`` calls. Keys that are neither simulator
    options nor noise fields are dropped with a warning.

    Raises:
        ValueError: If an enum code is not one of maestro's, or a recorded
            noise-model call is not a ``set_*`` method.
    """
    fields: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in data.items():
        if key not in _PAYLOAD_KEYS:
            unknown.append(key)
            continue
        if key in _ENUMS:
            value = _ENUMS[key](value)
        elif key == "noise_model":
            value = _noise_model_from_payload(value)
        fields[key] = value
    if unknown:
        warnings.warn(
            "Ignoring stored maestro_config keys this version of divi does not "
            f"support: {sorted(unknown)}. Please report this to the divi "
            "maintainers.",
            stacklevel=2,
        )
    return MaestroConfig(**fields)
