# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The saved Qoro Service defaults ``divi-config-helper`` writes and ``QoroService`` reads."""

import json
import os
import tempfile
from pathlib import Path

import platformdirs

from ._config import DeviceConfig, JobConfig, describe_unknown
from ._systems import QPUSystem, SimulatorCluster
from .runners._maestro import MaestroConfig
from .runners._maestro_payload import (
    _PAYLOAD_KEYS,
    maestro_config_from_payload,
    maestro_config_to_payload,
)

SavedDefaults = tuple[JobConfig | None, MaestroConfig | None, DeviceConfig | None]

_SECTIONS = ("job_config", "maestro_config", "device_config")


def defaults_path() -> Path:
    """``qoro.json`` in divi's user config folder."""
    return platformdirs.user_config_path("divi") / "qoro.json"


def _job_config_to_json(config: JobConfig) -> dict:
    return {
        name: value.name if isinstance(value, QPUSystem | SimulatorCluster) else value
        for name, value in config._set_fields().items()
        if value is not None
    }


def _maestro_config_from_json(data: dict) -> MaestroConfig:
    if unknown := describe_unknown(data, _PAYLOAD_KEYS):
        raise ValueError(f"maestro_config has unknown options {unknown}.")
    try:
        return maestro_config_from_payload(data)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"maestro_config is not valid: {exc!r}.") from exc


def load_defaults(path: Path | None = None) -> SavedDefaults:
    """Read the saved ``job_config``, ``maestro_config`` and ``device_config``.

    Each is ``None`` when not saved, all three when the file does not exist. The
    ``JobConfig`` holds only the fields that were saved, with targets as names.

    Args:
        path: The file to read; :func:`defaults_path` when omitted.

    Raises:
        ValueError: If the file cannot be read or is not valid, naming the file.
    """
    path = path or defaults_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None, None
    except OSError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object.")
        if unknown := describe_unknown(data, _SECTIONS):
            raise ValueError(f"unknown sections {unknown}.")
        for name, section in data.items():
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a JSON object.")
        job, maestro_config, device = (data.get(name) for name in _SECTIONS)
        return (
            None if job is None else JobConfig(**job),
            (
                None
                if maestro_config is None
                else _maestro_config_from_json(maestro_config)
            ),
            None if device is None else DeviceConfig.from_payload(device),
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def save_defaults(
    path: Path,
    job_config: JobConfig | None = None,
    maestro_config: MaestroConfig | None = None,
    device_config: DeviceConfig | None = None,
) -> None:
    """Write the configs to ``path``, leaving out ``None`` ones.

    The file is replaced in one step, so a failed write leaves the old one intact.
    """
    data = {}
    if job_config is not None:
        data["job_config"] = _job_config_to_json(job_config)
    if maestro_config is not None:
        data["maestro_config"] = maestro_config_to_payload(maestro_config)
    if device_config is not None:
        data["device_config"] = device_config.to_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise
