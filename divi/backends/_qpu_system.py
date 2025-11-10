# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Data models for Quantum Processing Units (QPUs) and QPUSystems."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

_AVAILABLE_QPU_SYSTEMS: dict[str, QPUSystem] = {}


@dataclass(frozen=True, repr=True)
class QPU:
    """Represents a single Quantum Processing Unit (QPU).

    Attributes:
        nickname: The unique name or identifier for the QPU.
        q_bits: The number of qubits in the QPU.
        status: The current operational status of the QPU.
        system_kind: The type of technology the QPU uses.
    """

    nickname: str
    q_bits: int
    status: str
    system_kind: str


@dataclass(frozen=True, repr=True)
class QPUSystem:
    """Represents a collection of QPUs that form a quantum computing system.

    Attributes:
        name: The name of the QPU system.
        qpus: A list of QPU objects that are part of this system.
        access_level: The access level granted to the user for this system (e.g., 'PUBLIC').
        supports_expval: Whether the system supports expectation value jobs.
    """

    name: str
    qpus: list[QPU] = field(default_factory=list)
    access_level: str = "PUBLIC"
    supports_expval: bool = False


def parse_qpu_systems(json_data: list) -> list[QPUSystem]:
    """Parses a list of QPU system data from JSON into QPUSystem objects."""
    return [
        QPUSystem(
            name=system_data["name"],
            qpus=[QPU(**qpu) for qpu in system_data.get("qpus", [])],
            access_level=system_data["access_level"],
        )
        for system_data in json_data
    ]


def update_qpu_systems_cache(systems: list[QPUSystem]):
    """Updates the cache of available QPU systems."""
    _AVAILABLE_QPU_SYSTEMS.clear()
    for system in systems:
        if system.name == "qoro_maestro":
            system = replace(system, supports_expval=True)
        _AVAILABLE_QPU_SYSTEMS[system.name] = system


def get_qpu_system(name: str) -> QPUSystem:
    """
    Get a QPUSystem object by its name from the cache.

    Args:
        name: The name of the QPU system to retrieve.

    Returns:
        The QPUSystem object with the matching name.

    Raises:
        ValueError: If the cache is empty or the system is not found.
    """
    if not _AVAILABLE_QPU_SYSTEMS:
        raise ValueError(
            "QPU systems cache is empty. Call `QoroService.fetch_qpu_systems()` to populate it."
        )
    try:
        return _AVAILABLE_QPU_SYSTEMS[name]
    except KeyError:
        raise ValueError(f"QPUSystem with name '{name}' not found in cache.") from None


def get_available_qpu_systems() -> list[QPUSystem]:
    """Returns a list of all available QPU systems from the cache."""
    return list(_AVAILABLE_QPU_SYSTEMS.values())
