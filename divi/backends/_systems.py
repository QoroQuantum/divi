# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Data models for Quantum Processing Units (QPUs), QPUSystems, and SimulatorClusters."""

from __future__ import annotations

from threading import RLock

from pydantic import BaseModel, ConfigDict, Field

_AVAILABLE_QPU_SYSTEMS: dict[str, QPUSystem] = {}
_AVAILABLE_SIMULATOR_CLUSTERS: dict[str, SimulatorCluster] = {}
_CACHE_LOCK = RLock()


class _Target(BaseModel):
    """Base for the targets the Qoro API describes.

    Unknown fields are ignored rather than rejected: the service adds fields to
    these payloads over time, and a client that refuses them cannot be upgraded
    independently of the server. Frozen so they stay safe to cache.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")


class QPU(_Target):
    """Represents a single Quantum Processing Unit (QPU)."""

    nickname: str
    """The unique name or identifier for the QPU."""

    q_bits: int
    """The number of qubits in the QPU."""

    status: str
    """The current operational status of the QPU."""

    system_kind: str
    """The type of technology the QPU uses."""

    vendor_type: str | None = None
    """The vendor this QPU routes to, when the service reports one."""


class QPUSystem(_Target):
    """Represents a collection of QPUs that form a quantum computing system."""

    name: str
    """The name of the QPU system."""

    qpus: list[QPU] = Field(default_factory=list)
    """A list of QPU objects that are part of this system."""

    access_level: str = "PUBLIC"
    """The access level granted to the user for this system (e.g., 'PUBLIC')."""

    supports_expval: bool = False
    """Whether the system supports expectation value jobs."""


class SimulatorCluster(_Target):
    """Represents a simulator cluster for cloud-based quantum simulation."""

    name: str
    """The name of the simulator cluster."""

    access_level: str = "PUBLIC"
    """The access level for this cluster (e.g., 'PUBLIC', 'ADMIN')."""

    minimum_tier: str = "FREE"
    """The minimum token tier required to access this cluster."""

    supports_expval: bool = True
    """Whether the cluster supports expectation value jobs."""


def parse_qpu_systems(json_data: list) -> list[QPUSystem]:
    """Parses a list of QPU system data from JSON into QPUSystem objects."""
    return [QPUSystem.model_validate(system_data) for system_data in json_data]


def update_qpu_systems_cache(systems: list[QPUSystem]):
    """Updates the cache of available QPU systems."""
    with _CACHE_LOCK:
        _AVAILABLE_QPU_SYSTEMS.clear()
        for system in systems:
            if system.name == "qoro_maestro":
                system = system.model_copy(update={"supports_expval": True})
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
    with _CACHE_LOCK:
        if not _AVAILABLE_QPU_SYSTEMS:
            raise ValueError(
                "QPU systems cache is empty. Call `QoroService.fetch_qpu_systems()` to populate it."
            )
        try:
            return _AVAILABLE_QPU_SYSTEMS[name]
        except KeyError:
            raise ValueError(
                f"QPUSystem with name '{name}' not found in cache."
            ) from None


def get_available_qpu_systems() -> list[QPUSystem]:
    """Returns a list of all available QPU systems from the cache."""
    with _CACHE_LOCK:
        return list(_AVAILABLE_QPU_SYSTEMS.values())


def parse_simulator_clusters(json_data: list) -> list[SimulatorCluster]:
    """Parses a list of simulator cluster data from JSON into SimulatorCluster objects."""
    return [SimulatorCluster.model_validate(cluster_data) for cluster_data in json_data]


def update_simulator_clusters_cache(clusters: list[SimulatorCluster]):
    """Updates the cache of available simulator clusters."""
    with _CACHE_LOCK:
        _AVAILABLE_SIMULATOR_CLUSTERS.clear()
        for cluster in clusters:
            _AVAILABLE_SIMULATOR_CLUSTERS[cluster.name] = cluster


def get_simulator_cluster(name: str) -> SimulatorCluster:
    """Get a SimulatorCluster object by its name from the cache.

    Args:
        name: The name of the simulator cluster to retrieve.

    Returns:
        The SimulatorCluster object with the matching name.

    Raises:
        ValueError: If the cache is empty or the cluster is not found.
    """
    with _CACHE_LOCK:
        if not _AVAILABLE_SIMULATOR_CLUSTERS:
            raise ValueError(
                "Simulator clusters cache is empty. "
                "Call `QoroService.fetch_simulator_clusters()` to populate it."
            )
        try:
            return _AVAILABLE_SIMULATOR_CLUSTERS[name]
        except KeyError:
            raise ValueError(
                f"SimulatorCluster with name '{name}' not found in cache."
            ) from None


def get_available_simulator_clusters() -> list[SimulatorCluster]:
    """Returns a list of all available simulator clusters from the cache."""
    with _CACHE_LOCK:
        return list(_AVAILABLE_SIMULATOR_CLUSTERS.values())
