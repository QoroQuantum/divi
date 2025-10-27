# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Data models for Quantum Processing Units (QPUs) and QPUSystems."""

from dataclasses import dataclass


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
        access_level: The access level granted to the user for this system (e.g., 'basic').
    """

    name: str
    qpus: list[QPU]
    access_level: str
