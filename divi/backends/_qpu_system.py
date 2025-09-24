# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True, repr=True)
class QPU:
    nickname: str
    q_bits: int
    status: str
    system_kind: str


@dataclass(frozen=True, repr=True)
class QPUSystem:
    name: str
    qpus: list[QPU]
    access_level: str
