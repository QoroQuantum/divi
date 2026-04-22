# SPDX-FileCopyrightText: 2025 ORCA Computing Limited
# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored subset of ``orcacomputing/loop-progressive-simulator``.

Upstream: https://github.com/orcacomputing/loop-progressive-simulator
Commit: 941400e28594270391ec9bde14cb1bbf0af2a8a2 (2026-02-11)

Vendored because the upstream package uses absolute imports and is not
published as an installable distribution. Only the files needed for the
public TBI sampling API are included, with imports rewritten to relative.
"""

from .bscircuits import (
    AbstractBSCircuit,
    ConcreteBSCircuit,
    multiple_loops,
    progressive_decomposition,
)
from .fock_states import SparseBosonicFockState
from .step_simulator import progressive_simulation

__all__ = [
    "AbstractBSCircuit",
    "ConcreteBSCircuit",
    "SparseBosonicFockState",
    "multiple_loops",
    "progressive_decomposition",
    "progressive_simulation",
]
