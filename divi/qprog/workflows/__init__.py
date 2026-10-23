# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from ._partitioning_ensemble import PartitioningProgramEnsemble
from ._time_evolution_trajectory import TimeEvolutionTrajectory
from ._vqe_sweep import MoleculeTransformer, VQEHyperparameterSweep

if TYPE_CHECKING:
    from .lassqd import LASSQD

__all__ = [
    "LASSQD",
    "MoleculeTransformer",
    "PartitioningProgramEnsemble",
    "TimeEvolutionTrajectory",
    "VQEHyperparameterSweep",
]


def __getattr__(name: str) -> Any:
    """Resolve ``LASSQD`` on first access."""
    if name == "LASSQD":
        from . import lassqd

        value = lassqd.LASSQD
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
