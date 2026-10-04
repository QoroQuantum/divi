# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from divi._optional import import_optional

from ._partitioning_ensemble import PartitioningProgramEnsemble
from ._time_evolution_trajectory import TimeEvolutionTrajectory
from ._vqe_sweep import MoleculeTransformer, VQEHyperparameterSweep

if TYPE_CHECKING:
    from ._lassqd import (
        LASSQD,
        CCSDPreparation,
        FragmentationConfig,
        FragmentSpec,
        FragmentState,
        FullOrbitalSolve,
        LASSQDRoundReport,
        LASSQDState,
        LinearMethodPreparation,
        SecondOrderOrbitalSolve,
        SQDConfig,
        VQEPreparation,
    )

_LASSQD_EXPORTS = (
    "CCSDPreparation",
    "FragmentSpec",
    "FragmentState",
    "FragmentationConfig",
    "FullOrbitalSolve",
    "LASSQD",
    "LASSQDRoundReport",
    "LASSQDState",
    "LinearMethodPreparation",
    "SecondOrderOrbitalSolve",
    "SQDConfig",
    "VQEPreparation",
)

__all__ = [
    *_LASSQD_EXPORTS,
    "MoleculeTransformer",
    "PartitioningProgramEnsemble",
    "TimeEvolutionTrajectory",
    "VQEHyperparameterSweep",
]


def __getattr__(name: str) -> Any:
    """Resolve the LASSQD exports on first access."""
    if name in _LASSQD_EXPORTS:
        for module_name in ("pyscf", "ffsim"):
            import_optional(module_name, extra="chem", capability="LASSQD")
        from . import _lassqd

        value = getattr(_lassqd, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
