# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Localised active-space sample-based quantum diagonalisation."""

from divi._optional import import_optional

for _module_name in ("pyscf", "ffsim"):
    import_optional(_module_name, extra="chem", capability="LASSQD")

from ._config import (  # noqa: E402
    CCSDPreparation,
    FixedOrbitals,
    FragmentationConfig,
    LBFGSOrbitalUpdate,
    LinearMethodPreparation,
    OrbitalUpdate,
    Preparation,
    SecondOrderOrbitalUpdate,
    StateAveragedOrbitalUpdate,
    VQEPreparation,
)
from ._sqd import SQDConfig  # noqa: E402
from ._state import (  # noqa: E402
    ElectronicState,
    FragmentSpec,
    FragmentState,
    LASSQDRoundReport,
    LASSQDState,
    StateTransition,
    StateWavefunction,
)
from ._workflow import LASSQD  # noqa: E402

__all__ = [
    "CCSDPreparation",
    "ElectronicState",
    "FixedOrbitals",
    "FragmentationConfig",
    "FragmentSpec",
    "FragmentState",
    "LASSQD",
    "LASSQDRoundReport",
    "LASSQDState",
    "LBFGSOrbitalUpdate",
    "LinearMethodPreparation",
    "OrbitalUpdate",
    "Preparation",
    "SecondOrderOrbitalUpdate",
    "SQDConfig",
    "StateAveragedOrbitalUpdate",
    "StateTransition",
    "StateWavefunction",
    "VQEPreparation",
]
