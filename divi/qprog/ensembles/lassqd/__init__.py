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
from ._preparation import (  # noqa: E402
    LUCJInteraction,
    LUCJPreparation,
    prepare_lucj,
)
from ._sqd import (  # noqa: E402
    SQDConfig,
    SQDResult,
    SQDSolver,
    SQDState,
    compute_spatial_rdm1,
    filter_symmetry,
    probs_to_sqd_bitstrings,
)
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
    "LUCJInteraction",
    "LUCJPreparation",
    "OrbitalUpdate",
    "Preparation",
    "SecondOrderOrbitalUpdate",
    "SQDConfig",
    "SQDResult",
    "SQDSolver",
    "SQDState",
    "StateAveragedOrbitalUpdate",
    "StateTransition",
    "StateWavefunction",
    "VQEPreparation",
    "compute_spatial_rdm1",
    "filter_symmetry",
    "prepare_lucj",
    "probs_to_sqd_bitstrings",
]
