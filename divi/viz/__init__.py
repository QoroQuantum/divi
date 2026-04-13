# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Loss-landscape visualization for variational programs.

Provides one- and two-dimensional objective scans and a PCA-based plane scan.
Evaluations use Divi's batched cost path. On
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`
subclasses, the same entry points are available as ``program.viz``.

Attribution for orqviz-aligned geometry is recorded in
``LICENSES/ORQViz-Apache-2.0-acknowledgement.txt``.
"""

from ._api import (
    ProgramViz,
    compute_hessian,
    fourier_analysis_2d,
    run_neb,
    scan_1d,
    scan_2d,
    scan_interp_1d,
    scan_interp_2d,
    scan_pca,
)
from ._gradients import GradientMethod
from ._periodic import periodic_trajectory_wrap, periodic_wrap
from ._results import (
    Fourier2DResult,
    HessianResult,
    NEBResult,
    PCAScanResult,
    Scan1DResult,
    Scan2DResult,
)

__all__ = [
    "Fourier2DResult",
    "GradientMethod",
    "HessianResult",
    "NEBResult",
    "PCAScanResult",
    "ProgramViz",
    "Scan1DResult",
    "Scan2DResult",
    "compute_hessian",
    "fourier_analysis_2d",
    "periodic_trajectory_wrap",
    "periodic_wrap",
    "run_neb",
    "scan_1d",
    "scan_2d",
    "scan_interp_1d",
    "scan_interp_2d",
    "scan_pca",
]
