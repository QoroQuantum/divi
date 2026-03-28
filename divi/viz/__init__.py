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

from ._api import ProgramViz, scan_1d, scan_2d, scan_pca
from ._results import PCAScanResult, Scan1DResult, Scan2DResult

__all__ = [
    "ProgramViz",
    "PCAScanResult",
    "Scan1DResult",
    "Scan2DResult",
    "scan_1d",
    "scan_2d",
    "scan_pca",
]
