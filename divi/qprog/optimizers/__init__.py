# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizers for variational quantum algorithms.

Each optimizer lives in its own submodule; this package re-exports the public
API so ``from divi.qprog.optimizers import X`` resolves regardless of layout.
"""

from ._base import Optimizer, argmin_finite
from ._grid_search import GridSearchOptimizer
from ._monte_carlo import MonteCarloOptimizer, MonteCarloState
from ._pymoo import PymooMethod, PymooOptimizer, PymooState
from ._qng import QNGOptimizer
from ._rosalin import RosalinOptimizer
from ._scipy import ScipyMethod, ScipyOptimizer
from ._spsa import QNSPSAOptimizer, QUIVEROptimizer, SPSAOptimizer

__all__ = [
    "GridSearchOptimizer",
    "MonteCarloOptimizer",
    "MonteCarloState",
    "Optimizer",
    "PymooMethod",
    "PymooOptimizer",
    "PymooState",
    "QNGOptimizer",
    "QNSPSAOptimizer",
    "QUIVEROptimizer",
    "RosalinOptimizer",
    "SPSAOptimizer",
    "ScipyMethod",
    "ScipyOptimizer",
    "argmin_finite",
]
