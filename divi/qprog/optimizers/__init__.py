# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizers for variational quantum algorithms.

Each optimizer lives in its own submodule; this package re-exports the public
API so ``from divi.qprog.optimizers import X`` resolves regardless of layout.
The natural-gradient metric estimators live in :mod:`divi.qprog._metrics` and
are re-exported here for convenience.
"""

from divi.qprog._metrics import (
    FubiniStudyMetricEstimator,
    MetricEstimator,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
)
from divi.qprog.optimizers._base import Optimizer
from divi.qprog.optimizers._grid_search import GridSearchOptimizer
from divi.qprog.optimizers._monte_carlo import MonteCarloOptimizer, MonteCarloState
from divi.qprog.optimizers._pymoo import PymooMethod, PymooOptimizer, PymooState
from divi.qprog.optimizers._qng import QNGOptimizer
from divi.qprog.optimizers._scipy import ScipyMethod, ScipyOptimizer
from divi.qprog.optimizers._spsa import QNSPSAOptimizer, SPSAOptimizer

__all__ = [
    "FubiniStudyMetricEstimator",
    "GridSearchOptimizer",
    "MetricEstimator",
    "MonteCarloOptimizer",
    "MonteCarloState",
    "Optimizer",
    "PullbackMetricEstimator",
    "PymooMethod",
    "PymooOptimizer",
    "PymooState",
    "QNGOptimizer",
    "QNSPSAOptimizer",
    "SPSAOptimizer",
    "ScipyMethod",
    "ScipyOptimizer",
    "StochasticFidelityMetricEstimator",
]
