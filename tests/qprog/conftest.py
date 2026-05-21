# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for qprog tests."""

import pytest

from divi.qprog import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.optimizers import PymooMethod, PymooOptimizer

_ALL_OPTIMIZERS = [
    ("MonteCarlo", lambda: MonteCarloOptimizer(population_size=5, n_best_sets=2)),
    ("L_BFGS_B", lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B)),
    ("COBYLA", lambda: ScipyOptimizer(method=ScipyMethod.COBYLA)),
    ("NELDER_MEAD", lambda: ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)),
    ("CMAES", lambda: PymooOptimizer(method=PymooMethod.CMAES, population_size=10)),
    ("DE", lambda: PymooOptimizer(method=PymooMethod.DE, population_size=5)),
]
# Subset that supports save/load checkpointing.
_CHECKPOINTING_OPTIMIZERS = [
    (opt_id, factory)
    for opt_id, factory in _ALL_OPTIMIZERS
    if opt_id in {"MonteCarlo", "CMAES", "DE"}
]


def _fixture_kwargs(variants):
    ids, factories = zip(*variants)
    return {"params": list(factories), "ids": list(ids)}


@pytest.fixture(**_fixture_kwargs(_ALL_OPTIMIZERS))
def optimizer(request):
    """Parametrize over every supported optimizer variant. Returns a fresh instance."""
    return request.param()


@pytest.fixture(**_fixture_kwargs(_CHECKPOINTING_OPTIMIZERS))
def checkpointing_optimizer(request):
    """Parametrize over optimizers that support save/load checkpointing."""
    return request.param()
