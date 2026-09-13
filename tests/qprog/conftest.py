# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for qprog tests."""

import pytest

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.reporting._session import ProgressSession
from tests.qprog._helpers import RecordingProgressSession

_OPTIMIZER_FACTORIES = [
    ("monte-carlo", lambda: MonteCarloOptimizer(population_size=5, n_best_sets=2)),
    ("l-bfgs-b", lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B)),
    ("cobyla", lambda: ScipyOptimizer(method=ScipyMethod.COBYLA)),
    ("nelder-mead", lambda: ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)),
    ("cmaes", lambda: PymooOptimizer(method=PymooMethod.CMAES, population_size=10)),
    ("de", lambda: PymooOptimizer(method=PymooMethod.DE, population_size=5)),
]
_CHECKPOINTING_OPTIMIZERS = [
    variant
    for variant in _OPTIMIZER_FACTORIES
    if variant[0] in {"monte-carlo", "cmaes", "de"}
]


@pytest.fixture
def recording_direct_sessions(monkeypatch):
    """Route direct ownership through real sessions exposed to the test."""
    sessions: list[RecordingProgressSession] = []

    def make_direct(state, **kwargs):
        del kwargs
        session = RecordingProgressSession(state)
        sessions.append(session)
        return session

    monkeypatch.setattr(ProgressSession, "direct", staticmethod(make_direct))
    return sessions


def _fixture_kwargs(variants):
    ids, factories = zip(*variants)
    return {"params": list(factories), "ids": list(ids)}


@pytest.fixture(**_fixture_kwargs(_OPTIMIZER_FACTORIES))
def optimizer(request):
    """Parametrize over every general optimizer used by qprog tests."""
    return request.param()


@pytest.fixture(
    **_fixture_kwargs(
        [variant for variant in _OPTIMIZER_FACTORIES if variant[0] != "l-bfgs-b"]
    )
)
def gradient_free_optimizer(request):
    """Parametrize over general optimizers that need no exact gradient.

    For programs with no exact parameter-shift rule, such as
    :class:`~divi.qprog.algorithms.QAOA`.
    """
    return request.param()


@pytest.fixture(**_fixture_kwargs(_CHECKPOINTING_OPTIMIZERS))
def checkpointing_optimizer(request):
    """Parametrize over optimizers that support save/load checkpointing."""
    return request.param()
