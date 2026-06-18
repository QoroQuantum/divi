# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for qprog tests."""

import pytest

from tests.qprog.optimizers._contracts import (
    CHECKPOINTING_VARIANT_IDS,
    OPTIMIZER_VARIANTS,
)

# Subset that supports save/load checkpointing.
_CHECKPOINTING_OPTIMIZERS = [
    (opt_id, factory)
    for opt_id, factory in OPTIMIZER_VARIANTS
    if opt_id in CHECKPOINTING_VARIANT_IDS
]


def _fixture_kwargs(variants):
    ids, factories = zip(*variants)
    return {"params": list(factories), "ids": list(ids)}


@pytest.fixture(**_fixture_kwargs(OPTIMIZER_VARIANTS))
def optimizer(request):
    """Parametrize over every supported optimizer variant. Returns a fresh instance."""
    return request.param()


@pytest.fixture(**_fixture_kwargs(_CHECKPOINTING_OPTIMIZERS))
def checkpointing_optimizer(request):
    """Parametrize over optimizers that support save/load checkpointing."""
    return request.param()
