# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contract fixtures for optimizer tests."""

import pytest

from tests.qprog.optimizers._helpers import small_vqe
from tests.qprog.optimizers._optimizer_contracts import (
    GRADIENT_OPTIMIZER_CONTRACTS,
    NOISY_OPTIMIZER_CONTRACTS,
    OPTIMIZER_CONTRACTS,
)


@pytest.fixture
def toy_vqe(default_test_simulator, default_optimizer):
    """:func:`small_vqe` on a real analytic backend."""
    return small_vqe(default_test_simulator, default_optimizer)


@pytest.fixture
def injectable_vqe(dummy_simulator, default_optimizer):
    """:func:`small_vqe` whose metric measurement seam is meant to be patched."""
    return small_vqe(dummy_simulator, default_optimizer)


def _contract_id(contract):
    return contract.__name__.removeprefix("verify_")


@pytest.fixture(params=OPTIMIZER_CONTRACTS, ids=_contract_id)
def optimizer_contract(request):
    return request.param


@pytest.fixture(params=GRADIENT_OPTIMIZER_CONTRACTS, ids=_contract_id)
def gradient_optimizer_contract(request):
    return request.param


@pytest.fixture(params=NOISY_OPTIMIZER_CONTRACTS, ids=_contract_id)
def noisy_optimizer_contract(request):
    return request.param
