# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contract fixtures for optimizer tests."""

import pytest

from tests.qprog.optimizers._optimizer_contracts import (
    GRADIENT_OPTIMIZER_CONTRACTS,
    NOISY_OPTIMIZER_CONTRACTS,
    OPTIMIZER_CONTRACTS,
)


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
