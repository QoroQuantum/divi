# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the visualisation tests.

Every landscape tool (scans, NEB, Hessian) needs the same thing: a cheap
program with a smooth, analytically known cost surface.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.qprog import VQE, GenericLayerAnsatz
from divi.qprog.problems import HamiltonianProblem


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def mock_landscape(mocker):
    """Replace a program's cost evaluation with ``fn`` applied to each parameter row."""

    def install(program, fn):
        def evaluate(param_sets, **kwargs):
            return {i: float(fn(p)) for i, p in enumerate(np.atleast_2d(param_sets))}

        return mocker.patch.object(
            program, "_evaluate_cost_param_sets", side_effect=evaluate
        )

    return install


@pytest.fixture
def basic_ansatz():
    return GenericLayerAnsatz([RYGate, RZGate])


@pytest.fixture
def vqe_program(dummy_simulator, basic_ansatz, default_optimizer):
    """Single-qubit ``<Z>`` VQE — the smallest program with a real cost surface."""
    return VQE(
        HamiltonianProblem(SparsePauliOp("Z"), n_electrons=1),
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
