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

from divi.qprog.algorithms import PCE, QNN, VQE, AngleEmbedding, GenericLayerAnsatz
from divi.qprog.problems import BinaryOptimizationProblem, HamiltonianProblem


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def mock_landscape(mocker):
    """Replace a program's loss evaluation with ``fn`` applied to each parameter
    row; ``GradientMethod.PROGRAM`` derivatives bypass it."""

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


@pytest.fixture
def make_two_qubit_vqe(basic_ansatz, default_optimizer):
    """Build a two-qubit VQE, whose loss is linear in ``<H>``, on ``backend``."""

    def build(backend):
        return VQE(
            HamiltonianProblem(SparsePauliOp(["ZZ", "XI", "IX"], [1.0, 0.7, -0.4])),
            ansatz=basic_ansatz,
            n_layers=2,
            backend=backend,
            optimizer=default_optimizer,
        )

    return build


@pytest.fixture
def exact_vqe(make_two_qubit_vqe, default_test_simulator):
    return make_two_qubit_vqe(default_test_simulator)


@pytest.fixture
def make_supervised_qnn(basic_ansatz, default_optimizer):
    """Build a labelled QNN, whose squared-error loss is non-linear in ``<Z>``,
    on ``backend``."""

    def build(backend):
        return QNN(
            1,
            AngleEmbedding(rotation="Y"),
            basic_ansatz,
            np.array([[0.1], [0.7]]),
            labels=np.array([0.5, -0.5]),
            backend=backend,
            optimizer=default_optimizer,
        )

    return build


@pytest.fixture
def supervised_qnn(make_supervised_qnn, default_test_simulator):
    return make_supervised_qnn(default_test_simulator)


@pytest.fixture
def make_pce(basic_ansatz, default_optimizer):
    """Build a PCE program, whose objective is non-linear in its expectation
    values, on ``backend``."""

    def build(backend):
        return PCE(
            problem=BinaryOptimizationProblem(np.array([[1.0, -0.6], [-0.6, 2.0]])),
            ansatz=basic_ansatz,
            n_layers=2,
            backend=backend,
            optimizer=default_optimizer,
        )

    return build


@pytest.fixture
def pce_program(make_pce, default_test_simulator):
    return make_pce(default_test_simulator)
