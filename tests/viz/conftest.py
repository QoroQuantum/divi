# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the visualisation tests.

Every landscape tool (scans, NEB, Hessian) needs the same thing: a cheap
program with a smooth, analytically known cost surface.
"""

import pytest
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.qprog import VQE, GenericLayerAnsatz


@pytest.fixture
def basic_ansatz():
    return GenericLayerAnsatz([RYGate, RZGate])


@pytest.fixture
def vqe_program(dummy_simulator, basic_ansatz, default_optimizer):
    """Single-qubit ``<Z>`` VQE — the smallest program with a real cost surface."""
    return VQE(
        hamiltonian=SparsePauliOp("Z"),
        n_electrons=1,
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
