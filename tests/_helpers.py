# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across the whole test suite."""

import re

import numpy as np
from qiskit import QuantumCircuit


def exact_match(message: str) -> str:
    """``pytest.raises(match=...)`` pattern matching ``message`` and nothing else."""
    return f"^{re.escape(message)}$"


def unitary(simulator, circuit: QuantumCircuit) -> np.ndarray:
    """``circuit``'s unitary, column ``j`` its output state on basis input ``j``,
    with qubit 0 the least-significant bit."""
    n = circuit.num_qubits
    columns = []
    for index in range(2**n):
        prepared = circuit.copy_empty_like()
        prepared.global_phase = 0
        for qubit in range(n):
            if index >> qubit & 1:
                prepared.x(qubit)
        columns.append(simulator.statevector(prepared.compose(circuit)))
    return np.column_stack(columns)


def same_unitary(
    simulator, a: QuantumCircuit, b: QuantumCircuit, atol: float = 1e-8
) -> bool:
    """Whether circuits ``a`` and ``b`` implement one unitary, up to a global
    phase."""
    return equal_up_to_phase(unitary(simulator, a), unitary(simulator, b), atol)


def equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """Whether matrices or states ``a`` and ``b`` differ only by a global phase."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pivot = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(a[pivot]) < atol:
        return False
    phase = b[pivot] / a[pivot]
    return abs(abs(phase) - 1.0) <= atol and np.allclose(a * phase, b, atol=atol)
