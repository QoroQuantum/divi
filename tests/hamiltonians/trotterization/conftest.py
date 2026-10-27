# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from qiskit.quantum_info import SparsePauliOp

from tests.hamiltonians.trotterization._helpers import SIMPLE_HAMILTONIAN


@pytest.fixture
def simple_hamiltonian() -> SparsePauliOp:
    """Three-term SPO over 2 qubits (coeffs 1.0, 2.0, 3.0)."""
    return SIMPLE_HAMILTONIAN
