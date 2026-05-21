# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Hamiltonian tests."""

import numpy as np
import pennylane as qp
from qiskit.quantum_info import SparsePauliOp


def assert_matches_pennylane(actual: SparsePauliOp, expected_pl) -> None:
    """Matrix-equality check against a PennyLane operator.

    ``wire_order=reversed(range(n))`` reconciles PennyLane's big-endian
    wire convention with Qiskit's little-endian ``to_matrix()`` output.
    """
    n = actual.num_qubits
    expected_mat = qp.matrix(expected_pl, wire_order=list(reversed(range(n))))
    np.testing.assert_allclose(actual.to_matrix(), expected_mat, atol=1e-10)
