# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi.utils import _is_sanitized, convert_qubo_matrix_to_pennylane_ising


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.array([[1, 2], [2, 3]]), True),  # Symmetric dense
        (np.array([[1, 2], [0, 3]]), True),  # Upper triangular dense
        (np.array([[1, 2], [1, 3]]), False),  # Not sanitized dense
        (sps.csr_matrix([[1, 2], [2, 3]]), True),  # Symmetric sparse
        (sps.csr_matrix([[1, 2], [0, 3]]), True),  # Upper triangular sparse
        (sps.csr_matrix([[1, 2], [1, 3]]), False),  # Not sanitized sparse
    ],
)
def test_is_sanitized(matrix, expected):
    """
    Tests the _is_sanitized function with various matrix types.
    """
    assert _is_sanitized(matrix) == expected


class TestQuboToIsingConversion:
    """
    A test class for the convert_qubo_matrix_to_pennylane_ising function.
    """

    def test_symmetric_dense_qubo(self):
        """
        Tests conversion with a simple symmetric dense QUBO matrix.
        """
        qubo_matrix = np.array([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_upper_triangular_dense_qubo(self):
        """
        Tests conversion with an upper triangular dense QUBO matrix.
        The result should be the same as the symmetric equivalent.
        """
        qubo_matrix = np.array([[-1.0, 4.0], [0.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]]
        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_non_sanitized_qubo_raises_warning(self):
        """
        Tests that a non-sanitized QUBO matrix raises a warning and is correctly
        symmetrized.
        """
        qubo_matrix = np.array([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]]
        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_diagonal_qubo(self):
        """
        Tests a purely diagonal QUBO matrix, which should result in only Z terms.
        """
        qubo_matrix = np.array([[1.0, 0.0], [0.0, -2.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = -0.5 * qml.Z(0) + 1.0 * qml.Z(1)
        expected_constant = -0.5

        assert np.isclose(constant, expected_constant)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_sparse_qubo(self):
        """
        Tests conversion with a sparse QUBO matrix.
        """
        qubo_matrix = sps.csr_matrix([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_3x3_qubo(self):
        """
        Tests a larger 3x3 QUBO matrix to ensure correct term summation.
        """
        qubo_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_constant = 8.0
        expected_hamiltonian = (
            0.5 * (qml.Z(0) @ qml.Z(1))
            + 0.75 * (qml.Z(0) @ qml.Z(2))
            + 1.25 * (qml.Z(1) @ qml.Z(2))
            - 1.75 * qml.Z(0)
            - 3.75 * qml.Z(1)
            - 5.0 * qml.Z(2)
        )

        assert np.isclose(constant, expected_constant)

        hamiltonian = hamiltonian.simplify()
        expected_hamiltonian = expected_hamiltonian.simplify()

        h_coeffs, h_ops = hamiltonian.terms()
        e_coeffs, e_ops = expected_hamiltonian.terms()

        # Create dictionaries for easy lookup and comparison for robust comparison
        h_map = {op.hash: coeff for coeff, op in zip(h_coeffs, h_ops)}
        e_map = {op.hash: coeff for coeff, op in zip(e_coeffs, e_ops)}

        assert h_map.keys() == e_map.keys()
        for key in h_map:
            assert np.isclose(h_map[key], e_map[key])
