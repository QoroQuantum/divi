# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi.utils import (
    _is_sanitized,
    clean_hamiltonian,
    convert_qubo_matrix_to_pennylane_ising,
    hamiltonian_to_pauli_string,
)


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


class TestCleanHamiltonian:
    """
    A test class for the clean_hamiltonian function.
    """

    def test_clean_hamiltonian_none(self):
        """
        Tests that clean_hamiltonian returns empty Hamiltonian and 0.0 constant
        when given None.
        """
        hamiltonian, constant = clean_hamiltonian(None)
        assert isinstance(hamiltonian, qml.Hamiltonian)
        assert len(hamiltonian.terms()[0]) == 0
        assert constant == 0.0

    @pytest.mark.parametrize(
        "hamiltonian, expected_op, expected_constant",
        [
            # Test case 1: Hamiltonian with no constant part
            (
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                0.0,
            ),
            # Test case 2: Hamiltonian with only constant parts
            (
                qml.sum(
                    qml.s_prod(2.5, qml.Identity(0)), qml.s_prod(1.5, qml.Identity(1))
                ),
                qml.Hamiltonian([], []),
                4.0,
            ),
            # Test case 3: Mixed Hamiltonian
            (
                qml.sum(
                    qml.s_prod(2, qml.PauliX(0)),
                    qml.s_prod(3, qml.Identity(0)),
                    qml.PauliZ(1),
                ),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                3.0,
            ),
            # Test case 4: Single Identity operator
            (qml.Identity(0), qml.Hamiltonian([], []), 1.0),
            # Test case 5: Single scaled Identity operator
            (qml.s_prod(5.0, qml.Identity(0)), qml.Hamiltonian([], []), 5.0),
            # Test case 6: Product of Identities
            (qml.prod(qml.Identity(0), qml.Identity(1)), qml.Hamiltonian([], []), 1.0),
            # Test case 7: Single non-constant operator
            (qml.PauliZ(0), qml.PauliZ(0), 0.0),
            # Test case 8: Empty Hamiltonian
            (qml.Hamiltonian([], []), qml.Hamiltonian([], []), 0.0),
        ],
    )
    def test_clean_hamiltonian_various_cases(
        self, hamiltonian, expected_op, expected_constant
    ):
        """
        Tests the clean_hamiltonian function with various scenarios.
        """
        new_hamiltonian, constant = clean_hamiltonian(hamiltonian)

        assert np.isclose(constant, expected_constant)

        try:
            # qml.equal can be sensitive, so we compare terms for robustness
            new_coeffs, new_ops = new_hamiltonian.terms()
            exp_coeffs, exp_ops = expected_op.terms()

            assert len(new_coeffs) == len(exp_coeffs)

            if len(new_coeffs) > 0:
                # Create dictionaries for easy lookup and comparison
                new_map = {op.hash: coeff for coeff, op in zip(new_coeffs, new_ops)}
                exp_map = {op.hash: coeff for coeff, op in zip(exp_coeffs, exp_ops)}

                assert new_map.keys() == exp_map.keys()
                for key in new_map:
                    assert np.isclose(new_map[key], exp_map[key])
        except qml.operation.TermsUndefinedError:
            # Fallback for operators that don't have .terms()
            assert qml.equal(new_hamiltonian, expected_op)


class TestHamiltonianToPauliString:
    """
    A test class for the hamiltonian_to_pauli_string function.
    """

    @pytest.mark.parametrize(
        "hamiltonian, n_qubits, expected",
        [
            # Test single Pauli operators
            (qml.Hamiltonian([1.0], [qml.PauliX(0)]), 1, "X"),
            (qml.Hamiltonian([1.0], [qml.PauliY(0)]), 1, "Y"),
            (qml.Hamiltonian([1.0], [qml.PauliZ(0)]), 1, "Z"),
            # Test multiple qubits
            (
                qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)]),
                3,
                "XYZ",
            ),
            # Test multiple terms separated by semicolon
            (
                qml.Hamiltonian([1.0, 1.0], [qml.PauliX(0), qml.PauliY(1)]),
                2,
                "XI;IY",
            ),
            # Test scaled operator (SProd)
            (
                qml.Hamiltonian([1.0], [qml.s_prod(2.5, qml.PauliX(0))]),
                1,
                "X",
            ),
            # Test product operator
            (
                qml.Hamiltonian([1.0], [qml.prod(qml.PauliX(0), qml.PauliZ(1))]),
                2,
                "XZ",
            ),
            # Test identity fills remaining qubits
            (
                qml.Hamiltonian([1.0], [qml.PauliX(2)]),
                3,
                "IIX",
            ),
            # Test complex Hamiltonian with multiple terms
            (
                qml.Hamiltonian(
                    [1.0, 1.0, 1.0],
                    [
                        qml.PauliX(0) @ qml.PauliY(1),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                        qml.PauliX(2),
                    ],
                ),
                3,
                "XYI;IZZ;IIX",
            ),
            # Test a single operator not wrapped in a Hamiltonian
            (qml.PauliZ(0), 1, "Z"),
            # Test with qml.sum
            (qml.sum(qml.PauliX(0), qml.PauliZ(1)), 2, "XI;IZ"),
            # Test with nested s_prod (coefficients are ignored)
            (qml.s_prod(2.0, qml.s_prod(1.5, qml.PauliX(0))), 1, "X"),
            # Test empty Hamiltonian
            (qml.Hamiltonian([], []), 3, ""),
            # Test that qml.Identity is simplified away in a product
            (qml.PauliX(0) @ qml.Identity(1), 2, "XI"),
            # Test sum of prods
            (
                qml.sum(
                    qml.prod(qml.PauliX(0), qml.PauliY(1)),
                    qml.prod(qml.PauliZ(1), qml.PauliZ(2)),
                ),
                3,
                "XYI;IZZ",
            ),
            # Test that Identity is handled correctly
            (qml.Identity(0), 2, "II"),
            (qml.Hamiltonian([1.0], [qml.Identity(0)]), 1, "I"),
        ],
    )
    def test_hamiltonian_conversions(self, hamiltonian, n_qubits, expected):
        """Test various Hamiltonian conversion scenarios."""
        result = hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)
        assert result == expected

    def test_unknown_pauli_operator_raises_error(self):
        """Test that unknown Pauli operators raise ValueError."""

        # Create a mock operator with an unknown name
        class UnknownPauli(qml.operation.Operator):
            def __init__(self, wires):
                super().__init__(wires=wires)
                self.name = "UnknownOperator"

        unknown_op = UnknownPauli(wires=[0])
        hamiltonian = qml.Hamiltonian([1.0], [unknown_op])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

    def test_pauli_with_no_wires_raises_error(self):
        """Test that Pauli operators with no wires raise ValueError."""

        # Create a mock operator with no wires
        class NoWirePauli(qml.operation.Operator):
            def __init__(self):
                super().__init__(wires=[])
                self.name = "PauliX"

        no_wire_op = NoWirePauli()
        hamiltonian = qml.Hamiltonian([1.0], [no_wire_op])

        with pytest.raises(ValueError, match="has no wires"):
            hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

    @pytest.mark.parametrize(
        "wire, n_qubits",
        [
            (-1, 1),  # Negative wire index
            (2, 2),  # Wire index >= n_qubits
            (5, 3),  # Wire index much larger than n_qubits
        ],
    )
    def test_wire_index_out_of_range_raises_error(self, wire, n_qubits):
        """Test that wire indices out of range raise ValueError."""
        hamiltonian = qml.Hamiltonian([1.0], [qml.PauliX(wire)])
        with pytest.raises(ValueError, match="Wire index.*out of range"):
            hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)
