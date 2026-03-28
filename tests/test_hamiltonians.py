# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dimod
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi import hamiltonians
from divi.hamiltonians import (
    ExactTrotterization,
    IsingResult,
    NativeIsingConverter,
    QDrift,
    QuadratizedIsingConverter,
    _clean_hamiltonian,
    _dense_to_sparse,
    _hamiltonian_term_count,
    _is_sanitized,
    _resolve_ising_converter,
    _sort_hamiltonian_terms,
    compress_ham_ops,
    convert_hamiltonian_to_pauli_string,
    convert_qubo_matrix_to_pennylane_ising,
    encode_ham_ops,
    normalize_binary_polynomial_problem,
    qubo_to_ising,
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


class TestHamiltonianTermCount:
    """Tests for _hamiltonian_term_count helper."""

    def test_multi_term_sum(self, simple_hamiltonian):
        """Multi-term Sum returns correct count."""
        assert _hamiltonian_term_count(simple_hamiltonian) == 3

    def test_single_term_sprod(self):
        """Single-term SProd (e.g. 0.5*Z(0)) returns 1."""
        single_sprod = 0.5 * qml.Z(0)
        assert _hamiltonian_term_count(single_sprod) == 1

    def test_single_term_bare_pauli(self):
        """Single bare Pauli operator returns 1."""
        assert _hamiltonian_term_count(qml.Z(0)) == 1

    def test_empty_hamiltonian(self):
        """Empty Hamiltonian returns 0."""
        empty = qml.Hamiltonian([], [])
        assert _hamiltonian_term_count(empty) == 0


class TestQuboToIsingConversion:
    """
    A test class for the convert_qubo_matrix_to_pennylane_ising function.
    """

    def test_symmetric_dense_qubo(self):
        """
        Tests conversion with a simple symmetric dense QUBO matrix.

        Q = [[-1, 2], [2, -1]] → E(x) = -x₀ + 4x₀x₁ - x₁
        Hand-derived Ising: H = Z₀Z₁ - 0.5·Z₀ - 0.5·Z₁, offset = 0.
        """
        qubo_matrix = np.array([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1)) - 0.5 * qml.Z(0) - 0.5 * qml.Z(1)
        )

        assert np.isclose(constant, 0.0)
        assert qml.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_upper_triangular_dense_qubo(self):
        """
        Tests conversion with an upper triangular dense QUBO matrix.
        The result should be the same as the symmetric equivalent.

        Q = [[-1, 4], [0, -1]] → symmetrized [[-1, 2], [2, -1]] → same Ising.
        """
        qubo_matrix = np.array([[-1.0, 4.0], [0.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1)) - 0.5 * qml.Z(0) - 0.5 * qml.Z(1)
        )

        assert np.isclose(constant, 0.0)
        assert qml.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_non_sanitized_qubo_raises_warning(self):
        """
        Tests that a non-sanitized QUBO matrix raises a warning and is correctly
        symmetrized.
        """
        qubo_matrix = np.array([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]] → same Ising as symmetric test.
        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1)) - 0.5 * qml.Z(0) - 0.5 * qml.Z(1)
        )

        assert np.isclose(constant, 0.0)
        assert qml.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_non_sanitized_sparse_qubo_raises_warning(self):
        """Non-sanitized sparse QUBO raises warning and is symmetrized."""
        qubo_matrix = sps.csc_matrix([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1)) - 0.5 * qml.Z(0) - 0.5 * qml.Z(1)
        )
        assert np.isclose(constant, 0.0)
        assert qml.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

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

        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1)) - 0.5 * qml.Z(0) - 0.5 * qml.Z(1)
        )

        assert np.isclose(constant, 0.0)
        assert qml.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_3x3_qubo(self):
        """
        Tests a larger 3x3 QUBO matrix to ensure correct term summation.
        """
        qubo_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_constant = 10.5
        expected_hamiltonian = (
            1.0 * (qml.Z(0) @ qml.Z(1))
            + 1.5 * (qml.Z(0) @ qml.Z(2))
            + 2.5 * (qml.Z(1) @ qml.Z(2))
            - 3.0 * qml.Z(0)
            - 5.5 * qml.Z(1)
            - 7.0 * qml.Z(2)
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

    @pytest.mark.parametrize(
        "qubo_matrix, expect_warning",
        [
            # Dense symmetric
            (np.array([[-1.0, 2.0], [2.0, -1.0]]), False),
            # Dense upper-triangular
            (np.array([[-1.0, 4.0], [0.0, -1.0]]), False),
            # Dense 3×3 symmetric
            (np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float), False),
            # Sparse symmetric (CSR)
            (sps.csr_matrix([[-1.0, 2.0], [2.0, -1.0]]), False),
            # Sparse upper-triangular (CSC)
            (sps.csc_matrix([[-1.0, 4.0], [0.0, -1.0]]), False),
            # Sparse 3×3 symmetric (COO)
            (
                sps.coo_matrix(
                    np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
                ),
                False,
            ),
            # Non-sanitized dense (triggers warning + double symmetrization)
            (np.array([[-1.0, 3.0], [1.0, -1.0]]), True),
            # Non-sanitized sparse
            (sps.csc_matrix([[-1.0, 3.0], [1.0, -1.0]]), True),
            # Larger 5×5 symmetric
            (
                np.array(
                    [
                        [2, -1, 0, 3, 0],
                        [-1, 4, 1, 0, -2],
                        [0, 1, -3, 2, 1],
                        [3, 0, 2, 1, -1],
                        [0, -2, 1, -1, 5],
                    ],
                    dtype=float,
                ),
                False,
            ),
        ],
        ids=[
            "2x2_sym_dense",
            "2x2_upper_dense",
            "3x3_sym_dense",
            "2x2_sym_sparse",
            "2x2_upper_sparse",
            "3x3_sym_sparse",
            "2x2_nonsanitized_dense",
            "2x2_nonsanitized_sparse",
            "5x5_sym_dense",
        ],
    )
    def test_ising_energies_match_qubo(self, qubo_matrix, expect_warning):
        """Ising energy for every bitstring matches the original QUBO energy.

        This is the definitive correctness test: for each x ∈ {0,1}^n, the
        Ising energy <σ|H|σ> + constant must equal x^T Q x, where σ and x
        are related by x_i = (1 - σ_i) / 2.

        Covers dense, sparse, upper-triangular, non-sanitized (double
        symmetrization), and larger matrix inputs.
        """
        from itertools import product as iterproduct

        q_dense = qubo_matrix.toarray() if sps.issparse(qubo_matrix) else qubo_matrix
        n = q_dense.shape[0]
        Q_sym = (q_dense + q_dense.T) / 2

        if expect_warning:
            with pytest.warns(
                UserWarning, match="neither symmetric nor upper triangular"
            ):
                hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(
                    qubo_matrix
                )
        else:
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        h_matrix = np.real(np.diag(qml.matrix(hamiltonian, wire_order=range(n))))

        for bits in iterproduct((0, 1), repeat=n):
            x = np.array(bits, dtype=float)
            qubo_energy = float(x @ Q_sym @ x)
            # x_i = (1-σ_i)/2 → σ_i=+1 when x_i=0, σ_i=-1 when x_i=1
            # Computational basis index: bit 0 is qubit 0
            idx = sum(b << (n - 1 - i) for i, b in enumerate(bits))
            ising_energy = h_matrix[idx] + constant
            assert ising_energy == pytest.approx(
                qubo_energy
            ), f"x={list(bits)}: ising={ising_energy}, qubo={qubo_energy}"


class TestBinaryToIsingConverters:
    """Tests for Protocol-based binary-to-Ising converters."""

    @staticmethod
    def _eval_polynomial_energy(problem, assignment):
        energy = 0.0
        for term, coeff in problem.terms.items():
            if len(term) == 0:
                energy += coeff
                continue
            prod = 1.0
            for var in term:
                prod *= assignment[var]
            energy += coeff * prod
        return float(energy)

    @staticmethod
    def _eval_ising_energy(operator, constant, assignment, variable_to_idx):
        if _hamiltonian_term_count(operator) == 0:
            return float(constant)

        coeffs, ops = operator.terms()
        z_vals = {i: 1 - (2 * assignment[var]) for var, i in variable_to_idx.items()}
        energy = float(constant)
        for coeff, op in zip(coeffs, ops):
            if isinstance(op, qml.ops.Prod):
                wires = [int(wire) for wire in op.wires]
                parity = np.prod([z_vals[w] for w in wires])
            else:
                parity = z_vals[int(op.wires[0])]
            energy += float(coeff) * float(parity)
        return float(energy)

    def test_native_converter_matches_polynomial_energy(self):
        hubo = {
            ("x0",): -1.0,
            ("x0", "x1"): 2.0,
            ("x0", "x1", "x2"): 4.0,
            (): 0.5,
        }
        problem = normalize_binary_polynomial_problem(
            hubo, variable_order=("x0", "x1", "x2")
        )

        result = NativeIsingConverter().convert(problem)
        assert result.metadata["strategy"] == "native"
        assert len(result.operator.wires) == 3

        # Test decode_fn returns correct assignments
        decoded = result.decode_fn("101")
        assert list(decoded) == [1, 0, 1]

        for x0 in (0, 1):
            for x1 in (0, 1):
                for x2 in (0, 1):
                    assignment = {"x0": x0, "x1": x1, "x2": x2}
                    poly_energy = self._eval_polynomial_energy(problem, assignment)
                    ising_energy = self._eval_ising_energy(
                        result.operator,
                        result.constant,
                        assignment,
                        problem.variable_to_idx,
                    )
                    assert ising_energy == pytest.approx(poly_energy)

    def test_quadratized_converter_introduces_ancillas_for_cubic_term(self):
        problem = normalize_binary_polynomial_problem(
            {("x0", "x1", "x2"): 1.0},
            variable_order=("x0", "x1", "x2"),
        )
        result = QuadratizedIsingConverter(strength=5.0).convert(problem)

        assert result.metadata["strategy"] == "quadratized"
        assert result.metadata["ancilla_count"] >= 1

        # decode_fn should return only original variables
        n_measure_qubits = len(result.operator.wires)
        bitstring = "1" * n_measure_qubits
        decoded = result.decode_fn(bitstring)
        assert len(decoded) == problem.n_vars

        if _hamiltonian_term_count(result.operator) > 0:
            _, ops = result.operator.terms()
            for op in ops:
                locality = len(op.wires)
                assert locality <= 2

    def test_quadratized_converter_has_no_ancillas_for_quadratic_input(self):
        qubo = np.array([[1.0, -0.5], [0.0, 2.0]])
        problem = normalize_binary_polynomial_problem(qubo)
        result = QuadratizedIsingConverter(strength=5.0).convert(problem)

        assert result.metadata["ancilla_count"] == 0

    def test_quadratized_converter_matches_polynomial_energy(self):
        """Ising energy at optimal ancilla setting matches polynomial energy.

        For a cubic polynomial, quadratization introduces ancilla variables.
        For each original-variable assignment, the minimum Ising energy over
        all ancilla assignments should equal the polynomial energy (with
        sufficiently large strength).
        """
        hubo = {
            ("x0",): -1.0,
            ("x0", "x1"): 2.0,
            ("x0", "x1", "x2"): 4.0,
            (): 0.5,
        }
        problem = normalize_binary_polynomial_problem(
            hubo, variable_order=("x0", "x1", "x2")
        )

        result = QuadratizedIsingConverter(strength=100.0).convert(problem)
        n_total = len(result.operator.wires)
        assert result.metadata["ancilla_count"] >= 1

        # Use PennyLane matrix representation for reliable energy evaluation.
        # Explicit wire_order ensures consistent bitstring↔energy mapping
        # regardless of operator wire ordering (which varies with dimod's
        # non-deterministic ancilla naming).
        h_matrix = qml.matrix(result.operator, wire_order=range(n_total))

        # Group all computational-basis energies by decoded original assignment,
        # then verify the minimum matches the polynomial energy.
        from collections import defaultdict
        from itertools import product as iterproduct

        energies_by_assignment = defaultdict(list)
        for val in range(2**n_total):
            bs = format(val, f"0{n_total}b")
            decoded = tuple(int(d) for d in result.decode_fn(bs))
            # Energy = <val|H|val> + constant
            ising_e = float(np.real(h_matrix[val, val])) + result.constant
            energies_by_assignment[decoded].append(ising_e)

        for x0, x1, x2 in iterproduct((0, 1), repeat=3):
            assignment = {"x0": x0, "x1": x1, "x2": x2}
            poly_energy = self._eval_polynomial_energy(problem, assignment)
            best_ising = min(energies_by_assignment[(x0, x1, x2)])
            assert best_ising == pytest.approx(
                poly_energy
            ), f"x={[x0, x1, x2]}: best_ising={best_ising}, poly={poly_energy}"


class TestCleanHamiltonian:
    """
    A test class for the clean_hamiltonian function.
    """

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
        new_hamiltonian, constant = _clean_hamiltonian(hamiltonian)

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
        result = convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)
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
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

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
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

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
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)


class TestCompressedObservables:
    """Tests for sparse+gzip Hamiltonian observable compression."""

    # -- _dense_to_sparse ------------------------------------------------

    @pytest.mark.parametrize(
        "dense, expected",
        [
            ("Z", "Z0"),
            ("IZ", "Z1"),
            ("ZZ", "Z0Z1"),
            ("XYZ", "X0Y1Z2"),
            ("IIII", "I"),
            ("ZIIZ", "Z0Z3"),
            ("XI", "X0"),
        ],
    )
    def test_dense_to_sparse(self, dense, expected):
        assert _dense_to_sparse(dense) == expected

    # -- encode_ham_ops --------------------------------------------------

    def test_encode_prefix(self):
        encoded = encode_ham_ops("ZZII;IZIZ;IIII")
        assert encoded.startswith("@gzs4:")

    def test_encode_large_qubit_count(self):
        dense = "Z" + "I" * 63
        encoded = encode_ham_ops(dense)
        assert encoded.startswith("@gzs64:")

    def test_compress_ham_ops_multi_group(self):
        """Pipe-delimited groups are each independently compressed."""
        group_a = "ZZII;IZIZ"
        group_b = "XXII;IYIZ"
        compressed = compress_ham_ops(f"{group_a}|{group_b}")
        parts = compressed.split("|")
        assert len(parts) == 2
        assert parts[0] == encode_ham_ops(group_a)
        assert parts[1] == encode_ham_ops(group_b)

    def test_compression_ratio(self):
        """Encoding should produce a string shorter than the dense input for large Hamiltonians."""
        # 64-qubit Hamiltonian with 100 sparse terms
        terms = []
        for i in range(100):
            paulis = ["I"] * 64
            paulis[i % 64] = "Z"
            paulis[(i + 1) % 64] = "Z"
            terms.append("".join(paulis))
        dense = ";".join(terms)
        encoded = encode_ham_ops(dense)
        assert len(encoded) < len(dense)


@pytest.fixture
def simple_hamiltonian():
    """Hamiltonian with three terms (coefficients 1.0, 2.0, 3.0) for Trotterization tests."""
    return (1.0 * qml.Z(0) + 2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify()


class TestExactTrotterization:
    """
    Tests for ExactTrotterization strategy (public API and specified behavior).
    """

    def test_both_keep_fraction_and_keep_top_n_raises(self):
        """At most one of keep_fraction or keep_top_n may be provided."""
        with pytest.raises(
            ValueError, match="At most one of keep_fraction or keep_top_n"
        ):
            ExactTrotterization(keep_fraction=0.5, keep_top_n=2)

    @pytest.mark.parametrize("keep_fraction", [-0.1, 0, 1.5])
    def test_keep_fraction_out_of_range_raises(self, keep_fraction):
        """keep_fraction must be in (0, 1]."""
        with pytest.raises(ValueError, match="keep_fraction must be in \\(0, 1\\]"):
            ExactTrotterization(keep_fraction=keep_fraction)

    @pytest.mark.parametrize("keep_top_n", [-1, 0, 0.5])
    def test_keep_top_n_invalid_raises(self, keep_top_n):
        """keep_top_n must be a positive integer (>= 1)."""
        with pytest.raises(ValueError, match="keep_top_n must be a positive integer"):
            ExactTrotterization(keep_top_n=keep_top_n)

    def test_no_truncation_returns_simplified_hamiltonian(self, simple_hamiltonian):
        """When both keep_fraction and keep_top_n are None, returns Hamiltonian unchanged."""
        result = ExactTrotterization().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "strategy_kwargs,warn_match",
        [
            ({"keep_fraction": 1.0}, "keep_fraction is 1.0.*no truncation"),
            ({"keep_top_n": 10}, "keep_top_n is greater than or equal"),
        ],
        ids=["keep_fraction_one", "keep_top_n_exceeds_terms"],
    )
    def test_early_return_returns_full_and_warns(
        self, simple_hamiltonian, strategy_kwargs, warn_match
    ):
        """When keep_fraction=1.0 or keep_top_n >= terms, returns full Hamiltonian and warns."""
        with pytest.warns(UserWarning, match=warn_match):
            result = ExactTrotterization(**strategy_kwargs).process_hamiltonian(
                simple_hamiltonian
            )
        assert qml.equal(result, simple_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_hamiltonian):
        """Single-term operators (SProd, bare Pauli) work with keep_top_n; no len() error."""
        strategy = ExactTrotterization(keep_top_n=1)
        with pytest.warns(UserWarning, match="keep_top_n is greater than or equal"):
            result = strategy.process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_fraction(self, single_term_hamiltonian):
        """Single-term operators work with keep_fraction; returns full operator."""
        strategy = ExactTrotterization(keep_fraction=0.5)
        result = strategy.process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    def test_constant_only_hamiltonian_raises(self):
        """Constant-only Hamiltonian raises ValueError; rejected at boundary."""
        constant_only = qml.Identity(0) * 5.0
        strategy = ExactTrotterization(keep_fraction=0.5)
        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            strategy.process_hamiltonian(constant_only)

    @pytest.mark.parametrize(
        "keep_top_n,expected",
        [
            (
                1,
                (3.0 * (qml.Z(0) @ qml.Z(1))).simplify(),
            ),
            (
                2,
                (2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify(),
            ),
        ],
        ids=["keep_top_n_1", "keep_top_n_2"],
    )
    def test_keep_top_n_keeps_largest_terms(
        self, simple_hamiltonian, keep_top_n, expected
    ):
        """keep_top_n keeps that many largest-magnitude terms (plus constant)."""
        result = ExactTrotterization(keep_top_n=keep_top_n).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, _ = result.terms()
        assert len(coeffs) == keep_top_n
        assert qml.equal(result, expected)

    def test_keep_fraction_reduces_term_count(self, simple_hamiltonian):
        """keep_fraction < 1 yields fewer terms; kept terms have total |coeff| >= fraction of full."""
        result = ExactTrotterization(keep_fraction=0.5).process_hamiltonian(
            simple_hamiltonian
        )
        full_coeffs, _ = simple_hamiltonian.terms()
        result_coeffs, _ = result.terms()
        assert len(result_coeffs) <= len(full_coeffs)
        # For keep_fraction=0.5, total |coeff| in simple_hamiltonian is 6; we keep terms summing to >= 3
        full_sum_abs = np.sum(np.abs(full_coeffs))
        result_sum_abs = np.sum(np.abs(result_coeffs))
        assert result_sum_abs >= 0.5 * full_sum_abs
        # With 0.5 we keep exactly the largest term (3.0*Z(0)@Z(1))
        expected = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        assert qml.equal(result, expected)

    def test_exact_trotterization_stateful_is_false(self):
        """ExactTrotterization reports stateful=False (cache is memoization only, not state)."""
        assert ExactTrotterization(keep_top_n=2).stateful is False

    @pytest.mark.parametrize(
        "strategy_kwargs",
        [{"keep_top_n": 2}, {"keep_fraction": 0.5}],
        ids=["keep_top_n", "keep_fraction"],
    )
    def test_exact_trotterization_caches_result(
        self, mocker, simple_hamiltonian, strategy_kwargs
    ):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; returns same object."""
        spy = mocker.spy(hamiltonians, "_sort_hamiltonian_terms")
        strategy = ExactTrotterization(**strategy_kwargs)
        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(simple_hamiltonian)
        assert result1 is result2
        assert spy.call_count == 1

    def test_exact_trotterization_different_hamiltonians_separate_cache(
        self, simple_hamiltonian
    ):
        """Different Hamiltonians get separate cache entries; each returns correct result."""
        strategy = ExactTrotterization(keep_top_n=1)
        ham2 = (4.0 * qml.Z(0) + 5.0 * qml.Z(1)).simplify()

        result1 = strategy.process_hamiltonian(simple_hamiltonian)
        result2 = strategy.process_hamiltonian(ham2)
        result3 = strategy.process_hamiltonian(simple_hamiltonian)

        # Cached: result1 and result3 are same object
        assert result1 is result3
        # Different Hamiltonians yield different results
        assert not qml.equal(result1, result2)
        # Correct truncation: simple_hamiltonian -> 3.0*Z(0)@Z(1); ham2 -> 5.0*Z(1)
        expected1 = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        expected2 = (5.0 * qml.Z(1)).simplify()
        assert qml.equal(result1, expected1)
        assert qml.equal(result2, expected2)

    @pytest.mark.parametrize(
        "strategy_kwargs,warn_match",
        [
            ({}, None),
            ({"keep_fraction": 1.0}, "keep_fraction is 1.0"),
            ({"keep_top_n": 10}, "keep_top_n is greater than or equal"),
        ],
        ids=["no_truncation", "keep_fraction_one", "keep_top_n_exceeds_terms"],
    )
    def test_exact_trotterization_no_cache_on_early_return(
        self, mocker, simple_hamiltonian, strategy_kwargs, warn_match
    ):
        """Early-return paths do not use cache; _sort_hamiltonian_terms never called."""
        spy = mocker.spy(hamiltonians, "_sort_hamiltonian_terms")
        strategy = ExactTrotterization(**strategy_kwargs)
        for _ in range(2):
            if warn_match is not None:
                with pytest.warns(UserWarning, match=warn_match):
                    strategy.process_hamiltonian(simple_hamiltonian)
            else:
                strategy.process_hamiltonian(simple_hamiltonian)
        assert spy.call_count == 0


class TestQDrift:
    """
    Tests for QDrift strategy (public API and specified behavior).
    """

    def test_invalid_sampling_strategy_raises(self):
        """sampling_strategy must be 'uniform' or 'weighted'."""
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            QDrift(sampling_budget=5, sampling_strategy="invalid")

    def test_seed_non_int_raises(self):
        """seed must be an integer when provided."""
        with pytest.raises(ValueError, match="seed must be an integer"):
            QDrift(sampling_budget=5, seed=1.5)

    def test_all_none_returns_unchanged_and_warns(self, simple_hamiltonian):
        """When keep_fraction, keep_top_n and sampling_budget are all None, returns Hamiltonian unchanged."""
        with pytest.warns(
            UserWarning, match="Neither keep_fraction, keep_top_n, nor sampling_budget"
        ):
            result = QDrift().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), result is a valid operator with terms."""
        result = QDrift(sampling_budget=4, seed=42).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, ops = result.terms()
        assert len(coeffs) >= 1
        assert len(coeffs) == len(ops)

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical first sample across fresh instances."""
        s1 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        s2 = QDrift(sampling_budget=5, seed=123, sampling_strategy="uniform")
        r1 = s1.process_hamiltonian(simple_hamiltonian)
        r2 = s2.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(r1, r2)

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(simple_hamiltonian)
        coeffs, _ = result.terms()
        # Kept terms (from 0.5 fraction) + 3 sampled from the rest
        assert len(coeffs) >= 3

    def test_keep_fraction_one_and_sample_budget_warns_no_terms_to_sample(
        self, simple_hamiltonian
    ):
        """When keep_fraction=1.0 and sample_budget set, all terms kept so no sampling; warns."""
        with pytest.warns(UserWarning) as record:
            result = QDrift(
                keep_fraction=1.0, sampling_budget=5, seed=0
            ).process_hamiltonian(simple_hamiltonian)
        # ExactTrotterization may warn "keep_fraction is 1.0..."; QDrift warns "no terms left to sample"
        messages = [str(w.message) for w in record]
        assert any("no terms left to sample" in m for m in messages)
        assert qml.equal(result, simple_hamiltonian)

    def test_sample_budget_none_equivalent_to_exact_trotterization(
        self, simple_hamiltonian
    ):
        """When sample_budget is None but keep_fraction set, result equals ExactTrotterization."""
        with pytest.warns(UserWarning, match="sampling_budget is not set"):
            qdrift = QDrift(keep_fraction=0.5)
        exact = ExactTrotterization(keep_fraction=0.5)
        qdrift_result = qdrift.process_hamiltonian(simple_hamiltonian)
        exact_result = exact.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(qdrift_result, exact_result)

    def test_qdrift_caches_keep_hamiltonian(self, mocker, simple_hamiltonian):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; ExactTrotterization called once."""
        spy = mocker.spy(ExactTrotterization, "process_hamiltonian")
        strategy = QDrift(
            keep_fraction=0.5, sampling_budget=3, seed=42, sampling_strategy="weighted"
        )
        strategy.process_hamiltonian(simple_hamiltonian)
        strategy.process_hamiltonian(simple_hamiltonian)
        assert spy.call_count == 1

    def test_keep_fraction_one_warns_only_on_first_call(self, simple_hamiltonian):
        """'No terms left to sample' warning is emitted only on first call, not on cached calls."""
        strategy = QDrift(keep_fraction=1.0, sampling_budget=5, seed=0)
        with pytest.warns(UserWarning) as first_record:
            first_result = strategy.process_hamiltonian(simple_hamiltonian)
        first_messages = [str(w.message) for w in first_record]
        assert any("no terms left to sample" in m for m in first_messages)
        assert qml.equal(first_result, simple_hamiltonian)

        with warnings.catch_warnings(record=True) as second_record:
            warnings.simplefilter("always")
            second_result = strategy.process_hamiltonian(simple_hamiltonian)
        second_messages = [str(w.message) for w in second_record]
        assert not any("no terms left to sample" in m for m in second_messages)
        assert qml.equal(second_result, simple_hamiltonian)

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_keep_top_n(self, single_term_hamiltonian):
        """Single-term operators work with QDrift keep_top_n; no len() error."""
        with pytest.warns(
            UserWarning,
            match="keep_top_n is greater than or equal|All terms were kept",
        ):
            result = QDrift(
                keep_top_n=1, sampling_budget=2, seed=42
            ).process_hamiltonian(single_term_hamiltonian)
        assert qml.equal(result, single_term_hamiltonian.simplify())

    @pytest.mark.parametrize(
        "single_term_hamiltonian",
        [0.5 * qml.Z(0), qml.Z(0)],
        ids=["sprod", "bare_pauli"],
    )
    def test_single_term_hamiltonian_with_sampling_budget_only(
        self, single_term_hamiltonian
    ):
        """Single-term operators work with QDrift sampling_budget only; returns term unchanged."""
        result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(
            single_term_hamiltonian
        )
        assert qml.equal(result.simplify(), single_term_hamiltonian.simplify())

    def test_empty_hamiltonian_warns_and_returns_kept(self):
        """Empty to_sample_hamiltonian (no terms) warns and returns empty Hamiltonian."""
        empty = qml.Hamiltonian([], [])
        with pytest.warns(UserWarning, match="No terms to sample"):
            result = QDrift(sampling_budget=3, seed=42).process_hamiltonian(empty)
        assert qml.equal(result, empty)

    def test_qdrift_stateful_is_true(self):
        """QDrift reports stateful=True (caches intermediate results)."""
        assert QDrift(sampling_budget=5, seed=0).stateful is True

    def test_n_hamiltonians_per_iteration_less_than_one_raises(self):
        """n_hamiltonians_per_iteration must be >= 1."""
        with pytest.raises(
            ValueError, match="n_hamiltonians_per_iteration must be >= 1"
        ):
            QDrift(sampling_budget=5, n_hamiltonians_per_iteration=0)

    def test_rng_produces_different_samples_on_repeated_calls(self, simple_hamiltonian):
        """Instance RNG produces different Hamiltonian samples on repeated calls."""
        strategy = QDrift(sampling_budget=4, seed=42, sampling_strategy="uniform")
        r0 = strategy.process_hamiltonian(simple_hamiltonian)
        r1 = strategy.process_hamiltonian(simple_hamiltonian)
        r2 = strategy.process_hamiltonian(simple_hamiltonian)
        # With 3 terms and sample_budget=4, sampling with replacement can produce
        # different orderings; at least two of the three should differ
        results = [r0, r1, r2]
        assert not all(qml.equal(results[0], r) for r in results)


class TestResolveIsingConverter:
    """Tests for _resolve_ising_converter."""

    def test_native_returns_native_converter(self):
        """'native' returns a NativeIsingConverter instance."""
        converter = _resolve_ising_converter("native", quadratization_strength=10.0)
        assert isinstance(converter, NativeIsingConverter)

    def test_quadratized_returns_quadratized_converter(self):
        """'quadratized' returns a QuadratizedIsingConverter with the given strength."""
        converter = _resolve_ising_converter("quadratized", quadratization_strength=5.0)
        assert isinstance(converter, QuadratizedIsingConverter)
        assert converter.strength == 5.0

    def test_invalid_builder_raises(self):
        """An unrecognized builder string raises ValueError."""
        with pytest.raises(ValueError, match="hamiltonian_builder must be either"):
            _resolve_ising_converter("unknown", quadratization_strength=1.0)


class TestSortHamiltonianTerms:
    """Tests for _sort_hamiltonian_terms."""

    def test_absolute_order_sorts_by_literal_coefficient(self):
        """'absolute' sorts by the literal coefficient value (ascending)."""
        ham = qml.Hamiltonian([0.5, -0.3, 0.1], [qml.Z(0), qml.Z(1), qml.Z(2)])
        result = _sort_hamiltonian_terms(ham, order="absolute")
        coeffs, _ = result.terms()
        assert list(coeffs) == pytest.approx([-0.3, 0.1, 0.5])

    def test_magnitude_order_sorts_by_absolute_value(self):
        """'magnitude' sorts by abs(coefficient) (ascending)."""
        ham = qml.Hamiltonian([0.5, -0.3, 0.1], [qml.Z(0), qml.Z(1), qml.Z(2)])
        result = _sort_hamiltonian_terms(ham, order="magnitude")
        coeffs, _ = result.terms()
        assert [abs(c) for c in coeffs] == pytest.approx([0.1, 0.3, 0.5])

    def test_single_term_operator_passes_through(self):
        """A single PauliZ (not a Sum/Hamiltonian) is returned unchanged."""
        op = qml.PauliZ(0)
        result = _sort_hamiltonian_terms(op)
        assert qml.equal(result, op)

    def test_negative_coefficients_preserved(self):
        """Negative coefficients are preserved after sorting, not lost."""
        ham = qml.Hamiltonian([-2.0, 1.0], [qml.Z(0), qml.Z(1)])
        result = _sort_hamiltonian_terms(ham, order="absolute")
        coeffs, _ = result.terms()
        assert -2.0 in [float(c) for c in coeffs]
        assert 1.0 in [float(c) for c in coeffs]


# ---------------------------------------------------------------------------
# qubo_to_ising
# ---------------------------------------------------------------------------


class TestQuboToIsing:
    """Tests for the qubo_to_ising helper."""

    # -- Happy path: dict QUBO --

    def test_dict_qubo_returns_ising_result(self):
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        assert isinstance(result, IsingResult)
        assert result.n_qubits == 2
        assert result.cost_hamiltonian is not None

    def test_loss_constant_includes_encoding_and_ham_constants(self):
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        # The loss_constant should be a finite number (sum of encoding.constant + ham_constant)
        assert np.isfinite(result.loss_constant)

    def test_decode_fn_returns_binary_array(self):
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        decoded = result.encoding.decode_fn("11")
        assert all(b in (0, 1) for b in decoded)

    # -- Numpy matrix input --

    def test_numpy_matrix_qubo(self):
        Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        result = qubo_to_ising(Q)
        assert result.n_qubits == 2

    # -- Sparse matrix input --

    def test_sparse_matrix_qubo(self):
        Q = sps.csr_matrix(np.array([[-1.0, 2.0], [0.0, -1.0]]))
        result = qubo_to_ising(Q)
        assert result.n_qubits == 2

    # -- HUBO (higher-order) with quadratization --

    def test_hubo_native(self):
        hubo = {(0, 1, 2): 1.0, (0,): -1.0, (1,): -1.0, (2,): -1.0}
        result = qubo_to_ising(hubo, hamiltonian_builder="native")
        assert result.n_qubits >= 3

    def test_hubo_quadratized(self):
        hubo = {(0, 1, 2): 1.0, (0,): -1.0, (1,): -1.0, (2,): -1.0}
        result = qubo_to_ising(
            hubo, hamiltonian_builder="quadratized", quadratization_strength=5.0
        )
        # Quadratization adds auxiliary qubits
        assert result.n_qubits >= 3

    # -- Cost hamiltonian is cleaned (no identity terms) --

    def test_cost_hamiltonian_has_no_identity(self):
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        # The cleaned hamiltonian should not be a pure Identity
        # (constant terms are absorbed into loss_constant)
        assert not isinstance(result.cost_hamiltonian, qml.Identity)

    # -- n_qubits matches wire count --

    def test_n_qubits_matches_wires(self):
        qubo = {(0,): -1.0, (1,): -1.0, (2,): -1.0, (0, 1): 1.0, (1, 2): 1.0}
        result = qubo_to_ising(qubo)
        assert result.n_qubits == len(result.cost_hamiltonian.wires)

    # -- Edge cases --

    def test_single_variable(self):
        qubo = {(0,): -3.0}
        result = qubo_to_ising(qubo)
        assert result.n_qubits == 1

    def test_constant_only_raises(self):
        """A QUBO that produces only constant terms should raise."""
        # An empty QUBO dict normalises to a trivial problem
        with pytest.raises((ValueError, Exception)):
            qubo_to_ising({})

    def test_invalid_hamiltonian_builder_raises(self):
        qubo = {(0,): -1.0}
        with pytest.raises(ValueError, match="native.*quadratized"):
            qubo_to_ising(qubo, hamiltonian_builder="invalid")

    # -- BQM input --

    def test_dimod_bqm(self):
        bqm = dimod.BinaryQuadraticModel(
            {0: -1.0, 1: -1.0}, {(0, 1): 2.0}, 0.0, vartype="BINARY"
        )
        result = qubo_to_ising(bqm)
        assert result.n_qubits == 2
