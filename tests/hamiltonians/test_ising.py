# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for binary-to-Ising conversion (_ising.py)."""

import dimod
import numpy as np
import pennylane as qp
import pytest
import scipy.sparse as sps

from divi.hamiltonians import (
    IsingResult,
    NativeIsingConverter,
    QuadratizedIsingConverter,
    _hamiltonian_term_count,
    _is_sanitized,
    _resolve_ising_converter,
    convert_qubo_matrix_to_pennylane_ising,
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
    """Tests the _is_sanitized function with various matrix types."""
    assert _is_sanitized(matrix) == expected


class TestQuboToIsingConversion:
    """Tests for convert_qubo_matrix_to_pennylane_ising."""

    def test_symmetric_dense_qubo(self):
        """
        Tests conversion with a simple symmetric dense QUBO matrix.

        Q = [[-1, 2], [2, -1]] → E(x) = -x₀ + 4x₀x₁ - x₁
        Hand-derived Ising: H = Z₀Z₁ - 0.5·Z₀ - 0.5·Z₁, offset = 0.
        """
        qubo_matrix = np.array([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1)

        assert np.isclose(constant, 0.0)
        assert qp.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_upper_triangular_dense_qubo(self):
        """
        Tests conversion with an upper triangular dense QUBO matrix.
        The result should be the same as the symmetric equivalent.

        Q = [[-1, 4], [0, -1]] → symmetrized [[-1, 2], [2, -1]] → same Ising.
        """
        qubo_matrix = np.array([[-1.0, 4.0], [0.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1)

        assert np.isclose(constant, 0.0)
        assert qp.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_non_sanitized_qubo_raises_warning(self):
        """
        Tests that a non-sanitized QUBO matrix raises a warning and is correctly
        symmetrized.
        """
        qubo_matrix = np.array([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]] → same Ising as symmetric test.
        expected_hamiltonian = 1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1)

        assert np.isclose(constant, 0.0)
        assert qp.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_non_sanitized_sparse_qubo_raises_warning(self):
        """Non-sanitized sparse QUBO raises warning and is symmetrized."""
        qubo_matrix = sps.csc_matrix([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1)
        assert np.isclose(constant, 0.0)
        assert qp.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_diagonal_qubo(self):
        """
        Tests a purely diagonal QUBO matrix, which should result in only Z terms.
        """
        qubo_matrix = np.array([[1.0, 0.0], [0.0, -2.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = -0.5 * qp.Z(0) + 1.0 * qp.Z(1)
        expected_constant = -0.5

        assert np.isclose(constant, expected_constant)
        assert qp.equal(hamiltonian, expected_hamiltonian)

    def test_sparse_qubo(self):
        """Tests conversion with a sparse QUBO matrix."""
        qubo_matrix = sps.csr_matrix([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1)

        assert np.isclose(constant, 0.0)
        assert qp.equal(hamiltonian.simplify(), expected_hamiltonian.simplify())

    def test_3x3_qubo(self):
        """Tests a larger 3x3 QUBO matrix to ensure correct term summation."""
        qubo_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_constant = 10.5
        expected_hamiltonian = (
            1.0 * (qp.Z(0) @ qp.Z(1))
            + 1.5 * (qp.Z(0) @ qp.Z(2))
            + 2.5 * (qp.Z(1) @ qp.Z(2))
            - 3.0 * qp.Z(0)
            - 5.5 * qp.Z(1)
            - 7.0 * qp.Z(2)
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
            # Non-sanitized dense (triggers warning + symmetrization)
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

        Covers dense, sparse, upper-triangular, non-sanitized (triggers
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

        h_matrix = np.real(np.diag(qp.matrix(hamiltonian, wire_order=range(n))))

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
            if isinstance(op, qp.ops.Prod):
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
        h_matrix = qp.matrix(result.operator, wire_order=range(n_total))

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
        assert not isinstance(result.cost_hamiltonian, qp.Identity)

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
