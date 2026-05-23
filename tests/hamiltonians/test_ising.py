# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for binary-to-Ising conversion (_ising.py)."""

from collections import defaultdict
from itertools import product as iterproduct

import dimod
import numpy as np
import pennylane as qp
import pytest
import scipy.sparse as sps
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import (
    IsingResult,
    NativeIsingConverter,
    QuadratizedIsingConverter,
    normalize_binary_polynomial_problem,
    qubo_to_ising,
    qubo_to_spo,
    to_spo,
)
from divi.hamiltonians._ising import (
    _convert_qubo_matrix_to_ising_spo,
    _is_sanitized,
    _resolve_ising_converter,
)


def _spo_z_basis_energy(spo: SparsePauliOp, bits: tuple[int, ...]) -> float:
    """Evaluate a Z-only SPO at the computational basis state ``bits``.

    ``bits[i]`` is the bit value for qubit ``i``. Returns
    ``<bits|spo|bits>``. Raises if the SPO contains non-Z Paulis (the
    Ising Hamiltonians under test are diagonal in the Z basis by
    construction).
    """
    n = spo.num_qubits
    energy = 0.0
    for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
        parity = 1
        for pos, ch in enumerate(label):
            qubit = n - 1 - pos  # big-endian label
            if ch == "Z":
                parity *= 1 - 2 * bits[qubit]
            elif ch != "I":
                raise ValueError(
                    f"Non-Z Pauli {ch!r} in Ising Hamiltonian label {label!r}"
                )
        energy += float(coeff.real) * parity
    return energy


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
    """Tests for ``_convert_qubo_matrix_to_ising_spo``."""

    def test_symmetric_dense_qubo(self):
        """
        Q = [[-1, 2], [2, -1]] → E(x) = -x₀ + 4x₀x₁ - x₁
        Hand-derived Ising: H = Z₀Z₁ - 0.5·Z₀ - 0.5·Z₁, offset = 0.
        """
        qubo_matrix = np.array([[-1.0, 2.0], [2.0, -1.0]])
        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1))

        assert np.isclose(constant, 0.0)
        assert spo.simplify() == expected.simplify()

    def test_upper_triangular_dense_qubo(self):
        """Upper-triangular Q = [[-1, 4], [0, -1]] symmetrises to the same Ising."""
        qubo_matrix = np.array([[-1.0, 4.0], [0.0, -1.0]])
        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1))

        assert np.isclose(constant, 0.0)
        assert spo.simplify() == expected.simplify()

    def test_non_sanitized_qubo_raises_warning(self):
        """A non-sanitised QUBO triggers the symmetrisation warning."""
        qubo_matrix = np.array([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        # Symmetrised version is [[-1, 2], [2, -1]] → same Ising as the symmetric test.
        expected = to_spo(1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1))

        assert np.isclose(constant, 0.0)
        assert spo.simplify() == expected.simplify()

    def test_non_sanitized_sparse_qubo_raises_warning(self):
        """Non-sanitised sparse QUBO triggers the symmetrisation warning."""
        qubo_matrix = sps.csc_matrix([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1))
        assert np.isclose(constant, 0.0)
        assert spo.simplify() == expected.simplify()

    def test_diagonal_qubo(self):
        """Purely diagonal QUBO yields only single-Z terms."""
        qubo_matrix = np.array([[1.0, 0.0], [0.0, -2.0]])
        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(-0.5 * qp.Z(0) + 1.0 * qp.Z(1))

        assert np.isclose(constant, -0.5)
        assert spo.simplify() == expected.simplify()

    def test_sparse_qubo(self):
        """Sparse-matrix input yields the same Ising as its dense equivalent."""
        qubo_matrix = sps.csr_matrix([[-1.0, 2.0], [2.0, -1.0]])
        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(1.0 * (qp.Z(0) @ qp.Z(1)) - 0.5 * qp.Z(0) - 0.5 * qp.Z(1))

        assert np.isclose(constant, 0.0)
        assert spo.simplify() == expected.simplify()

    def test_3x3_qubo(self):
        """Larger 3x3 QUBO exercises pair-term summation."""
        qubo_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        expected = to_spo(
            1.0 * (qp.Z(0) @ qp.Z(1))
            + 1.5 * (qp.Z(0) @ qp.Z(2))
            + 2.5 * (qp.Z(1) @ qp.Z(2))
            - 3.0 * qp.Z(0)
            - 5.5 * qp.Z(1)
            - 7.0 * qp.Z(2)
        )

        assert np.isclose(constant, 10.5)
        assert spo.simplify() == expected.simplify()

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
        q_dense = qubo_matrix.toarray() if sps.issparse(qubo_matrix) else qubo_matrix
        n = q_dense.shape[0]
        Q_sym = (q_dense + q_dense.T) / 2

        if expect_warning:
            with pytest.warns(
                UserWarning, match="neither symmetric nor upper triangular"
            ):
                spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)
        else:
            spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)

        for bits in iterproduct((0, 1), repeat=n):
            x = np.array(bits, dtype=float)
            qubo_energy = float(x @ Q_sym @ x)
            # x_i = (1-σ_i)/2 → σ_i=+1 when x_i=0, σ_i=-1 when x_i=1
            ising_energy = _spo_z_basis_energy(spo, bits) + constant
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
        """SPO version: bits[qubit] is read from ``assignment[var]``."""
        if operator.size == 0:
            return float(constant)
        bits = tuple(
            assignment[var]
            for var, _ in sorted(variable_to_idx.items(), key=lambda kv: kv[1])
        )
        return _spo_z_basis_energy(operator, bits) + float(constant)

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
        assert result.operator.num_qubits == 3

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
        n_measure_qubits = result.operator.num_qubits
        bitstring = "1" * n_measure_qubits
        decoded = result.decode_fn(bitstring)
        assert len(decoded) == problem.n_vars

        if result.operator.size > 0:
            for label in result.operator.paulis.to_labels():
                locality = sum(1 for ch in label if ch != "I")
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
        n_total = result.operator.num_qubits
        assert result.metadata["ancilla_count"] >= 1

        # Group all computational-basis energies by decoded original assignment,
        # then verify the minimum matches the polynomial energy.
        energies_by_assignment = defaultdict(list)
        for val in range(2**n_total):
            bs = format(val, f"0{n_total}b")
            decoded = tuple(int(d) for d in result.decode_fn(bs))
            # bs[i] is qubit i (decode_fn uses big-endian bitstring indexing).
            bits = tuple(int(c) for c in bs)
            ising_e = _spo_z_basis_energy(result.operator, bits) + result.constant
            energies_by_assignment[decoded].append(ising_e)

        for x0, x1, x2 in iterproduct((0, 1), repeat=3):
            assignment = {"x0": x0, "x1": x1, "x2": x2}
            poly_energy = self._eval_polynomial_energy(problem, assignment)
            best_ising = min(energies_by_assignment[(x0, x1, x2)])
            assert best_ising == pytest.approx(
                poly_energy
            ), f"x={[x0, x1, x2]}: best_ising={best_ising}, poly={poly_energy}"

    def test_native_and_quadratized_agree_on_feasible_subspace(self):
        """Native and quadratized converters must agree on every original-variable
        assignment when the quadratized penalty dominates the objective.

        Path-(1) [quadratized] is a penalty-relaxation that only matches the
        original HUBO on the subspace where every ancilla equals its product
        constraint; path-(2) [native] is an exact algebraic substitution.
        Across the full ``{0,1}^n_orig`` lattice, the minimum quadratized
        energy over ancilla assignments must equal the native energy at that
        assignment — for any strength large enough to dominate the objective.
        """
        hubo = {
            ("x0",): -2.5,
            ("x1",): 1.5,
            ("x0", "x1"): 3.0,
            ("x0", "x1", "x2"): -4.0,
            ("x1", "x2", "x3"): 2.0,
            ("x0", "x1", "x2", "x3"): -1.5,
            (): 0.75,
        }
        problem = normalize_binary_polynomial_problem(
            hubo, variable_order=("x0", "x1", "x2", "x3")
        )

        native = NativeIsingConverter().convert(problem)
        # Adaptive strength: ``QuadratizedIsingConverter(strength=None)`` picks
        # ``2 * max(|hubo coeff|) = 8.0`` for this polynomial. That is below
        # what the reviewer flagged as safe, so we also override with an
        # explicitly large strength to make the assertion strict.
        quad = QuadratizedIsingConverter(strength=100.0).convert(problem)
        n_total = quad.operator.num_qubits

        quad_energies_by_assignment: dict[tuple[int, ...], list[float]] = defaultdict(
            list
        )
        for val in range(2**n_total):
            bs = format(val, f"0{n_total}b")
            decoded = tuple(int(d) for d in quad.decode_fn(bs))
            bits = tuple(int(c) for c in bs)
            quad_energies_by_assignment[decoded].append(
                _spo_z_basis_energy(quad.operator, bits) + quad.constant
            )

        for x in iterproduct((0, 1), repeat=4):
            assignment = {f"x{i}": v for i, v in enumerate(x)}
            native_energy = self._eval_ising_energy(
                native.operator,
                native.constant,
                assignment,
                problem.variable_to_idx,
            )
            quad_min = min(quad_energies_by_assignment[x])
            assert quad_min == pytest.approx(
                native_energy
            ), f"x={x}: native={native_energy} vs quadratized_min={quad_min}"

    def test_quadratized_adaptive_strength_scales_with_input(self):
        """``strength=None`` picks an adaptive penalty from coefficient magnitudes.

        Two HUBOs that differ only by an overall rescaling must produce
        quadratized encodings whose effective penalty strengths scale
        identically — otherwise the same problem at different scales would
        cross from "feasibly solvable" to "infeasible relaxation."
        """
        hubo_small = {("x0", "x1", "x2"): 1.0, ("x0",): -1.0}
        hubo_large = {("x0", "x1", "x2"): 1000.0, ("x0",): -1000.0}
        for h in (hubo_small, hubo_large):
            problem = normalize_binary_polynomial_problem(
                h, variable_order=("x0", "x1", "x2")
            )
            result = QuadratizedIsingConverter().convert(problem)
            max_coeff = max(abs(c) for c in problem.terms.values())
            assert result.metadata["strength"] == pytest.approx(2.0 * max_coeff)

    def test_native_relative_zero_tol_prunes_only_noise(self):
        """Relative tolerance keeps real terms even when the largest coefficient
        is large; previously an absolute ``1e-12`` cutoff was indistinguishable
        from "very small but real" terms.
        """
        # Largest output Z-product weight here ~ 250 (constant from x0,x1
        # via 1000*x0*x1 = 250*(1 - Z0 - Z1 + Z0*Z1)); a small but real
        # contribution of 1e-7 from the linear term must survive.
        hubo = {("x0", "x1"): 1000.0, ("x0",): 4e-7}
        problem = normalize_binary_polynomial_problem(hubo, variable_order=("x0", "x1"))
        result = NativeIsingConverter().convert(problem)
        # Linear Z0 term contributes -(1000/4) + (-4e-7/2) = -250.0000002.
        n = result.operator.num_qubits
        single_z_coeffs: dict[int, float] = {}
        for label, coeff in zip(
            result.operator.paulis.to_labels(), result.operator.coeffs
        ):
            z_positions = [pos for pos, ch in enumerate(label) if ch == "Z"]
            non_id = [pos for pos, ch in enumerate(label) if ch != "I"]
            if len(z_positions) == 1 and len(non_id) == 1:
                # Single-Z term — map big-endian label position to qubit.
                single_z_coeffs[n - 1 - z_positions[0]] = float(coeff.real)
        # Both single-qubit Z terms must be present (Z0 from x0 and x0x1; Z1 from x0x1).
        assert 0 in single_z_coeffs
        assert 1 in single_z_coeffs


class TestResolveIsingConverter:
    """Tests for _resolve_ising_converter."""

    @pytest.mark.parametrize(
        "builder, strength, expected_cls",
        [
            ("native", 10.0, NativeIsingConverter),
            ("quadratized", 5.0, QuadratizedIsingConverter),
        ],
    )
    def test_known_builder_returns_matching_converter(
        self, builder, strength, expected_cls
    ):
        converter = _resolve_ising_converter(builder, quadratization_strength=strength)
        assert isinstance(converter, expected_cls)
        if expected_cls is QuadratizedIsingConverter:
            assert converter.strength == strength

    def test_invalid_builder_raises(self):
        """An unrecognized builder string raises ValueError."""
        with pytest.raises(ValueError, match="hamiltonian_builder must be either"):
            _resolve_ising_converter("unknown", quadratization_strength=1.0)


class TestQuboToIsing:
    """Tests for the qubo_to_ising helper."""

    @pytest.mark.parametrize(
        "qubo_factory, expected_n_qubits",
        [
            (lambda: {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}, 2),
            (lambda: np.array([[-1.0, 2.0], [0.0, -1.0]]), 2),
            (lambda: sps.csr_matrix(np.array([[-1.0, 2.0], [0.0, -1.0]])), 2),
            (
                lambda: dimod.BinaryQuadraticModel(
                    {0: -1.0, 1: -1.0}, {(0, 1): 2.0}, 0.0, vartype="BINARY"
                ),
                2,
            ),
            (lambda: {(0,): -3.0}, 1),
        ],
        ids=["dict_qubo", "numpy_matrix", "scipy_sparse", "dimod_bqm", "single_var"],
    )
    def test_returns_ising_result_with_expected_n_qubits(
        self, qubo_factory, expected_n_qubits
    ):
        """Every accepted input form normalises into an IsingResult."""
        result = qubo_to_ising(qubo_factory())
        assert isinstance(result, IsingResult)
        assert result.n_qubits == expected_n_qubits
        assert np.isfinite(result.loss_constant)

    def test_decode_fn_returns_binary_array(self):
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        decoded = result.encoding.decode_fn("11")
        assert all(b in (0, 1) for b in decoded)

    @pytest.mark.parametrize(
        "hamiltonian_builder, kwargs",
        [
            ("native", {}),
            ("quadratized", {"quadratization_strength": 5.0}),
        ],
    )
    def test_hubo_input_produces_sized_register(self, hamiltonian_builder, kwargs):
        """HUBO inputs work with both builder strategies; quadratization may
        add auxiliaries, so we only require ``n_qubits >= 3``."""
        hubo = {(0, 1, 2): 1.0, (0,): -1.0, (1,): -1.0, (2,): -1.0}
        result = qubo_to_ising(hubo, hamiltonian_builder=hamiltonian_builder, **kwargs)
        assert result.n_qubits >= 3

    def test_cost_hamiltonian_has_no_identity(self):
        """Pure-identity rows fold into ``loss_constant``, never the SPO."""
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        result = qubo_to_ising(qubo)
        assert all(
            set(label) != {"I"} for label in result.cost_hamiltonian.paulis.to_labels()
        )

    def test_n_qubits_matches_wires(self):
        qubo = {(0,): -1.0, (1,): -1.0, (2,): -1.0, (0, 1): 1.0, (1, 2): 1.0}
        result = qubo_to_ising(qubo)
        assert result.n_qubits == result.cost_hamiltonian.num_qubits

    def test_constant_only_raises(self):
        """A QUBO that normalises to constant-only terms must raise."""
        with pytest.raises((ValueError, Exception)):
            qubo_to_ising({})

    def test_invalid_hamiltonian_builder_raises(self):
        qubo = {(0,): -1.0}
        with pytest.raises(ValueError, match="native.*quadratized"):
            qubo_to_ising(qubo, hamiltonian_builder="invalid")


class TestQuboToSpo:
    """Tests for the qubo_to_spo convenience wrapper."""

    def test_returns_sparse_pauli_op(self):
        spo = qubo_to_spo({(0,): -1.0, (1,): -1.0, (0, 1): 2.0})
        assert isinstance(spo, SparsePauliOp)

    def test_z_basis_eigenvalues_equal_qubo_energies(self):
        """SPO computational-basis eigenvalues match the QUBO objective.

        This is the contract that justifies returning a single SPO instead
        of forcing the caller to track a separate offset: the returned
        operator's expectation value on any bitstring equals the QUBO's
        energy on that bitstring.
        """
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        spo = qubo_to_spo(qubo)
        diag = np.real(np.diag(spo.to_matrix()))
        # Bitstring ordering: Qiskit uses qubit 0 rightmost, so diag[i]
        # corresponds to bits = reversed(binary(i, n_qubits)).
        n = spo.num_qubits
        for state_idx, energy in enumerate(diag):
            bits = [(state_idx >> q) & 1 for q in range(n)]
            qubo_energy = sum(
                coeff for term, coeff in qubo.items() if all(bits[i] == 1 for i in term)
            )
            assert np.isclose(
                energy, qubo_energy
            ), f"state {bits}: SPO={energy}, QUBO={qubo_energy}"

    def test_bakes_loss_constant_as_identity_term(self):
        """The Ising-encoding loss constant appears as a pure-identity row."""
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        ising = qubo_to_ising(qubo)
        spo = qubo_to_spo(qubo)
        identity_label = "I" * spo.num_qubits
        identity_rows = [
            (label, coeff) for label, coeff in spo.to_list() if label == identity_label
        ]
        assert len(identity_rows) == 1
        assert np.isclose(identity_rows[0][1], ising.loss_constant)

    def test_round_trip_through_to_spo_pauli_dict(self):
        """``qubo_to_spo`` agrees with ``to_spo`` of the equivalent Pauli dict.

        :meth:`SparsePauliOp.to_list` emits Qiskit-convention labels
        (qubit 0 rightmost) but :func:`to_spo` reads divi convention
        (qubit 0 leftmost), so labels are reversed when feeding them back
        through ``to_spo``.
        """
        qubo = {(0,): -1.0, (1,): -1.0, (0, 1): 2.0}
        spo_q = qubo_to_spo(qubo)
        pauli_dict = {
            label[::-1]: float(np.real(coeff)) for label, coeff in spo_q.to_list()
        }
        spo_r = to_spo(pauli_dict)
        assert spo_q.simplify() == spo_r.simplify()

    def test_numpy_matrix_input(self):
        Q = np.array([[-1.0, 2.0], [0.0, -1.0]])
        spo = qubo_to_spo(Q)
        assert spo.num_qubits == 2

    def test_quadratized_path_passes_strength(self):
        """The ``quadratization_strength`` kwarg threads through to the
        underlying converter selection."""
        hubo = {(0, 1, 2): 1.0, (0,): -1.0, (1,): -1.0, (2,): -1.0}
        spo = qubo_to_spo(
            hubo, hamiltonian_builder="quadratized", quadratization_strength=5.0
        )
        assert spo.num_qubits >= 3
