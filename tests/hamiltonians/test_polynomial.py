# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for binary polynomial types, normalization, and JIT evaluation kernels.

Covers:
- ``qubo_to_matrix`` (BQM, sparse, unsupported input branches)
- ``hubo_to_binary_polynomial`` / ``qubo_to_binary_polynomial``
- ``normalize_binary_polynomial_problem``
- ``compile_problem`` and the underlying JIT kernels
  (``_eval_poly_1d_jit``, ``_eval_poly_2d_jit``, ``_compute_hard_cvar_energy_jit``).
"""

import dimod
import numpy as np
import pytest
import scipy.sparse as sps

from divi.hamiltonians import (
    _compute_hard_cvar_energy_jit,
    _eval_poly_1d_jit,
    _eval_poly_2d_jit,
    _evaluate_binary_polynomial,
    compile_problem,
    hubo_to_binary_polynomial,
    normalize_binary_polynomial_problem,
    qubo_to_binary_polynomial,
    qubo_to_matrix,
)
from divi.pipeline.stages._pce_cost_stage import _compute_hard_cvar_energy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_problem(qubo_matrix):
    """Build a BinaryPolynomialProblem from a QUBO matrix."""
    return normalize_binary_polynomial_problem(np.asarray(qubo_matrix, dtype=float))


def _make_hubo_problem(hubo_dict):
    """Build a BinaryPolynomialProblem from a HUBO dict."""
    return normalize_binary_polynomial_problem(hubo_dict)


# ---------------------------------------------------------------------------
# qubo_to_matrix
# ---------------------------------------------------------------------------


class TestQuboToMatrixBQM:
    """Cover the BinaryQuadraticModel branch."""

    def test_valid_binary_bqm(self):
        """A BINARY BQM should be converted to a dense ndarray."""
        bqm = dimod.BinaryQuadraticModel(
            {"a": 1.0, "b": -2.0},
            {("a", "b"): 0.5},
            vartype=dimod.BINARY,
        )
        result = qubo_to_matrix(bqm)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # Diagonal should match linear biases
        variables = list(bqm.variables)
        idx_a = variables.index("a")
        idx_b = variables.index("b")
        assert result[idx_a, idx_a] == pytest.approx(1.0)
        assert result[idx_b, idx_b] == pytest.approx(-2.0)
        # Off-diagonal should be half the quadratic bias (symmetric convention:
        # Q[i,j] = Q[j,i] = J_ij/2 so that x^T Q x = BQM energy - offset)
        assert result[idx_a, idx_b] == pytest.approx(0.25)
        assert result[idx_b, idx_a] == pytest.approx(0.25)

    def test_non_binary_bqm_raises(self):
        """A non-BINARY (SPIN) BQM should raise ValueError."""
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, vartype=dimod.SPIN)
        with pytest.raises(ValueError, match="vartype='BINARY'"):
            qubo_to_matrix(bqm)


class TestQuboToMatrixSparse:
    """Cover the sparse matrix branch."""

    def test_valid_square_sparse(self):
        """A square sparse matrix should be returned as-is."""
        sparse = sps.csr_matrix(np.eye(3))
        result = qubo_to_matrix(sparse)
        assert sps.issparse(result)
        assert result.shape == (3, 3)

    def test_non_square_sparse_raises(self):
        """A non-square sparse matrix should raise ValueError."""
        sparse = sps.csr_matrix(np.ones((2, 3)))
        with pytest.raises(ValueError, match="Must be a square matrix"):
            qubo_to_matrix(sparse)


class TestQuboToMatrixUnsupported:
    """Cover the unsupported-type fallback."""

    def test_unsupported_type_raises(self):
        """A string (or any unsupported type) should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported QUBO type"):
            qubo_to_matrix("not a matrix")

    def test_unsupported_type_int_raises(self):
        """An integer should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported QUBO type"):
            qubo_to_matrix(42)


# ---------------------------------------------------------------------------
# hubo_to_binary_polynomial / qubo_to_binary_polynomial
# ---------------------------------------------------------------------------


class TestHuboToBinaryPolynomial:
    """Coverage for HUBO normalization to BinaryPolynomial."""

    def test_hubo_dict_with_constant_converts(self):
        hubo = {
            ("x0",): -1.0,
            ("x0", "x1", "x2"): 2.5,
            (): 3.0,
        }

        polynomial = hubo_to_binary_polynomial(hubo)
        assert isinstance(polynomial, dimod.BinaryPolynomial)
        assert polynomial.vartype == dimod.Vartype.BINARY
        assert polynomial[frozenset({"x0"})] == pytest.approx(-1.0)
        assert polynomial[frozenset({"x0", "x1", "x2"})] == pytest.approx(2.5)
        assert polynomial[frozenset()] == pytest.approx(3.0)

    def test_hubo_dict_duplicate_variables_raises(self):
        with pytest.raises(ValueError, match="duplicate variables"):
            hubo_to_binary_polynomial({("x0", "x0"): 1.0})

    def test_unsupported_hubo_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported HUBO type"):
            hubo_to_binary_polynomial(123)  # type: ignore[arg-type]


class TestQuboToBinaryPolynomial:
    """Coverage for QUBO conversion to BinaryPolynomial."""

    def test_dense_matrix_converts_to_quadratic_polynomial(self):
        qubo = np.array([[1.0, 2.0], [0.0, 3.0]])
        polynomial = qubo_to_binary_polynomial(qubo)

        assert polynomial[frozenset({0})] == pytest.approx(1.0)
        assert polynomial[frozenset({0, 1})] == pytest.approx(2.0)
        assert polynomial[frozenset({1})] == pytest.approx(3.0)

    def test_binary_quadratic_model_keeps_labels(self):
        bqm = dimod.BinaryQuadraticModel(
            {"a": 1.0, "b": -2.0},
            {("a", "b"): 0.75},
            1.25,
            dimod.Vartype.BINARY,
        )
        polynomial = qubo_to_binary_polynomial(bqm)

        assert polynomial[frozenset({"a"})] == pytest.approx(1.0)
        assert polynomial[frozenset({"b"})] == pytest.approx(-2.0)
        assert polynomial[frozenset({"a", "b"})] == pytest.approx(0.75)
        assert polynomial[frozenset()] == pytest.approx(1.25)


class TestNormalizeBinaryPolynomialProblem:
    """Coverage for canonical internal problem normalization."""

    def test_normalize_hubo_with_custom_variable_order(self):
        hubo = {("x2", "x0"): 1.5, ("x1",): -3.0, (): 0.5}
        normalized = normalize_binary_polynomial_problem(
            hubo,
            variable_order=("x0", "x1", "x2"),
        )

        assert normalized.variable_order == ("x0", "x1", "x2")
        assert normalized.variable_to_idx == {"x0": 0, "x1": 1, "x2": 2}
        assert normalized.terms[("x0", "x2")] == pytest.approx(1.5)
        assert normalized.terms[("x1",)] == pytest.approx(-3.0)
        assert normalized.constant == pytest.approx(0.5)
        assert normalized.n_vars == 3

    def test_normalize_uses_deterministic_default_order(self):
        hubo = {("b",): 1.0, ("a",): 2.0}
        normalized = normalize_binary_polynomial_problem(hubo)

        # Sorted by repr for deterministic behavior across mixed types.
        assert normalized.variable_order == ("a", "b")
        assert normalized.variable_to_idx == {"a": 0, "b": 1}

    def test_variable_order_mismatch_raises(self):
        hubo = {("a",): 1.0, ("b",): 2.0}
        with pytest.raises(ValueError, match="must contain exactly the variables"):
            normalize_binary_polynomial_problem(hubo, variable_order=("a", "c"))

    def test_variable_order_duplicates_raise(self):
        hubo = {("a",): 1.0}
        with pytest.raises(ValueError, match="must not contain duplicates"):
            normalize_binary_polynomial_problem(hubo, variable_order=("a", "a"))


# ---------------------------------------------------------------------------
# compile_problem
# ---------------------------------------------------------------------------


class TestCompileProblem:
    def test_diagonal_qubo(self):
        """Diagonal QUBO: only degree-1 terms."""
        problem = _make_problem(np.diag([1.0, 2.0, 3.0]))
        ti, to, tc, const = compile_problem(problem)

        assert const == 0.0
        assert len(tc) == 3
        # Each term has exactly 1 index
        for t in range(len(tc)):
            assert to[t + 1] - to[t] == 1

    def test_full_qubo(self):
        """Full QUBO: degree-1 (diagonal) + degree-2 (off-diagonal) terms."""
        Q = np.array([[1.0, -0.5], [-0.5, 2.0]])
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        # 2 diagonal + 1 off-diagonal = 3 terms (symmetric → one off-diag)
        assert len(tc) >= 2
        assert const == 0.0

    def test_empty_problem(self):
        """Zero-coefficient QUBO produces empty arrays."""
        problem = _make_problem(np.zeros((2, 2)))
        ti, to, tc, const = compile_problem(problem)

        assert len(tc) == 0
        assert len(ti) == 0
        assert const == 0.0

    def test_hubo_cubic(self):
        """HUBO with cubic term produces degree-3 entries."""
        hubo = {(0,): -1.0, (1,): -2.0, (0, 1, 2): 3.0}
        problem = _make_hubo_problem(hubo)
        ti, to, tc, const = compile_problem(problem)

        degrees = [to[t + 1] - to[t] for t in range(len(tc))]
        assert 3 in degrees  # cubic term present


# ---------------------------------------------------------------------------
# _eval_poly_1d_jit
# ---------------------------------------------------------------------------


class TestEvalPoly1dJit:
    def test_matches_python_loop(self):
        """JIT 1D result matches the Python fallback for a random QUBO."""
        Q = np.array([[1.0, -0.2], [-0.2, 0.5]])
        problem = _make_problem(Q)
        compiled = compile_problem(problem)
        ti, to, tc, const = compiled

        x = np.array([0.7, 0.3])
        expected = _evaluate_binary_polynomial(x, problem)
        result = float(_eval_poly_1d_jit(x, ti, to, tc, const))

        assert result == pytest.approx(expected, rel=1e-12)

    def test_binary_assignment(self):
        """JIT gives correct energy for binary {0,1} assignments."""
        Q = np.diag([1.0, 2.0])
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        # x = [1, 0] → energy = 1*1² + 2*0² = 1.0
        assert float(
            _eval_poly_1d_jit(np.array([1.0, 0.0]), ti, to, tc, const)
        ) == pytest.approx(1.0)
        # x = [0, 1] → energy = 1*0² + 2*1² = 2.0
        assert float(
            _eval_poly_1d_jit(np.array([0.0, 1.0]), ti, to, tc, const)
        ) == pytest.approx(2.0)
        # x = [0, 0] → energy = 0.0
        assert float(
            _eval_poly_1d_jit(np.array([0.0, 0.0]), ti, to, tc, const)
        ) == pytest.approx(0.0)

    def test_hubo_evaluation(self):
        """JIT evaluates cubic HUBO correctly."""
        hubo = {(0,): -3.0, (1,): -3.0, (2,): -3.0, (0, 1, 2): 2.0}
        problem = _make_hubo_problem(hubo)
        ti, to, tc, const = compile_problem(problem)

        x = np.array([1.0, 1.0, 1.0])
        expected = _evaluate_binary_polynomial(x, problem)
        result = float(_eval_poly_1d_jit(x, ti, to, tc, const))

        assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# _eval_poly_2d_jit
# ---------------------------------------------------------------------------


class TestEvalPoly2dJit:
    def test_matches_python_loop(self):
        """JIT 2D result matches the Python fallback for batched assignments."""
        Q = np.array([[1.0, -0.2, 0.0], [-0.2, 0.5, 0.0], [0.0, 0.0, -1.0]])
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        x = np.random.default_rng(42).random((3, 100))
        expected = _evaluate_binary_polynomial(x, problem)
        result = _eval_poly_2d_jit(x, ti, to, tc, const)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_single_state_matches_1d(self):
        """2D kernel with one state matches 1D kernel."""
        Q = np.array([[1.0, -0.5], [-0.5, 2.0]])
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        x_1d = np.array([0.6, 0.4])
        x_2d = x_1d.reshape(-1, 1)

        result_1d = float(_eval_poly_1d_jit(x_1d, ti, to, tc, const))
        result_2d = float(_eval_poly_2d_jit(x_2d, ti, to, tc, const)[0])

        assert result_2d == pytest.approx(result_1d, rel=1e-12)

    def test_large_batch(self):
        """2D kernel handles 5000 states without error."""
        Q = np.eye(10)
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        x = np.random.default_rng(42).random((10, 5000))
        result = _eval_poly_2d_jit(x, ti, to, tc, const)

        assert result.shape == (5000,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# _compute_hard_cvar_energy_jit
# ---------------------------------------------------------------------------


class TestComputeHardCvarEnergyJit:
    def test_matches_python_fallback(self):
        """JIT CVaR matches the Python fallback path."""
        Q = np.diag([1.0, 2.0])
        problem = _make_problem(Q)
        compiled = compile_problem(problem)
        ti, to, tc, const = compiled

        parities = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.uint8)
        counts = np.array([10.0, 20.0, 30.0, 40.0])
        total_shots = 100.0
        alpha_cvar = 0.25

        expected = _compute_hard_cvar_energy(
            parities, counts, total_shots, problem, alpha_cvar
        )
        x_vals = np.ascontiguousarray(1.0 - parities.astype(np.float64))
        result = float(
            _compute_hard_cvar_energy_jit(
                x_vals, counts, total_shots, alpha_cvar, ti, to, tc, const
            )
        )

        assert result == pytest.approx(expected, rel=1e-10)

    def test_all_equal_energies(self):
        """When all states have the same energy, CVaR equals that energy."""
        Q = np.zeros((2, 2))
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        x_vals = np.ones((2, 4), dtype=np.float64)
        counts = np.array([25.0, 25.0, 25.0, 25.0])

        result = float(
            _compute_hard_cvar_energy_jit(
                x_vals, counts, 100.0, 0.25, ti, to, tc, const
            )
        )
        assert result == pytest.approx(0.0)

    def test_alpha_cvar_selects_tail(self):
        """Smaller alpha_cvar selects lower-energy tail."""
        Q = np.diag([1.0, 2.0])
        problem = _make_problem(Q)
        ti, to, tc, const = compile_problem(problem)

        # 4 states with different energies
        parities = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.uint8)
        x_vals = np.ascontiguousarray(1.0 - parities.astype(np.float64))
        counts = np.array([25.0, 25.0, 25.0, 25.0])

        e_25 = float(
            _compute_hard_cvar_energy_jit(
                x_vals, counts, 100.0, 0.25, ti, to, tc, const
            )
        )
        e_75 = float(
            _compute_hard_cvar_energy_jit(
                x_vals, counts, 100.0, 0.75, ti, to, tc, const
            )
        )
        # 25% tail should be <= 75% tail (CVaR picks lower energies)
        assert e_25 <= e_75
