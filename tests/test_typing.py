# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.typing module."""

import dimod
import numpy as np
import pytest
import scipy.sparse as sps

from divi.hamiltonians import (
    hubo_to_binary_polynomial,
    normalize_binary_polynomial_problem,
    qubo_to_binary_polynomial,
)
from divi.typing import (
    qubo_to_matrix,
)


class TestQuboToMatrixBQM:
    """Cover the BinaryQuadraticModel branch (L40-51)."""

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
        # Off-diagonal should match quadratic bias (symmetric)
        assert result[idx_a, idx_b] == pytest.approx(0.5)
        assert result[idx_b, idx_a] == pytest.approx(0.5)

    def test_non_binary_bqm_raises(self):
        """A non-BINARY (SPIN) BQM should raise ValueError."""
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, vartype=dimod.SPIN)
        with pytest.raises(ValueError, match="vartype='BINARY'"):
            qubo_to_matrix(bqm)


class TestQuboToMatrixSparse:
    """Cover the sparse matrix branch (L65-72)."""

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
    """Cover the unsupported type fallback (L74)."""

    def test_unsupported_type_raises(self):
        """A string (or any unsupported type) should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported QUBO type"):
            qubo_to_matrix("not a matrix")

    def test_unsupported_type_int_raises(self):
        """An integer should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported QUBO type"):
            qubo_to_matrix(42)


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
