# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.typing module — focuses on qubo_to_matrix edge cases."""

import dimod
import numpy as np
import pytest
import scipy.sparse as sps

from divi.typing import qubo_to_matrix


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
