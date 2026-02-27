# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from dataclasses import dataclass

import dimod
import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse as sps

# ── Pipeline type aliases ────────────────────────────────────────────

AxisLabel = tuple[
    str, Hashable
]  # A single (axis_name, value) pair used in batch and branch keys.

QASMTag = tuple[AxisLabel, ...]  # Sequence of AxisLabels labelling a QASM body variant.

# ── QProg type aliases ───────────────────────────────────────────────

GraphProblemTypes = nx.Graph | rx.PyGraph
QUBOProblemTypes = list | np.ndarray | sps.spmatrix | dimod.BinaryQuadraticModel
HUBOTerm = tuple[Hashable, ...]
HUBOProblemTypes = dict[HUBOTerm, float] | dimod.BinaryPolynomial


@dataclass(frozen=True)
class BinaryPolynomialProblem:
    """Canonical internal representation for binary polynomial optimization problems."""

    polynomial: dimod.BinaryPolynomial
    variable_order: tuple[Hashable, ...]
    variable_to_idx: dict[Hashable, int]
    constant: float

    @property
    def terms(self) -> dict[HUBOTerm, float]:
        """Terms with tuple keys ordered by variable_order (derived from polynomial)."""
        result: dict[HUBOTerm, float] = {}
        for term, coeff in self.polynomial.items():
            ordered = tuple(sorted(term, key=lambda var: self.variable_to_idx[var]))
            result[ordered] = float(coeff)
        return result

    @property
    def n_vars(self) -> int:
        return len(self.variable_order)


def qubo_to_matrix(qubo: QUBOProblemTypes) -> np.ndarray | sps.spmatrix:
    """Convert supported QUBO inputs to a square matrix.

    Args:
        qubo: QUBO input as list, ndarray, sparse matrix, or BinaryQuadraticModel.

    Returns:
        Square QUBO matrix as a dense ndarray or sparse matrix.

    Raises:
        ValueError: If the input cannot be converted to a square matrix or the
            BinaryQuadraticModel is not binary.
    """
    if isinstance(qubo, dimod.BinaryQuadraticModel):
        if qubo.vartype != dimod.Vartype.BINARY:
            raise ValueError(
                f"BinaryQuadraticModel must have vartype='BINARY', got {qubo.vartype}"
            )
        variables = list(qubo.variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}
        matrix = np.diag([qubo.linear.get(v, 0) for v in variables])
        for (u, v), coeff in qubo.quadratic.items():
            i, j = var_to_idx[u], var_to_idx[v]
            matrix[i, j] = matrix[j, i] = coeff
        return matrix

    if isinstance(qubo, list):
        qubo = np.asarray(qubo)

    if isinstance(qubo, np.ndarray):
        if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
            raise ValueError(
                "Invalid QUBO matrix."
                f" Got array of shape {qubo.shape}."
                " Must be a square matrix."
            )
        return qubo

    if sps.isspmatrix(qubo):
        if qubo.shape[0] != qubo.shape[1]:
            raise ValueError(
                "Invalid QUBO matrix."
                f" Got sparse matrix of shape {qubo.shape}."
                " Must be a square matrix."
            )
        return qubo

    raise ValueError(f"Unsupported QUBO type: {type(qubo)}")
