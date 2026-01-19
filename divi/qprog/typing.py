# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse as sps

GraphProblemTypes = nx.Graph | rx.PyGraph
QUBOProblemTypes = list | np.ndarray | sps.spmatrix | dimod.BinaryQuadraticModel


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
