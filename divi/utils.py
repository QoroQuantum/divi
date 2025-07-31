# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from warnings import warn

import numpy as np
import pennylane as qml
import scipy.sparse as sps


def _is_sanitized(
    qubo_matrix: np.ndarray | sps.spmatrix,
) -> np.ndarray | sps.spmatrix:
    # Sanitize the QUBO matrix to ensure it is either symmetric or upper triangular.

    is_sparse = sps.issparse(qubo_matrix)

    return (
        (
            ((qubo_matrix != qubo_matrix.T).nnz == 0)
            or ((qubo_matrix != sps.triu(qubo_matrix)).nnz == 0)
        )
        if is_sparse
        else (
            np.allclose(qubo_matrix, qubo_matrix.T)
            or np.allclose(qubo_matrix, np.triu(qubo_matrix))
        )
    )


def convert_qubo_matrix_to_pennylane_ising(
    qubo_matrix: np.ndarray | sps.spmatrix,
) -> tuple[qml.operation.Operator, float]:
    """Convert QUBO matrix to Ising Hamiltonian in Pennylane.

    The conversion follows the mapping:
    - QUBO variables x_i ∈ {0,1} map to Ising variables s_i ∈ {-1,1} via s_i = 2x_i - 1
    - This transforms a QUBO problem into an equivalent Ising problem

    Args:
        qubo_matrix: The QUBO matrix Q where the objective is to minimize x^T Q x

    Returns:
        A tuple of (Ising Hamiltonian as a PennyLane operator, constant term)
    """

    if not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    is_sparse = sps.issparse(qubo_matrix)
    backend = sps if is_sparse else np

    # Gather non-zero indices in the upper triangle of the matrix
    triu_matrix = backend.triu(
        qubo_matrix,
        **(
            {"format": qubo_matrix.format if qubo_matrix.format != "coo" else "csc"}
            if is_sparse
            else {}
        ),
    )
    rows, cols = triu_matrix.nonzero()
    values = triu_matrix[rows, cols].A1 if is_sparse else triu_matrix[rows, cols]

    n = qubo_matrix.shape[0]
    linear_terms = np.zeros(n)
    constant_term = 0.0
    ising_terms = []
    ising_weights = []

    for i, j, weight in zip(rows, cols, values):
        weight = float(weight)
        i, j = int(i), int(j)

        if i == j:
            # Diagonal elements
            linear_terms[i] -= weight / 2
            constant_term += weight / 2
        else:
            # Off-diagonal elements (i < j since we're using triu)
            ising_terms.append([i, j])
            ising_weights.append(weight / 4)

            # Update linear terms
            linear_terms[i] -= weight / 4
            linear_terms[j] -= weight / 4

            # Update constant term
            constant_term += weight / 4

    # Add the linear terms (Z operators)
    for i, curr_lin_term in filter(lambda x: x[1] != 0, enumerate(linear_terms)):
        ising_terms.append([i])
        ising_weights.append(float(curr_lin_term))

    # Construct the Ising Hamiltonian as a PennyLane operator
    pauli_string = qml.Identity(0) * 0
    for term, weight in zip(ising_terms, ising_weights):
        if len(term) == 1:
            # Single-qubit term (Z operator)
            curr_term = qml.Z(term[0]) * weight
        else:
            # Two-qubit term (ZZ interaction)
            curr_term = (
                reduce(lambda x, y: x @ y, map(lambda x: qml.Z(x), term)) * weight
            )

        pauli_string += curr_term

    return pauli_string.simplify(), constant_term
