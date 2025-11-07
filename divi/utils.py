# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from warnings import warn

import numpy as np
import pennylane as qml
import scipy.sparse as sps


def hamiltonian_to_pauli_string(hamiltonian: qml.Hamiltonian, n_qubits: int) -> str:
    """
    Convert a QNode Hamiltonian to a semicolon-separated string of Pauli operators.

    Each term in the Hamiltonian is represented as a string of Pauli letters ('I', 'X', 'Y', 'Z'),
    one per qubit. Multiple terms are separated by semicolons.

    Args:
        hamiltonian (qml.Hamiltonian): The Pennylane Hamiltonian object to convert.
        n_qubits (int): Number of qubits to represent in the string.

    Returns:
        str: The Hamiltonian as a semicolon-separated string of Pauli operators.

    Raises:
        ValueError: If an unknown Pauli operator is encountered or wire index is out of range.
    """
    pauli_letters = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}
    identity_row = np.full(n_qubits, "I", dtype="<U1")

    terms = []
    for term in hamiltonian.operands:
        op = term.base if isinstance(term, qml.ops.SProd) else term
        ops = op.operands if isinstance(op, qml.ops.Prod) else [op]

        paulis = identity_row.copy()
        for p in ops:
            # Better fallback logic with validation
            if p.name in pauli_letters:
                pauli = pauli_letters[p.name]
            elif p.name.startswith("Pauli") and len(p.name) > 5:
                pauli = p.name[5:]  # Only if safe to slice
            else:
                raise ValueError(
                    f"Unknown Pauli operator: {p.name}. "
                    "Expected 'PauliX', 'PauliY', 'PauliZ', or a name starting with 'Pauli'."
                )

            # Bounds checking for wire indices
            if not p.wires:
                raise ValueError(f"Pauli operator {p.name} has no wires")

            wire = int(p.wires[0])
            if wire < 0 or wire >= n_qubits:
                raise ValueError(
                    f"Wire index {wire} out of range for {n_qubits} qubits. "
                    f"Valid range: [0, {n_qubits - 1}]"
                )

            paulis[wire] = pauli
        terms.append("".join(paulis))

    return ";".join(terms)


def reverse_dict_endianness(
    probs_dict: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Reverse endianness of all bitstrings in a dictionary of probability distributions."""
    return {
        tag: {bitstring[::-1]: prob for bitstring, prob in probs.items()}
        for tag, probs in probs_dict.items()
    }


def _is_sanitized(
    qubo_matrix: np.ndarray | sps.spmatrix,
) -> np.ndarray | sps.spmatrix:
    """
    Check if a QUBO matrix is either symmetric or upper triangular.

    This function validates that the input QUBO matrix is in a proper format
    for conversion to an Ising Hamiltonian. The matrix should be either
    symmetric (equal to its transpose) or upper triangular.

    Args:
        qubo_matrix (np.ndarray | sps.spmatrix): The QUBO matrix to validate.
            Can be a dense NumPy array or a sparse SciPy matrix.

    Returns:
        bool: True if the matrix is symmetric or upper triangular, False otherwise.
    """
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
    """
    Convert a QUBO matrix to an Ising Hamiltonian in PennyLane format.

    The conversion follows the mapping from QUBO variables x_i ∈ {0,1} to
    Ising variables σ_i ∈ {-1,1} via the transformation x_i = (1 - σ_i)/2. This
    transforms a QUBO minimization problem into an equivalent Ising minimization
    problem.

    The function handles both dense NumPy arrays and sparse SciPy matrices efficiently.
    If the input matrix is neither symmetric nor upper triangular, it will be
    symmetrized automatically with a warning.

    Args:
        qubo_matrix (np.ndarray | sps.spmatrix): The QUBO matrix Q where the
            objective is to minimize x^T Q x. Can be a dense NumPy array or a
            sparse SciPy matrix (any format). Should be square and either
            symmetric or upper triangular.

    Returns:
        tuple[qml.operation.Operator, float]: A tuple containing:
            - Ising Hamiltonian as a PennyLane operator (sum of Pauli Z terms)
            - Constant offset term to be added to energy calculations

    Raises:
        UserWarning: If the QUBO matrix is neither symmetric nor upper triangular.

    Example:
        >>> import numpy as np
        >>> qubo = np.array([[1, 2], [0, 3]])
        >>> hamiltonian, offset = convert_qubo_matrix_to_pennylane_ising(qubo)
        >>> print(f"Offset: {offset}")
    """

    if not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    is_sparse = sps.issparse(qubo_matrix)
    backend = sps if is_sparse else np

    symmetrized_qubo = (qubo_matrix + qubo_matrix.T) / 2

    # Gather non-zero indices in the upper triangle of the matrix
    triu_matrix = backend.triu(
        symmetrized_qubo,
        **(
            {"format": qubo_matrix.format if qubo_matrix.format != "coo" else "csc"}
            if is_sparse
            else {}
        ),
    )

    if is_sparse:
        coo_mat = triu_matrix.tocoo()
        rows, cols, values = coo_mat.row, coo_mat.col, coo_mat.data
    else:
        rows, cols = triu_matrix.nonzero()
        values = triu_matrix[rows, cols]

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
