# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary-polynomial-to-Ising Hamiltonian conversion."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, NamedTuple, Protocol
from warnings import warn

import dimod
import numpy as np
import numpy.typing as npt
import pennylane as qml
import scipy.sparse as sps

from divi.hamiltonians._polynomial import normalize_binary_polynomial_problem
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian,
    _is_empty_hamiltonian,
    _z_product,
)
from divi.hamiltonians._types import BinaryPolynomialProblem


class IsingEncoding(NamedTuple):
    """Result of converting a binary polynomial problem to an Ising Hamiltonian."""

    operator: qml.operation.Operator
    constant: float
    decode_fn: Callable[[str], Any]
    metadata: dict[str, object] | None = None


class BinaryToIsingConverter(Protocol):
    """Protocol for pluggable binary-to-Ising converters."""

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        """Convert a canonical binary-polynomial problem to an Ising Hamiltonian."""
        ...


@dataclass(frozen=True)
class NativeIsingConverter(BinaryToIsingConverter):
    """Convert binary polynomials to Ising operators by exact substitution x=(1-Z)/2."""

    zero_tol: float = 1e-12

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        n_qubits = problem.n_vars
        if n_qubits == 0:
            return IsingEncoding(
                operator=qml.Hamiltonian([], []),
                constant=problem.constant,
                decode_fn=lambda bitstring: np.array([], dtype=np.int32),
                metadata={"strategy": "native", "term_count": 0},
            )

        z_term_weights: dict[tuple[int, ...], float] = {}
        constant = 0.0

        for term, coeff in problem.terms.items():
            if abs(coeff) <= self.zero_tol:
                continue

            if len(term) == 0:
                constant += float(coeff)
                continue

            term_indices = tuple(problem.variable_to_idx[var] for var in term)
            scale = float(coeff) / (2 ** len(term_indices))

            for subset_size in range(len(term_indices) + 1):
                for subset in combinations(term_indices, subset_size):
                    contribution = scale if subset_size % 2 == 0 else -scale
                    if subset_size == 0:
                        constant += contribution
                        continue
                    z_term_weights[subset] = (
                        z_term_weights.get(subset, 0.0) + contribution
                    )

        weighted_terms = [
            weight * _z_product(indices)
            for indices, weight in z_term_weights.items()
            if abs(weight) > self.zero_tol
        ]
        operator = (
            qml.sum(*weighted_terms).simplify()
            if weighted_terms
            else qml.Hamiltonian([], [])
        )

        variable_to_idx = problem.variable_to_idx
        variable_order = problem.variable_order

        def _decode(bitstring: str) -> np.ndarray:
            return np.fromiter(
                (int(bitstring[variable_to_idx[var]]) for var in variable_order),
                dtype=np.int32,
            )

        return IsingEncoding(
            operator=operator,
            constant=float(constant),
            decode_fn=_decode,
            metadata={"strategy": "native", "term_count": len(weighted_terms)},
        )


@dataclass(frozen=True)
class QuadratizedIsingConverter(BinaryToIsingConverter):
    """Convert binary polynomials to Ising operators via quadratization to QUBO/BQM."""

    strength: float = 10.0

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        if problem.n_vars == 0:
            return IsingEncoding(
                operator=qml.Hamiltonian([], []),
                constant=problem.constant,
                decode_fn=lambda bitstring: np.array([], dtype=np.int32),
                metadata={"strategy": "quadratized", "ancilla_count": 0},
            )

        bqm = dimod.make_quadratic(problem.polynomial, self.strength, dimod.BINARY)
        qubo_terms, offset = bqm.to_qubo()

        variable_order = tuple(sorted(bqm.variables, key=repr))
        var_to_idx = {var: idx for idx, var in enumerate(variable_order)}
        qubo_matrix = np.zeros((len(variable_order), len(variable_order)), dtype=float)

        for (u, v), coeff in qubo_terms.items():
            i, j = var_to_idx[u], var_to_idx[v]
            qubo_matrix[i, j] += float(coeff)

        # Quadratization output can be asymmetric depending on term ordering.
        # Normalize once here so converter-level warning semantics stay reserved
        # for direct external unsanitized inputs.
        if not _is_sanitized(qubo_matrix):
            qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

        operator, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)
        original_vars = set(problem.variable_order)
        original_variable_order = problem.variable_order
        ancilla_variables = [var for var in variable_order if var not in original_vars]

        # Build decode function that maps measured bitstring (over all variables
        # including ancillas) back to the original variable assignments only.
        measure_variable_to_idx = var_to_idx

        def _decode(bitstring: str) -> np.ndarray:
            return np.fromiter(
                (
                    int(bitstring[measure_variable_to_idx[var]])
                    for var in original_variable_order
                ),
                dtype=np.int32,
            )

        return IsingEncoding(
            operator=operator,
            constant=float(constant + offset),
            decode_fn=_decode,
            metadata={
                "strategy": "quadratized",
                "strength": self.strength,
                "ancilla_count": len(ancilla_variables),
                "ancilla_variables": ancilla_variables,
            },
        )


def _resolve_ising_converter(
    hamiltonian_builder: Literal["native", "quadratized"],
    *,
    quadratization_strength: float,
) -> BinaryToIsingConverter:
    """Resolve a converter selector string."""
    if hamiltonian_builder == "native":
        return NativeIsingConverter()
    if hamiltonian_builder == "quadratized":
        return QuadratizedIsingConverter(strength=quadratization_strength)
    raise ValueError("hamiltonian_builder must be either 'native' or 'quadratized'.")


class IsingResult(NamedTuple):
    """Result of converting a QUBO/HUBO to a cleaned Ising Hamiltonian."""

    cost_hamiltonian: qml.operation.Operator
    loss_constant: float
    n_qubits: int
    encoding: IsingEncoding


def qubo_to_ising(
    qubo,
    *,
    hamiltonian_builder: str = "native",
    quadratization_strength: float = 10.0,
) -> IsingResult:
    """Convert a QUBO/HUBO to a cleaned Ising Hamiltonian.

    Normalizes the input, converts via the selected Ising converter,
    cleans constant terms, and validates the result.

    Args:
        qubo: QUBO dict, HUBO dict, numpy matrix, BQM, or BinaryPolynomial.
        hamiltonian_builder: ``"native"`` or ``"quadratized"``.
        quadratization_strength: Penalty for quadratization.

    Returns:
        :class:`IsingResult` with cost Hamiltonian, loss constant,
        qubit count, and full Ising encoding.

    Raises:
        ValueError: If the Hamiltonian contains only constant terms.
    """
    canonical = normalize_binary_polynomial_problem(qubo)
    converter = _resolve_ising_converter(
        hamiltonian_builder, quadratization_strength=quadratization_strength
    )
    encoding = converter.convert(canonical)

    raw_ham = encoding.operator
    cleaned, ham_constant = _clean_hamiltonian(raw_ham)
    if _is_empty_hamiltonian(cleaned):
        raise ValueError("Hamiltonian contains only constant terms.")

    return IsingResult(
        cost_hamiltonian=cleaned,
        loss_constant=encoding.constant + ham_constant,
        n_qubits=len(raw_ham.wires),
        encoding=encoding,
    )


def _is_sanitized(
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
) -> npt.NDArray[np.float64] | sps.spmatrix:
    """
    Check if a QUBO matrix is either symmetric or upper triangular.

    This function validates that the input QUBO matrix is in a proper format
    for conversion to an Ising Hamiltonian. The matrix should be either
    symmetric (equal to its transpose) or upper triangular.

    Args:
        qubo_matrix (npt.NDArray[np.float64] | sps.spmatrix): The QUBO matrix to validate.
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
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
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
        qubo_matrix (npt.NDArray[np.float64] | sps.spmatrix): The QUBO matrix Q where the
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
            # Factor of weight/2 because x^T Q x for symmetric Q counts both
            # (i,j) and (j,i), so triu entry is half the total interaction.
            ising_terms.append([i, j])
            ising_weights.append(weight / 2)

            # Update linear terms
            linear_terms[i] -= weight / 2
            linear_terms[j] -= weight / 2

            # Update constant term
            constant_term += weight / 2

    # Add the linear terms (Z operators)
    for i, curr_lin_term in filter(lambda x: x[1] != 0, enumerate(linear_terms)):
        ising_terms.append([i])
        ising_weights.append(float(curr_lin_term))

    # Construct the Ising Hamiltonian as a PennyLane operator
    pauli_string = qml.Identity(0) * 0
    for term, weight in zip(ising_terms, ising_weights):
        curr_term = _z_product(tuple(term)) * weight
        pauli_string += curr_term

    return pauli_string.simplify(), constant_term
