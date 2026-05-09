# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary-polynomial-to-Ising Hamiltonian conversion."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Any, Literal, Protocol
from warnings import warn

import dimod
import numpy as np
import numpy.typing as npt
import pennylane as qp
import scipy.sparse as sps
from qiskit.quantum_info import SparsePauliOp

from ._polynomial import normalize_binary_polynomial_problem
from ._term_ops import (
    _clean_hamiltonian_spo,
    _empty_spo,
    _from_spo,
    _num_qubits,
)
from ._types import BinaryPolynomialProblem


def _empty_decode(bitstring: str) -> np.ndarray:
    """Decode function for the n_vars == 0 / empty-problem case."""
    return np.array([], dtype=np.int32)


@dataclass(eq=False)
class IsingEncoding:
    """Result of converting a binary polynomial problem to an Ising Hamiltonian.

    The cost operator is held as a ``SparsePauliOp`` and materialised to a
    PennyLane :class:`~pennylane.operation.Operator` on first access via
    the :attr:`operator` property.
    """

    _operator_spo: SparsePauliOp
    constant: float
    decode_fn: Callable[[str], Any]
    metadata: dict[str, object] | None = None

    @property
    def wires(self) -> tuple:
        """Canonical wire mapping aligned with the SPO (always ``range(num_qubits)``)."""
        return tuple(range(_num_qubits(self._operator_spo)))

    @cached_property
    def operator(self) -> qp.operation.Operator:
        """Cost Hamiltonian as a PennyLane operator (built lazily on first access)."""
        return _from_spo(self._operator_spo, self.wires)


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
                _operator_spo=_empty_spo(0),
                constant=problem.constant,
                decode_fn=_empty_decode,
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

        sparse_terms = [
            ("Z" * len(indices), list(indices), float(weight))
            for indices, weight in z_term_weights.items()
            if abs(weight) > self.zero_tol
        ]
        if sparse_terms:
            spo = SparsePauliOp.from_sparse_list(
                sparse_terms, num_qubits=n_qubits
            ).simplify()
        else:
            spo = _empty_spo(n_qubits)

        variable_to_idx = problem.variable_to_idx
        variable_order = problem.variable_order

        def _decode(bitstring: str) -> np.ndarray:
            return np.fromiter(
                (int(bitstring[variable_to_idx[var]]) for var in variable_order),
                dtype=np.int32,
            )

        return IsingEncoding(
            _operator_spo=spo,
            constant=float(constant),
            decode_fn=_decode,
            metadata={"strategy": "native", "term_count": len(sparse_terms)},
        )


@dataclass(frozen=True)
class QuadratizedIsingConverter(BinaryToIsingConverter):
    """Convert binary polynomials to Ising operators via quadratization to QUBO/BQM."""

    strength: float = 10.0

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        if problem.n_vars == 0:
            return IsingEncoding(
                _operator_spo=_empty_spo(0),
                constant=problem.constant,
                decode_fn=_empty_decode,
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

        # Quadratization output can be asymmetric; symmetrise silently here
        # so the user-facing warning stays reserved for direct unsanitised inputs.
        if not _is_sanitized(qubo_matrix):
            qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

        spo, constant = _convert_qubo_matrix_to_ising_spo(qubo_matrix)
        original_vars = set(problem.variable_order)
        original_variable_order = problem.variable_order
        ancilla_variables = [var for var in variable_order if var not in original_vars]

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
            _operator_spo=spo,
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


@dataclass(eq=False)
class IsingResult:
    """Result of converting a QUBO/HUBO to a cleaned Ising Hamiltonian.

    Mirrors :class:`IsingEncoding` in that ``cost_hamiltonian`` is built
    lazily on first access from an internal ``SparsePauliOp``.
    """

    _cost_hamiltonian_spo: SparsePauliOp
    loss_constant: float
    n_qubits: int
    encoding: IsingEncoding

    @property
    def wires(self) -> tuple:
        """Canonical wire mapping aligned with the SPO."""
        return tuple(range(_num_qubits(self._cost_hamiltonian_spo)))

    @cached_property
    def cost_hamiltonian(self) -> qp.operation.Operator:
        """Cleaned cost Hamiltonian as a PennyLane operator (lazy on first access)."""
        return _from_spo(self._cost_hamiltonian_spo, self.wires)


def qubo_to_ising(
    qubo,
    *,
    hamiltonian_builder: Literal["native", "quadratized"] = "native",
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

    cleaned_spo, ham_constant = _clean_hamiltonian_spo(encoding._operator_spo)
    if cleaned_spo.size == 0:
        raise ValueError("Hamiltonian contains only constant terms.")

    return IsingResult(
        _cost_hamiltonian_spo=cleaned_spo,
        loss_constant=encoding.constant + ham_constant,
        n_qubits=_num_qubits(encoding._operator_spo),
        encoding=encoding,
    )


def _is_sanitized(
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
) -> bool:
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
    if sps.issparse(qubo_matrix):
        sparse_m = sps.csr_matrix(qubo_matrix)
        return bool(
            (sparse_m != sparse_m.T).nnz == 0
            or (sparse_m != sps.triu(sparse_m)).nnz == 0
        )
    dense_m = np.asarray(qubo_matrix)
    return bool(
        np.allclose(dense_m, dense_m.T) or np.allclose(dense_m, np.triu(dense_m))
    )


def _convert_qubo_matrix_to_ising_spo(
    qubo_matrix: npt.NDArray[np.float64] | sps.spmatrix,
) -> tuple[SparsePauliOp, float]:
    """Convert a QUBO matrix to an Ising Hamiltonian as a ``SparsePauliOp``.

    The mapping ``x_i = (1 - σ_i)/2`` rewrites ``min x^T Q x`` as a
    Pauli-Z Ising minimisation.

    Args:
        qubo_matrix: Square QUBO matrix Q (dense or scipy.sparse). Symmetrised
            with a warning if neither symmetric nor upper-triangular.

    Returns:
        ``(spo, constant_offset)`` — Pauli-Z-only Ising Hamiltonian and the
        energy offset to add back when scoring.
    """
    if not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )

    if sps.issparse(qubo_matrix):
        sparse_m = sps.csr_matrix(qubo_matrix)
        symmetrized_qubo = (sparse_m + sparse_m.T) / 2
        coo_mat = sps.triu(symmetrized_qubo).tocoo()
        rows, cols, values = coo_mat.row, coo_mat.col, coo_mat.data
    else:
        dense_m = np.asarray(qubo_matrix)
        symmetrized_qubo = (dense_m + dense_m.T) / 2
        triu_matrix = np.triu(symmetrized_qubo)
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
            linear_terms[i] -= weight / 2
            constant_term += weight / 2
        else:
            # x^T Q x for symmetric Q counts both (i,j) and (j,i), so the
            # triu entry contributes half the total interaction.
            ising_terms.append([i, j])
            ising_weights.append(weight / 2)
            linear_terms[i] -= weight / 2
            linear_terms[j] -= weight / 2
            constant_term += weight / 2

    for i, curr_lin_term in ((i, v) for i, v in enumerate(linear_terms) if v != 0):
        ising_terms.append([i])
        ising_weights.append(float(curr_lin_term))

    if not ising_terms:
        return _empty_spo(n), constant_term

    sparse_terms = [
        ("Z" * len(term), [int(i) for i in term], float(weight))
        for term, weight in zip(ising_terms, ising_weights)
    ]
    spo = SparsePauliOp.from_sparse_list(sparse_terms, num_qubits=n).simplify()
    return spo, constant_term
