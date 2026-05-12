# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary-polynomial-to-Ising Hamiltonian conversion."""

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Protocol
from warnings import warn

import dimod
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from qiskit.quantum_info import PauliList, SparsePauliOp

from ._polynomial import normalize_binary_polynomial_problem
from ._term_ops import _clean_hamiltonian_spo, _empty_spo, _n_qubits
from ._types import BinaryPolynomialProblem


def _wires_from_spo(spo: SparsePauliOp) -> tuple:
    return tuple(range(_n_qubits(spo)))


def _make_decode(idx_map: dict, var_order: tuple) -> Callable[[str], np.ndarray]:
    """Build a decode function that pulls ``var_order`` bits from a bitstring
    via ``idx_map``. Empty ``var_order`` produces the empty-problem decoder."""

    def _decode(bitstring: str) -> np.ndarray:
        return np.fromiter(
            (int(bitstring[idx_map[var]]) for var in var_order),
            dtype=np.int32,
            count=len(var_order),
        )

    return _decode


_empty_decode = _make_decode({}, ())


def _max_abs_input_coeff(problem: BinaryPolynomialProblem) -> float:
    """Largest absolute coefficient across all terms of ``problem`` (0.0 if empty)."""
    return max((abs(float(c)) for c in problem.terms.values()), default=0.0)


@dataclass(eq=False)
class IsingEncoding:
    """Result of converting a binary polynomial problem to an Ising Hamiltonian.

    The cost operator is held as a ``SparsePauliOp`` over qubits
    ``range(num_qubits)``.
    """

    operator: SparsePauliOp
    constant: float
    decode_fn: Callable[[str], Any]
    metadata: dict[str, object] | None = None

    @property
    def wires(self) -> tuple:
        """Canonical wire mapping aligned with the SPO (always ``range(num_qubits)``)."""
        return _wires_from_spo(self.operator)


class BinaryToIsingConverter(Protocol):
    """Protocol for pluggable binary-to-Ising converters."""

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        """Convert a canonical binary-polynomial problem to an Ising Hamiltonian."""
        ...


@dataclass(frozen=True)
class NativeIsingConverter(BinaryToIsingConverter):
    """Convert binary polynomials to Ising operators by exact substitution x=(1-Z)/2.

    :attr:`zero_tol` is applied *relative* to the largest accumulated
    coefficient on both the input and output sides.
    """

    zero_tol: float = 1e-12
    """Relative tolerance for pruning accumulated Z-product weights and
    input-polynomial coefficients."""

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        n_qubits = problem.n_vars
        if n_qubits == 0:
            return IsingEncoding(
                operator=_empty_spo(0),
                constant=problem.constant,
                decode_fn=_empty_decode,
                metadata={"strategy": "native", "term_count": 0},
            )

        # Collect signed contributions per Z-product subset in a list and
        # finalize via ``math.fsum``; naive ``+=`` loses precision over the
        # 2**k signed terms each degree-k monomial injects.
        z_term_parts: defaultdict[tuple[int, ...], list[float]] = defaultdict(list)
        constant_parts: list[float] = []

        max_input_coeff = _max_abs_input_coeff(problem)
        input_threshold = self.zero_tol * max_input_coeff

        for term, coeff in problem.terms.items():
            if abs(coeff) <= input_threshold:
                continue

            if len(term) == 0:
                constant_parts.append(float(coeff))
                continue

            term_indices = tuple(problem.variable_to_idx[var] for var in term)
            scale = float(coeff) / (2 ** len(term_indices))

            for subset_size in range(len(term_indices) + 1):
                for subset in combinations(term_indices, subset_size):
                    contribution = scale if subset_size % 2 == 0 else -scale
                    if subset_size == 0:
                        constant_parts.append(contribution)
                        continue
                    z_term_parts[subset].append(contribution)

        constant = math.fsum(constant_parts)
        z_term_weights = {k: math.fsum(parts) for k, parts in z_term_parts.items()}

        max_output_weight = max((abs(w) for w in z_term_weights.values()), default=0.0)
        output_threshold = self.zero_tol * max_output_weight
        filtered = [
            (subset, weight)
            for subset, weight in z_term_weights.items()
            if abs(weight) > output_threshold
        ]
        if not filtered:
            spo = _empty_spo(n_qubits)
        else:
            n_terms = len(filtered)
            z_arr = np.zeros((n_terms, n_qubits), dtype=bool)
            x_arr = np.zeros((n_terms, n_qubits), dtype=bool)
            coeffs = np.empty(n_terms, dtype=complex)
            for i, (subset, weight) in enumerate(filtered):
                z_arr[i, list(subset)] = True
                coeffs[i] = weight
            # Explicit absolute atol relative to the largest output weight.
            spo = SparsePauliOp(
                PauliList.from_symplectic(z_arr, x_arr),
                coeffs=coeffs,
            ).simplify(atol=output_threshold, rtol=0)

        return IsingEncoding(
            operator=spo,
            constant=float(constant),
            decode_fn=_make_decode(problem.variable_to_idx, problem.variable_order),
            metadata={"strategy": "native", "term_count": spo.size},
        )


@dataclass(frozen=True)
class QuadratizedIsingConverter(BinaryToIsingConverter):
    """Convert binary polynomials to Ising operators via quadratization to QUBO/BQM.

    The penalty term enforcing ``p == u·v`` for each aux-var substitution
    must dominate the objective for the QUBO's global minimum to coincide
    with the original HUBO's. When :attr:`strength` is ``None`` the
    converter picks ``strength_multiplier * max(|hubo coefficient|)``
    (with a floor of ``strength_multiplier``).
    """

    strength: float | None = None
    """Explicit penalty strength. ``None`` triggers adaptive sizing from the
    input HUBO's coefficient magnitudes."""

    strength_multiplier: float = 2.0
    """Multiplier applied to ``max(|hubo coefficient|)`` when
    :attr:`strength` is ``None``. Values ≥ 2 keep the penalty strictly
    larger than the objective."""

    def _resolve_strength(self, problem: BinaryPolynomialProblem) -> float:
        if self.strength is not None:
            return float(self.strength)
        max_coeff = _max_abs_input_coeff(problem)
        if max_coeff == 0.0:
            # Degenerate constant-only problem — any positive strength works.
            return float(self.strength_multiplier)
        return float(self.strength_multiplier) * max_coeff

    def convert(self, problem: BinaryPolynomialProblem) -> IsingEncoding:
        if problem.n_vars == 0:
            return IsingEncoding(
                operator=_empty_spo(0),
                constant=problem.constant,
                decode_fn=_empty_decode,
                metadata={"strategy": "quadratized", "ancilla_count": 0},
            )

        strength = self._resolve_strength(problem)
        bqm = dimod.make_quadratic(problem.polynomial, strength, dimod.BINARY)
        qubo_terms, offset = bqm.to_qubo()

        variable_order = tuple(sorted(bqm.variables, key=repr))
        var_to_idx = {var: idx for idx, var in enumerate(variable_order)}
        qubo_matrix = np.zeros((len(variable_order), len(variable_order)), dtype=float)

        for (u, v), coeff in qubo_terms.items():
            i, j = var_to_idx[u], var_to_idx[v]
            qubo_matrix[i, j] += float(coeff)

        # Quadratization output may be asymmetric; suppress the user-facing
        # warning since the asymmetry isn't user-driven here.
        spo, constant = _convert_qubo_matrix_to_ising_spo(
            qubo_matrix, warn_on_asymmetric=False
        )
        original_vars = set(problem.variable_order)
        ancilla_variables = [var for var in variable_order if var not in original_vars]

        return IsingEncoding(
            operator=spo,
            constant=float(constant + offset),
            decode_fn=_make_decode(var_to_idx, problem.variable_order),
            metadata={
                "strategy": "quadratized",
                "strength": strength,
                "ancilla_count": len(ancilla_variables),
                "ancilla_variables": ancilla_variables,
            },
        )


def _resolve_ising_converter(
    hamiltonian_builder: Literal["native", "quadratized"],
    *,
    quadratization_strength: float | None,
) -> BinaryToIsingConverter:
    """Resolve a converter selector string."""
    if hamiltonian_builder == "native":
        return NativeIsingConverter()
    if hamiltonian_builder == "quadratized":
        return QuadratizedIsingConverter(strength=quadratization_strength)
    raise ValueError("hamiltonian_builder must be either 'native' or 'quadratized'.")


@dataclass(eq=False)
class IsingResult:
    """Result of converting a QUBO/HUBO to a cleaned Ising Hamiltonian."""

    cost_hamiltonian: SparsePauliOp
    loss_constant: float
    n_qubits: int
    encoding: IsingEncoding

    @property
    def wires(self) -> tuple:
        """Canonical wire mapping aligned with the SPO."""
        return _wires_from_spo(self.cost_hamiltonian)


def qubo_to_ising(
    qubo,
    *,
    hamiltonian_builder: Literal["native", "quadratized"] = "native",
    quadratization_strength: float | None = None,
) -> IsingResult:
    """Convert a QUBO/HUBO to a cleaned Ising Hamiltonian.

    Normalizes the input, converts via the selected Ising converter,
    cleans constant terms, and validates the result.

    Args:
        qubo: QUBO dict, HUBO dict, numpy matrix, BQM, or BinaryPolynomial.
        hamiltonian_builder: ``"native"`` or ``"quadratized"``.
        quadratization_strength: Penalty for quadratization. ``None``
            (default) picks an adaptive strength ``2 * max(|hubo coeff|)``
            so the penalty dominates the objective for arbitrary problem
            scales — see :class:`QuadratizedIsingConverter`. Ignored when
            ``hamiltonian_builder="native"``.

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

    cleaned_spo, ham_constant = _clean_hamiltonian_spo(encoding.operator)
    if cleaned_spo.size == 0:
        raise ValueError("Hamiltonian contains only constant terms.")

    return IsingResult(
        cost_hamiltonian=cleaned_spo,
        loss_constant=encoding.constant + ham_constant,
        n_qubits=_n_qubits(encoding.operator),
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
    *,
    warn_on_asymmetric: bool = True,
) -> tuple[SparsePauliOp, float]:
    """Convert a QUBO matrix to an Ising Hamiltonian as a ``SparsePauliOp``.

    The mapping ``x_i = (1 - σ_i)/2`` rewrites ``min x^T Q x`` as a
    Pauli-Z Ising minimisation.

    Args:
        qubo_matrix: Square QUBO matrix Q (dense or scipy.sparse). Symmetrised
            internally regardless of input shape.
        warn_on_asymmetric: When ``True`` (default), emit a ``UserWarning`` if
            the input is neither symmetric nor upper-triangular. Internal
            callers producing known-asymmetric matrices should pass ``False``.

    Returns:
        ``(spo, constant_offset)`` — Pauli-Z-only Ising Hamiltonian and the
        energy offset to add back when scoring.
    """
    if warn_on_asymmetric and not _is_sanitized(qubo_matrix):
        warn(
            "The QUBO matrix is neither symmetric nor upper triangular."
            " Symmetrizing it for the Ising Hamiltonian creation."
        )

    if sps.issparse(qubo_matrix):
        sparse_m = sps.csr_matrix(qubo_matrix)
        symmetrized_qubo = (sparse_m + sparse_m.T) / 2
        coo_mat = sps.triu(symmetrized_qubo).tocoo()
        rows = np.asarray(coo_mat.row, dtype=np.int64)
        cols = np.asarray(coo_mat.col, dtype=np.int64)
        values = np.asarray(coo_mat.data, dtype=np.float64)
    else:
        dense_m = np.asarray(qubo_matrix)
        symmetrized_qubo = (dense_m + dense_m.T) / 2
        triu_matrix = np.triu(symmetrized_qubo)
        rows, cols = (a.astype(np.int64) for a in triu_matrix.nonzero())
        values = triu_matrix[rows, cols].astype(np.float64)

    n = qubo_matrix.shape[0]
    linear_terms = np.zeros(n, dtype=np.float64)

    # ``np.add.at`` is required (not fancy indexing) — a qubit index can
    # appear in many off-diagonal pairs and the accumulations must combine.
    diag_mask = rows == cols
    diag_w = values[diag_mask]
    np.add.at(linear_terms, rows[diag_mask], -diag_w / 2)
    constant_term = float(np.sum(diag_w) / 2)

    off_rows = rows[~diag_mask]
    off_cols = cols[~diag_mask]
    # x^T Q x for symmetric Q counts both (i,j) and (j,i), so the triu
    # entry contributes half the total interaction.
    off_half_w = values[~diag_mask] / 2
    np.add.at(linear_terms, off_rows, -off_half_w)
    np.add.at(linear_terms, off_cols, -off_half_w)
    constant_term += float(np.sum(off_half_w))

    # Threshold both the linear-term filter and the final ``simplify`` call
    # relative to the input scale.
    max_input_value = float(np.max(np.abs(values))) if values.size else 0.0
    threshold = 1e-12 * max_input_value

    nonzero_linear = np.flatnonzero(np.abs(linear_terms) > threshold)
    n_2body = off_rows.size
    n_1body = nonzero_linear.size
    n_terms = n_2body + n_1body
    if n_terms == 0:
        return _empty_spo(n), constant_term

    # Z-only Hamiltonian: x_arr stays all-False.
    z_arr = np.zeros((n_terms, n), dtype=bool)
    x_arr = np.zeros((n_terms, n), dtype=bool)
    coeffs = np.empty(n_terms, dtype=complex)

    if n_2body:
        idx = np.arange(n_2body)
        z_arr[idx, off_rows] = True
        z_arr[idx, off_cols] = True
        coeffs[:n_2body] = off_half_w
    if n_1body:
        idx_1 = np.arange(n_1body) + n_2body
        z_arr[idx_1, nonzero_linear] = True
        coeffs[n_2body:] = linear_terms[nonzero_linear]

    spo = SparsePauliOp(
        PauliList.from_symplectic(z_arr, x_arr),
        coeffs=coeffs,
    ).simplify(atol=threshold, rtol=0)
    return spo, constant_term
