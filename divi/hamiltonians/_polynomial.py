# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary polynomial types, normalization, and JIT-compiled evaluation kernels.

Handles HUBO/QUBO inputs, normalizes them to :class:`BinaryPolynomialProblem`,
and provides compiled evaluators used by both ``divi.qprog.algorithms._pce``
and the ``PCECostStage`` pipeline stage.
"""

from collections.abc import Hashable
from typing import Any

import dimod
import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

from divi.hamiltonians._types import (
    BinaryPolynomialProblem,
    CompiledBinaryPolynomial,
    HUBOProblemTypes,
    HUBOTerm,
    QUBOProblemTypes,
)


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
            matrix[i, j] = matrix[j, i] = coeff / 2
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


def _default_variable_order(variables: set[Hashable]) -> tuple[Hashable, ...]:
    """Build deterministic variable order for mixed, potentially incomparable labels.

    Sorts integers numerically (0, 1, 2, ..., 10, 11) rather than
    lexicographically ("0", "1", "10", "11", ..., "2").  Non-integer
    labels fall back to ``repr``-based ordering.
    """
    if all(isinstance(v, int) for v in variables):
        return tuple(sorted(variables))
    return tuple(sorted(variables, key=repr))


def _normalize_hubo_term_key(term: Any) -> HUBOTerm:
    """Validate and normalize a HUBO term key to a tuple."""
    if isinstance(term, tuple):
        term_tuple = term
    elif isinstance(term, frozenset):
        term_tuple = tuple(term)
    else:
        raise ValueError(
            "HUBO term keys must be tuples (or frozensets from BinaryPolynomial internals), "
            f"got {type(term)}"
        )

    if len(set(term_tuple)) != len(term_tuple):
        raise ValueError(
            f"Invalid HUBO term {term_tuple}: duplicate variables in a monomial are not allowed."
        )

    for variable in term_tuple:
        if not isinstance(variable, Hashable):
            raise ValueError(f"HUBO variable must be hashable, got {type(variable)}")

    return term_tuple


def hubo_to_binary_polynomial(hubo: HUBOProblemTypes) -> dimod.BinaryPolynomial:
    """Convert HUBO input to a dimod BinaryPolynomial with BINARY vartype."""
    if isinstance(hubo, dimod.BinaryPolynomial):
        if hubo.vartype != dimod.Vartype.BINARY:
            raise ValueError(
                f"BinaryPolynomial must have vartype='BINARY', got {hubo.vartype}"
            )
        return hubo

    if isinstance(hubo, dict):
        poly_terms: dict[frozenset[Hashable], float] = {}
        for raw_term, coeff in hubo.items():
            term = _normalize_hubo_term_key(raw_term)
            key = frozenset(term)
            poly_terms[key] = poly_terms.get(key, 0.0) + float(coeff)
        return dimod.BinaryPolynomial(poly_terms, dimod.Vartype.BINARY)

    raise ValueError(f"Unsupported HUBO type: {type(hubo)}")


def qubo_to_binary_polynomial(qubo: QUBOProblemTypes) -> dimod.BinaryPolynomial:
    """Convert supported QUBO inputs to a binary polynomial."""
    if isinstance(qubo, dimod.BinaryQuadraticModel):
        if qubo.vartype != dimod.Vartype.BINARY:
            raise ValueError(
                f"BinaryQuadraticModel must have vartype='BINARY', got {qubo.vartype}"
            )
        poly_terms: dict[frozenset[Hashable], float] = {
            frozenset({var}): float(coeff)
            for var, coeff in qubo.linear.items()
            if coeff != 0
        }
        for (u, v), coeff in qubo.quadratic.items():
            key = frozenset({u, v})
            poly_terms[key] = poly_terms.get(key, 0.0) + float(coeff)
        if qubo.offset != 0:
            poly_terms[frozenset()] = float(qubo.offset)
        return dimod.BinaryPolynomial(poly_terms, dimod.Vartype.BINARY)

    matrix = qubo_to_matrix(qubo)
    if sps.isspmatrix(matrix):
        coo_matrix = matrix.tocoo()
        rows, cols, values = coo_matrix.row, coo_matrix.col, coo_matrix.data
        n_vars = matrix.shape[0]
    else:
        rows, cols = matrix.nonzero()
        values = matrix[rows, cols]
        n_vars = matrix.shape[0]

    poly_terms: dict[frozenset[Hashable], float] = {}
    for i, j, coeff in zip(rows, cols, values):
        if coeff == 0:
            continue
        key = frozenset({int(i)}) if i == j else frozenset({int(i), int(j)})
        poly_terms[key] = poly_terms.get(key, 0.0) + float(coeff)

    for idx in range(n_vars):
        poly_terms.setdefault(frozenset({idx}), 0.0)

    return dimod.BinaryPolynomial(poly_terms, dimod.Vartype.BINARY)


def normalize_binary_polynomial_problem(
    problem: QUBOProblemTypes | HUBOProblemTypes,
    *,
    variable_order: tuple[Hashable, ...] | list[Hashable] | None = None,
) -> BinaryPolynomialProblem:
    """Normalize QUBO/HUBO input into canonical binary-polynomial representation."""
    if isinstance(problem, QUBOProblemTypes):
        polynomial = qubo_to_binary_polynomial(problem)
    else:
        polynomial = hubo_to_binary_polynomial(problem)

    variables = set(polynomial.variables)
    if variable_order is None:
        resolved_order = _default_variable_order(variables)
    else:
        resolved_order = tuple(variable_order)
        if len(set(resolved_order)) != len(resolved_order):
            raise ValueError("variable_order must not contain duplicates.")
        if set(resolved_order) != variables:
            raise ValueError(
                "variable_order must contain exactly the variables present in the problem."
            )

    variable_to_idx = {var: idx for idx, var in enumerate(resolved_order)}
    constant = float(polynomial.get(frozenset(), 0.0))
    return BinaryPolynomialProblem(
        polynomial=polynomial,
        variable_order=resolved_order,
        variable_to_idx=variable_to_idx,
        constant=constant,
    )


# ---------------------------------------------------------------------------
# JIT-compiled polynomial evaluation kernels
#
# Shared math over ``BinaryPolynomialProblem``.  Consumed by both
# ``divi.qprog.algorithms._pce`` and the ``PCECostStage`` pipeline stage.
# ---------------------------------------------------------------------------


def compile_problem(
    problem: BinaryPolynomialProblem,
) -> CompiledBinaryPolynomial:
    """Pre-extract polynomial terms into Numba-friendly CSR-style arrays.

    Returns:
        Tuple of ``(term_indices, term_offsets, coeffs, constant)`` where:

        - *term_indices* — flat ``int32`` array of variable indices for all terms
        - *term_offsets* — ``int32`` array of length ``n_terms + 1``; term *t*
          uses ``term_indices[term_offsets[t]:term_offsets[t+1]]``
        - *coeffs* — ``float64`` coefficient per non-constant term
        - *constant* — sum of constant (empty-key) terms
    """
    constant = 0.0
    offsets = [0]
    all_indices: list[int] = []
    all_coeffs: list[float] = []

    for term, coeff in problem.terms.items():
        c = float(coeff)
        if c == 0.0:
            continue
        if len(term) == 0:
            constant += c
            continue
        indices = [problem.variable_to_idx[var] for var in term]
        all_indices.extend(indices)
        offsets.append(len(all_indices))
        all_coeffs.append(c)

    term_indices = (
        np.array(all_indices, dtype=np.int32)
        if all_indices
        else np.empty(0, dtype=np.int32)
    )
    term_offsets = np.array(offsets, dtype=np.int32)
    coeffs = (
        np.array(all_coeffs, dtype=np.float64)
        if all_coeffs
        else np.empty(0, dtype=np.float64)
    )
    return term_indices, term_offsets, coeffs, constant


@numba.njit(cache=True)
def _eval_poly_1d_jit(
    x_vals: npt.NDArray[np.float64],
    term_indices: npt.NDArray[np.int32],
    term_offsets: npt.NDArray[np.int32],
    coeffs: npt.NDArray[np.float64],
    constant: float,
) -> float:
    """Evaluate binary polynomial for a single variable assignment (1D)."""
    n_terms = len(coeffs)
    energy = constant
    for t in range(n_terms):
        start = term_offsets[t]
        end = term_offsets[t + 1]
        degree = end - start
        c = coeffs[t]
        if degree == 1:
            energy += c * x_vals[term_indices[start]] ** 2
        else:
            prod = 1.0
            for k in range(start, end):
                prod *= x_vals[term_indices[k]]
            energy += c * prod
    return energy


@numba.njit(cache=True, parallel=True)
def _eval_poly_2d_jit(
    x_vals: npt.NDArray[np.float64],
    term_indices: npt.NDArray[np.int32],
    term_offsets: npt.NDArray[np.int32],
    coeffs: npt.NDArray[np.float64],
    constant: float,
) -> npt.NDArray[np.float64]:
    """Evaluate binary polynomial for batched variable assignments (2D).

    Uses ``prange`` to parallelise over states.  Each thread computes the
    full polynomial for a subset of states independently — no shared writes.
    On single-core or constrained environments, ``prange`` degrades to
    sequential execution with no overhead beyond a thread-pool check.

    SIMD auto-vectorisation alone is insufficient here because the inner
    gather pattern (``x_vals[term_indices[k], s]``) defeats LLVM's
    vectoriser.  ``prange`` provides the needed throughput scaling.

    Args:
        x_vals: Shape ``(n_vars, n_states)``.

    Returns:
        Energy array of shape ``(n_states,)``.
    """
    n_terms = len(coeffs)
    n_states = x_vals.shape[1]
    energies = np.empty(n_states, dtype=np.float64)
    for s in numba.prange(n_states):
        e = constant
        for t in range(n_terms):
            start = term_offsets[t]
            end = term_offsets[t + 1]
            degree = end - start
            c = coeffs[t]
            if degree == 1:
                e += c * x_vals[term_indices[start], s] ** 2
            else:
                prod = 1.0
                for k in range(start, end):
                    prod *= x_vals[term_indices[k], s]
                e += c * prod
        energies[s] = e
    return energies


@numba.njit(cache=True)
def _compute_hard_cvar_energy_jit(
    x_vals: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    total_shots: float,
    alpha_cvar: float,
    term_indices: npt.NDArray[np.int32],
    term_offsets: npt.NDArray[np.int32],
    coeffs: npt.NDArray[np.float64],
    constant: float,
) -> float:
    """Fused CVaR energy: evaluate polynomial + sort + accumulate in one kernel.

    Equivalent to calling ``_eval_poly_2d_jit`` followed by the CVaR
    partial-sort accumulation, but avoids materialising intermediate NumPy
    arrays for ``argsort``, ``cumsum``, and ``searchsorted``.

    Args:
        x_vals: Shape ``(n_vars, n_states)`` — binary variable assignments.
        counts: Shape ``(n_states,)`` — shot counts per state.
        total_shots: Sum of *counts*.
        alpha_cvar: CVaR tail fraction.
        term_indices, term_offsets, coeffs, constant: compiled problem arrays.
    """
    len(coeffs)
    n_states = x_vals.shape[1]

    # Note: _eval_poly_2d_jit uses ``parallel=True`` / ``prange``, so this
    # call crosses an implicit threading boundary.  Numba's thread pool is
    # reused (not re-spawned) and degrades to sequential on single-core.
    energies = _eval_poly_2d_jit(x_vals, term_indices, term_offsets, coeffs, constant)

    sorted_indices = np.argsort(energies)
    cutoff_count = int(np.ceil(alpha_cvar * total_shots))

    cvar_energy = 0.0
    count_sum = 0.0
    for i in range(n_states):
        idx = sorted_indices[i]
        c = counts[idx]
        if count_sum + c <= cutoff_count:
            cvar_energy += energies[idx] * c
            count_sum += c
        else:
            remaining = cutoff_count - count_sum
            cvar_energy += energies[idx] * remaining
            count_sum += remaining
            break

    return cvar_energy / cutoff_count


def _evaluate_binary_polynomial(
    x_vals: npt.NDArray[np.float64],
    problem: BinaryPolynomialProblem,
    _compiled: CompiledBinaryPolynomial | None = None,
) -> npt.NDArray[np.float64] | float:
    """Evaluate binary polynomial energy for one or many assignments.

    Degree-1 terms are evaluated as ``c * x_i²`` rather than ``c * x_i`` to
    undo the linearization (``x_i² → x_i``) applied during polynomial
    normalization.  This is a no-op for binary values (``x² = x``) but
    produces correct energies for continuous soft-relaxed values.

    Args:
        x_vals: Variable assignments. Shape ``(n_vars,)`` for one assignment
            or ``(n_vars, n_states)`` for many.
        problem: Canonical binary polynomial problem.
        _compiled: Pre-compiled CSR arrays from :func:`compile_problem`.
            When provided the Numba JIT kernel is used instead of the
            Python loop.
    """
    if _compiled is not None:
        term_indices, term_offsets, coeffs, constant = _compiled
        x = np.ascontiguousarray(x_vals, dtype=np.float64)
        if x.ndim == 1:
            return float(
                _eval_poly_1d_jit(x, term_indices, term_offsets, coeffs, constant)
            )
        return _eval_poly_2d_jit(x, term_indices, term_offsets, coeffs, constant)

    is_single = x_vals.ndim == 1
    energy = 0.0 if is_single else np.zeros(x_vals.shape[1], dtype=np.float64)

    for term, coeff in problem.terms.items():
        coeff = float(coeff)
        if coeff == 0:
            continue
        if len(term) == 0:
            energy = energy + coeff
            continue

        indices = [problem.variable_to_idx[var] for var in term]
        if len(term) == 1:
            # De-linearise: evaluate as c * x_i² instead of c * x_i.
            idx = indices[0]
            monomial = x_vals[idx] ** 2 if is_single else x_vals[idx, :] ** 2
        elif is_single:
            monomial = np.prod(x_vals[indices])
        else:
            monomial = np.prod(x_vals[indices, :], axis=0)
        energy = energy + (coeff * monomial)

    return float(energy) if is_single else energy
