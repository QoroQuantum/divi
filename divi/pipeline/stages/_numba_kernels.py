# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Numba JIT-compiled kernels for PCE binary polynomial evaluation.

Provides a compiled (CSR-style) representation of ``BinaryPolynomialProblem``
and JIT-accelerated energy evaluation for both single and batched variable
assignments, plus a fused CVaR energy kernel.
"""

import numba
import numpy as np
import numpy.typing as npt

from divi.typing import BinaryPolynomialProblem


def compile_problem(
    problem: BinaryPolynomialProblem,
) -> tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64], float
]:
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

    # --- Evaluate energies ---
    # Note: _eval_poly_2d_jit uses ``parallel=True`` / ``prange``, so this
    # call crosses an implicit threading boundary.  Numba's thread pool is
    # reused (not re-spawned) and degrades to sequential on single-core.
    energies = _eval_poly_2d_jit(x_vals, term_indices, term_offsets, coeffs, constant)

    # --- CVaR accumulation (fused sort + partial sum) ---
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
