# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Cost-function domain logic: PennyLane Hamiltonian ops, Trotterization,
binary-polynomial normalization + evaluation, and QUBO/HUBO-to-Ising
conversion.
"""

from divi.hamiltonians._ising import (
    BinaryToIsingConverter,
    IsingEncoding,
    IsingResult,
    NativeIsingConverter,
    QuadratizedIsingConverter,
    _is_sanitized,
    _resolve_ising_converter,
    convert_qubo_matrix_to_pennylane_ising,
    qubo_to_ising,
)
from divi.hamiltonians._polynomial import (
    _compute_hard_cvar_energy_jit,
    _eval_poly_1d_jit,
    _eval_poly_2d_jit,
    _evaluate_binary_polynomial,
    compile_problem,
    hubo_to_binary_polynomial,
    normalize_binary_polynomial_problem,
    qubo_to_binary_polynomial,
    qubo_to_matrix,
)
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian,
    _get_terms_iterable,
    _hamiltonian_term_count,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
    _sort_hamiltonian_terms,
)
from divi.hamiltonians._trotterization import (
    ExactTrotterization,
    QDrift,
    TrotterizationStrategy,
)
from divi.hamiltonians._types import (
    BinaryPolynomialProblem,
    CompiledBinaryPolynomial,
    HUBOProblemTypes,
    HUBOTerm,
    QUBOProblemTypes,
)

__all__ = [
    "BinaryPolynomialProblem",
    "BinaryToIsingConverter",
    "CompiledBinaryPolynomial",
    "ExactTrotterization",
    "HUBOProblemTypes",
    "HUBOTerm",
    "IsingEncoding",
    "IsingResult",
    "NativeIsingConverter",
    "QDrift",
    "QUBOProblemTypes",
    "QuadratizedIsingConverter",
    "TrotterizationStrategy",
    "compile_problem",
    "convert_qubo_matrix_to_pennylane_ising",
    "hubo_to_binary_polynomial",
    "normalize_binary_polynomial_problem",
    "qubo_to_binary_polynomial",
    "qubo_to_ising",
    "qubo_to_matrix",
]
