# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Cost-function domain logic: PennyLane Hamiltonian ops, Trotterization,
binary-polynomial normalization + evaluation, and QUBO/HUBO-to-Ising
conversion.
"""

from ._types import (
    BinaryPolynomialProblem,
    CompiledBinaryPolynomial,
    HUBOProblemTypes,
    HUBOTerm,
    QUBOProblemTypes,
)
from ._polynomial import (
    compile_problem,
    hubo_to_binary_polynomial,
    normalize_binary_polynomial_problem,
    qubo_to_binary_polynomial,
    qubo_to_matrix,
)
from ._ising import (
    BinaryToIsingConverter,
    IsingEncoding,
    IsingResult,
    NativeIsingConverter,
    QuadratizedIsingConverter,
    qubo_to_ising,
)
from ._trotterization import (
    ExactTrotterization,
    QDrift,
    TrotterizationStrategy,
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
    "hubo_to_binary_polynomial",
    "normalize_binary_polynomial_problem",
    "qubo_to_binary_polynomial",
    "qubo_to_ising",
    "qubo_to_matrix",
]
