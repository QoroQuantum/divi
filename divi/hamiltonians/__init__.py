# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Cost-function domain logic: PennyLane Hamiltonian ops, Trotterization,
binary-polynomial normalization + evaluation, and QUBO/HUBO-to-Ising
conversion.
"""

from ._ising import (
    BinaryToIsingConverter,
    IsingEncoding,
    IsingResult,
    NativeIsingConverter,
    QuadratizedIsingConverter,
    qubo_to_ising,
    qubo_to_spo,
)
from ._mixers import (
    bit_driver,
    bit_flip_mixer,
    edge_driver,
    x_mixer,
    xy_mixer,
)
from ._polynomial import (
    compile_problem,
    hubo_to_binary_polynomial,
    normalize_binary_polynomial_problem,
    qubo_to_binary_polynomial,
    qubo_to_matrix,
)
from ._term_ops import to_spo
from ._trotterization import (
    ExactTrotterization,
    QDrift,
    TrotterizationResult,
    TrotterizationStrategy,
)
from ._types import (
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
    "TrotterizationResult",
    "bit_driver",
    "to_spo",
    "bit_flip_mixer",
    "compile_problem",
    "edge_driver",
    "hubo_to_binary_polynomial",
    "normalize_binary_polynomial_problem",
    "qubo_to_binary_polynomial",
    "qubo_to_ising",
    "qubo_to_matrix",
    "qubo_to_spo",
    "x_mixer",
    "xy_mixer",
]
