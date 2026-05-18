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
)
from ._mixers import (
    bit_driver_spo,
    bit_flip_mixer_spo,
    edge_driver_spo,
    x_mixer_spo,
    xy_mixer_spo,
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
    "bit_driver_spo",
    "to_spo",
    "bit_flip_mixer_spo",
    "compile_problem",
    "edge_driver_spo",
    "hubo_to_binary_polynomial",
    "normalize_binary_polynomial_problem",
    "qubo_to_binary_polynomial",
    "qubo_to_ising",
    "qubo_to_matrix",
    "x_mixer_spo",
    "xy_mixer_spo",
]
