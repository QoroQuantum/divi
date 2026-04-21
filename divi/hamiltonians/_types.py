# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases and domain dataclasses for the hamiltonians package."""

from collections.abc import Hashable
from dataclasses import dataclass

import dimod
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps

QUBOProblemTypes = list | np.ndarray | sps.spmatrix | dimod.BinaryQuadraticModel
HUBOTerm = tuple[Hashable, ...]
HUBOProblemTypes = dict[HUBOTerm, float] | dimod.BinaryPolynomial

# CSR-style compact form of a BinaryPolynomialProblem, returned by
# ``compile_problem`` and consumed by the JIT evaluators.
CompiledBinaryPolynomial = tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64], float
]


@dataclass(frozen=True)
class BinaryPolynomialProblem:
    """Canonical internal representation for binary polynomial optimization problems."""

    polynomial: dimod.BinaryPolynomial
    variable_order: tuple[Hashable, ...]
    variable_to_idx: dict[Hashable, int]
    constant: float

    @property
    def terms(self) -> dict[HUBOTerm, float]:
        """Terms with tuple keys ordered by variable_order (derived from polynomial)."""
        result: dict[HUBOTerm, float] = {}
        for term, coeff in self.polynomial.items():
            ordered = tuple(sorted(term, key=lambda var: self.variable_to_idx[var]))
            result[ordered] = float(coeff)
        return result

    @property
    def n_vars(self) -> int:
        return len(self.variable_order)
