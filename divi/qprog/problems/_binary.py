# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Binary optimization (QUBO / HUBO) problem class for QAOA."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pennylane as qml
import pennylane.qaoa as pqaoa

from divi.hamiltonians import normalize_binary_polynomial_problem, qubo_to_ising
from divi.qprog.problems._base import QAOAProblem
from divi.typing import HUBOProblemTypes, QUBOProblemTypes


class BinaryOptimizationProblem(QAOAProblem):
    """Generic QUBO or HUBO problem.

    Normalises the input, converts to an Ising Hamiltonian, and provides
    a standard X-mixer with equal superposition initial state.

    Args:
        problem: QUBO matrix, BinaryQuadraticModel, HUBO dict, or
            BinaryPolynomial.
        hamiltonian_builder: Ising conversion backend (``"native"`` or
            ``"quadratized"``).
        quadratization_strength: Penalty strength for quadratization.
    """

    def __init__(
        self,
        problem: QUBOProblemTypes | HUBOProblemTypes,
        *,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        quadratization_strength: float = 10.0,
    ):
        self._canonical_problem = normalize_binary_polynomial_problem(problem)
        self._ising = qubo_to_ising(
            problem,
            hamiltonian_builder=hamiltonian_builder,
            quadratization_strength=quadratization_strength,
        )
        self._mixer_hamiltonian = pqaoa.x_mixer(range(self._ising.n_qubits))

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._ising.loss_constant

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        base_decode = self._ising.encoding.decode_fn
        vo = self._canonical_problem.variable_order

        if vo != tuple(range(self._canonical_problem.n_vars)):

            def _decode_with_names(bitstring: str) -> dict | None:
                decoded = base_decode(bitstring)
                if decoded is None:
                    return None
                return dict(zip(vo, decoded))

            return _decode_with_names
        return base_decode

    @property
    def metadata(self) -> dict[str, Any]:
        return self._ising.encoding.metadata or {}

    @property
    def canonical_problem(self):
        """The normalised ``BinaryPolynomialProblem``."""
        return self._canonical_problem
