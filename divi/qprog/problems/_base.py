# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOAProblem protocol — base class for all QAOA-compatible problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Any

import pennylane as qml

from divi.qprog.algorithms._initial_state import InitialState, SuperpositionState


class QAOAProblem(ABC):
    """Base class for all QAOA-compatible problems.

    Subclasses must implement the four abstract properties that provide
    the ingredients QAOA needs.  Default implementations of
    ``recommended_initial_state``, ``is_feasible``, ``repair``, and
    ``compute_energy`` are provided and can be overridden.
    """

    @property
    @abstractmethod
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian encoding the optimisation objective."""

    @property
    @abstractmethod
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        """The mixer Hamiltonian for exploring the solution space."""

    @property
    @abstractmethod
    def loss_constant(self) -> float:
        """Constant offset added back to the expectation value."""

    @property
    @abstractmethod
    def decode_fn(self) -> Callable[[str], Any]:
        """Map a measurement bitstring to a domain-level solution.

        Bitstrings use **left-to-right** qubit ordering: the character at
        index *i* corresponds to qubit *i* of the cost Hamiltonian.
        """

    @property
    def recommended_initial_state(self) -> InitialState:
        """Recommended initial quantum state for this problem.

        Defaults to :class:`SuperpositionState` (ground state of the
        standard X mixer).
        """
        return SuperpositionState()

    def is_feasible(self, bitstring: str) -> bool:
        """Check whether a bitstring represents a feasible solution.

        Defaults to ``True`` (unconstrained).
        """
        return True

    def repair_infeasible_bitstring(
        self, bitstring: str
    ) -> tuple[str, Any, float | None]:
        """Repair an infeasible bitstring into a feasible one.

        Returns:
            A three-element tuple ``(repaired_bitstring, decoded, energy)``:

            - **repaired_bitstring**: The feasible bitstring after repair.
            - **decoded**: Domain-level solution (e.g. tour list, routes),
              or ``None`` if unavailable.
            - **energy**: Objective value of the repaired solution,
              or ``None`` if unknown.

        The default implementation returns the bitstring unchanged.
        """
        return bitstring, None, None

    def compute_energy(self, bitstring: str) -> float | None:
        """Evaluate the objective energy for a bitstring.

        Defaults to ``None`` (unknown).
        """
        return None

    # ------------------------------------------------------------------
    # Decomposition hooks (override to enable partitioned workflows)
    # ------------------------------------------------------------------

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        """Decompose this problem into sub-problems for partitioned solving.

        Returns a dict mapping program IDs (any hashable) to sub-Problems.
        The decomposition strategy should be configured at construction time.

        Raises:
            NotImplementedError: If the subclass does not support decomposition.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support decomposition. "
            "Override decompose() to enable partitioning workflows."
        )

    def initial_solution_size(self) -> int:
        """Size of the global solution vector (e.g. number of graph nodes)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement initial_solution_size()."
        )

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: Hashable,
        candidate_decoded: list[int],
    ) -> list[int]:
        """Map a sub-solution's decoded bits into the global solution vector.

        Must return a **new** list — do not mutate *current_solution*.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extend_solution()."
        )

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Score a complete global solution. Lower is better."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evaluate_global_solution()."
        )

    def finalize_solution(self, score: float, solution: list[int]) -> tuple[Any, float]:
        """Post-process the best beam search result.

        Returns:
            A ``(result, energy)`` tuple.  Default: ``(solution, score)``.
        """
        return solution, score

    def format_top_solutions(
        self, results: list[tuple[float, list[int]]]
    ) -> list[tuple[Any, float]]:
        """Format beam search output for get_top_solutions().

        Returns:
            A list of ``(result, energy)`` tuples.  Default: returns as-is.
        """
        return results
