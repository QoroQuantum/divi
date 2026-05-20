# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOAProblem protocol — base class for all QAOA-compatible problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Any

from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import x_mixer
from divi.hamiltonians._term_ops import _require_qiskit_num_qubits
from divi.qprog.algorithms import InitialState, SuperpositionState


class QAOAProblem(ABC):
    """Base class for all QAOA-compatible problems.

    Subclasses must implement the abstract properties that provide
    the ingredients QAOA needs.  Default implementations of
    ``mixer_hamiltonian``, ``recommended_initial_state``, ``is_feasible``,
    ``repair_infeasible_bitstring``, and ``compute_energy`` are provided and
    can be overridden.
    """

    @property
    @abstractmethod
    def cost_hamiltonian(self) -> SparsePauliOp:
        """The cost Hamiltonian encoding the optimization objective."""

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """Mixer Hamiltonian for exploring the solution space.

        Defaults to the standard X mixer over the cost Hamiltonian qubits,
        suitable for unconstrained binary optimization. Override this for
        constrained feasible subspaces or problem-specific mixers.
        """
        return x_mixer(_require_qiskit_num_qubits(self.cost_hamiltonian.num_qubits))

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

        Defaults to :class:`~divi.qprog.algorithms.SuperpositionState` (ground state of the
        standard X mixer).
        """
        return SuperpositionState()

    @property
    def wire_labels(self) -> tuple:
        """Qubit-position-ordered labels for the cost Hamiltonian.

        Defaults to dense ``range(num_qubits)``. Override for problems whose
        users expect domain-level identifiers (e.g. graph node names) at
        ``QAOA._circuit_wires`` and through ``decode_fn``.
        """
        return tuple(
            range(_require_qiskit_num_qubits(self.cost_hamiltonian.num_qubits))
        )

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

        Returns a dict mapping program IDs to sub-problems. Program IDs must
        be hashable and stable because later partitioning hooks receive the
        same IDs when extending candidates into a global solution.

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

        ``prog_id`` is one of the keys returned by :meth:`decompose`.
        Must return a **new** list — do not mutate *current_solution*.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extend_solution()."
        )

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Score a complete global solution for beam search.

        Lower scores are better. Problems with maximization objectives should
        usually return a negated score here, then expose the natural objective
        value from :meth:`postprocess_candidates`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evaluate_global_solution()."
        )

    def postprocess_candidates(
        self, candidates: list[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[Any, float]]:
        """Post-process raw partition-aggregation candidates.

        Args:
            candidates: Beam-search ``(score, solution)`` pairs, where
                ``score`` is the value returned by
                :meth:`evaluate_global_solution` and lower is better.
            strict: If supported by a constrained problem, reject invalid raw
                solutions instead of repairing them. The default implementation
                ignores this flag.

        Returns:
            Problem-specific ``(result, objective_value)`` tuples ready to
            return from partitioned aggregation APIs. The objective value uses
            the problem's public convention, which may differ from the
            beam-search score. Default: ``(solution, score)``.
        """
        return [(solution, score) for score, solution in candidates]
