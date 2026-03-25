# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOAProblem protocol — base class for all QAOA-compatible problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
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
        """Map a measurement bitstring to a domain-level solution."""

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


def draw_graph_solution_nodes(main_graph: nx.Graph, partition_nodes):
    """Visualize a graph with solution nodes highlighted.

    Draws the graph with nodes colored to distinguish solution nodes (red) from
    other nodes (light blue).

    Args:
        main_graph (nx.Graph): NetworkX graph to visualize.
        partition_nodes: Collection of node indices that are part of the solution.
    """
    node_colors = [
        "red" if node in partition_nodes else "lightblue" for node in main_graph.nodes()
    ]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(main_graph)
    nx.draw_networkx_nodes(main_graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(main_graph, pos)
    nx.draw_networkx_labels(main_graph, pos, font_size=10, font_weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
