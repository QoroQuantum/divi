# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph problem classes for QAOA."""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as pqaoa

from divi.hamiltonians import (
    _clean_hamiltonian,
    _get_terms_iterable,
    _is_empty_hamiltonian,
)
from divi.qprog.algorithms._initial_state import (
    InitialState,
    OnesState,
    SuperpositionState,
    ZerosState,
)
from divi.qprog.problems._base import QAOAProblem
from divi.qprog.problems._graph_partitioning_utils import (
    GraphPartitioningConfig,
    _edge_partition_graph,
    _node_partition_graph,
)
from divi.typing import GraphProblemTypes


class _GraphProblemBase(QAOAProblem):
    """Shared logic for PennyLane-backed graph problems.

    Subclasses only need to set ``_pl_func_name`` and the two
    ``_*_state_cls`` class attributes, then call ``super().__init__``.
    """

    _pl_func_name: str
    _constrained_state_cls: type[InitialState]
    _unconstrained_state_cls: type[InitialState]

    def __init__(
        self,
        graph: GraphProblemTypes,
        *,
        is_constrained: bool = True,
        config: GraphPartitioningConfig | None = None,
    ):
        self._graph = graph
        self._is_constrained = is_constrained

        cost_ham, self._mixer_hamiltonian, *self._metadata = self._resolve(
            graph, is_constrained
        )

        cleaned, ham_constant = _clean_hamiltonian(cost_ham)
        if _is_empty_hamiltonian(cleaned):
            raise ValueError("Hamiltonian contains only constant terms.")

        self._cost_hamiltonian = cleaned
        self._loss_constant = ham_constant
        self._wire_labels = tuple(cleaned.wires)
        self._initial_state = (
            self._constrained_state_cls
            if is_constrained
            else self._unconstrained_state_cls
        )()
        self._config = config
        self._reverse_index_maps = {}

    @classmethod
    def _resolve(cls, graph, is_constrained):
        """Call the PennyLane QAOA function for this problem type."""
        pl_fn = getattr(pqaoa, cls._pl_func_name)
        try:
            return pl_fn(graph, constrained=is_constrained)
        except TypeError:
            return pl_fn(graph)

    @property
    def graph(self) -> GraphProblemTypes:
        """The underlying graph."""
        return self._graph

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._loss_constant

    @property
    def recommended_initial_state(self) -> InitialState:
        return self._initial_state

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        wires = self._wire_labels

        def _decode(bitstring: str) -> list:
            return [
                wires[idx]
                for idx, bit in enumerate(bitstring)
                if bit == "1" and idx < len(wires)
            ]

        return _decode

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata[0] if self._metadata else {}

    def decompose(self) -> dict[tuple[str, int], QAOAProblem]:
        if self._config is None:
            raise ValueError(
                "Cannot decompose: no config was provided at construction."
            )

        # Warn if this problem type has known partitioning risks
        tier = _PARTITIONING_COMPATIBILITY_TIERS.get(type(self))
        if tier is not None:
            risk_level, rationale = tier
            prefix = "High-risk" if risk_level == "high-risk" else "Heuristic-risk"
            detail = (
                "Aggregation is heuristic and may miss globally valid/high-quality "
                f"solutions because {rationale}"
                if risk_level == "high-risk"
                else "Results may be sensitive to partition boundaries because "
                f"{rationale}"
            )
            warn(
                f"{prefix} graph partitioning objective: "
                f"{type(self).__name__}. {detail}",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(self, EdgePartitioningProblem):
            subgraphs = _edge_partition_graph(
                self.graph,
                n_max_nodes_per_cluster=self._config.max_n_nodes_per_cluster,
            )
            subgraphs = [sg for sg in subgraphs if sg.size() > 0]
        else:
            subgraphs = _node_partition_graph(
                self.graph,
                partitioning_config=self._config,
            )

        self._reverse_index_maps = {}
        sub_problems = {}

        for i, subgraph in enumerate(subgraphs):
            prog_id = (string.ascii_uppercase[i], subgraph.number_of_nodes())

            index_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
            self._reverse_index_maps[prog_id] = {v: k for k, v in index_map.items()}

            relabeled = nx.relabel_nodes(subgraph, index_map)
            sub_problems[prog_id] = type(self)(
                relabeled, is_constrained=self._is_constrained
            )

        return sub_problems

    def initial_solution_size(self) -> int:
        return self.graph.number_of_nodes()

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: tuple[str, int],
        candidate_decoded: list[int],
    ) -> list[int]:
        extended = list(current_solution)
        reverse_map = self._reverse_index_maps[prog_id]

        # Reset all positions belonging to this partition to 0
        for global_idx in reverse_map.values():
            extended[global_idx] = 0

        # Set positions for nodes in the candidate's decoded solution to 1
        for local_node in candidate_decoded:
            global_idx = reverse_map[local_node]
            extended[global_idx] = 1

        return extended

    def evaluate_global_solution(self, solution: list[int]) -> float:
        hamiltonian = self.cost_hamiltonian
        wire_to_bit = {w: solution[w] for w in hamiltonian.wires}

        energy = self.loss_constant
        for term in _get_terms_iterable(hamiltonian):
            coeff = 1.0
            base_op = term

            if isinstance(term, qml.ops.SProd):
                coeff = float(term.scalar)
                base_op = term.base

            eigenvalue = 1.0
            for wire in base_op.wires:
                eigenvalue *= 1 - 2 * wire_to_bit[wire]

            energy += coeff * eigenvalue

        return energy

    def finalize_solution(
        self, score: float, solution: list[int]
    ) -> tuple[list[int], float]:
        return list(np.where(solution)[0]), score

    def format_top_solutions(
        self, results: list[tuple[float, list[int]]]
    ) -> list[tuple[list[int], float]]:
        return [(list(np.where(solution)[0]), score) for score, solution in results]


class MaxCutProblem(_GraphProblemBase):
    """MaxCut problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
    """

    _pl_func_name = "maxcut"
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState


class MaxCliqueProblem(_GraphProblemBase):
    """Max clique problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_clique"
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MaxIndependentSetProblem(_GraphProblemBase):
    """Max independent set problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_independent_set"
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MinVertexCoverProblem(_GraphProblemBase):
    """Min vertex cover problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "min_vertex_cover"
    _constrained_state_cls = OnesState
    _unconstrained_state_cls = SuperpositionState


class MaxWeightCycleProblem(_GraphProblemBase):
    """Max weight cycle problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _pl_func_name = "max_weight_cycle"
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState


class EdgePartitioningProblem(_GraphProblemBase):
    """Placeholder for edge-partitioning problems.

    Edge partitioning operates on directed graphs and uses weak connectivity.
    The Hamiltonian construction is not yet implemented — this class exists
    so that :class:`PartitioningProgramEnsemble` can detect the partitioning mode
    from the Problem type.

    .. note:: This is incomplete.  Passing an ``EdgePartitioningProblem``
       to :class:`QAOA` directly will raise ``AttributeError`` because
       ``_pl_func_name`` is not set.
    """

    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState

    #: Signals that the graph is directed and needs weak-connectivity checks.
    is_edge_problem: bool = True

    def __init__(self, graph: GraphProblemTypes):
        # Skip _GraphProblemBase.__init__ — no PennyLane function to call.
        self._graph = graph
        self._is_constrained = True

    @property
    def cost_hamiltonian(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def mixer_hamiltonian(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def loss_constant(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support Hamiltonian construction."
        )

    @property
    def decode_fn(self):
        raise NotImplementedError(
            "EdgePartitioningProblem does not yet support decoding."
        )


# Partitioning is most robust for cut-style objectives (e.g. MaxCut).
# Structure-dependent objectives may lose cross-partition constraints.
_PARTITIONING_COMPATIBILITY_TIERS = {
    MaxWeightCycleProblem: (
        "high-risk",
        "partitioning can break cycles across cluster boundaries.",
    ),
    MaxCliqueProblem: (
        "heuristic-risk",
        "partitioning can hide cross-partition adjacency needed for global cliques.",
    ),
    MaxIndependentSetProblem: (
        "heuristic-risk",
        "partitioning can hide cross-partition conflicts between selected vertices.",
    ),
    MinVertexCoverProblem: (
        "heuristic-risk",
        "partitioning can hide cross-partition edges that must be covered globally.",
    ),
}


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
