# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph problem classes for QAOA."""

from collections.abc import Callable, Hashable
from functools import cached_property
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import _clean_hamiltonian_spo
from divi.qprog import GraphProblemTypes
from divi.qprog.algorithms import (
    InitialState,
    OnesState,
    SuperpositionState,
    ZerosState,
)
from divi.qprog.problems import GraphPartitioningConfig, QAOAProblem
from divi.qprog.problems._graph_hamiltonians import (
    max_clique_hamiltonians,
    max_independent_set_hamiltonians,
    max_weight_cycle_hamiltonians,
    maxcut_hamiltonians,
    min_vertex_cover_hamiltonians,
)
from divi.qprog.problems._graph_partitioning_utils import _node_partition_graph


class _GraphProblemBase(QAOAProblem):
    """Shared logic for graph problems built directly from ``SparsePauliOp``.

    Subclasses set ``_resolver`` (a function returning ``(cost_spo, mixer_spo)``
    or ``(cost_spo, mixer_spo, metadata)``) and the two ``_*_state_cls`` class
    attributes, then call ``super().__init__``.
    """

    _resolver: staticmethod
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

        cost_spo, self._mixer_hamiltonian, *self._metadata = self._resolve(
            graph, is_constrained
        )

        cleaned, ham_constant = _clean_hamiltonian_spo(cost_spo, raise_on_constant=True)

        self._cost_hamiltonian = cleaned
        self._loss_constant = ham_constant
        self._wire_labels = self._compute_wire_labels(graph)
        self._initial_state = (
            self._constrained_state_cls
            if is_constrained
            else self._unconstrained_state_cls
        )()
        self._config = config
        self._reverse_index_maps = {}

    @classmethod
    def _resolve(cls, graph, is_constrained):
        """Build cost/mixer SPOs for this problem type."""
        try:
            return cls._resolver(graph, constrained=is_constrained)
        except TypeError:
            return cls._resolver(graph)

    @staticmethod
    def _compute_wire_labels(graph: GraphProblemTypes) -> tuple:
        """Map qubit positions back to original node values in node-iteration order."""
        if isinstance(graph, nx.Graph):
            return tuple(graph.nodes())
        # rustworkx graph: edge_list() / node values; mirror the relabeling done
        # inside the SPO builders.
        return tuple(graph.nodes())

    @property
    def graph(self) -> GraphProblemTypes:
        """The underlying graph.

        Treat as read-only: the cost Hamiltonian is fixed at construction, and
        ``evaluate_global_solution`` caches the Pauli term list on first call.
        Mutating this graph (adding nodes/edges, changing weights) regenerates
        neither, so scores would go stale. Build a new problem instead.
        """
        return self._graph

    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        return self._cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        return self._mixer_hamiltonian

    @property
    def wire_labels(self) -> tuple:
        return self._wire_labels

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

    def decompose(self) -> dict[Hashable, QAOAProblem]:
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

        subgraphs = _node_partition_graph(
            self.graph,
            partitioning_config=self._config,
        )

        self._reverse_index_maps = {}
        sub_problems: dict[Hashable, QAOAProblem] = {}

        for i, (subgraph, cluster_ids) in enumerate(subgraphs):
            prog_id = (f"P{i}", len(subgraph))
            # ``cluster_ids[local_idx] == original_node_id``; the partitioner
            # has already relabeled each subgraph to ``0..M-1``.
            self._reverse_index_maps[prog_id] = dict(enumerate(cluster_ids))
            sub_problems[prog_id] = type(self)(
                subgraph, is_constrained=self._is_constrained
            )

        return sub_problems

    def initial_solution_size(self) -> int:
        return len(self.graph)

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: Hashable,
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

    @cached_property
    def _diagonal_terms(self) -> list[tuple[float, tuple[int, ...]]]:
        """``(coeff, z_qubit_indices)`` per cost-Hamiltonian term.

        Computed once from the (immutable) cost Hamiltonian so
        :meth:`evaluate_global_solution` need not rebuild Pauli labels on every
        call. Validates the Hamiltonian is diagonal (Z/I only) up front.
        """
        spo: SparsePauliOp = self.cost_hamiltonian
        terms: list[tuple[float, tuple[int, ...]]] = []
        for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
            z_qubits = []
            for qubit, char in enumerate(reversed(label)):
                if char == "I":
                    continue
                if char != "Z":
                    raise ValueError(
                        f"Cost Hamiltonian contains non-diagonal term {label!r}; "
                        f"evaluate_global_solution requires Z-only operators."
                    )
                z_qubits.append(qubit)
            terms.append((float(np.real(coeff)), tuple(z_qubits)))
        return terms

    def evaluate_global_solution(self, solution: list[int]) -> float:
        energy = self.loss_constant
        for coeff, z_qubits in self._diagonal_terms:
            eigenvalue = 1.0
            for qubit in z_qubits:
                eigenvalue *= 1 - 2 * solution[qubit]
            energy += coeff * eigenvalue
        return energy

    def postprocess_candidates(
        self, candidates: list[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[list[int], float]]:
        return [(list(np.where(solution)[0]), score) for score, solution in candidates]


class MaxCutProblem(_GraphProblemBase):
    """MaxCut problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
    """

    _resolver = staticmethod(maxcut_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState


class MaxCliqueProblem(_GraphProblemBase):
    """Max clique problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _resolver = staticmethod(max_clique_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MaxIndependentSetProblem(_GraphProblemBase):
    """Max independent set problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _resolver = staticmethod(max_independent_set_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState


class MinVertexCoverProblem(_GraphProblemBase):
    """Min vertex cover problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph.
        is_constrained: Use constrained mixer. Defaults to True.
    """

    _resolver = staticmethod(min_vertex_cover_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = OnesState
    _unconstrained_state_cls = SuperpositionState


class MaxWeightCycleProblem(_GraphProblemBase):
    """Max weight cycle problem on a directed graph.

    Args:
        graph: NetworkX DiGraph or RustworkX PyDiGraph with weighted edges.
        is_constrained: Use cycle-mixer (preserves valid cycles). Defaults to True.
    """

    _resolver = staticmethod(max_weight_cycle_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState

    @staticmethod
    def _compute_wire_labels(graph: GraphProblemTypes) -> tuple:
        # Cycle problems use edge variables; wires are 0-indexed by edge count.
        if hasattr(graph, "number_of_edges"):
            return tuple(range(graph.number_of_edges()))
        return tuple(range(len(graph.edge_list())))  # type: ignore[attr-defined]


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
