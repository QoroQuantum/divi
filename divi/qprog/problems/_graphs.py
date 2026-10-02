# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph problem classes for QAOA."""

import itertools
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
from divi.qprog.problems import QAOAProblem
from divi.qprog.problems._graph_hamiltonians import (
    _node_ids,
    _qubit_edges,
    max_clique_hamiltonians,
    max_independent_set_hamiltonians,
    max_weight_cycle_hamiltonians,
    maxcut_hamiltonians,
    min_vertex_cover_hamiltonians,
)
from divi.qprog.problems._graph_partitioning_utils import (
    _drawable,
    _node_partition_graph,
)
from divi.qprog.problems._partitioning_config import GraphPartitioningConfig


def _selected_wires(bitstring: str, n_wires: int) -> list[int]:
    """Positions set to ``1`` in ``bitstring``, ignoring bits past the last wire."""
    return [idx for idx, bit in enumerate(bitstring) if bit == "1" and idx < n_wires]


class _GraphProblemBase(QAOAProblem):
    """Shared logic for graph problems built directly from ``SparsePauliOp``.

    Subclasses set ``_resolver`` (a function returning ``(cost_spo, mixer_spo)``
    or ``(cost_spo, mixer_spo, metadata)``) and the two ``_*_state_cls`` class
    attributes. Problems without a constrained mixer set
    ``_supports_constrained_mixer = False`` and ignore ``use_constrained_mixer``.
    """

    _resolver: staticmethod
    _constrained_state_cls: type[InitialState]
    _unconstrained_state_cls: type[InitialState]
    _supports_constrained_mixer = True
    _supports_partitioning = True

    def __init__(
        self,
        graph: GraphProblemTypes,
        *,
        use_constrained_mixer: bool = True,
        config: GraphPartitioningConfig | None = None,
    ):
        if config is not None and not self._supports_partitioning:
            raise ValueError(
                f"{type(self).__name__} does not support graph partitioning: its "
                "variables are edges, while partitioning splits the graph by "
                "node. Build it without 'config'."
            )
        use_constrained_mixer = (
            use_constrained_mixer and self._supports_constrained_mixer
        )
        self._graph = graph
        self._use_constrained_mixer = use_constrained_mixer

        if self._supports_constrained_mixer:
            resolved = self._resolver(graph, constrained=use_constrained_mixer)
        else:
            resolved = self._resolver(graph)
        cost_spo, self._mixer_hamiltonian, *self._metadata = resolved

        cleaned, ham_constant = _clean_hamiltonian_spo(cost_spo, raise_on_constant=True)

        self._cost_hamiltonian = cleaned
        self._loss_constant = ham_constant
        self._wire_labels = self._compute_wire_labels()
        self._initial_state = (
            self._constrained_state_cls
            if use_constrained_mixer
            else self._unconstrained_state_cls
        )()
        self._config = config
        self._reverse_index_maps = {}

    def _compute_wire_labels(self) -> tuple:
        """Node of each qubit: networkx node labels in iteration order, or
        rustworkx node indices in ascending order."""
        return tuple(_node_ids(self._graph))

    @property
    def graph(self) -> GraphProblemTypes:
        """The underlying graph.

        Treat as read-only: the cost Hamiltonian, the Pauli terms cached by
        ``evaluate_global_solution`` and the adjacency cached by
        ``is_feasible`` are built from it once. Build a new problem instead of
        mutating it.
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
            return [wires[idx] for idx in _selected_wires(bitstring, len(wires))]

        return _decode

    def _selected_qubits(self, bitstring: str) -> list[int]:
        return _selected_wires(bitstring, len(self._wire_labels))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata[0] if self._metadata else {}

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        if self._config is None:
            raise ValueError(
                "Cannot decompose: no config was provided at construction."
            )

        rationale = _PARTITIONING_RISKS.get(type(self))
        if rationale is not None:
            warn(
                f"Heuristic-risk graph partitioning objective: "
                f"{type(self).__name__}. Results may be sensitive to partition "
                f"boundaries because {rationale}",
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
                subgraph, use_constrained_mixer=self._use_constrained_mixer
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
        position = self._node_positions

        for global_node in reverse_map.values():
            extended[position[global_node]] = 0

        for local_node in candidate_decoded:
            extended[position[reverse_map[local_node]]] = 1

        return extended

    @cached_property
    def _node_positions(self) -> dict:
        """Global solution position of each node id used by the partitioner."""
        return {node: i for i, node in enumerate(_node_ids(self._graph))}

    @cached_property
    def _qubit_edges(self) -> list[tuple[int, int]]:
        """The graph's edges as pairs of qubit positions."""
        return _qubit_edges(self._graph)[1]

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
        nodes = self._wire_labels
        return [
            ([nodes[i] for i in np.flatnonzero(solution)], score)
            for score, solution in candidates
        ]


class MaxCutProblem(_GraphProblemBase):
    """MaxCut problem on a graph.

    The cut is unweighted: every edge counts once, and edge ``weight``
    attributes are ignored. For weighted MaxCut, pass the QUBO to
    :class:`~divi.qprog.problems.BinaryOptimizationProblem`.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Ignored; MaxCut has no constrained mixer.
    """

    _resolver = staticmethod(maxcut_hamiltonians)  # type: ignore[assignment, bad-override]
    _unconstrained_state_cls = SuperpositionState
    _supports_constrained_mixer = False


class MaxCliqueProblem(_GraphProblemBase):
    """Max clique problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _resolver = staticmethod(max_clique_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState

    def is_feasible(self, bitstring: str) -> bool:
        """Whether every pair of selected nodes is joined by an edge."""
        adjacent = self._adjacent_pairs
        return all(
            pair in adjacent
            for pair in itertools.combinations(self._selected_qubits(bitstring), 2)
        )

    @cached_property
    def _adjacent_pairs(self) -> frozenset[tuple[int, int]]:
        """Qubit pairs ``(i, j)`` with ``i < j`` joined by an edge."""
        return frozenset((min(u, v), max(u, v)) for u, v in self._qubit_edges)

    def compute_energy(self, bitstring: str) -> float:
        """Negated number of selected nodes, so larger cliques score lower."""
        return -float(len(self.decode_fn(bitstring)))


class MaxIndependentSetProblem(_GraphProblemBase):
    """Max independent set problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _resolver = staticmethod(max_independent_set_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState

    def is_feasible(self, bitstring: str) -> bool:
        """Whether no two selected nodes are joined by an edge."""
        selected = set(self._selected_qubits(bitstring))
        return not any(u in selected and v in selected for u, v in self._qubit_edges)

    def compute_energy(self, bitstring: str) -> float:
        """Negated number of selected nodes, so larger sets score lower."""
        return -float(len(self.decode_fn(bitstring)))


class MinVertexCoverProblem(_GraphProblemBase):
    """Min vertex cover problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _resolver = staticmethod(min_vertex_cover_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = OnesState
    _unconstrained_state_cls = SuperpositionState

    def is_feasible(self, bitstring: str) -> bool:
        """Whether every edge has at least one selected endpoint."""
        selected = set(self._selected_qubits(bitstring))
        return all(u in selected or v in selected for u, v in self._qubit_edges)

    def compute_energy(self, bitstring: str) -> float:
        """Number of selected nodes, so smaller covers score lower."""
        return float(len(self.decode_fn(bitstring)))


class MaxWeightCycleProblem(_GraphProblemBase):
    """Max weight cycle problem on a directed graph.

    Args:
        graph: NetworkX DiGraph or RustworkX PyDiGraph with weighted edges. A
            RustworkX edge's weight is its numeric payload or the ``"weight"``
            entry of its dict payload, and ``metadata`` names edges by node
            index.
        use_constrained_mixer: Use the cycle mixer, which preserves valid cycles.
            Defaults to ``True``.
    """

    _resolver = staticmethod(max_weight_cycle_hamiltonians)  # type: ignore[assignment, bad-override]
    _constrained_state_cls = SuperpositionState
    _unconstrained_state_cls = SuperpositionState
    _supports_partitioning = False

    def _compute_wire_labels(self) -> tuple:
        """Edge-variable wires ``0..W-1``, one per edge of ``metadata``."""
        return tuple(range(len(self.metadata)))


# Partitioning is most robust for cut-style objectives (e.g. MaxCut).
# Structure-dependent objectives may lose cross-partition constraints.
_PARTITIONING_RISKS = {
    MaxCliqueProblem: "partitioning can hide cross-partition adjacency needed for global cliques.",
    MaxIndependentSetProblem: "partitioning can hide cross-partition conflicts between selected vertices.",
    MinVertexCoverProblem: "partitioning can hide cross-partition edges that must be covered globally.",
}


def draw_graph_solution_nodes(main_graph: GraphProblemTypes, partition_nodes):
    """Visualise a graph with solution nodes highlighted.

    Draws the graph with nodes coloured to distinguish solution nodes (red) from
    other nodes (light blue).

    Args:
        main_graph: NetworkX or RustworkX graph to visualise. RustworkX nodes
            are drawn and labelled by node index.
        partition_nodes: Collection of the solution's nodes, as reported by the
            problem's solutions.
    """
    main_graph = _drawable(main_graph)
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
