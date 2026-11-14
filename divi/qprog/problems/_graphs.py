# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Graph problem classes for QAOA."""

import itertools
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from functools import cached_property
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sps
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians import bit_driver, bit_flip_mixer, x_mixer
from divi.qprog.initial_states import (
    InitialState,
    OnesState,
    SuperpositionState,
    ZerosState,
)

from ._base import QAOAProblem, place_by_coupling
from ._graph_hamiltonians import (
    _build_vertex_subset_cost,
    _node_ids,
    _qubit_edges,
    _qubit_graph,
    _read_edge_weight,
    _undirected_wire_edges,
    apply_bit_flip_mixer,
    max_weight_cycle_hamiltonians,
    maxcut_hamiltonians,
)
from ._graph_partitioning_utils import (
    _drawable,
    _partition_graph_by_nodes,
)
from ._partitioning_config import GraphPartitioningConfig
from ._types import GraphProblemTypes


def _selected_wires(bitstring: str, n_wires: int) -> list[int]:
    """Positions set to ``1`` in ``bitstring``, ignoring bits past the last wire."""
    return [idx for idx, bit in enumerate(bitstring) if bit == "1" and idx < n_wires]


class GraphProblemBase(QAOAProblem):
    """Shared logic for graph problems built directly from ``SparsePauliOp``.

    Subclasses set ``_resolver`` (a function returning ``(cost_spo, mixer_spo)``
    or ``(cost_spo, mixer_spo, metadata)``), or override
    ``_resolve_hamiltonians``, and the two ``_*_state_cls`` class attributes. Problems without a constrained mixer set
    ``_supports_constrained_mixer = False`` and ignore ``use_constrained_mixer``.
    """

    _resolver: staticmethod
    _constrained_state_cls: type[InitialState]
    _unconstrained_state_cls: type[InitialState]
    _supports_constrained_mixer = True
    _supports_partitioning = True
    _isolated_value = 0
    _self_loop_value = 0

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

        resolved = self._resolve_hamiltonians(graph, use_constrained_mixer)
        cost_spo, self._mixer_hamiltonian, *self._metadata = resolved
        self._cost_hamiltonian = cost_spo.simplify(atol=0)
        self._wire_labels = self._compute_wire_labels()
        self._initial_state = (
            self._constrained_state_cls
            if use_constrained_mixer
            else self._unconstrained_state_cls
        )()
        self._config = config
        self._reverse_index_maps = {}

    def _resolve_hamiltonians(
        self, graph: GraphProblemTypes, constrained: bool
    ) -> tuple:
        """``(cost, mixer, *metadata)`` from the class's resolver."""
        if self._supports_constrained_mixer:
            return self._resolver(graph, constrained=constrained)
        return self._resolver(graph)

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

        subgraphs = _partition_graph_by_nodes(
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

    def order_partitions(self, prog_ids: Sequence[Hashable]) -> list[Hashable]:
        """Partitions ordered so each next one has the largest summed ``|w|``
        edge weight to those already placed."""
        position = self._node_positions
        groups = [
            [position[node] for node in self._reverse_index_maps[pid].values()]
            for pid in prog_ids
        ]
        _ids, edges = _undirected_wire_edges(self._graph)
        couplings = ((u, v, _read_edge_weight(payload)) for u, v, payload in edges)
        return place_by_coupling(prog_ids, groups, couplings)

    def initial_solution_size(self) -> int:
        return len(self.graph)

    def extend_solution(
        self,
        current_solution: Sequence[int],
        prog_id: Hashable,
        candidate_decoded: Sequence[int],
    ) -> list[int]:
        extended = list(current_solution)
        reverse_map = self._reverse_index_maps[prog_id]
        position = self._node_positions

        for global_node in reverse_map.values():
            extended[position[global_node]] = 0

        for local_node in candidate_decoded:
            extended[position[reverse_map[local_node]]] = 1

        for isolated, value in zip(self._isolated_positions, self._isolated_values):
            extended[isolated] = value

        return extended

    @cached_property
    def _isolated_positions(self) -> np.ndarray:
        """Qubit positions of the nodes without an edge to another node, which
        partitioning leaves out of every cluster."""
        if not self._supports_partitioning:
            return np.empty(0, dtype=np.intp)
        linked = {qubit for u, v in self._qubit_edges if u != v for qubit in (u, v)}
        return np.array(
            [qubit for qubit in range(len(self._wire_labels)) if qubit not in linked],
            dtype=np.intp,
        )

    @cached_property
    def _isolated_values(self) -> np.ndarray:
        """Trivially optimal value of each isolated node: ``_self_loop_value`` for
        one with a self-loop, ``_isolated_value`` otherwise."""
        if not self._isolated_positions.size:
            return np.empty(0, dtype=np.int64)
        looped = {u for u, v in self._qubit_edges if u == v}
        return np.array(
            [
                self._self_loop_value if qubit in looped else self._isolated_value
                for qubit in self._isolated_positions
            ],
            dtype=np.int64,
        )

    def _fill_isolated(self, solution: Sequence[int] | np.ndarray) -> np.ndarray:
        """``solution`` with every isolated node at its trivially optimal value."""
        bits = np.array(solution, dtype=np.int64)
        bits[self._isolated_positions] = self._isolated_values
        return bits

    @property
    def _scoring_hamiltonian(self) -> SparsePauliOp:
        """The diagonal Hamiltonian that scores global solutions."""
        return self.cost_hamiltonian

    @cached_property
    def _node_positions(self) -> dict:
        """Global solution position of each node id used by the partitioner."""
        return {node: i for i, node in enumerate(_node_ids(self._graph))}

    @cached_property
    def _qubit_edges(self) -> list[tuple[int, int]]:
        """The graph's edges as pairs of qubit positions."""
        return _qubit_edges(self._graph)[1]

    @cached_property
    def _diagonal_terms(self) -> tuple[np.ndarray, sps.csr_matrix]:
        """``(coeffs, z_masks)`` of the cost Hamiltonian's terms.

        ``z_masks[t, q]`` is 1 when term ``t`` has a Z on qubit ``q``. Computed
        once from the (immutable) cost Hamiltonian. Validates the Hamiltonian is
        diagonal (Z/I only) up front.
        """
        spo: SparsePauliOp = self._scoring_hamiltonian
        non_diagonal = np.flatnonzero(spo.paulis.x.any(axis=1))
        if non_diagonal.size:
            label = spo.paulis.to_labels()[non_diagonal[0]]
            raise ValueError(
                f"Cost Hamiltonian contains non-diagonal term {label!r}; "
                f"evaluate_global_solution requires Z-only operators."
            )
        return (
            np.real(spo.coeffs).astype(np.float64),
            sps.csr_matrix(spo.paulis.z.astype(np.int64)),
        )

    def evaluate_global_solution(self, solution: Sequence[int]) -> float:
        coeffs, z_masks = self._diagonal_terms
        x = np.asarray(solution, dtype=np.int64)
        if x.shape != (z_masks.shape[1],):
            raise ValueError(
                f"Expected a solution of shape ({z_masks.shape[1]},), "
                f"got shape {x.shape}."
            )
        parities = (z_masks @ self._fill_isolated(x)) & 1
        return float(coeffs @ (1 - 2 * parities))

    def postprocess_candidates(
        self, candidates: Sequence[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[list[int], float]]:
        nodes = self._wire_labels
        return [
            ([nodes[i] for i in np.flatnonzero(self._fill_isolated(solution))], score)
            for score, solution in candidates
        ]


class MaxCutProblem(GraphProblemBase):
    """Weighted MaxCut problem on a graph.

    Each cut edge contributes its weight: the ``"weight"`` edge attribute in
    networkx, and a numeric payload or the ``"weight"`` entry of a dict payload
    in rustworkx. An edge without a weight counts ``1``, a negative weight
    penalises cutting that edge, and self-loops never count.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Ignored; MaxCut has no constrained mixer.

    Raises:
        ValueError: If an edge weight is NaN or infinite.
    """

    _resolver = staticmethod(maxcut_hamiltonians)  # type: ignore[assignment, bad-override]
    _unconstrained_state_cls = SuperpositionState
    _supports_constrained_mixer = False


class _VertexSubsetProblem(GraphProblemBase):
    """Shared logic for problems that select a feasible subset of the nodes.

    Global solutions are scored by the penalised cost, which counts violated
    edges whatever the mixer, and post-processing repairs or drops infeasible
    candidates.
    """

    _cost_bit: int
    _penalised_edge_states: tuple[str, ...]
    _constraints_on_complement = False

    def _resolve_hamiltonians(
        self, graph: GraphProblemTypes, constrained: bool
    ) -> tuple:
        """The constrained cost, or the penalised cost and the X mixer; the
        constrained mixer is applied as gates, and expanded to a Hamiltonian only
        on request."""
        n_qubits, edges = _qubit_edges(graph)
        constraints = _qubit_graph(n_qubits, edges)
        if self._constraints_on_complement:
            constraints = nx.complement(constraints)
        self._constraint_graph = constraints
        self._penalised_cost = _build_vertex_subset_cost(
            constraints,
            n_qubits,
            cost_bit=self._cost_bit,
            penalised_edge_states=self._penalised_edge_states,
        ).simplify(atol=0)
        if constrained:
            return bit_driver(n_qubits, b=self._cost_bit), None
        return self._penalised_cost, x_mixer(n_qubits)

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """The mixer as a Hamiltonian; the constrained one is built on first access.

        Raises:
            ValueError: If the constrained mixer would expand into too many
                Pauli terms (see :func:`~divi.hamiltonians.bit_flip_mixer`).
        """
        if self._mixer_hamiltonian is None:
            self._mixer_hamiltonian = bit_flip_mixer(
                self._constraint_graph, b=1 - self._cost_bit
            )
        return self._mixer_hamiltonian

    @property
    def _applies_mixer_as_gates(self) -> bool:
        return self._use_constrained_mixer

    def apply_mixer(self, circuit, beta, qubits: Sequence[int]) -> None:
        """The constrained mixer as one multi-controlled ``Rx(2·beta)`` per node
        (see :func:`apply_bit_flip_mixer`); otherwise the X mixer."""
        if not self._use_constrained_mixer:
            super().apply_mixer(circuit, beta, qubits)
            return
        apply_bit_flip_mixer(
            circuit, self._constraint_graph, 1 - self._cost_bit, beta, qubits
        )

    def mixer_parameter_frequency(self) -> tuple[float, int]:
        """``(1, 2·n_nodes)`` for the constrained mixer, else the X mixer's."""
        if not self._use_constrained_mixer:
            return super().mixer_parameter_frequency()
        return 1.0, 2 * len(self._wire_labels)

    def compute_energy(self, bitstring: str) -> float:
        """Selected-node count, signed so that better subsets score lower."""
        raise NotImplementedError

    def _repair_selection(self, selected: set[int]) -> set[int]:
        """A feasible subset reached greedily from the qubits in ``selected``."""
        raise NotImplementedError

    @property
    def _scoring_hamiltonian(self) -> SparsePauliOp:
        return self._penalised_cost

    def postprocess_candidates(
        self, candidates: Sequence[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[list, float]]:
        """Distinct feasible ``(nodes, objective)`` pairs, best first.

        Infeasible candidates are repaired greedily, or dropped when ``strict``
        is set. The objective is :meth:`compute_energy`.
        """
        n_qubits = len(self._wire_labels)
        results: dict[str, tuple[list, float]] = {}
        for _, solution in candidates:
            selected = set(np.flatnonzero(self._fill_isolated(solution)).tolist())
            if not self.is_feasible(self._to_bitstring(selected, n_qubits)):
                if strict:
                    continue
                selected = self._repair_selection(selected)
            bitstring = self._to_bitstring(selected, n_qubits)
            results[bitstring] = (
                self.decode_fn(bitstring),
                self.compute_energy(bitstring),
            )
        return sorted(results.values(), key=lambda entry: entry[1])

    @staticmethod
    def _to_bitstring(selected: set[int], n_qubits: int) -> str:
        return "".join("1" if qubit in selected else "0" for qubit in range(n_qubits))

    @staticmethod
    def _pick_by_count(counts: Counter) -> int:
        """The qubit with the largest count, the lowest position on ties."""
        return max(counts, key=lambda qubit: (counts[qubit], -qubit))


class MaxCliqueProblem(_VertexSubsetProblem):
    """Max clique problem on a graph.

    An isolated node is a clique of one, so it is selected only when nothing
    else is.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState
    _cost_bit = 1
    _penalised_edge_states = ("10", "01", "00")
    _constraints_on_complement = True

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

    def _fill_isolated(self, solution: Sequence[int] | np.ndarray) -> np.ndarray:
        """``solution``, with the first isolated node selected when nothing is."""
        bits = super()._fill_isolated(solution)
        if not bits.any() and self._isolated_positions.size:
            bits[self._isolated_positions[0]] = 1
        return bits

    def compute_energy(self, bitstring: str) -> float:
        """Negated number of selected nodes, so larger cliques score lower."""
        return -float(len(self.decode_fn(bitstring)))

    def _repair_selection(self, selected: set[int]) -> set[int]:
        """Drop the node with the most non-neighbours until a clique remains."""
        selected = set(selected)
        while True:
            missing = Counter(
                qubit
                for pair in itertools.combinations(sorted(selected), 2)
                if pair not in self._adjacent_pairs
                for qubit in pair
            )
            if not missing:
                return selected
            selected.discard(self._pick_by_count(missing))


class MaxIndependentSetProblem(_VertexSubsetProblem):
    """Max independent set problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _constrained_state_cls = ZerosState
    _unconstrained_state_cls = SuperpositionState
    _isolated_value = 1
    _cost_bit = 1
    _penalised_edge_states = ("10", "01", "00")

    def is_feasible(self, bitstring: str) -> bool:
        """Whether no two selected nodes are joined by an edge."""
        selected = set(self._selected_qubits(bitstring))
        return not any(u in selected and v in selected for u, v in self._qubit_edges)

    def compute_energy(self, bitstring: str) -> float:
        """Negated number of selected nodes, so larger sets score lower."""
        return -float(len(self.decode_fn(bitstring)))

    def _repair_selection(self, selected: set[int]) -> set[int]:
        """Drop the node in the most conflicts until no edge has both ends selected."""
        selected = set(selected)
        while True:
            conflicts = Counter(
                qubit
                for edge in self._qubit_edges
                if all(end in selected for end in edge)
                for qubit in edge
            )
            if not conflicts:
                return selected
            selected.discard(self._pick_by_count(conflicts))


class MinVertexCoverProblem(_VertexSubsetProblem):
    """Min vertex cover problem on a graph.

    Args:
        graph: NetworkX or RustworkX graph. RustworkX nodes are identified by
            node index, not payload, in wire labels and decoded solutions.
        use_constrained_mixer: Use the constrained mixer and a feasible
            initial state. Defaults to ``True``.
    """

    _constrained_state_cls = OnesState
    _unconstrained_state_cls = SuperpositionState
    _self_loop_value = 1
    _cost_bit = 0
    _penalised_edge_states = ("11", "10", "01")

    def is_feasible(self, bitstring: str) -> bool:
        """Whether every edge has at least one selected endpoint."""
        selected = set(self._selected_qubits(bitstring))
        return all(u in selected or v in selected for u, v in self._qubit_edges)

    def compute_energy(self, bitstring: str) -> float:
        """Number of selected nodes, so smaller covers score lower."""
        return float(len(self.decode_fn(bitstring)))

    def _repair_selection(self, selected: set[int]) -> set[int]:
        """Add the node covering the most uncovered edges until every edge is covered."""
        selected = set(selected)
        while True:
            uncovered = Counter(
                qubit
                for edge in self._qubit_edges
                if not any(end in selected for end in edge)
                for qubit in set(edge)
            )
            if not uncovered:
                return selected
            selected.add(self._pick_by_count(uncovered))


class MaxWeightCycleProblem(GraphProblemBase):
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
