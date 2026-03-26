# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Weighted matching problem for QAOA-based quantum optimization."""

from __future__ import annotations

import string
from collections.abc import Callable, Hashable
from functools import partial
from typing import Any, Literal

import networkx as nx
import numpy as np

from divi.qprog.problems._base import QAOAProblem
from divi.qprog.problems._binary import BinaryOptimizationProblem

# ------------------------------------------------------------------
# Matching utility functions
# ------------------------------------------------------------------


def _construct_matching_qubo(
    graph: nx.Graph,
    edge_to_qubit: dict[tuple, int],
    penalty_scale: float = 10.0,
) -> np.ndarray:
    """Build a QUBO matrix encoding the maximum-weight matching problem.

    Linear terms ``-w_e`` maximize edge weight.  Quadratic penalty terms
    ``+lambda`` for each pair of incident edges enforce the matching
    constraint (at most one edge per node).

    Args:
        graph: Weighted graph.
        edge_to_qubit: Mapping from ``(u, v)`` edge tuples to qubit indices.
        penalty_scale: Multiplier for the penalty strength.  The actual
            penalty is ``penalty_scale * sum(all_edge_weights)``.

    Returns:
        Symmetric QUBO matrix of shape ``(n_edges, n_edges)``.
    """
    n = len(set(edge_to_qubit.values()))
    qubo = np.zeros((n, n), dtype=float)

    total_weight = sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True))
    penalty = penalty_scale * total_weight

    # Linear terms: -w_e on the diagonal
    for (u, v), idx in edge_to_qubit.items():
        if u > v:
            continue  # skip reverse entries
        w = graph[u][v].get("weight", 1.0)
        qubo[idx, idx] = -w

    # Quadratic terms: +penalty for pairs of incident edges
    edges_by_idx = {}
    for (u, v), idx in edge_to_qubit.items():
        if u > v:
            continue
        edges_by_idx[idx] = (u, v)

    node_to_qubits: dict[Any, list[int]] = {}
    for idx, (u, v) in edges_by_idx.items():
        node_to_qubits.setdefault(u, []).append(idx)
        node_to_qubits.setdefault(v, []).append(idx)

    for _node, qubits in node_to_qubits.items():
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                qi, qj = qubits[i], qubits[j]
                qubo[qi, qj] += penalty
                qubo[qj, qi] += penalty

    return qubo


def _sort_matching(matching: list[tuple]) -> list[tuple]:
    """Canonical sort: sort nodes within each edge, then sort edges."""
    return sorted(tuple(sorted(edge)) for edge in matching)


def is_valid_matching(edges: list[tuple]) -> bool:
    """Check that no node appears in more than one selected edge."""
    seen: set = set()
    for u, v in edges:
        if u in seen or v in seen:
            return False
        seen.add(u)
        seen.add(v)
    return True


def _bitstring_to_matching(
    bitstring: str, edge_to_qubit: dict[tuple, int]
) -> list[tuple]:
    """Decode a measurement bitstring into a list of matching edges.

    Uses right-to-left indexing (qubit 0 = rightmost bit).
    """
    n = len(bitstring)
    matching = []
    for edge, qubit in edge_to_qubit.items():
        if edge[0] > edge[1]:
            continue  # skip reverse entries
        if bitstring[n - qubit - 1] == "1":
            matching.append(edge)
    return _sort_matching(matching)


def check_matching_matrix(M: np.ndarray, A: np.ndarray) -> bool:
    """Validate that adjacency matrix *M* is a valid matching in graph *A*.

    Checks:
        1. ``M`` has no edges where ``A`` has none.
        2. Each row and column sum of ``M`` is at most 1.
    """
    if np.any(M[A == 0] != 0):
        return False
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    return bool(np.all(row_sums <= 1) and np.all(col_sums <= 1))


# ------------------------------------------------------------------
# Edge-based graph partitioning
# ------------------------------------------------------------------


def _partition_graph_by_edges(
    graph: nx.Graph,
    max_edges: int,
    algorithm: Literal["kernighan_lin", "spectral"] = "kernighan_lin",
    seed: int | None = None,
) -> list[nx.Graph]:
    """Recursively partition a graph until each subgraph has <= *max_edges* edges.

    Args:
        graph: The graph to partition.
        max_edges: Maximum number of edges per partition.
        algorithm: ``"kernighan_lin"`` (weight-aware) or ``"spectral"``
            (topology-based Fiedler vector).
        seed: Random seed for reproducibility.

    Returns:
        List of subgraph copies.
    """
    if graph.size() <= max_edges:
        return [graph.copy()]

    if graph.number_of_nodes() < 2:
        return [graph.copy()]

    if algorithm == "kernighan_lin":
        part_a, part_b = _kl_bisect(graph, seed=seed)
    elif algorithm == "spectral":
        part_a, part_b = _spectral_bisect(graph)
    else:
        raise ValueError(
            f"Unsupported partitioning algorithm: {algorithm!r}. "
            "Supported: 'kernighan_lin', 'spectral'."
        )

    sg_a = graph.subgraph(part_a).copy()
    sg_b = graph.subgraph(part_b).copy()

    result = []
    for sg in (sg_a, sg_b):
        if sg.size() == 0:
            continue
        result.extend(_partition_graph_by_edges(sg, max_edges, algorithm, seed=seed))
    return result


def _kl_bisect(graph: nx.Graph, seed: int | None = None) -> tuple[set, set]:
    """Kernighan-Lin bisection with weight-negated edges.

    Negates edge weights so KL preferentially cuts low-weight edges,
    keeping high-weight edges within partitions.
    """
    G_neg = graph.copy()
    max_w = max(
        (d.get("weight", 1.0) for _, _, d in G_neg.edges(data=True)),
        default=1.0,
    )
    for u, v, d in G_neg.edges(data=True):
        d["kl_weight"] = max_w + 1 - d.get("weight", 1.0)

    part_a, part_b = nx.community.kernighan_lin_bisection(
        G_neg, weight="kl_weight", seed=seed
    )
    return set(part_a), set(part_b)


def _spectral_bisect(graph: nx.Graph) -> tuple[set, set]:
    """Fiedler-vector bisection on the graph Laplacian."""
    import scipy.sparse.linalg as spla

    L = nx.laplacian_matrix(graph).astype(float)
    _eigenvalues, eigenvectors = spla.eigsh(L, k=2, which="SM")
    fiedler = eigenvectors[:, 1]
    median = np.median(fiedler)

    nodes = list(graph.nodes())
    part_a = {nodes[i] for i in range(len(nodes)) if fiedler[i] <= median}
    part_b = set(nodes) - part_a
    return part_a, part_b


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _count_conflicts(solution: list[int], edges: list[tuple]) -> int:
    """Count matching constraint violations in a solution vector."""
    node_count: dict = {}
    for idx, bit in enumerate(solution):
        if bit:
            u, v = edges[idx]
            node_count[u] = node_count.get(u, 0) + 1
            node_count[v] = node_count.get(v, 0) + 1
    return sum(max(0, c - 1) for c in node_count.values())


def _classical_cleanup(
    solution: list[int],
    graph: nx.Graph,
    edges: list[tuple],
    edge_to_qubit: dict[tuple, int],
) -> list[int]:
    """Fill unmatched nodes using exact classical matching on the residual graph.

    Identifies nodes not covered by the quantum solution, builds the
    residual subgraph, and runs :func:`nx.max_weight_matching` on it.
    """
    matched_nodes: set = set()
    for idx, bit in enumerate(solution):
        if bit:
            u, v = edges[idx]
            matched_nodes.add(u)
            matched_nodes.add(v)

    residual_nodes = [n for n in graph.nodes() if n not in matched_nodes]
    if not residual_nodes:
        return solution

    residual = graph.subgraph(residual_nodes)
    if residual.number_of_edges() == 0:
        return solution

    extra_edges = nx.max_weight_matching(residual, maxcardinality=False)

    result = list(solution)
    for u, v in extra_edges:
        key = (u, v) if (u, v) in edge_to_qubit else (v, u)
        if key in edge_to_qubit:
            result[edge_to_qubit[key]] = 1
    return result


def _repair_matching(edges: list[tuple], graph: nx.Graph) -> list[tuple]:
    """Greedily repair an invalid matching by keeping highest-weight edges first."""
    weighted = sorted(
        edges,
        key=lambda e: graph[e[0]][e[1]].get("weight", 1.0),
        reverse=True,
    )
    valid: list[tuple] = []
    used: set = set()
    for u, v in weighted:
        if u not in used and v not in used:
            valid.append((u, v))
            used.add(u)
            used.add(v)
    return valid


# ------------------------------------------------------------------
# MaxWeightMatchingProblem
# ------------------------------------------------------------------


class MaxWeightMatchingProblem(QAOAProblem):
    """Maximum-weight matching problem for QAOA.

    Given a weighted graph, finds a set of edges (matching) that maximizes
    total weight while ensuring no two selected edges share a node.

    Can be used directly with :class:`~divi.qprog.algorithms.QAOA` for
    small graphs, or with
    :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` for large
    graphs via edge-based partitioning.

    Args:
        graph: Weighted undirected graph.
        penalty_scale: Strength of matching constraint penalties in the
            QUBO formulation.  Higher values enforce constraints more
            strictly.
        max_edges_per_partition: Maximum edges per partition.  Setting
            this enables :meth:`decompose` for partitioned solving.
        partition_algorithm: Edge partitioning strategy.
            ``"kernighan_lin"`` (default, weight-aware) or ``"spectral"``.
        use_classical_cleanup: If ``True`` (default), fill unmatched
            residual nodes via :func:`nx.max_weight_matching` during
            :meth:`finalize_solution`.
        seed: Random seed for partitioning reproducibility.

    Example::

        from divi.qprog.problems import MaxWeightMatchingProblem
        from divi.qprog import QAOA
        from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
        from divi.backends import MaestroSimulator

        import networkx as nx

        G = nx.gnm_random_graph(8, 12, seed=42)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0

        problem = MaxWeightMatchingProblem(G, penalty_scale=10.0)
        qaoa = QAOA(problem, n_layers=2,
                     optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
                     max_iterations=20,
                     backend=MaestroSimulator())
        qaoa.run()
    """

    def __init__(
        self,
        graph: nx.Graph,
        penalty_scale: float = 10.0,
        *,
        max_edges_per_partition: int | None = None,
        partition_algorithm: Literal["kernighan_lin", "spectral"] = "kernighan_lin",
        use_classical_cleanup: bool = True,
        seed: int | None = None,
    ):
        self._graph = graph
        self._penalty_scale = penalty_scale
        self._max_edges_per_partition = max_edges_per_partition
        self._partition_algorithm = partition_algorithm
        self._use_classical_cleanup = use_classical_cleanup
        self._seed = seed

        # Build edge-to-qubit mapping (canonical: u < v)
        self._edges = [(u, v) if u < v else (v, u) for u, v in graph.edges()]
        self._edge_to_qubit: dict[tuple, int] = {}
        for i, (u, v) in enumerate(self._edges):
            self._edge_to_qubit[(u, v)] = i
            self._edge_to_qubit[(v, u)] = i

        # Build full-graph QUBO and delegate Hamiltonian to BinaryOptimizationProblem
        qubo_matrix = _construct_matching_qubo(
            graph, self._edge_to_qubit, penalty_scale
        )
        self._bop = BinaryOptimizationProblem(qubo_matrix)

        # Decomposition state (populated by decompose())
        self._edge_index_maps: dict[Hashable, list[int]] = {}

    # ------------------------------------------------------------------
    # QAOAProblem interface (delegated to internal BinaryOptimizationProblem)
    # ------------------------------------------------------------------

    @property
    def cost_hamiltonian(self):
        return self._bop.cost_hamiltonian

    @property
    def mixer_hamiltonian(self):
        return self._bop.mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._bop.loss_constant

    @property
    def decode_fn(self) -> Callable[[str], list[tuple]]:
        return partial(_bitstring_to_matching, edge_to_qubit=self._edge_to_qubit)

    @property
    def graph(self) -> nx.Graph:
        """The input graph."""
        return self._graph

    # ------------------------------------------------------------------
    # Decomposition hooks
    # ------------------------------------------------------------------

    def decompose(self) -> dict[tuple[str, int], QAOAProblem]:
        if self._max_edges_per_partition is None:
            raise ValueError(
                "Cannot decompose: max_edges_per_partition was not set at construction."
            )

        subgraphs = _partition_graph_by_edges(
            self._graph,
            max_edges=self._max_edges_per_partition,
            algorithm=self._partition_algorithm,
            seed=self._seed,
        )

        self._edge_index_maps = {}
        sub_problems: dict[tuple[str, int], QAOAProblem] = {}

        for i, subgraph in enumerate(subgraphs):
            prog_id = (string.ascii_uppercase[i], subgraph.size())

            # Local edge-to-qubit mapping for this partition
            local_edges = [(u, v) if u < v else (v, u) for u, v in subgraph.edges()]
            local_e2q: dict[tuple, int] = {}
            for j, (u, v) in enumerate(local_edges):
                local_e2q[(u, v)] = j
                local_e2q[(v, u)] = j

            # Map local indices → global indices
            self._edge_index_maps[prog_id] = [
                self._edge_to_qubit[e] for e in local_edges
            ]

            # Build per-partition QUBO
            qubo = _construct_matching_qubo(subgraph, local_e2q, self._penalty_scale)
            sub_problems[prog_id] = BinaryOptimizationProblem(qubo)

        return sub_problems

    def initial_solution_size(self) -> int:
        return len(self._edges)

    def extend_solution(
        self,
        current_solution: list[int],
        prog_id: Hashable,
        candidate_decoded: list[int],
    ) -> list[int]:
        extended = list(current_solution)
        global_indices = self._edge_index_maps[prog_id]
        for local_idx, global_idx in enumerate(global_indices):
            extended[global_idx] = int(candidate_decoded[local_idx])
        return extended

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Score a solution: negative (weight - conflict_penalty * conflicts).

        Lower is better for beam search.  Maximizing weight while minimizing
        conflicts.
        """
        weight = 0.0
        node_count: dict = {}
        for idx, bit in enumerate(solution):
            if bit:
                u, v = self._edges[idx]
                weight += self._graph[u][v].get("weight", 1.0)
                node_count[u] = node_count.get(u, 0) + 1
                node_count[v] = node_count.get(v, 0) + 1

        conflicts = sum(max(0, c - 1) for c in node_count.values())
        avg_weight = sum(
            d.get("weight", 1.0) for _, _, d in self._graph.edges(data=True)
        ) / max(self._graph.number_of_edges(), 1)

        # Negate: beam search keeps lowest scores
        return -(weight - avg_weight * conflicts)

    def _postprocess_solution(self, solution: list[int]) -> tuple[list[tuple], float]:
        """Repair conflicts, apply cleanup, compute weight."""
        # Repair first (fix conflicts), then cleanup (fill gaps)
        matching = [self._edges[i] for i, bit in enumerate(solution) if bit]
        if not is_valid_matching(matching):
            matching = _repair_matching(matching, self._graph)
            # Rebuild solution vector from repaired matching
            solution = [0] * len(self._edges)
            for edge in matching:
                solution[self._edge_to_qubit[edge]] = 1

        if self._use_classical_cleanup:
            solution = _classical_cleanup(
                solution, self._graph, self._edges, self._edge_to_qubit
            )
            matching = [self._edges[i] for i, bit in enumerate(solution) if bit]

        weight = sum(self._graph[u][v].get("weight", 1.0) for u, v in matching)
        return _sort_matching(matching), weight

    def finalize_solution(
        self, score: float, solution: list[int]
    ) -> tuple[list[tuple], float]:
        return self._postprocess_solution(solution)

    def format_top_solutions(
        self, results: list[tuple[float, list[int]]]
    ) -> list[tuple[list[tuple], float]]:
        formatted = [
            self._postprocess_solution(solution) for _score, solution in results
        ]

        # Sort by weight descending, then deduplicate
        formatted.sort(key=lambda x: x[1], reverse=True)
        seen: set[tuple] = set()
        deduped = []
        for edges, w in formatted:
            key = tuple(edges)
            if key not in seen:
                seen.add(key)
                deduped.append((edges, w))
        return deduped
