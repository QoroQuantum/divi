# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Weighted matching problem for QAOA-based quantum optimisation."""

import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from functools import cached_property, partial
from itertools import combinations
from typing import Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
import rustworkx as rx
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from divi.qprog.problems._base import QAOAProblem
from divi.qprog.problems._binary import BinaryOptimizationProblem
from divi.qprog.problems._graph_hamiltonians import _edge_weight, _wire_edges

_Edge = tuple[int, int]

#: Bit length of the heaviest edge weight once scaled to an integer for
#: :func:`rustworkx.max_weight_matching`.
_MATCHING_WEIGHT_BITS = 100

# ------------------------------------------------------------------
# Canonical edge representation
# ------------------------------------------------------------------


def _matching_edges(
    graph: nx.Graph | rx.PyGraph,
) -> tuple[list, list[_Edge], list[float]]:
    """Node ids, the edges as sorted position pairs ``(u, v)`` with ``u <= v``, and their weights.

    Endpoints are positions into the node ids, so a rustworkx graph and the
    networkx graph labelled by its node indices give identical output. Weights
    follow :func:`_edge_weight` and default to 1; parallel edges collapse into
    one carrying the last weight.
    """
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) or not isinstance(
        graph, (nx.Graph, rx.PyGraph)
    ):
        raise TypeError(
            f"Expected an undirected graph (nx.Graph or rx.PyGraph), got "
            f"{type(graph).__name__}."
        )
    ids, wire_edges = _wire_edges(graph)
    weighted = sorted(
        ((min(u, v), max(u, v), _edge_weight(payload)) for u, v, payload in wire_edges),
        key=lambda edge: edge[:2],
    )
    edges = [(u, v) for u, v, _ in weighted]
    weights = [1.0 if weight is None else weight for _, _, weight in weighted]
    return ids, edges, weights


def _construct_matching_qubo(
    edges: Sequence[_Edge],
    weights: Sequence[float],
    penalty_weight: float = 10.0,
) -> dict[tuple[Hashable, ...], float]:
    """QUBO terms encoding maximum-weight matching over edge qubits ``0..len(edges)-1``.

    Linear terms ``-w_e`` reward edge weight. Each pair of edges sharing a node
    costs ``penalty_weight * sum(weights)``, enforcing at most one edge per node.

    Args:
        edges: Edges as node pairs; qubit ``i`` is ``edges[i]``.
        weights: Weight of each edge.
        penalty_weight: Multiplier for the penalty strength.

    Returns:
        Polynomial terms ``{(i,): -w_i, (i, j): penalty}`` with ``i < j``.
    """
    penalty = penalty_weight * sum(weights)
    terms: dict[tuple[Hashable, ...], float] = {(i,): -w for i, w in enumerate(weights)}
    incident: defaultdict[int, list[int]] = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        incident[u].append(i)
        incident[v].append(i)
    for qubits in incident.values():
        terms.update((pair, penalty) for pair in combinations(qubits, 2))
    return terms


def is_valid_matching(edges: list[tuple]) -> bool:
    """Check that no node appears in more than one selected edge."""
    seen: set = set()
    for u, v in edges:
        if u in seen or v in seen:
            return False
        seen.add(u)
        seen.add(v)
    return True


def _bitstring_to_matching(bitstring: str, edges: Sequence[tuple]) -> list[tuple]:
    """Edges selected by a measurement bitstring, in qubit order.

    Uses left-to-right qubit ordering: ``bitstring[i]`` selects ``edges[i]``.
    """
    return [edge for edge, bit in zip(edges, bitstring) if bit == "1"]


def check_matching_matrix(M: np.ndarray, A: np.ndarray) -> bool:
    """Validate that adjacency matrix *M* is a valid matching in graph *A*.

    Checks:
        1. ``M`` has no edges where ``A`` has none.
        2. Each row and column sum of ``M`` is at most 1.

    Raises:
        ValueError: If ``M`` or ``A`` is not a symmetric square matrix.
    """
    for name, matrix in (("M", M), ("A", A)):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"{name} must be a square matrix, got shape {matrix.shape}."
            )
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name} must be symmetric.")
    if M.shape != A.shape:
        raise ValueError(
            f"M and A must have the same shape, got {M.shape} and {A.shape}."
        )
    if np.any(M[A == 0] != 0):
        return False
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    return bool(np.all(row_sums <= 1) and np.all(col_sums <= 1))


# ------------------------------------------------------------------
# Edge-based graph partitioning
# ------------------------------------------------------------------


def _edge_index_graph(edges: Sequence[_Edge]) -> rx.PyGraph:
    """rustworkx graph on positions ``0..max(v)``, node payloads positions, edge payloads edge indices."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(1 + max((v for _, v in edges), default=-1)))
    graph.add_edges_from([(u, v, i) for i, (u, v) in enumerate(edges)])
    return graph


def _partition_graph_by_edges(
    edges: Sequence[_Edge],
    weights: Sequence[float],
    max_edges: int,
    algorithm: Literal["kernighan_lin", "spectral"] = "kernighan_lin",
    seed: int | None = 0,
) -> list[list[int]]:
    """Recursively partition a graph until each part has <= *max_edges* edges.

    Isolated nodes are dropped and connected components are partitioned
    separately. Edges cut by a bisection belong to no partition.

    Args:
        edges: Edges as position pairs ``(u, v)`` with ``u < v``.
        weights: Weight of each edge.
        max_edges: Maximum number of edges per partition.
        algorithm: ``"kernighan_lin"`` (weight-aware) or ``"spectral"``
            (weighted-Laplacian Fiedler vector).
        seed: Random seed for reproducibility.

    Returns:
        Ascending edge indices of each partition, each with at least one and
        at most *max_edges* edges.

    Raises:
        ValueError: If ``algorithm`` is unsupported, or a connected subgraph
            above *max_edges* edges cannot be bisected.
    """
    if algorithm not in ("kernighan_lin", "spectral"):
        raise ValueError(
            f"Unsupported partitioning algorithm: {algorithm!r}. "
            "Supported: 'kernighan_lin', 'spectral'."
        )
    return _split_by_edges(
        _edge_index_graph(edges), weights, max_edges, algorithm, seed
    )


def _split_by_edges(
    graph: rx.PyGraph,
    weights: Sequence[float],
    max_edges: int,
    algorithm: Literal["kernighan_lin", "spectral"],
    seed: int | None,
) -> list[list[int]]:
    """:func:`_partition_graph_by_edges` on an :func:`_edge_index_graph` subgraph."""
    n_edges = graph.num_edges()
    if n_edges <= max_edges:
        return [sorted(graph.edges())] if n_edges else []

    components = sorted(
        sorted(component)
        for component in rx.connected_components(graph)
        if len(component) > 1
    )
    if len(components) > 1 or len(components[0]) < graph.num_nodes():
        return [
            part
            for component in components
            for part in _split_by_edges(
                graph.subgraph(component), weights, max_edges, algorithm, seed
            )
        ]

    if algorithm == "kernighan_lin":
        part_a, part_b = _kl_bisect(graph, weights, seed=seed)
    else:
        part_a, part_b = _spectral_bisect(graph, weights)

    if sorted(part_b) < sorted(part_a):
        part_a, part_b = part_b, part_a

    if not part_a or not part_b:
        raise ValueError(
            f"Cannot split a connected subgraph of {n_edges} edges to meet "
            f"max_edges={max_edges}: {algorithm!r} bisection left one side empty."
        )

    index_of = {graph[node]: node for node in graph.node_indices()}
    return [
        part
        for side in (part_a, part_b)
        for part in _split_by_edges(
            graph.subgraph(sorted(index_of[position] for position in side)),
            weights,
            max_edges,
            algorithm,
            seed,
        )
    ]


def _kl_bisect(
    graph: rx.PyGraph, weights: Sequence[float], seed: int | None = None
) -> tuple[set[int], set[int]]:
    """Kernighan-Lin bisection minimising the total weight of cut edges.

    Low-weight edges are cut in preference, keeping high-weight edges within
    partitions. Runs :func:`networkx.algorithms.community.kernighan_lin_bisection`
    on the position-labelled graph and returns the positions of each side.
    """
    kl_graph = nx.Graph()
    kl_graph.add_nodes_from(sorted(graph.nodes()))
    kl_graph.add_weighted_edges_from(
        (graph[u], graph[v], weights[index])
        for u, v, index in sorted(graph.weighted_edge_list(), key=lambda e: e[2])
    )
    part_a, part_b = nx.community.kernighan_lin_bisection(
        kl_graph, weight="weight", seed=seed
    )
    return set(part_a), set(part_b)


def _spectral_bisect(
    graph: rx.PyGraph, weights: Sequence[float]
) -> tuple[set[int], set[int]]:
    """Fiedler-vector bisection on the weighted graph Laplacian.

    The ``ceil(n / 2)`` nodes with the lowest Fiedler entries form the first
    part, so both parts are non-empty even when entries tie. Returns the
    positions of each side.
    """
    nodes = list(graph.node_indices())
    local = {node: i for i, node in enumerate(nodes)}
    edge_list = graph.weighted_edge_list()
    rows = [local[u] for u, _, _ in edge_list]
    cols = [local[v] for _, v, _ in edge_list]
    data = [weights[index] for _, _, index in edge_list]
    n = len(nodes)
    adjacency = sps.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adjacency = adjacency + adjacency.T
    laplacian = sps.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    v0 = np.linspace(1.0, 2.0, n)
    _eigenvalues, eigenvectors = spla.eigsh(laplacian, k=2, which="SM", v0=v0)
    fiedler = eigenvectors[:, 1]

    order = np.argsort(fiedler, kind="stable")
    positions = [graph[node] for node in nodes]
    part_a = {positions[i] for i in order[: (n + 1) // 2]}
    part_b = set(positions) - part_a
    return part_a, part_b


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _count_conflicts(solution: npt.ArrayLike, edges: npt.ArrayLike) -> int:
    """Count matching constraint violations in a solution vector."""
    selected = np.asarray(edges, dtype=np.intp).reshape(-1, 2)[
        np.asarray(solution, dtype=bool)
    ]
    counts = np.bincount(selected.ravel())
    return int(counts.sum() - np.count_nonzero(counts))


def _integer_weights(weights: Sequence[float]) -> list[int]:
    """Weights scaled by a common power of two and rounded, for :func:`rustworkx.max_weight_matching`.

    The heaviest weight becomes a :data:`_MATCHING_WEIGHT_BITS`-bit integer, so
    weights down to ``2**-47`` of it scale exactly and lighter ones round to a
    step of at most ``2**-99`` of it. Non-positive weights, which never improve
    a matching, map to 0.
    """
    heaviest = max(weights, default=0.0)
    if not heaviest > 0:
        return [0] * len(weights)
    shift = _MATCHING_WEIGHT_BITS - math.frexp(heaviest)[1]
    return [round(math.ldexp(w, shift)) if w > 0 else 0 for w in weights]


def _classical_cleanup(
    solution: list[int],
    edges: Sequence[_Edge],
    integer_weights: Sequence[int],
) -> list[int]:
    """Fill unmatched nodes using exact classical matching on the residual graph.

    The residual graph holds the positive-weight edges between nodes the
    solution leaves unmatched; :func:`rustworkx.max_weight_matching` adds its
    maximum-weight matching under ``integer_weights``.
    """
    matched = {node for idx, bit in enumerate(solution) if bit for node in edges[idx]}
    residual_edges = [
        i
        for i, (u, v) in enumerate(edges)
        if integer_weights[i] > 0 and u not in matched and v not in matched
    ]
    if not residual_edges:
        return solution

    residual = _edge_index_graph([edges[i] for i in residual_edges])
    extra = rx.max_weight_matching(
        residual,
        max_cardinality=False,
        weight_fn=lambda local: integer_weights[residual_edges[local]],
    )

    edge_at = {edges[i]: i for i in residual_edges}
    result = list(solution)
    for u, v in extra:
        result[edge_at[(min(u, v), max(u, v))]] = 1
    return result


def _repair_matching(
    selected: Sequence[int], edges: Sequence[_Edge], weights: Sequence[float]
) -> list[int]:
    """Greedily repair an invalid matching by keeping highest-weight edges first.

    Returns the kept edge indices in ascending order.
    """
    valid: list[int] = []
    used: set[int] = set()
    for i in sorted(selected, key=lambda i: weights[i], reverse=True):
        u, v = edges[i]
        if u not in used and v not in used:
            valid.append(i)
            used.add(u)
            used.add(v)
    return sorted(valid)


# ------------------------------------------------------------------
# MaxWeightMatchingProblem
# ------------------------------------------------------------------


class MaxWeightMatchingProblem(QAOAProblem):
    """Maximum-weight matching problem for QAOA.

    Given a weighted graph, finds a set of edges (matching) that maximises
    total weight while ensuring no two selected edges share a node.

    Can be used directly with :class:`~divi.qprog.algorithms.QAOA` for
    small graphs, or with
    :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` for large
    graphs via edge-based partitioning.

    Qubit ``i`` is the ``i``-th edge in ascending order of its endpoints'
    node positions. A decoded matching lists each edge as a pair of node
    labels for ``nx.Graph`` and node indices for ``rx.PyGraph``.

    Args:
        graph: Weighted undirected graph, ``nx.Graph`` or ``rx.PyGraph``.
            ``rx.PyGraph`` nodes are identified by node index, and an edge payload
            that is a number, or a dict with a ``"weight"`` entry, is its
            weight; other edges weigh 1.
        penalty_weight: Strength of matching constraint penalties in the
            QUBO formulation.  Higher values enforce constraints more
            strictly.
        max_edges_per_partition: Maximum edges per partition.  Setting
            this enables :meth:`decompose` for partitioned solving.
        partition_algorithm: Edge partitioning strategy.
            ``"kernighan_lin"`` (default, weight-aware) or ``"spectral"``.
        use_classical_cleanup: If ``True`` (default), fill unmatched
            residual nodes via :func:`~rustworkx.max_weight_matching` during
            :meth:`postprocess_candidates`. It runs on integer weights scaled
            so the heaviest edge has 100 bits: weights down to ``2**-47`` of
            the heaviest are exact and lighter ones round to a step of at most
            ``2**-99`` of it.
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

        problem = MaxWeightMatchingProblem(G, penalty_weight=10.0)
        qaoa = QAOA(problem, n_layers=2,
                     optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
                     max_iterations=20,
                     backend=MaestroSimulator())
        qaoa.run()
    """

    def __init__(
        self,
        graph: nx.Graph | rx.PyGraph,
        penalty_weight: float = 10.0,
        *,
        max_edges_per_partition: int | None = None,
        partition_algorithm: Literal["kernighan_lin", "spectral"] = "kernighan_lin",
        use_classical_cleanup: bool = True,
        seed: int | None = 0,
    ):
        self._input_graph = graph
        node_ids, self._edges, self._weights = _matching_edges(graph)
        if any(u == v for u, v in self._edges):
            raise ValueError("Graph contains self-loops")
        if max_edges_per_partition is not None and max_edges_per_partition < 1:
            raise ValueError(
                "max_edges_per_partition must be at least 1, got "
                f"{max_edges_per_partition}."
            )
        self._penalty_weight = penalty_weight
        self._max_edges_per_partition = max_edges_per_partition
        self._partition_algorithm = partition_algorithm
        self._use_classical_cleanup = use_classical_cleanup
        self._seed = seed

        self._edge_to_qubit: dict[_Edge, int] = {
            edge: i for i, edge in enumerate(self._edges)
        }
        self._labelled_edges = [(node_ids[u], node_ids[v]) for u, v in self._edges]
        self._edge_array = np.array(self._edges, dtype=np.intp).reshape(-1, 2)
        self._weight_array = np.array(self._weights, dtype=float)

        self._bop = BinaryOptimizationProblem(
            _construct_matching_qubo(self._edges, self._weights, penalty_weight)
        )

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
        return partial(_bitstring_to_matching, edges=self._labelled_edges)

    @property
    def graph(self) -> nx.Graph | rx.PyGraph:
        """The input graph.

        Treat as read-only: edge weights are read into cached state at
        construction. Changing edge weights or structure afterwards will not
        update the cached average-weight penalty, so ``evaluate_global_solution``
        would return stale scores. Build a new problem instead.
        """
        return self._input_graph

    def _is_matching(self, selected: Sequence[int]) -> bool:
        return is_valid_matching([self._edges[i] for i in selected])

    def _matching_and_weight(
        self, selected: Sequence[int]
    ) -> tuple[list[tuple], float]:
        """The labelled edges at ascending ``selected`` indices and their total weight."""
        return (
            [self._labelled_edges[i] for i in selected],
            math.fsum(self._weights[i] for i in selected),
        )

    def is_feasible(self, bitstring: str) -> bool:
        """Check that the decoded matching has no node appearing in more than one edge."""
        return self._is_matching([i for i, bit in enumerate(bitstring) if bit == "1"])

    def compute_energy(self, bitstring: str) -> float | None:
        """Compute matching weight (negated, since lower is better).

        Returns ``None`` for infeasible bitstrings.
        """
        selected = [i for i, bit in enumerate(bitstring) if bit == "1"]
        if not self._is_matching(selected):
            return None
        return -math.fsum(self._weights[i] for i in selected)

    # ------------------------------------------------------------------
    # Decomposition hooks
    # ------------------------------------------------------------------

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        if self._max_edges_per_partition is None:
            raise ValueError(
                "Cannot decompose: max_edges_per_partition was not set at construction."
            )

        parts = _partition_graph_by_edges(
            self._edges,
            self._weights,
            max_edges=self._max_edges_per_partition,
            algorithm=self._partition_algorithm,
            seed=self._seed,
        )

        self._edge_index_maps = {}
        sub_problems: dict[Hashable, QAOAProblem] = {}
        for i, indices in enumerate(parts):
            prog_id = (f"P{i}", len(indices))
            self._edge_index_maps[prog_id] = indices
            qubo = _construct_matching_qubo(
                [self._edges[j] for j in indices],
                [self._weights[j] for j in indices],
                self._penalty_weight,
            )
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

    @cached_property
    def _avg_weight(self) -> float:
        """Mean edge weight of the (immutable) graph; the conflict penalty."""
        return sum(self._weights) / max(len(self._weights), 1)

    @cached_property
    def _matching_weights(self) -> list[int]:
        """Edge weights as integers for the classical cleanup."""
        return _integer_weights(self._weights)

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Score a solution: negative (weight - conflict_penalty * conflicts).

        Lower is better for beam search.  Maximising weight while minimising
        conflicts.
        """
        mask = np.asarray(solution, dtype=bool)
        weight = float(self._weight_array[mask].sum())
        conflicts = _count_conflicts(mask, self._edge_array)

        # Negate: beam search keeps lowest scores
        return -(weight - self._avg_weight * conflicts)

    def _postprocess_solution(self, solution: list[int]) -> tuple[list[tuple], float]:
        """Repair conflicts, apply cleanup, compute weight."""
        # Repair first (fix conflicts), then cleanup (fill gaps)
        selected = [i for i, bit in enumerate(solution) if bit]
        if not self._is_matching(selected):
            selected = _repair_matching(selected, self._edges, self._weights)
            solution = [0] * len(self._edges)
            for i in selected:
                solution[i] = 1

        if self._use_classical_cleanup:
            solution = _classical_cleanup(solution, self._edges, self._matching_weights)
            selected = [i for i, bit in enumerate(solution) if bit]

        return self._matching_and_weight(selected)

    def postprocess_candidates(
        self, candidates: list[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[list[tuple], float]]:
        """Post-process matching candidates, optionally hard-filtering invalid ones.

        With ``strict=False``, invalid raw candidates are repaired and may be
        improved by classical cleanup. With ``strict=True``, invalid raw
        candidates are discarded before repair or cleanup.
        """
        formatted = []
        invalid_seen = False
        for _score, solution in candidates:
            selected = [i for i, bit in enumerate(solution) if bit]
            valid = self._is_matching(selected)
            if strict:
                if valid:
                    formatted.append(self._matching_and_weight(selected))
            else:
                invalid_seen = invalid_seen or not valid
                formatted.append(self._postprocess_solution(solution))

        if strict and not formatted:
            warnings.warn(
                "No valid matching candidates found under strict=True. "
                "Consider widening the aggregation strategy parameters, "
                "or running with strict=False to inspect repaired output.",
                UserWarning,
                stacklevel=2,
            )
        if invalid_seen:
            warnings.warn(
                "At least one partition aggregate was not a valid matching "
                "and was repaired. Use get_top_solutions(..., strict=True) "
                "to discard invalid raw candidates instead.",
                UserWarning,
                stacklevel=2,
            )

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
