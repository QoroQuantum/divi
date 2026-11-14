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
from typing import Any, Literal

import networkx as nx
import numpy as np
import rustworkx as rx

from ._base import QAOAProblem, place_by_coupling
from ._binary import BinaryOptimizationProblem
from ._graph_hamiltonians import _read_edge_weight
from ._graph_partitioning_utils import _canonical_edges, _partition_graph_by_nodes
from ._partitioning_config import GraphPartitioningConfig

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
    follow :func:`_canonical_edges` and default to 1; parallel edges collapse
    into one carrying the last weight.

    Raises:
        TypeError: If ``graph`` is directed or not a graph.
        ValueError: If an edge weight is NaN or infinite.
    """
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) or not isinstance(
        graph, (nx.Graph, rx.PyGraph)
    ):
        raise TypeError(
            f"Expected an undirected graph (nx.Graph or rx.PyGraph), got "
            f"{type(graph).__name__}."
        )
    ids, weighted = _canonical_edges(graph)
    edges = [(u, v) for u, v, _ in weighted]
    weights = [_read_edge_weight(weight) for _, _, weight in weighted]
    return ids, edges, weights


def _construct_matching_qubo(
    edges: Sequence[_Edge],
    weights: Sequence[float],
    penalty_weight: float = 10.0,
) -> dict[tuple[Hashable, ...], float]:
    """QUBO terms encoding maximum-weight matching over edge qubits ``0..len(edges)-1``.

    Linear terms ``-w_e`` reward edge weight. Each pair of edges sharing a node
    costs ``penalty_weight * max|w|`` (``penalty_weight`` when every weight is
    0); with ``penalty_weight > 1`` every selection with a shared node costs
    more than the same selection without one of its edges.

    Args:
        edges: Edges as node pairs; qubit ``i`` is ``edges[i]``.
        weights: Weight of each edge.
        penalty_weight: Multiplier for the penalty strength.

    Returns:
        Polynomial terms ``{(i,): -w_i, (i, j): penalty}`` with ``i < j``.
    """
    penalty = penalty_weight * (max(map(abs, weights), default=0.0) or 1.0)
    terms: dict[tuple[Hashable, ...], float] = {(i,): -w for i, w in enumerate(weights)}
    incident: defaultdict[int, list[int]] = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        incident[u].append(i)
        incident[v].append(i)
    for qubits in incident.values():
        terms.update((pair, penalty) for pair in combinations(qubits, 2))
    return terms


def is_valid_matching(edges: Sequence[tuple]) -> bool:
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

    Raises:
        ValueError: If ``bitstring`` does not hold one bit per edge.
    """
    if len(bitstring) != len(edges):
        raise ValueError(
            f"Expected a bitstring of {len(edges)} bits, one per edge, "
            f"got {len(bitstring)}."
        )
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


def _conflict_graph(edges: Sequence[_Edge], weights: Sequence[float]) -> rx.PyGraph:
    """Graph on edge indices joining two positive-weight edges that share a node,
    weighted by the lighter of the two.

    Non-positive edges, which no maximum-weight matching needs, join nothing.
    """
    graph = rx.PyGraph()
    graph.add_nodes_from(range(len(edges)))
    incident: defaultdict[int, list[int]] = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        if weights[i] > 0:
            incident[u].append(i)
            incident[v].append(i)
    graph.add_edges_from(
        [
            (i, j, min(weights[i], weights[j]))
            for indices in incident.values()
            for i, j in combinations(indices, 2)
        ]
    )
    return graph


def _partition_edges(
    conflicts: rx.PyGraph,
    max_edges: int,
    algorithm: Literal["spectral", "metis", "kernighan_lin"] = "spectral",
) -> list[list[int]]:
    """Group the edges joined in the :func:`_conflict_graph` ``conflicts`` into
    partitions of at most ``max_edges`` edges.

    Partitions ``conflicts`` with :func:`_partition_graph_by_nodes`, cutting the
    lightest conflicts. Every joined edge lands in exactly one partition;
    conflicts between partitions are left to aggregation.

    Returns:
        Ascending edge indices of each partition, ordered by first index; empty
        when ``conflicts`` has no edges.

    Raises:
        ValueError: If ``algorithm`` is unsupported.
    """
    config = GraphPartitioningConfig(
        max_n_nodes_per_cluster=max_edges, partitioning_algorithm=algorithm
    )
    if conflicts.num_edges() == 0:
        return []
    clusters = _partition_graph_by_nodes(conflicts, config, merge_edgeless=False)
    return sorted(sorted(ids) for _sub, ids in clusters)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


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


def _fill_unmatched_nodes(
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


class _MatchingPartition(BinaryOptimizationProblem):
    """The matching QUBO of one partition's edges, whose candidates aggregation
    repairs into matchings and ranks by weight."""

    candidate_feasibility = "repair"

    def __init__(
        self, edges: Sequence[_Edge], weights: Sequence[float], penalty_weight: float
    ):
        super().__init__(_construct_matching_qubo(edges, weights, penalty_weight))
        self._edges = list(edges)
        self._weights = list(weights)

    def _selected(self, bitstring: str) -> list[int]:
        return [i for i, bit in enumerate(bitstring) if bit == "1"]

    def is_feasible(self, bitstring: str) -> bool:
        """Whether the selected edges form a matching."""
        return is_valid_matching([self._edges[i] for i in self._selected(bitstring)])

    def compute_energy(self, bitstring: str) -> float:
        """Negated weight of the selected edges."""
        return -math.fsum(self._weights[i] for i in self._selected(bitstring))

    def repair_infeasible_bitstring(self, bitstring: str) -> tuple[str, Any, float]:
        """The matching kept by :func:`_repair_matching`, its decoding and energy."""
        kept = set(
            _repair_matching(self._selected(bitstring), self._edges, self._weights)
        )
        repaired = "".join("1" if i in kept else "0" for i in range(len(bitstring)))
        return repaired, self.decode_fn(repaired), self.compute_energy(repaired)


# ------------------------------------------------------------------
# MaxWeightMatchingProblem
# ------------------------------------------------------------------


class MaxWeightMatchingProblem(QAOAProblem):
    """Maximum-weight matching problem for QAOA.

    Given a weighted graph, finds a set of edges (matching) that maximises
    total weight while ensuring no two selected edges share a node.

    Can be used directly with :class:`~divi.qprog.algorithms.QAOA` for
    small graphs, or with
    :class:`~divi.qprog.ensembles.PartitioningProgramEnsemble` for large
    graphs via edge-based partitioning: every positive-weight edge that shares
    a node with another belongs to exactly one partition, and conflicts between
    partitions are resolved when their solutions are combined. Each
    partition's candidates are repaired into matchings and ranked by weight,
    and combinations are scored by their weight after the same repair. Edges no
    partition solves are added to every combination, and non-positive edges
    are never selected.

    Qubit ``i`` is the ``i``-th edge in ascending order of its endpoints'
    node positions. A decoded matching lists each edge as a pair of node
    labels for ``nx.Graph`` and node indices for ``rx.PyGraph``.

    Args:
        graph: Weighted undirected graph, ``nx.Graph`` or ``rx.PyGraph``.
            ``rx.PyGraph`` nodes are identified by node index, and an edge payload
            that is a number, or a dict with a ``"weight"`` entry, is its
            weight; other edges weigh 1.
        penalty_weight: Cost of two selected edges sharing a node, as a
            multiple of the largest edge weight magnitude. Must be greater
            than 1, so that every shared node costs more than any edge earns.
        max_edges_per_partition: Maximum edges per partition, at least 2.
            Setting this enables :meth:`decompose` for partitioned solving.
        partition_algorithm: Algorithm that splits the edges, cutting the
            lightest conflicts: ``"spectral"`` (default), ``"metis"`` or
            ``"kernighan_lin"``, as in
            :class:`~divi.qprog.problems.GraphPartitioningConfig`.
        use_classical_cleanup: If ``True`` (default), fill unmatched
            residual nodes via :func:`~rustworkx.max_weight_matching` during
            :meth:`postprocess_candidates`. It runs on integer weights scaled
            so the heaviest edge has 100 bits: weights down to ``2**-47`` of
            the heaviest are exact and lighter ones round to a step of at most
            ``2**-99`` of it.

    Raises:
        TypeError: If ``graph`` is directed.
        ValueError: If the graph has a self-loop or a non-finite edge weight,
            ``penalty_weight`` is not a finite number above 1, or
            ``max_edges_per_partition`` is below 2.

    Example::

        from divi.qprog.problems import MaxWeightMatchingProblem
        from divi.qprog.algorithms import QAOA
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
        partition_algorithm: Literal["spectral", "metis", "kernighan_lin"] = "spectral",
        use_classical_cleanup: bool = True,
    ):
        self._input_graph = graph
        node_ids, self._edges, self._weights = _matching_edges(graph)
        if any(u == v for u, v in self._edges):
            raise ValueError("Graph contains self-loops")
        penalty_weight = float(penalty_weight)
        if not math.isfinite(penalty_weight) or penalty_weight <= 1:
            raise ValueError(
                f"penalty_weight must be a finite number above 1, got {penalty_weight}."
            )
        if max_edges_per_partition is not None and max_edges_per_partition < 2:
            raise ValueError(
                "max_edges_per_partition must be at least 2, got "
                f"{max_edges_per_partition}."
            )
        self._penalty_weight = penalty_weight
        self._max_edges_per_partition = max_edges_per_partition
        self._partition_algorithm = partition_algorithm
        self._use_classical_cleanup = use_classical_cleanup

        self._labelled_edges = [(node_ids[u], node_ids[v]) for u, v in self._edges]

        self._bop = BinaryOptimizationProblem(
            _construct_matching_qubo(self._edges, self._weights, penalty_weight)
        )

        self._edge_index_maps: dict[Hashable, list[int]] = {}
        self._preset_edges: list[int] = []

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
    def decode_fn(self) -> Callable[[str], list[tuple]]:
        return partial(_bitstring_to_matching, edges=self._labelled_edges)

    @property
    def graph(self) -> nx.Graph | rx.PyGraph:
        """The input graph.

        Treat as read-only: edge weights are read into cached state at
        construction. Changing edge weights or structure afterwards will not
        update the cached conflict penalty, so ``evaluate_global_solution``
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

        parts = _partition_edges(
            self._conflicts,
            max_edges=self._max_edges_per_partition,
            algorithm=self._partition_algorithm,
        )
        if not parts:
            raise ValueError(
                "Cannot decompose: no two positive-weight edges share a node, so "
                "the best matching holds every positive-weight edge. Solve the "
                "problem without partitioning."
            )

        conflicted = [
            indices
            for indices in parts
            if self._conflicts.subgraph(indices).num_edges() > 0
        ]
        solved = {i for indices in conflicted for i in indices}
        self._preset_edges = [
            i
            for i, weight in enumerate(self._weights)
            if weight > 0 and i not in solved
        ]
        self._edge_index_maps = {}
        sub_problems: dict[Hashable, QAOAProblem] = {}
        for i, indices in enumerate(conflicted):
            prog_id = (f"P{i}", len(indices))
            self._edge_index_maps[prog_id] = indices
            sub_problems[prog_id] = _MatchingPartition(
                [self._edges[j] for j in indices],
                [self._weights[j] for j in indices],
                self._penalty_weight,
            )

        return sub_problems

    def order_partitions(self, prog_ids: Sequence[Hashable]) -> list[Hashable]:
        """Partitions ordered so each next one shares the most conflict weight
        with those already placed."""
        groups = [self._edge_index_maps[pid] for pid in prog_ids]
        return place_by_coupling(prog_ids, groups, self._conflicts.weighted_edge_list())

    @cached_property
    def _conflicts(self) -> rx.PyGraph:
        return _conflict_graph(self._edges, self._weights)

    def initial_solution_size(self) -> int:
        return len(self._edges)

    def extend_solution(
        self,
        current_solution: Sequence[int],
        prog_id: Hashable,
        candidate_decoded: Sequence[int],
    ) -> list[int]:
        extended = list(current_solution)
        global_indices = self._edge_index_maps[prog_id]
        for local_idx, global_idx in enumerate(global_indices):
            extended[global_idx] = int(candidate_decoded[local_idx])
        return extended

    def _complete(self, selected: Sequence[int]) -> list[int]:
        """``selected`` plus every preset edge, repaired greedily by weight when
        two share a node."""
        combined = sorted(set(selected).union(self._preset_edges))
        if self._is_matching(combined):
            return combined
        return _repair_matching(combined, self._edges, self._weights)

    @cached_property
    def _matching_weights(self) -> list[int]:
        """Edge weights as integers for the classical cleanup."""
        return _integer_weights(self._weights)

    def evaluate_global_solution(self, solution: Sequence[int]) -> float:
        """Negated weight of the matching :meth:`postprocess_candidates` makes of
        ``solution`` before classical cleanup. Lower is better; a solution the
        repair changes scores ``1e-9 * max|w|`` worse per dropped edge, so
        valid solutions win ties."""
        raw = [i for i, bit in enumerate(solution) if bit]
        selected = self._complete(raw)
        dropped = len(set(raw).union(self._preset_edges)) - len(selected)
        return -math.fsum(self._weights[i] for i in selected) + dropped * self._tie_step

    @cached_property
    def _tie_step(self) -> float:
        return 1e-9 * (max(map(abs, self._weights), default=0.0) or 1.0)

    def _postprocess_solution(self, solution: list[int]) -> tuple[list[tuple], float]:
        """Complete and repair ``solution``, then apply the classical cleanup."""
        selected = self._complete([i for i, bit in enumerate(solution) if bit])
        if self._use_classical_cleanup:
            bits = [0] * len(self._edges)
            for i in selected:
                bits[i] = 1
            bits = _fill_unmatched_nodes(bits, self._edges, self._matching_weights)
            selected = [i for i, bit in enumerate(bits) if bit]
        return self._matching_and_weight(selected)

    def postprocess_candidates(
        self, candidates: Sequence[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[list[tuple], float]]:
        """Post-process matching candidates, optionally hard-filtering invalid ones.

        Every candidate gains the positive-weight edges no partition solves,
        keeping the heaviest edge at each shared node. With ``strict=False``,
        invalid raw candidates are repaired and may be improved by classical
        cleanup. With ``strict=True``, invalid raw candidates are discarded
        before repair or cleanup.
        """
        formatted = []
        invalid_seen = False
        for _score, solution in candidates:
            selected = [i for i, bit in enumerate(solution) if bit]
            valid = self._is_matching(selected)
            if strict:
                if valid:
                    formatted.append(
                        self._matching_and_weight(self._complete(selected))
                    )
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
