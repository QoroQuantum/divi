# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
import sys
from collections.abc import Callable, Sequence
from typing import Literal
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse as sps
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from sklearn.cluster import SpectralClustering

from divi._optional import optional_module
from divi.qprog import GraphProblemTypes
from divi.qprog.problems._graph_hamiltonians import _edge_weight, _wire_edges
from divi.qprog.problems._partitioning_config import GraphPartitioningConfig

# TODO: Make this dynamic through an interaction with usher
# once a proper endpoint is exposed
_MAXIMUM_AVAILABLE_QUBITS = 30


_WeightedEdge = tuple[int, int, float | None]


def _canonical_edges(graph: GraphProblemTypes) -> tuple[list, list[_WeightedEdge]]:
    """Node ids and the undirected edges as sorted ``(u, v, weight)``, ``u <= v``.

    Endpoints are positions into the node ids, so a rustworkx graph and the
    networkx graph labelled by its node indices give identical output. Weights
    follow :func:`_edge_weight`, and parallel rustworkx edges collapse into one
    carrying the last weight, as repeated ``add_edge`` calls do in networkx.
    Weight data that is present but not a real number emits a ``UserWarning``
    and leaves its edge unweighted.
    """
    ids, wire_edges = _wire_edges(graph)
    edges: list[_WeightedEdge] = []
    unusable: set[str] = set()
    for u, v, payload in wire_edges:
        weight = _edge_weight(payload)
        raw = payload.get("weight") if isinstance(payload, dict) else payload
        if weight is None and raw is not None:
            unusable.add(type(raw).__name__)
        edges.append((min(u, v), max(u, v), weight))
    if unusable:
        warn(
            f"Edge weights of type(s) {sorted(unusable)} are not real numbers; "
            "graph partitioning treats those edges as unweighted.",
            UserWarning,
            stacklevel=3,
        )
    edges.sort(key=lambda edge: edge[:2])
    return ids, edges


def _spectral_inputs(graph: GraphProblemTypes) -> tuple[sps.csr_matrix, list]:
    """Return ``(adjacency_matrix, node_ids)`` for spectral clustering.

    Row ``i`` of the CSR adjacency matrix corresponds to ``node_ids[i]``; an
    unweighted edge counts ``1`` and a self-loop sits once on the diagonal.
    """
    node_ids, edges = _canonical_edges(graph)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, weight in edges:
        value = 1.0 if weight is None else weight
        rows.append(u)
        cols.append(v)
        data.append(value)
        if u != v:
            rows.append(v)
            cols.append(u)
            data.append(value)
    n_nodes = len(node_ids)
    adj_matrix = sps.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    return adj_matrix, node_ids


def _metis_inputs(graph: GraphProblemTypes) -> tuple[list[list[int]], list]:
    """Return ``(adjacency_list, node_ids)`` for METIS.

    The adjacency list is keyed by positions ``0..N-1`` with each neighbour list
    ascending; ``node_ids`` maps positions back to the graph's node ids.
    """
    node_ids, edges = _canonical_edges(graph)
    adj_list: list[list[int]] = [[] for _ in node_ids]
    for u, v, _weight in edges:
        adj_list[u].append(v)
        if u != v:
            adj_list[v].append(u)
    return adj_list, node_ids


def _kernighan_lin_input(graph: GraphProblemTypes) -> tuple[nx.Graph, list]:
    """Position-labelled networkx graph for Kernighan–Lin, with node ids."""
    node_ids, edges = _canonical_edges(graph)
    kl_graph = nx.Graph()
    kl_graph.add_nodes_from(range(len(node_ids)))
    kl_graph.add_edges_from(
        (u, v, {} if weight is None else {"weight": weight}) for u, v, weight in edges
    )
    return kl_graph, node_ids


def _relabeled_subgraph_with_ids(
    graph: GraphProblemTypes, cluster: list
) -> tuple[GraphProblemTypes, list]:
    """Build a ``0..M-1``-indexed subgraph for ``cluster`` and return it
    alongside the original IDs.  ``cluster_ids[i]`` is the parent-graph
    identifier for the subgraph's local node ``i``; local nodes follow the
    parent's node order for both graph types.
    """
    if isinstance(graph, rx.PyGraph):
        sub, node_map = graph.subgraph_with_nodemap(list(cluster))
        return sub, [node_map[i] for i in range(len(node_map))]
    position = {node: i for i, node in enumerate(graph)}
    ordered = sorted(cluster, key=position.__getitem__)
    local = {node: i for i, node in enumerate(ordered)}
    sub = graph.__class__()
    sub.add_nodes_from((i, dict(graph.nodes[node])) for i, node in enumerate(ordered))
    sub.add_edges_from(
        (local[u], local[v], dict(data))
        for u, v, data in graph.edges(ordered, data=True)
        if v in local
    )
    return sub, ordered


_SubgraphWithIds = tuple[GraphProblemTypes, list]


def _apply_split_with_relabel(
    graph: GraphProblemTypes,
    algorithm: Literal["spectral", "metis"],
    n_clusters: int,
) -> tuple[_SubgraphWithIds, ...]:
    """Spectral or METIS split.

    Returns a tuple of ``(relabeled_subgraph, cluster_ids)`` pairs where
    each subgraph is locally indexed ``0..M-1`` and ``cluster_ids[i]``
    is the parent-graph identifier for local node ``i``.
    """
    if algorithm == "spectral":
        adj_matrix, node_ids = _spectral_inputs(graph)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            n_init=100,
            assign_labels="discretize",
            random_state=0,
        )
        parts = sc.fit_predict(adj_matrix)
    elif algorithm == "metis":
        if sys.platform == "win32" and optional_module("pymetis") is None:
            raise ImportError(
                "The 'metis' partitioning algorithm needs pymetis, which is not "
                "installed on Windows by default; install it via conda: "
                "conda install -c conda-forge pymetis. Otherwise use 'spectral' "
                "or 'kernighan_lin' instead."
            )
        from pymetis import part_graph

        adj_list, node_ids = _metis_inputs(graph)
        _, parts = part_graph(n_clusters, adjacency=adj_list)
    else:
        raise RuntimeError("Relabeling only needed for `spectral` and `metis`.")

    clusters: list[list] = [[] for _ in range(n_clusters)]
    for idx, part in enumerate(parts):
        clusters[part].append(node_ids[idx])

    non_empty_clusters = [clstr for clstr in clusters if clstr]
    if len(non_empty_clusters) != n_clusters:
        warn(
            f"_apply_split_with_relabel: {algorithm!r} requested {n_clusters} "
            f"clusters but produced {len(non_empty_clusters)} non-empty "
            "cluster(s); empty clusters were dropped from the result.",
            UserWarning,
            stacklevel=2,
        )

    return tuple(
        _relabeled_subgraph_with_ids(graph, clstr) for clstr in non_empty_clusters
    )


def _split_graph(
    graph: GraphProblemTypes, partitioning_config: GraphPartitioningConfig
) -> Sequence[_SubgraphWithIds]:
    """
    Splits a graph.

    If the requested partitioning algorithm is either "spectral" or "metis",
    then the requested `min_n_clusters` will be returned.
    For "kernighan_lin", a bisection will be returned

    Args:
        graph: The input graph to be partitioned (``nx.Graph`` or ``rx.PyGraph``).
        partitioning_config (GraphPartitioningConfig): The configuration to follow.

    Returns:
        Sequence of ``(relabeled_subgraph, cluster_ids)`` tuples.  Each
        subgraph is locally indexed ``0..M-1``; ``cluster_ids[i]`` is the
        parent-graph identifier for local node ``i``.
    """
    if (algorithm := partitioning_config.partitioning_algorithm) in (
        "spectral",
        "metis",
    ):
        return _apply_split_with_relabel(
            graph,
            algorithm,
            # If minimum clusters isn't a constraint, then default to bisection
            partitioning_config.minimum_n_clusters or 2,
        )
    elif partitioning_config.partitioning_algorithm == "kernighan_lin":
        kl_graph, node_ids = _kernighan_lin_input(graph)
        return tuple(
            _relabeled_subgraph_with_ids(graph, [node_ids[i] for i in sorted(part)])
            for part in nx.algorithms.community.kernighan_lin_bisection(
                kl_graph, seed=0
            )
        )
    else:
        raise ValueError(
            f"Unsupported partitioning algorithm: "
            f"{partitioning_config.partitioning_algorithm!r}."
        )


def _num_edges(graph: GraphProblemTypes) -> int:
    """Edge count for either supported graph type."""
    if isinstance(graph, rx.PyGraph):
        return graph.num_edges()
    return graph.number_of_edges()


def _merge_edgeless_clusters(
    graph: GraphProblemTypes,
    clusters: Sequence[_SubgraphWithIds],
    partitioning_config: GraphPartitioningConfig,
) -> list[_SubgraphWithIds]:
    """Fold clusters with no internal edges into a neighbouring cluster.

    Recursive bisection of a dense graph peels off lone vertices to satisfy the
    size constraints.  Such a cluster carries no intra-cluster objective, and
    for cut-style problems its Hamiltonian is constant, which problem
    construction rejects.  Each one is merged into the cluster it has the most
    connecting edges to, restricted to clusters with room under
    ``max_n_nodes_per_cluster``; ties break towards the smaller cluster, then
    the earlier one, so the result stays deterministic.

    Args:
        graph: The original graph the clusters were partitioned from.
        clusters: ``(relabeled_subgraph, cluster_ids)`` pairs to repair.
        partitioning_config: The configuration the partition was built under.

    Returns:
        The repaired cluster list, relabeled ``0..M-1`` as on input.

    Raises:
        ValueError: If every cluster is edgeless, or no cluster has room for an
            edgeless one under ``max_n_nodes_per_cluster``.
    """
    edgeless_positions = [
        i for i, (sub, _ids) in enumerate(clusters) if _num_edges(sub) == 0
    ]
    if not edgeless_positions:
        return list(clusters)

    target_ids: dict[int, list] = {
        i: list(ids) for i, (sub, ids) in enumerate(clusters) if _num_edges(sub) != 0
    }
    if not target_ids:
        raise ValueError(
            "Cannot repair the partition: every cluster is edgeless, so there is "
            "no cluster with internal edges to merge into. The graph has too few "
            "edges for the requested partitioning."
        )

    cap = partitioning_config.max_n_nodes_per_cluster

    for position in edgeless_positions:
        orphan_ids = list(clusters[position][1])
        candidates = [
            i
            for i, ids in target_ids.items()
            if cap is None or len(ids) + len(orphan_ids) <= cap
        ]
        if not candidates:
            raise ValueError(
                f"Cannot merge edgeless cluster {sorted(orphan_ids)}: no cluster "
                f"has room under 'max_n_nodes_per_cluster'={cap}. The graph is "
                "too dense to split this small — raise "
                "'max_n_nodes_per_cluster' or lower 'minimum_n_clusters'."
            )

        # Prefer the most strongly connected target, then the smallest, then the
        # earliest, keeping the choice independent of dict iteration accidents.
        scored = [
            (
                -sum(
                    len(set(graph.neighbors(node)) & set(target_ids[i]))
                    for node in orphan_ids
                ),
                len(target_ids[i]),
                i,
            )
            for i in candidates
        ]
        best = min(scored)[2]
        target_ids[best] = target_ids[best] + orphan_ids

    merged = [
        (
            clusters[i]
            if ids == list(clusters[i][1])
            else _relabeled_subgraph_with_ids(graph, ids)
        )
        for i, ids in target_ids.items()
    ]

    minimum = partitioning_config.minimum_n_clusters
    if minimum is not None and len(merged) < minimum:
        warn(
            f"Merging edgeless clusters left {len(merged)} cluster(s), below the "
            f"requested 'minimum_n_clusters'={minimum}.",
            UserWarning,
            stacklevel=2,
        )

    return merged


HeapEntry = tuple[int, int, GraphProblemTypes, list]


def _bisect_with_predicate(
    initial_partitions: list[HeapEntry],
    predicate: Callable[[GraphProblemTypes, Sequence[HeapEntry]], bool],
    partitioning_config: GraphPartitioningConfig,
) -> list[HeapEntry]:
    """
    Recursively bisects a list of graph partitions based on a user-defined predicate.

    This helper function repeatedly applies a partitioning strategy to a sequence of graph
    subgraphs. At each iteration, it evaluates a predicate to determine whether a subgraph
    should be further split. The process continues until no subgraphs satisfy the predicate,
    at which point the resulting collection of subgraphs is returned.

    The predicate is expected to accept two arguments:
        - The current subgraph under consideration.
        - A list of the other heap entries in the current iteration (both previously
          processed and yet to be processed), serving as the context for the decision.

    Returns the final list of subgraphs as a heapified sequence, ordered by descending
    node count.
    """
    subgraphs: list[HeapEntry] = initial_partitions
    heapq.heapify(subgraphs)
    # Strictly monotonic — breaks heap ties before falling through to
    # the graph object or cluster_ids list.
    entry_counter = (
        max((counter for _, counter, _, _ in initial_partitions), default=-1) + 1
    )

    while True:
        new_subgraphs: list[HeapEntry] = []
        changed = False

        while subgraphs:
            entry = heapq.heappop(subgraphs)
            _, _, subgraph, parent_ids = entry

            if predicate(subgraph, new_subgraphs + subgraphs):
                for child, child_local_ids in _split_graph(
                    subgraph, partitioning_config
                ):
                    child_global_ids = [parent_ids[i] for i in child_local_ids]
                    new_subgraphs.append(
                        (-len(child), entry_counter, child, child_global_ids)
                    )
                    entry_counter += 1
                changed = True
            else:
                new_subgraphs.append(entry)

        subgraphs = new_subgraphs
        heapq.heapify(subgraphs)

        if not changed:
            break

    return subgraphs


def _node_partition_graph(
    graph: GraphProblemTypes, partitioning_config: GraphPartitioningConfig
) -> list[_SubgraphWithIds]:
    """Partition ``graph`` into subgraphs honouring the configured constraints.

    Returns a list of ``(relabeled_subgraph, cluster_ids)`` pairs.  Each
    subgraph is locally indexed ``0..M-1``; ``cluster_ids[i]`` is the
    original-graph identifier for local node ``i``.
    """
    n_nodes = len(graph)
    # Splits return local positions, so the root must be indexed 0..N-1 too.
    positional: GraphProblemTypes
    if isinstance(graph, rx.PyGraph):
        initial_ids: list = list(graph.node_indexes())
        positional = graph.subgraph(initial_ids)
    else:
        initial_ids = list(graph.nodes())
        positional = nx.convert_node_labels_to_integers(graph)
    subgraphs: list[HeapEntry] = [(-n_nodes, 0, positional, initial_ids)]

    if partitioning_config.minimum_n_clusters:
        if partitioning_config.minimum_n_clusters >= n_nodes:
            raise ValueError(
                "Number of requested clusters must be smaller than the size of "
                "the graph, since every cluster needs at least one edge."
            )

        subgraphs = _bisect_with_predicate(
            subgraphs,
            lambda _, subgraphs: len(subgraphs)
            < partitioning_config.minimum_n_clusters - 1,
            partitioning_config,
        )

    if partitioning_config.max_n_nodes_per_cluster:
        subgraphs = _bisect_with_predicate(
            subgraphs,
            lambda subgraph, _: (
                len(subgraph) > partitioning_config.max_n_nodes_per_cluster
            ),
            partitioning_config,
        )

    # Repair before the budget check: merging changes cluster sizes.
    clusters = _merge_edgeless_clusters(
        graph,
        [(sub, ids) for (_, _, sub, ids) in subgraphs],
        partitioning_config,
    )

    if any(len(sub) > _MAXIMUM_AVAILABLE_QUBITS for sub, _ids in clusters):
        warn(
            "At least one cluster has more nodes than what can be executed on "
            f"the available backends: {_MAXIMUM_AVAILABLE_QUBITS} qubits."
        )

    return clusters


def _drawable(graph: GraphProblemTypes) -> nx.Graph:
    """networkx view for plotting; a rustworkx graph is keyed by node index."""
    if isinstance(graph, rx.PyGraph):
        view = nx.Graph()
        view.add_nodes_from(graph.node_indexes())
        view.add_edges_from(graph.edge_list())
        return view
    return graph


def draw_partitions(
    graph: GraphProblemTypes,
    reverse_index_maps: dict,
    pos: dict | None = None,
    figsize: tuple[int, int] | None = (10, 8),
    node_size: int = 300,
):
    """Draw a graph with nodes coloured by partition.

    Args:
        graph: The full graph. RustworkX nodes are drawn and labelled by node
            index.
        reverse_index_maps: Mapping ``{prog_id: {local_idx: global_node}}``,
            as built by ``_GraphProblemBase.decompose()``.
        pos: Node positions.  If *None*, uses spring layout.
        figsize: Figure size ``(width, height)``.
        node_size: Size of nodes.
    """
    if not reverse_index_maps:
        raise RuntimeError("There are no partitions to draw. Did you call decompose()?")
    graph = _drawable(graph)

    node_to_partition = {}
    for (partition_id, _), mapping in reverse_index_maps.items():
        for node in mapping.values():
            node_to_partition[node] = partition_id

    unique_partitions = sorted(set(node_to_partition.values()))
    n_partitions = len(unique_partitions)
    colors = colormaps["Set3"](np.linspace(0, 1, n_partitions))
    partition_colors = {pid: colors[i] for i, pid in enumerate(unique_partitions)}

    unassigned_colour = to_rgba("dimgray")
    node_colors = [
        (
            partition_colors[node_to_partition[node]]
            if node in node_to_partition
            else unassigned_colour
        )
        for node in graph.nodes()
    ]

    if pos is None:
        pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=figsize)
    nx.draw(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        with_labels=True,
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        alpha=0.8,
    )

    legend_entries = [
        (partition_colors[pid], f"Partition {pid}") for pid in unique_partitions
    ]
    if any(node not in node_to_partition for node in graph.nodes()):
        legend_entries.append((unassigned_colour, "Unassigned"))
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colour,
            markersize=10,
            label=label,
        )
        for colour, label in legend_entries
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.title("Graph Partitions Visualization")
    plt.axis("off")
    plt.show()
