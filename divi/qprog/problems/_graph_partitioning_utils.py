# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
import math
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

from divi._optional import import_if_available

from ._graph_hamiltonians import _edge_weight, _read_edge_weight, _wire_edges
from ._partitioning_config import GraphPartitioningConfig
from ._qubo_partitioning_utils import _merge_clusters_to_budget
from ._types import GraphProblemTypes

_WeightedEdge = tuple[int, int, float | None]


def _canonical_edges(graph: GraphProblemTypes) -> tuple[list, list[_WeightedEdge]]:
    """Node ids and the undirected edges as sorted ``(u, v, weight)``, ``u <= v``.

    Endpoints are positions into the node ids, so a rustworkx graph and the
    networkx graph labelled by its node indices give identical output. Weights
    follow :func:`_edge_weight`, and parallel rustworkx edges collapse into one
    carrying the last weight, as repeated ``add_edge`` calls do in networkx.
    Weight data that is present but not a real number emits a ``UserWarning``
    and leaves its edge unweighted.

    Raises:
        ValueError: If an edge weight is NaN or infinite.
    """
    ids, wire_edges = _wire_edges(graph)
    edges: list[_WeightedEdge] = []
    unusable: set[str] = set()
    for u, v, payload in wire_edges:
        weight = _edge_weight(payload)
        if weight is not None and not math.isfinite(weight):
            raise ValueError(
                f"Edge ({ids[u]!r}, {ids[v]!r}) has non-finite weight {weight}; "
                "edge weights must be finite."
            )
        raw = payload.get("weight") if isinstance(payload, dict) else payload
        if weight is None and raw is not None:
            unusable.add(type(raw).__name__)
        edges.append((min(u, v), max(u, v), weight))
    if unusable:
        warn(
            f"Edge weights of type(s) {sorted(unusable)} are not real numbers; "
            "those edges are treated as unweighted.",
            UserWarning,
            stacklevel=3,
        )
    edges.sort(key=lambda edge: edge[:2])
    return ids, edges


_SPECTRAL_ZERO_COUPLING = 1e-9


def _build_spectral_inputs(graph: GraphProblemTypes) -> tuple[sps.csr_matrix, list]:
    """Return ``(adjacency_matrix, node_ids)`` for spectral clustering.

    Row ``i`` of the CSR adjacency matrix corresponds to ``node_ids[i]``; each
    entry is the edge's coupling strength (``|weight|``, ``1`` when unweighted), a
    zero coupling counting ``1e-9`` of the strongest one, and a self-loop sits
    once on the diagonal.
    """
    node_ids, edges = _canonical_edges(graph)
    couplings = [abs(_read_edge_weight(weight)) for _, _, weight in edges]
    floor = _SPECTRAL_ZERO_COUPLING * max(couplings, default=0.0) or 1.0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (u, v, _weight), coupling in zip(edges, couplings):
        value = coupling or floor
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


_METIS_MEDIAN_WEIGHT = 1000
_METIS_MAX_WEIGHT = 1_000_000
_METIS_INT_MAX = 2**31 - 1


def _build_metis_inputs(
    graph: GraphProblemTypes,
) -> tuple[list[int], list[int], list[int] | None, list]:
    """Return ``(xadj, adjncy, eweights, node_ids)`` for METIS.

    ``xadj``/``adjncy`` hold the CSR adjacency over positions ``0..N-1``, each
    neighbour run ascending; self-loops are left out, as they never cross a cut.
    ``eweights`` holds each entry's coupling strength (``|weight|``, ``1`` when unweighted)
    as a positive integer, as METIS requires: the median non-zero coupling maps
    to 1000, and weights are clipped so that their sum fits METIS's 32-bit
    integers. It is ``None`` when the graph is unweighted or every coupling is
    zero. ``node_ids`` maps positions back to node ids.
    """
    node_ids, edges = _canonical_edges(graph)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    for u, v, weight in edges:
        if u != v:
            neighbours[u].append((v, abs(_read_edge_weight(weight))))
            neighbours[v].append((u, abs(_read_edge_weight(weight))))
    xadj, adjncy, couplings = [0], [], []
    for row in neighbours:
        for v, coupling in sorted(row):
            adjncy.append(v)
            couplings.append(coupling)
        xadj.append(len(adjncy))
    positive = [c for c in couplings if c > 0.0]
    if not positive or all(weight is None for _, _, weight in edges):
        return xadj, adjncy, None, node_ids
    scale = _METIS_MEDIAN_WEIGHT / float(np.median(positive))
    ceiling = min(_METIS_MAX_WEIGHT, _METIS_INT_MAX // len(couplings))
    return (
        xadj,
        adjncy,
        [max(1, round(min(c * scale, ceiling))) for c in couplings],
        node_ids,
    )


def _kernighan_lin_input(graph: GraphProblemTypes) -> tuple[nx.Graph, list]:
    """Position-labelled networkx graph for Kernighan–Lin, weighted by coupling
    strength (``|weight|``, ``1`` when unweighted), with node ids."""
    node_ids, edges = _canonical_edges(graph)
    kl_graph = nx.Graph()
    kl_graph.add_nodes_from(range(len(node_ids)))
    kl_graph.add_edges_from(
        (u, v, {"weight": abs(_read_edge_weight(weight))}) for u, v, weight in edges
    )
    return kl_graph, node_ids


def _build_relabeled_subgraph_with_ids(
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
        adj_matrix, node_ids = _build_spectral_inputs(graph)
        n_clusters = min(n_clusters, len(node_ids) - 1)
        if n_clusters < 2:
            n_clusters, parts = 1, [0] * len(node_ids)
        else:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels="discretize",
                random_state=0,
            )
            parts = sc.fit_predict(adj_matrix)
    elif algorithm == "metis":
        if sys.platform == "win32" and import_if_available("pymetis") is None:
            raise ImportError(
                "The 'metis' partitioning algorithm needs pymetis, which is not "
                "installed on Windows by default; install it via conda: "
                "conda install -c conda-forge pymetis. Otherwise use 'spectral' "
                "or 'kernighan_lin' instead."
            )
        from pymetis import CSRAdjacency, part_graph

        xadj, adjncy, eweights, node_ids = _build_metis_inputs(graph)
        _, parts = part_graph(
            n_clusters,
            adjacency=CSRAdjacency(adj_starts=xadj, adjacent=adjncy),
            eweights=eweights,
            recursive=True,
        )
    else:
        raise RuntimeError("Relabeling only needed for `spectral` and `metis`.")

    clusters: list[list] = [[] for _ in range(n_clusters)]
    for idx, part in enumerate(parts):
        clusters[part].append(node_ids[idx])

    return tuple(
        _build_relabeled_subgraph_with_ids(graph, clstr) for clstr in clusters if clstr
    )


def _split_graph(
    graph: GraphProblemTypes,
    partitioning_config: GraphPartitioningConfig,
    n_parts: int = 2,
) -> Sequence[_SubgraphWithIds]:
    """
    Splits a graph into ``n_parts`` parts.

    "spectral" and "metis" split into ``n_parts`` parts directly;
    "kernighan_lin" bisects into sizes in the ratio
    ``n_parts // 2 : n_parts - n_parts // 2``, so repeated bisection to a size
    cap fills parts up to the cap.

    Args:
        graph: The input graph to be partitioned (``nx.Graph`` or ``rx.PyGraph``).
        partitioning_config (GraphPartitioningConfig): The configuration to follow.
        n_parts: Number of parts for the k-way algorithms, at least 2.

    Returns:
        Sequence of ``(relabeled_subgraph, cluster_ids)`` tuples.  Each
        subgraph is locally indexed ``0..M-1``; ``cluster_ids[i]`` is the
        parent-graph identifier for local node ``i``.
    """
    if (algorithm := partitioning_config.partitioning_algorithm) in (
        "spectral",
        "metis",
    ):
        parts = _apply_split_with_relabel(graph, algorithm, n_parts)
    elif partitioning_config.partitioning_algorithm == "kernighan_lin":
        kl_graph, node_ids = _kernighan_lin_input(graph)
        order = [
            node
            for component in sorted(nx.connected_components(kl_graph), key=min)
            for node in nx.bfs_tree(kl_graph, min(component))
        ]
        first = round(len(order) * (n_parts // 2) / n_parts)
        parts = tuple(
            _build_relabeled_subgraph_with_ids(
                graph, [node_ids[i] for i in sorted(part)]
            )
            for part in nx.algorithms.community.kernighan_lin_bisection(
                kl_graph, partition=(set(order[:first]), set(order[first:])), seed=0
            )
        )
    else:
        raise ValueError(
            f"Unsupported partitioning algorithm: "
            f"{partitioning_config.partitioning_algorithm!r}."
        )
    return tuple(
        (component, [ids[i] for i in local_ids])
        for sub, ids in parts
        for component, local_ids in _separate_components(sub)
    )


def _separate_components(graph: GraphProblemTypes) -> list[_SubgraphWithIds]:
    """``graph``'s connected components, relabeled ``0..M-1``, in order of their
    first node."""
    components = (
        rx.connected_components(graph)
        if isinstance(graph, rx.PyGraph)
        else nx.connected_components(graph)
    )
    return [
        _build_relabeled_subgraph_with_ids(graph, sorted(component))
        for component in sorted(components, key=min)
    ]


def _num_edges(graph: GraphProblemTypes) -> int:
    """Edge count for either supported graph type."""
    if isinstance(graph, rx.PyGraph):
        return graph.num_edges()
    return graph.number_of_edges()


def _borrow_neighbour(
    graph: GraphProblemTypes,
    orphan_ids: list,
    target_ids: dict[int, list],
    cap: int | None,
) -> tuple[int, object] | None:
    """A ``(cluster position, node)`` whose node neighbours the orphan and can
    leave its cluster with an edge to spare, or ``None``.

    Moving that node gives the orphan an edge without growing any cluster.
    """
    if cap is not None and len(orphan_ids) + 1 > cap:
        return None
    orphan_neighbours = {
        neighbour for node in orphan_ids for neighbour in graph.neighbors(node)
    }
    for position, ids in target_ids.items():
        for node in ids:
            rest = set(ids) - {node}
            if node in orphan_neighbours and any(
                neighbour in rest for v in rest for neighbour in graph.neighbors(v)
            ):
                return position, node
    return None


def _merge_adjacent_clusters(
    graph: GraphProblemTypes,
    clusters: Sequence[_SubgraphWithIds],
    cap: int,
    minimum: int | None,
) -> list[_SubgraphWithIds]:
    """Merge each cluster into the neighbour it shares the most edge ``|weight|``
    with while the union has at most ``cap`` nodes, smallest cluster first,
    stopping at ``minimum`` clusters. Untouched clusters are returned as given."""
    adjacency, node_ids = _build_spectral_inputs(graph)
    position = {node: i for i, node in enumerate(node_ids)}
    by_positions = {
        tuple(sorted(position[node] for node in cluster[1])): cluster
        for cluster in clusters
    }
    merged = _merge_clusters_to_budget(
        adjacency, [np.array(key) for key in by_positions], cap, minimum
    )
    return [
        by_positions.get(tuple(cluster.tolist()))
        or _build_relabeled_subgraph_with_ids(
            graph, [node_ids[i] for i in cluster.tolist()]
        )
        for cluster in merged
    ]


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
    the earlier one, so the result stays deterministic.  When no cluster has
    room, the edgeless cluster instead takes one neighbouring node from a
    cluster that keeps an edge without it.

    Args:
        graph: The original graph the clusters were partitioned from.
        clusters: ``(relabeled_subgraph, cluster_ids)`` pairs to repair.
        partitioning_config: The configuration the partition was built under.

    Returns:
        The repaired cluster list, relabeled ``0..M-1`` as on input.

    Raises:
        ValueError: If every cluster is edgeless, or an edgeless cluster can
            neither fit into a cluster under ``max_n_nodes_per_cluster`` nor
            take a neighbour from one.
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
            borrowed = _borrow_neighbour(graph, orphan_ids, target_ids, cap)
            if borrowed is None:
                raise ValueError(
                    f"Cannot merge edgeless cluster {sorted(orphan_ids)}: no cluster "
                    f"has room under 'max_n_nodes_per_cluster'={cap}. The graph is "
                    "too dense to split this small — raise "
                    "'max_n_nodes_per_cluster' or lower 'minimum_n_clusters'."
                )
            donor, node = borrowed
            target_ids[donor] = [v for v in target_ids[donor] if v != node]
            target_ids[position] = orphan_ids + [node]
            continue

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
            else _build_relabeled_subgraph_with_ids(graph, ids)
        )
        for i, ids in target_ids.items()
    ]
    return merged


HeapEntry = tuple[int, int, GraphProblemTypes, list]


def _split_with_predicate(
    initial_partitions: list[HeapEntry],
    predicate: Callable[[GraphProblemTypes, Sequence[HeapEntry]], bool],
    partitioning_config: GraphPartitioningConfig,
    n_parts: Callable[[GraphProblemTypes], int] = lambda _graph: 2,
) -> list[HeapEntry]:
    """
    Recursively splits a list of graph partitions based on a user-defined predicate.

    This helper function repeatedly applies a partitioning strategy to a sequence of graph
    subgraphs. At each iteration, it evaluates a predicate to determine whether a subgraph
    should be further split, into ``n_parts(subgraph)`` parts. The process continues until
    no subgraphs satisfy the predicate, at which point the resulting collection of
    subgraphs is returned.

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

            children = (
                _split_graph(subgraph, partitioning_config, n_parts(subgraph))
                if predicate(subgraph, new_subgraphs + subgraphs)
                else ()
            )
            if len(children) > 1:
                for child, child_local_ids in children:
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


def _split_to_minimum(
    subgraphs: list[HeapEntry],
    minimum: int,
    partitioning_config: GraphPartitioningConfig,
) -> list[HeapEntry]:
    """Split the largest cluster into as many parts as are still missing until
    there are ``minimum`` clusters, or no cluster splits any further."""
    heap = list(subgraphs)
    heapq.heapify(heap)
    counter = max((entry[1] for entry in heap), default=-1) + 1
    while len(heap) < minimum:
        entry = heapq.heappop(heap)
        _, _, subgraph, parent_ids = entry
        n_parts = min(minimum - len(heap), len(subgraph))
        children = (
            _split_graph(subgraph, partitioning_config, n_parts) if n_parts > 1 else ()
        )
        if len(children) < 2:
            heapq.heappush(heap, entry)
            break
        for child, child_local_ids in children:
            heapq.heappush(
                heap,
                (-len(child), counter, child, [parent_ids[i] for i in child_local_ids]),
            )
            counter += 1
    return heap


def _partition_graph_by_nodes(
    graph: GraphProblemTypes,
    partitioning_config: GraphPartitioningConfig,
    *,
    merge_edgeless: bool = True,
) -> list[_SubgraphWithIds]:
    """Partition ``graph`` into subgraphs honouring the configured constraints.

    Returns a list of ``(relabeled_subgraph, cluster_ids)`` pairs.  Each
    subgraph is locally indexed ``0..M-1``; ``cluster_ids[i]`` is the
    original-graph identifier for local node ``i``.

    Nodes without an edge to another node belong to no cluster. Connected
    components are split apart before any algorithm runs, every split is
    separated into its connected components, and only clusters with edges count
    towards ``minimum_n_clusters``. Clusters left without an edge are merged
    away (see :func:`_merge_edgeless_clusters`) unless ``merge_edgeless`` is
    ``False``. Under ``max_n_nodes_per_cluster``, neighbouring clusters that fit
    together are then merged (see :func:`_merge_adjacent_clusters`).

    Raises:
        ValueError: If the graph has no edges, or the constraints cannot be met.
    """
    n_nodes = len(graph)
    if partitioning_config.minimum_n_clusters:
        if partitioning_config.minimum_n_clusters >= n_nodes:
            raise ValueError(
                "Number of requested clusters must be smaller than the size of "
                "the graph, since every cluster needs at least one edge."
            )

    # Splits return local positions, so the root must be indexed 0..N-1 too.
    positional: GraphProblemTypes
    if isinstance(graph, rx.PyGraph):
        initial_ids: list = list(graph.node_indexes())
        positional = graph.subgraph(initial_ids)
    else:
        initial_ids = list(graph.nodes())
        positional = nx.convert_node_labels_to_integers(graph)

    _, edges = _canonical_edges(positional)
    linked = nx.Graph()
    linked.add_nodes_from(range(len(initial_ids)))
    linked.add_edges_from((u, v) for u, v, _ in edges if u != v)

    subgraphs: list[HeapEntry] = []
    for component in sorted(nx.connected_components(linked), key=min):
        if len(component) > 1:
            sub, local_ids = _build_relabeled_subgraph_with_ids(
                positional, sorted(component)
            )
            subgraphs.append(
                (
                    -len(local_ids),
                    len(subgraphs),
                    sub,
                    [initial_ids[i] for i in local_ids],
                )
            )
    if not subgraphs:
        raise ValueError(
            "Cannot partition a graph without edges: no cluster could have an "
            "internal edge."
        )

    if partitioning_config.minimum_n_clusters:
        subgraphs = _split_to_minimum(
            subgraphs, partitioning_config.minimum_n_clusters, partitioning_config
        )

    if cap := partitioning_config.max_n_nodes_per_cluster:
        subgraphs = _split_with_predicate(
            subgraphs,
            lambda subgraph, _: len(subgraph) > cap,
            partitioning_config,
            n_parts=lambda subgraph: math.ceil(len(subgraph) / cap),
        )

    clusters = [(sub, ids) for (_, _, sub, ids) in subgraphs]
    if merge_edgeless:
        clusters = _merge_edgeless_clusters(graph, clusters, partitioning_config)
    if cap:
        clusters = _merge_adjacent_clusters(
            graph, clusters, cap, partitioning_config.minimum_n_clusters
        )
    minimum = partitioning_config.minimum_n_clusters
    if minimum is not None and len(clusters) < minimum:
        warn(
            f"Partitioning produced {len(clusters)} cluster(s), below the "
            f"requested 'minimum_n_clusters'={minimum}.",
            UserWarning,
            stacklevel=2,
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
            as built by ``GraphProblemBase.decompose()``.
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
