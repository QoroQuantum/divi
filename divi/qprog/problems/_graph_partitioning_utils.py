# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colormaps
from sklearn.cluster import SpectralClustering

# TODO: Make this dynamic through an interaction with usher
# once a proper endpoint is exposed
_MAXIMUM_AVAILABLE_QUBITS = 30


@dataclass(frozen=True, eq=True)
class GraphPartitioningConfig:
    """Configuration for graph partitioning algorithms.

    This class defines the parameters and constraints for partitioning large graphs
    into smaller subgraphs for quantum algorithm execution. It supports multiple
    partitioning algorithms and allows specification of size constraints.

    Attributes:
        max_n_nodes_per_cluster: Maximum number of nodes allowed in each cluster.
            If None, no upper limit is enforced. Must be a positive integer.
        minimum_n_clusters: Minimum number of clusters to create. If None, no
            lower limit is enforced. Must be a positive integer.
        partitioning_algorithm: Algorithm to use for partitioning. Options are:
            - "spectral": Spectral partitioning using Fiedler vector (default)
            - "metis": METIS graph partitioning library
            - "kernighan_lin": Kernighan-Lin algorithm

    Note:
        At least one of `max_n_nodes_per_cluster` or `minimum_n_clusters` must be
        specified. Both constraints cannot be None.

    Examples:
        >>> # Partition into clusters of at most 10 nodes
        >>> config = GraphPartitioningConfig(max_n_nodes_per_cluster=10)

        >>> # Create at least 5 clusters using METIS
        >>> config = GraphPartitioningConfig(
        ...     minimum_n_clusters=5,
        ...     partitioning_algorithm="metis"
        ... )

        >>> # Both constraints: clusters of max 8 nodes, min 3 clusters
        >>> config = GraphPartitioningConfig(
        ...     max_n_nodes_per_cluster=8,
        ...     minimum_n_clusters=3
        ... )
    """

    max_n_nodes_per_cluster: int | None = None
    minimum_n_clusters: int | None = None
    partitioning_algorithm: Literal["spectral", "metis", "kernighan_lin"] = "spectral"

    def __post_init__(self):
        if self.max_n_nodes_per_cluster is None and self.minimum_n_clusters is None:
            raise ValueError("At least one constraint must be specified.")

        if self.minimum_n_clusters is not None and self.minimum_n_clusters < 1:
            raise ValueError("'minimum_n_clusters' must be a positive integer.")

        if (
            self.max_n_nodes_per_cluster is not None
            and self.max_n_nodes_per_cluster < 1
        ):
            raise ValueError("'max_n_nodes_per_cluster' must be a positive number.")

        if self.partitioning_algorithm not in ("spectral", "metis", "kernighan_lin"):
            raise ValueError(
                f"Unsupported partitioning algorithm: {self.partitioning_algorithm}. "
                "Use 'spectral' or 'metis'."
            )


def _apply_split_with_relabel(
    graph: nx.Graph, algorithm: Literal["spectral", "metis"], n_clusters: int
) -> tuple[nx.Graph, ...]:
    """
    Relabels nodes of a graph to (0, ..., N-1) for algorithms that
    require this input/has output of this format and requires mapping
    back to original labels.
    """
    int_graph = nx.convert_node_labels_to_integers(graph, label_attribute="orig_label")

    if algorithm == "spectral":
        adj_matrix = nx.to_scipy_sparse_array(graph, format="csr")

        adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)
        adj_matrix.indices = adj_matrix.indices.astype(np.int32)

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            n_init=100,
            assign_labels="discretize",
        )
        parts = sc.fit_predict(adj_matrix)
    elif algorithm == "metis":
        try:
            from pymetis import part_graph
        except ImportError as e:
            raise ImportError(
                "pymetis is required for the 'metis' partitioning algorithm but could not be "
                "imported. On Windows, install via conda: conda install -c conda-forge pymetis. "
                "Otherwise use 'spectral' or 'kernighan_lin' instead."
            ) from e
        # After relabeling, nodes are integers (networkx stubs don't narrow this).
        adj_list = cast(list[list[int]], list(nx.to_dict_of_lists(int_graph).values()))
        _, parts = part_graph(n_clusters, adjacency=adj_list)
    else:
        raise RuntimeError("Relabeling only needed for `spectral` and `metis`.")

    clusters = [[] for _ in range(n_clusters)]
    for idx, part in enumerate(parts):
        orig_label = int_graph.nodes[idx]["orig_label"]
        clusters[part].append(orig_label)

    return tuple(graph.subgraph(clstr).copy() for clstr in clusters)


def _split_graph(
    graph: nx.Graph, partitioning_config: GraphPartitioningConfig
) -> Sequence[nx.Graph]:
    """
    Splits a graph.

    If the requested partitioning algorithm is either "spectral" or "metis",
    then the requested `min_n_clusters` will be returned.
    For "kernighan_lin", a bisection will be returned

    Args:
        graph (nx.Graph): The input graph to be partitioned.
        partitioning_config (GraphPartitioningConfig): The configuration to follow.

    Returns:
        subgraphs: a sequence of the generated partitions.
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
        part_1, part_2 = nx.algorithms.community.kernighan_lin_bisection(graph)
        return graph.subgraph(part_1).copy(), graph.subgraph(part_2).copy()
    else:
        raise ValueError(
            f"Unsupported partitioning algorithm: "
            f"{partitioning_config.partitioning_algorithm!r}."
        )


HeapEntry = tuple[int, int, nx.Graph]


def _bisect_with_predicate(
    initial_partitions: list[HeapEntry],
    predicate: Callable[[nx.Graph, Sequence[HeapEntry]], bool],
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
    entry_counter = len(subgraphs)

    while True:
        new_subgraphs: list[HeapEntry] = []
        changed = False

        while subgraphs:
            entry = heapq.heappop(subgraphs)
            subgraph = entry[2]

            if predicate(subgraph, new_subgraphs + subgraphs):
                for child in _split_graph(subgraph, partitioning_config):
                    new_subgraphs.append(
                        (-child.number_of_nodes(), entry_counter, child)
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
    graph: nx.Graph, partitioning_config: GraphPartitioningConfig
) -> list[nx.Graph]:

    subgraphs = [(-graph.number_of_nodes(), 0, graph)]

    # First generate the minimum number of clusters, requested by user
    # Initialize the graph as the initial subgraph
    # Add generic ID to break ties in heap
    if partitioning_config.minimum_n_clusters:
        if partitioning_config.minimum_n_clusters > graph.number_of_nodes():
            raise ValueError(
                "Number of requested clusters larger than the size of the graph."
            )

        subgraphs = _bisect_with_predicate(
            [(-graph.number_of_nodes(), 0, graph)],
            lambda _, subgraphs: len(subgraphs)
            < partitioning_config.minimum_n_clusters - 1,
            partitioning_config,
        )

    # Split oversized clusters
    if partitioning_config.max_n_nodes_per_cluster:
        subgraphs = _bisect_with_predicate(
            subgraphs,
            lambda subgraph, _: (
                subgraph.number_of_nodes() > partitioning_config.max_n_nodes_per_cluster
            ),
            partitioning_config,
        )

    if any(-sg[0] > _MAXIMUM_AVAILABLE_QUBITS for sg in subgraphs):
        warn(
            "At least one cluster has more nodes than what can be executed on "
            f"the available backends: {_MAXIMUM_AVAILABLE_QUBITS} qubits."
        )

    # Clean up on aisle 3
    return [graph for (_, _, graph) in subgraphs]


def draw_partitions(
    graph: nx.Graph,
    reverse_index_maps: dict,
    pos: dict | None = None,
    figsize: tuple[int, int] | None = (10, 8),
    node_size: int = 300,
):
    """Draw a graph with nodes colored by partition.

    Args:
        graph: The full graph.
        reverse_index_maps: Mapping ``{prog_id: {local_idx: global_node}}``,
            as built by ``_GraphProblemBase.decompose()``.
        pos: Node positions.  If *None*, uses spring layout.
        figsize: Figure size ``(width, height)``.
        node_size: Size of nodes.
    """
    if not reverse_index_maps:
        raise RuntimeError("There are no partitions to draw. Did you call decompose()?")

    node_to_partition = {}
    for (partition_id, _), mapping in reverse_index_maps.items():
        for node in mapping.values():
            node_to_partition[node] = partition_id

    unique_partitions = sorted(set(node_to_partition.values()))
    n_partitions = len(unique_partitions)
    colors = colormaps["Set3"](np.linspace(0, 1, n_partitions))
    partition_colors = {pid: colors[i] for i, pid in enumerate(unique_partitions)}

    node_colors = [
        partition_colors[node_to_partition.get(node, 0)] for node in graph.nodes()
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

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=partition_colors[pid],
            markersize=10,
            label=f"Partition {pid}",
        )
        for pid in unique_partitions
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.title("Graph Partitions Visualization")
    plt.axis("off")
    plt.show()
