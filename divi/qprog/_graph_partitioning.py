# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
import string
from collections.abc import Callable, Sequence, Set
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Literal
from warnings import warn

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse.linalg as spla
from pymetis import part_graph
from sklearn.cluster import SpectralClustering

from divi.interfaces import CircuitRunner
from divi.qprog import QAOA, ProgramBatch
from divi.qprog._qaoa import (
    _SUPPORTED_INITIAL_STATES_LITERAL,
    GraphProblem,
    GraphProblemTypes,
    draw_graph_solution_nodes,
)
from divi.qprog.quantum_program import QuantumProgram

from .optimizers import Optimizer

AggregateFn = Callable[
    [list[int], str, nx.Graph | rx.PyGraph, dict[int, int]], list[int]
]

# TODO: Make this dynamic through an interaction with usher
# once a proper endpoint is exposed
_MAXIMUM_AVAILABLE_QUBITS = 30


@dataclass(frozen=True, eq=True)
class PartitioningConfig:
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


def _divide_edges(
    graph: nx.DiGraph, edge_selection_predicate: Callable
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """
    Divides a graph into two subgraphs based on the provided edge selection criteria.

    Args:
        graph (nx.DiGraph): The input graph to be divided.
        edge_selection_predicate (Callable): A function which decides if an edge should be
                                            included in the selected subgraph.

    Returns:
        tuple[nx.DiGraph, nx.DiGraph]: A tuple containing two DiGraphs: the selected subgraph
                                        and the rest of the graph.
    """
    selected_edges = [
        (u, v)
        for u, v in graph.edges(data=False)
        if edge_selection_predicate(graph, u, v)
    ]
    rest_edges = [
        (u, v) for u, v in graph.edges(data=False) if (u, v) not in selected_edges
    ]

    selected_subgraph = graph.edge_subgraph(selected_edges).copy()
    rest_of_graph = graph.edge_subgraph(rest_edges).copy()

    rest_of_graph.remove_edges_from(selected_edges)  # to avoid overlap

    return selected_subgraph, rest_of_graph


def _fielder_laplacian_predicate(
    growing_graph: nx.DiGraph, src: int, dest: int
) -> bool:
    """
    Determines if an edge should be included in the selected subgraph based on spectral partitioning.

    This function uses the Fiedler vector of the graph's Laplacian matrix to divide
    the nodes into two partitions. An edge is included in the selected subgraph
    if both its source and destination nodes belong to the same partition.

    Args:
        growing_graph (nx.DiGraph): The graph containing the currently selected edges.
        src (int): The source node of the edge.
        dest (int): The destination node of the edge.

    Returns:
        bool: True if the edge should be included in the selected subgraph, False otherwise.
    """
    if growing_graph.number_of_edges() == 0:
        return True

    L = nx.laplacian_matrix(growing_graph).astype(float)

    # Create an initial random guess for the eigenvectors
    n = L.shape[0]
    X = np.random.rand(n, 2)
    X, _ = np.linalg.qr(X)  # Orthonormalize initial guess

    # Use LOBPCG to compute the two smallest eigenvalues and corresponding eigenvectors
    _, eigenvectors = spla.lobpcg(L, X, largest=False)

    fiedler_vector = eigenvectors[:, 1].real
    partition = set(i for i, v in enumerate(fiedler_vector) if v > 0)

    return (src in partition) == (dest in partition)


def _edge_partition_graph(
    graph: nx.DiGraph, n_max_nodes_per_cluster: int
) -> list[nx.DiGraph]:
    """
    Partitions a directed graph into smaller subgraphs using recursive bipartite spectral partitioning.

    The function repeatedly divides the input graph into two subgraphs based on the
    Fiedler vector of the graph's Laplacian matrix. This process is repeated
    until each of the subgraphs' no. of edges does not exceed the no. of qubits.

    Args:
        graph (nx.DiGraph): The input directed graph to be partitioned.
        n_max_nodes_per_cluster (int, optional): The maximum number of nodes per subgraph.
                                                Defaults to 8.

    Returns:
        list[nx.DiGraph]: A list of subgraphs resulting from the partitioning process.
    """
    subgraphs = [graph]

    while any(g.number_of_edges() > n_max_nodes_per_cluster for g in subgraphs):
        large_subgraphs = [
            g for g in subgraphs if g.number_of_edges() > n_max_nodes_per_cluster
        ]
        subgraphs = [
            g for g in subgraphs if g.number_of_edges() <= n_max_nodes_per_cluster
        ]

        if not large_subgraphs:
            break

        for large_subgraph in large_subgraphs:
            selected_subgraph, rest_of_graph = _divide_edges(
                large_subgraph, _fielder_laplacian_predicate
            )
            subgraphs.extend([selected_subgraph, rest_of_graph])

    return subgraphs


def _apply_split_with_relabel(
    graph: nx.Graph, algorithm: Literal["spectral", "metis"], n_clusters: int
) -> tuple[nx.Graph, nx.Graph]:
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
        adj_list = list(nx.to_dict_of_lists(int_graph).values())
        _, parts = part_graph(n_clusters, adjacency=adj_list)
    else:
        raise RuntimeError("Relabeling only needed for `spectral` and `metis`.")

    clusters = [[] for _ in range(n_clusters)]
    for idx, part in enumerate(parts):
        orig_label = int_graph.nodes[idx]["orig_label"]
        clusters[part].append(orig_label)

    return tuple(graph.subgraph(clstr) for clstr in clusters)


def _split_graph(
    graph: nx.Graph, partitioning_config: PartitioningConfig
) -> Sequence[nx.Graph]:
    """
    Splits a graph.

    If the requested partitioning algorithm is either "spectral" or "metis",
    then the requested `min_n_clusters` will be returned.
    For "kernighan_lin", a bisection will be returned

    Args:
        graph (nx.Graph): The input graph to be partitioned.
        partitioning_config (PartitioningConfig): The configuration to follow.

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
        return graph.subgraph(part_1), graph.subgraph(part_2)


def _bisect_with_predicate(
    initial_partitions: Sequence[nx.Graph],
    predicate: Callable[[nx.Graph | None, Sequence[nx.Graph] | None], bool],
    partitioning_config: PartitioningConfig,
) -> Sequence[nx.Graph]:
    """
    Recursively bisects a list of graph partitions based on a user-defined predicate.

    This helper function repeatedly applies a partitioning strategy to a sequence of graph
    subgraphs. At each iteration, it evaluates a predicate to determine whether a subgraph
    should be further split. The process continues until no subgraphs satisfy the predicate,
    at which point the resulting collection of subgraphs is returned.

    The predicate is expected to accept two arguments:
        - The current subgraph under consideration.
        - A list of other subgraphs in the current iteration (both previously processed
        and yet to be processed), serving as the context for the decision.

    Returns the final list of subgraphs as a heapified sequence, ordered by descending
    node count.
    """
    subgraphs = initial_partitions
    heapq.heapify(subgraphs)

    while True:
        new_subgraphs = []
        changed = False

        while subgraphs:
            (_, _, subgraph) = heapq.heappop(subgraphs)

            if predicate(subgraph, new_subgraphs + subgraphs):
                new_subgraphs.extend(_split_graph(subgraph, partitioning_config))
                changed = True
            else:
                new_subgraphs.append(subgraph)

        subgraphs = [
            (-sg.number_of_nodes(), i, sg) for (i, sg) in enumerate(new_subgraphs)
        ]
        heapq.heapify(subgraphs)

        if not changed:
            break

    return subgraphs


def _node_partition_graph(
    graph: nx.Graph, partitioning_config: PartitioningConfig
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
    return tuple(graph for (_, _, graph) in subgraphs)


def linear_aggregation(
    curr_solution: Sequence[Literal[0] | Literal[1]],
    subproblem_solution: Set[int],
    subproblem_reverse_index_map: dict[int, int],
):
    """Linearly combines a subproblem's solution into the main solution vector.

    This function iterates through each node of subproblem's solution. For each node,
    it uses the reverse index map to find its original index in the main graph,
    setting it to 1 in the current global solution, potentially overwriting any
    previous states.

    Args:
        curr_solution (Sequence[Literal[0] | Literal[1]]): The main solution
            vector being aggregated, represented as a sequence of 0s and 1s.
        subproblem_solution (Set[int]): A set containing the original indices of
            the nodes that form the solution for the subproblem.
        subproblem_reverse_index_map (dict[int, int]): A mapping from the
            subgraph's internal node labels back to their original indices in
            the main solution vector.

    Returns:
        The updated main solution vector.
    """
    for node in subproblem_solution:
        curr_solution[subproblem_reverse_index_map[node]] = 1

    return curr_solution


def dominance_aggregation(
    curr_solution: Sequence[Literal[0] | Literal[1]],
    subproblem_solution: Set[int],
    subproblem_reverse_index_map: dict[int, int],
):
    for node in subproblem_solution:
        original_index = subproblem_reverse_index_map[node]

        # Use existing assignment if dominant in previous solutions
        # (e.g., more 0s than 1s or vice versa)
        count_0 = curr_solution.count(0)
        count_1 = curr_solution.count(1)

        if (
            (count_0 > count_1 and curr_solution[original_index] == 0)
            or (count_1 > count_0 and curr_solution[original_index] == 1)
            or (count_0 == count_1)
        ):
            # Assign based on QAOA if tie
            curr_solution[original_index] = 1

    return curr_solution


def _run_and_compute_solution(program: QuantumProgram):

    program.run()

    final_sol_circuit_count, final_sol_run_time = program.compute_final_solution()

    return final_sol_circuit_count, final_sol_run_time


class GraphPartitioningQAOA(ProgramBatch):
    def __init__(
        self,
        graph: GraphProblemTypes,
        graph_problem: GraphProblem,
        n_layers: int,
        backend: CircuitRunner,
        partitioning_config: PartitioningConfig,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        aggregate_fn: AggregateFn = linear_aggregation,
        optimizer=Optimizer.MONTE_CARLO,
        max_iterations=10,
        **kwargs,
    ):
        """
        Initializes the graph partitioning class.

        Args:
            graph (nx.Graph | rx.PyGraph): The input graph to be partitioned.
            graph_problem (GraphProblem): The type of graph partitioning problem (e.g., EDGE_PARTITIONING).
            n_layers (int): Number of layers for the QAOA circuit.
            backend (CircuitRunner): Backend used to run quantum/classical circuits.
            partitioning_config (PartitioningConfig): the configuration of the partitioning as to the algorithm and
            expected output.
            initial_state ("Zeros", "Ones", "Superposition", "Recommended", optional): Initial state for the QAOA algorithm. Defaults to "Recommended".
            aggregate_fn (optional): Aggregation function to combine results. Defaults to `linear_aggregation`.
            optimizer (optional): Optimizer to use for QAOA. Defaults to `Optimizers.MONTE_CARLO`.
            max_iterations (int, optional): Maximum number of optimization iterations. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the QAOA constructor.

        """
        super().__init__(backend=backend)

        self.main_graph = graph
        self.is_edge_problem = graph_problem == GraphProblem.EDGE_PARTITIONING

        check_fn = (
            nx.is_connected if not self.is_edge_problem else nx.is_weakly_connected
        )
        if not check_fn(self.main_graph):
            raise ValueError("Provided graph is not fully connected.")

        self.partitioning_config = partitioning_config
        self.max_iterations = max_iterations

        self.solution = None
        self.aggregate_fn = aggregate_fn

        self._task_fn = _run_and_compute_solution

        self._constructor = partial(
            QAOA,
            initial_state=initial_state,
            graph_problem=graph_problem,
            optimizer=optimizer,
            max_iterations=self.max_iterations,
            backend=self.backend,
            n_layers=n_layers,
            **kwargs,
        )

    def create_programs(self):
        """
        Creates and initializes QAOA programs for each partitioned subgraph.

        The main graph is partitioned into node-based subgraphs
        according to the specified partitioning configuration. Each subgraph is relabeled with
        integer node labels for QAOA compatibility, and a reverse index map is stored for later
        result aggregation.

        Each program is assigned a unique program ID, which is a tuple of:
            - An uppercase letter (A, B, C, ...) corresponding to the partition index.
            - The number of nodes in the subgraph.

        Example program ID: ('A', 5) for the first partition with 5 nodes.

        The created QAOA programs are stored in the `self.programs` dictionary, keyed by their program IDs.

        """

        super().create_programs()

        if self.is_edge_problem:
            subgraphs = _edge_partition_graph(
                self.main_graph,
                n_max_nodes_per_cluster=self.partitioning_config.max_n_nodes_per_cluster,
            )
            cleaned_subgraphs = list(filter(lambda x: x.size() > 0, subgraphs))
        else:
            subgraphs = _node_partition_graph(
                self.main_graph,
                partitioning_config=self.partitioning_config,
            )

            self._bitstring_solution = [0] * self.main_graph.number_of_nodes()
            self.reverse_index_maps = {}

            for i, subgraph in enumerate(subgraphs):
                prog_id = (string.ascii_uppercase[i], subgraph.number_of_nodes())

                index_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
                self.reverse_index_maps[prog_id] = {v: k for k, v in index_map.items()}

                _subgraph = nx.relabel_nodes(subgraph, index_map)
                self.programs[prog_id] = self._constructor(
                    job_id=prog_id,
                    problem=_subgraph,
                    losses=self._manager.list(),
                    probs=self._manager.dict(),
                    final_params=self._manager.list(),
                    solution_nodes=self._manager.list(),
                    progress_queue=self._queue,
                )

    def compute_final_solutions(self):
        if self._executor is not None:
            self.join()

        if self._executor is not None:
            raise RuntimeError("A batch is already being run.")

        if len(self.programs) == 0:
            raise RuntimeError("No programs to run.")

        self._executor = ProcessPoolExecutor()

        self.futures = [
            self._executor.submit(program.compute_final_solution)
            for program in self.programs.values()
        ]

    def aggregate_results(self):
        """
        Aggregates the results from all QAOA subprograms to form a global solution.

        This method collects the final bitstring solutions from each partitioned subgraph's QAOA program,
        using the aggregation function specified at initialization (e.g., linear or dominance aggregation).
        It reconstructs the global solution by mapping each subgraph's solution back to the original node indices
        using the stored reverse index maps.

        The final solution is stored in `self.solution` as a list of node indices assigned to the selected partition.

        Raises:
            RuntimeError: If no programs exist, if programs have not been run, or if results are incomplete.
        Returns:
            list[int]: The list of node indices in the final aggregated solution.
        """
        super().aggregate_results()

        if any(len(program.probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. Please call `run()` first."
            )

        # Extract the solutions from each program
        for prog_id, program in self.programs.items():
            self._bitstring_solution = self.aggregate_fn(
                self._bitstring_solution,
                program.solution,
                self.reverse_index_maps[prog_id],
            )

        self.solution = list(np.where(self._bitstring_solution)[0])

        return self.solution

    def draw_partitions(
        self,
        pos: dict | None = None,
        figsize: tuple[int, int] | None = (10, 8),
        node_size: int | None = 300,
    ):
        """
        Draw a NetworkX graph with nodes colored by partition.

        Parameters:
        -----------
        pos : dict, optional
            Node positions. If None, uses spring layout
        figsize : tuple, optional
            Figure size (width, height)
        node_size : int, optional
            Size of nodes
        """

        if len(self.programs) == 0:
            raise RuntimeError(
                "There are no partitions to draw. Did you run create_programs()?"
            )

        # Convert partitions to node-to-partition mapping
        node_to_partition = {}
        for (partition_id, _), mapping in self.reverse_index_maps.items():
            for node in mapping.values():
                node_to_partition[node] = partition_id

        # Get unique partition IDs and create color map
        unique_partitions = sorted(list(set(node_to_partition.values())))
        n_partitions = len(unique_partitions)
        colors = cm.Set3(np.linspace(0, 1, n_partitions))
        partition_colors = {pid: colors[i] for i, pid in enumerate(unique_partitions)}

        # Create node color list
        node_colors = [
            partition_colors[node_to_partition.get(node, 0)]
            for node in self.main_graph.nodes()
        ]

        # Set positions
        if pos is None:
            pos = nx.spring_layout(self.main_graph, seed=42)

        # Draw the graph
        plt.figure(figsize=figsize)
        nx.draw(
            self.main_graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=True,
            font_size=8,
            font_weight="bold",
            edge_color="gray",
            alpha=0.8,
        )

        # Add legend
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

    def draw_solution(self):
        """
        Visualizes the main graph with nodes highlighted according to the final aggregated solution.

        If the solution has not yet been computed, this method calls `aggregate_results()` to obtain it.
        """

        if self.solution is None:
            self.aggregate_results()

        draw_graph_solution_nodes(self.main_graph, self.solution)
