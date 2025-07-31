import heapq
import re
import string
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Literal, Optional
from warnings import warn

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
    draw_graph_partitions,
)

from .optimizers import Optimizers

AggregateFn = Callable[
    [list[int], str, nx.Graph | rx.PyGraph, dict[int, int]], list[int]
]

# TODO: Make this dynamic through an interaction with usher
# once a proper endpoint is exposed
_MAXIMUM_AVAILABLE_QUBITS = 30


@dataclass(frozen=True, eq=True)
class PartitioningConfig:
    max_n_nodes_per_cluster: Optional[int] = None
    minimum_n_clusters: Optional[int] = None
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
            partitioning_config.minimum_n_clusters
            # If minimum clusters isn't a constraint, then default to a rough sectioning
            or ceil(
                graph.number_of_nodes() / partitioning_config.max_n_nodes_per_cluster
            ),
        )
    elif partitioning_config.partitioning_algorithm == "kernighan_lin":
        part_1, part_2 = nx.algorithms.community.kernighan_lin_bisection(graph)
        return graph.subgraph(part_1), graph.subgraph(part_2)


def _bisect_with_predicate(
    initial_partitions: Sequence[nx.Graph],
    predicate: Callable[[nx.Graph | None, Sequence[nx.Graph] | None], bool],
    partitioning_config: PartitioningConfig,
) -> Sequence[nx.Graph]:

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
    return tuple(sg[-1] for sg in subgraphs)


def linear_aggregation(curr_solution, solution_bitstring, graph, reverse_index_maps):
    for node in graph.nodes():
        solution_index = reverse_index_maps[node]
        curr_solution[solution_index] = int(solution_bitstring[node])

    return curr_solution


def domninance_aggregation(
    curr_solution, solution_bitstring, graph, reverse_index_maps
):
    for node in graph.nodes():
        solution_index = reverse_index_maps[node]

        # Use existing assignment if dominant in previous solutions
        # (e.g., more 0s than 1s or vice versa)
        count_0 = curr_solution.count(0)
        count_1 = curr_solution.count(1)

        if (
            (count_0 > count_1 and curr_solution[node] == 0)
            or (count_1 > count_0 and curr_solution[node] == 1)
            or (count_0 == count_1)
        ):
            # Assign based on QAOA if tie
            curr_solution[solution_index] = int(solution_bitstring[node])

    return curr_solution


class GraphPartitioningQAOA(ProgramBatch):
    def __init__(
        self,
        graph: nx.Graph | rx.PyGraph,
        graph_problem: GraphProblem,
        n_layers: int,
        backend: CircuitRunner,
        partitioning_config: PartitioningConfig,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        aggregate_fn: AggregateFn = linear_aggregation,
        optimizer=Optimizers.MONTE_CARLO,
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

        self.aggregate_fn = aggregate_fn

        self._solution_nodes = None

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
        if len(self.programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )

        super().create_programs()

        if self.is_edge_problem:
            subgraphs = _edge_partition_graph(
                self.main_graph, n_max_nodes_per_cluster=self.n_max_nodes_per_cluster
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
                index_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
                self.reverse_index_maps[i] = {v: k for k, v in index_map.items()}
                _subgraph = nx.relabel_nodes(subgraph, index_map)

                prog_id = (string.ascii_uppercase[i], subgraph.number_of_nodes())

                self.programs[prog_id] = self._constructor(
                    job_id=prog_id,
                    problem=_subgraph,
                    losses=self._manager.list(),
                    probs=self._manager.dict(),
                    final_params=self._manager.list(),
                    progress_queue=self._queue,
                )

    def compute_final_solutions(self):
        if self._executor is not None:
            self.wait_for_all()

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
        if len(self.programs) == 0:
            raise RuntimeError("No programs to aggregate. Run create_programs() first.")

        if self._executor is not None:
            self.wait_for_all()

        if any(len(program.losses) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Some/All programs have empty losses. Did you call run()?"
            )

        if any(len(program.probs) == 0 for program in self.programs.values()):
            raise RuntimeError(
                "Not all final probabilities computed yet. Please call `compute_final_solutions()` first."
            )

        # Extract the solutions from each program
        for program, reverse_index_maps in zip(
            self.programs.values(), self.reverse_index_maps.values()
        ):
            # Extract the final probabilities of the lowest energy
            last_iteration_losses = program.losses[-1]
            minimum_key = min(last_iteration_losses, key=last_iteration_losses.get)

            # Find the key matching the best_solution_idx with possible metadata in between
            pattern = re.compile(rf"^{minimum_key}(?:_[^_]*)*_0$")
            matching_keys = [k for k in program.probs.keys() if pattern.match(k)]

            if len(matching_keys) > 1:
                raise RuntimeError(f"More than one matching key found.")

            best_solution_key = matching_keys[0]

            minimum_probabilities = program.probs[best_solution_key]

            # The bitstring corresponding to the solution, with flip for correct endianness
            max_prob_key = max(minimum_probabilities, key=minimum_probabilities.get)[
                ::-1
            ]

            self._bitstring_solution = self.aggregate_fn(
                self._bitstring_solution,
                max_prob_key,
                program.problem,
                reverse_index_maps,
            )

        self.solution = list(np.where(self._bitstring_solution)[0])

        return self.solution

    def draw_solution(self):
        if self._solution_nodes is None:
            self.aggregate_results()

        draw_graph_partitions(self.main_graph, self.solution)
