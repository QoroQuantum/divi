from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
from typing import Optional
from warnings import warn

import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse.linalg as spla
from sklearn.cluster import SpectralClustering

from divi.qprog import QAOA, ProgramBatch
from divi.qprog._qaoa import _SUPPORTED_INITIAL_STATES_LITERAL, GraphProblem

from .optimizers import Optimizers

AggregateFn = Callable[
    [list[int], str, nx.Graph | rx.PyGraph, dict[int, int]], list[int]
]


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


def _edge_partition_graph(graph: nx.DiGraph, n_qubits: int = 8) -> list[nx.DiGraph]:
    """
    Partitions a directed graph into smaller subgraphs using recursive bipartite spectral partitioning.

    The function repeatedly divides the input graph into two subgraphs based on the
    Fiedler vector of the graph's Laplacian matrix. This process is repeated
    until each of the subgraphs' no. of edges does not exceed the no. of qubits.

    Args:
        graph (nx.DiGraph): The input directed graph to be partitioned.
        n_qubits (int, optional): The number of qubits per subgraph.
                                                Defaults to 8.

    Returns:
        list[nx.DiGraph]: A list of subgraphs resulting from the partitioning process.
    """
    subgraphs = [graph]

    while any(g.number_of_edges() > n_qubits for g in subgraphs):
        large_subgraphs = [g for g in subgraphs if g.number_of_edges() > n_qubits]
        subgraphs = [g for g in subgraphs if g.number_of_edges() <= n_qubits]

        if not large_subgraphs:
            break

        for large_subgraph in large_subgraphs:
            selected_subgraph, rest_of_graph = _divide_edges(
                large_subgraph, _fielder_laplacian_predicate
            )
            subgraphs.extend([selected_subgraph, rest_of_graph])

    return subgraphs


def _node_partition_graph(
    graph: nx.Graph,
    n_clusters: Optional[int] = None,
    avg_partition_size: Optional[float] = None,
) -> list[nx.Graph]:
    subgraphs = []
    total_edges = 0

    if not (bool(n_clusters) ^ bool(avg_partition_size)):
        raise RuntimeError(
            "Only one of 'n_clusters' and 'avg_partition_size' must be provided."
        )

    if n_clusters and n_clusters < 1:
        raise ValueError("'n_clusters' must be a positive integer.")

    if avg_partition_size and avg_partition_size < 1:
        raise ValueError("'avg_partition_size' must be a positive number.")

    if not n_clusters:
        n_clusters = int(graph.number_of_nodes() / avg_partition_size)

    adj_matrix = nx.to_numpy_array(graph)

    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
    partition = sc.fit_predict(adj_matrix)

    for i in range(n_clusters):
        subgraph = graph.subgraph(
            [node for node, cluster in enumerate(partition) if cluster == i]
        )

        if subgraph.number_of_edges() > 0:
            total_edges += subgraph.number_of_edges()
            subgraphs.append(subgraph)

    if total_edges != graph.number_of_edges():
        warn(
            f"Sum of edges count of partitions ({total_edges}) is different from the original graph {graph.number_of_edges()}"
        )
    return subgraphs


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
        n_qubits: int = None,
        n_clusters: int = None,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
        aggregate_fn: AggregateFn = linear_aggregation,
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        shots=5000,
        **kwargs,
    ):
        super().__init__()

        self.main_graph = graph

        self.n_qubits = n_qubits
        self.n_clusters = n_clusters

        if not (bool(self.n_qubits) ^ bool(self.n_clusters)):
            raise ValueError("One of `n_qubits` and `n_clusters` must be provided.")

        self.is_edge_problem = graph_problem == GraphProblem.EDGE_PARTITIONING

        self.aggregate_fn = aggregate_fn

        self._constructor = partial(
            QAOA,
            initial_state=initial_state,
            graph_problem=graph_problem,
            optimizer=optimizer,
            max_iterations=max_iterations,
            shots=shots,
            n_layers=n_layers,
            **kwargs,
        )

    def create_programs(self):
        if len(self.programs) > 0:
            raise RuntimeError(
                "Some programs already exist. "
                "Clear the program dictionary before creating new ones by using batch.reset()."
            )

        self.manager = Manager()

        if self.is_edge_problem:
            subgraphs = _edge_partition_graph(self.main_graph, n_qubits=self.n_qubits)
            cleaned_subgraphs = list(filter(lambda x: x.size() > 0, subgraphs))
        else:
            if not self.n_clusters:
                # Provide a smaller number than the available number of qubits
                # in case a bigger partition was provided (?)
                subgraphs = _node_partition_graph(
                    self.main_graph, avg_partition_size=self.n_qubits - 2
                )
            else:
                subgraphs = _node_partition_graph(
                    self.main_graph, n_clusters=self.n_clusters
                )

            self.solution = [0] * self.main_graph.number_of_nodes()
            self.reverse_index_maps = {}

            for i, subgraph in enumerate(subgraphs):
                index_map = {node: idx for idx, node in enumerate(subgraph.nodes())}
                self.reverse_index_maps[i] = {v: k for k, v in index_map.items()}
                _subgraph = nx.relabel_nodes(subgraph, index_map)

                self.programs[i] = self._constructor(
                    problem=_subgraph,
                    losses=self.manager.list(),
                    probs=self.manager.dict(),
                    final_params=self.manager.list(),
                )

        return

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

            minimum_probabilities = program.probs[f"{minimum_key}_0"]

            # The bitstring corresponding to the solution, with flip for correct endianness
            max_prob_key = max(minimum_probabilities, key=minimum_probabilities.get)[
                ::-1
            ]

            self.solution = self.aggregate_fn(
                self.solution, max_prob_key, program.problem, reverse_index_maps
            )

        return self.solution
