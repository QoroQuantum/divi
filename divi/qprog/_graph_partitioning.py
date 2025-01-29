from collections.abc import Callable
from functools import partial
from multiprocessing import Manager
from typing import Optional

import networkx as nx
import numpy as np
import rustworkx as rx
import scipy.sparse.linalg as spla
from sklearn.cluster import SpectralClustering

from divi.qprog import QAOA, ProgramBatch
from divi.qprog._qaoa import (
    _SUPPORTED_PROBLEMS,
    _SUPPORTED_PROBLEMS_LITERAL,
    _SUPPORTED_INITIAL_STATES_LITERAL,
)

from .optimizers import Optimizers


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

    # _, eigenvectors = spla.eigs(L, k=2, which="SM")
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
        n_clusters = int(graph.number_of_nodes / avg_partition_size)

    adj_matrix = nx.to_numpy_array(graph)

    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
    partition = sc.fit_predict(adj_matrix)

    for i in range(n_clusters):
        subgraph = graph.subgraph(
            [node for node, cluster in enumerate(partition) if cluster == i]
        )
        total_edges += subgraph.number_of_edges()
        subgraphs.append(subgraph)

    print(f"ATTENTION: Total edges are fewer than the original graph {total_edges}")
    return subgraphs


class GraphPartitioningQAOA(ProgramBatch):
    def __init__(
        self,
        problem: _SUPPORTED_PROBLEMS_LITERAL,
        graph: nx.Graph | rx.PyGraph,
        n_layers: int,
        n_qubits: int = None,
        n_clusters: int = None,
        initial_state: _SUPPORTED_INITIAL_STATES_LITERAL = "Recommended",
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

        self.is_edge_problem = problem not in _SUPPORTED_PROBLEMS

        self._constructor = partial(
            QAOA,
            initial_state=initial_state,
            problem=problem,
            optimizer=optimizer,
            max_iterations=max_iterations,
            shots=shots,
            n_layers=n_layers,
            **kwargs,
        )

    def create_programs(self):
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
                    graph=_subgraph,
                    losses=self.manager.list(),
                    probs=self.manager.list(),
                )

        return

    # for each in subgraphs:
    #     index_map = {node: idx for idx, node in enumerate(each.nodes())}
    #     subgraph = nx.relabel_nodes(each, index_map)
    #     probs = run_qaoa(subgraph, index_map)
    #     print(probs)
    #     for node in each.nodes():
    #         index = index_map[node]
    #         solutions[node] = int(probs[index])
    def aggregate_results(self):
        if self.executor is not None:
            self.wait_for_all()

        # Extract the solutions from each program
        for program, reverse_index_maps in zip(
            self.programs.values(), self.reverse_index_maps.values()
        ):
            # Extract the final probabilities of the lowest energy
            last_iteration_losses = program.losses[-1]
            minimum_key = min(last_iteration_losses, key=last_iteration_losses.get)

            minimum_probabilities = program.probs[-1][minimum_key]

            # The bitstring corresponding to the solution
            max_prob_key = max(minimum_probabilities, key=minimum_probabilities.get)

            for node in program.graph.nodes():
                solution_index = reverse_index_maps[node]
                self.solution[solution_index] = int(max_prob_key[node])

        return self.solution
