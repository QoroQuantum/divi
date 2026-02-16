# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import random
import warnings

import networkx as nx

from divi.qprog import GraphPartitioningQAOA, GraphProblem, PartitioningConfig
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def generate_random_graph(n_nodes: int, n_edges: int) -> nx.Graph:
    """
    Generate a random undirected graph with the specified number of nodes and edges.
    Ensures connectivity if possible. Extra edges are added randomly without duplicates.

    Args:
        n_nodes (int): Number of nodes.
        n_edges (int): Desired number of edges.

    Returns:
        nx.Graph: The resulting graph.
    """
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        warnings.warn(
            f"Requested {n_edges} edges, but max for {n_nodes} nodes is {max_edges}. Capping to max."
        )
        n_edges = max_edges

    # Start with a spanning tree (ensures connected)
    graph = nx.random_labeled_tree(n_nodes)
    current_edges = set(graph.edges())

    # Add random edges until desired count is reached
    while len(current_edges) < n_edges:
        u, v = random.sample(range(n_nodes), 2)
        edge = tuple(sorted((u, v)))
        if edge not in current_edges:
            graph.add_edge(*edge, weight=round(random.uniform(0.1, 1.0), 1))
            current_edges.add(edge)

    # Assign weights to initial tree edges (if not already)
    for u, v in graph.edges():
        if "weight" not in graph[u][v]:
            graph[u][v]["weight"] = round(random.uniform(0.1, 1.0), 1)

    return graph


def analyze_results(quantum_solution, classical_cut_size):
    cut_edges = 0

    for u, v in graph.edges():
        if (u in quantum_solution) != (v in quantum_solution):
            cut_edges += 1

    print(
        f"Quantum Cut Size to Classical Cut Size Ratio = {cut_edges / classical_cut_size}"
    )


if __name__ == "__main__":
    N_NODES = 30
    N_EDGES = 40

    graph = generate_random_graph(N_NODES, N_EDGES)

    qaoa_batch = GraphPartitioningQAOA(
        graph_problem=GraphProblem.MAXCUT,
        graph=graph,
        n_layers=1,
        partitioning_config=PartitioningConfig(
            max_n_nodes_per_cluster=10,
            partitioning_algorithm="metis",
        ),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=20,
        backend=get_backend(),
    )

    qaoa_batch.create_programs()
    qaoa_batch.run(blocking=True)
    quantum_solution = qaoa_batch.aggregate_results()

    classical_cut_size, classical_partition = nx.approximation.one_exchange(
        graph, seed=1
    )
    print(f"Total circuits: {qaoa_batch.total_circuit_count}")
    analyze_results(quantum_solution, classical_cut_size)
