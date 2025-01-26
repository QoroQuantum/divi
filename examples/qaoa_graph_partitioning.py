import random

import networkx as nx

from divi.qprog import GraphPartitioningQAOA
from divi.qprog.optimizers import Optimizers


def generate_random_graph(n_nodes: int, n_edges: int) -> nx.DiGraph:
    """
    Generate a random directed graph with the specified number of nodes and edges.

    Args:
        n_nodes (int): The number of nodes in the graph.
        n_edges (int): The number of edges in the graph.
    """
    graph = nx.Graph()

    nodes = range(n_nodes)
    graph.add_nodes_from(nodes)

    for _ in range(n_edges):
        u, v = random.sample(nodes, 2)

        # Avoid self-loops and duplicate edges
        while u == v or graph.has_edge(u, v):
            v = random.choice(nodes)
        weight = round(random.uniform(0.1, 1.0), 1)  # Round to 1 decimal place
        graph.add_edge(u, v, weight=weight)

    return graph


def analyze_results(classical_solution, quantum_solution):
    print(f"Classical Solution:\t{classical_solution}")
    print(f"Quantum Solution:\t{quantum_solution}")

    classical_ones = set([i for i, x in enumerate(classical_solution) if x == 1])
    quantum_ones = set([i for i, x in enumerate(quantum_solution) if x == 1])

    difference = len(quantum_ones.symmetric_difference(classical_ones))
    true_positives = classical_ones & quantum_ones

    print(f"No. of Mismatch in Nodes = {difference}")

    print(f"Recall = {len(true_positives) / len(classical_ones)}")
    print(f"Precision = {len(true_positives) / len(quantum_ones)}")
    print(f"Accuracy = {(N_NODES - difference) / N_NODES}")


if __name__ == "__main__":
    N_NODES = 25
    N_EDGES = 35

    graph = generate_random_graph(N_NODES, N_EDGES)

    curr_cut_size, partition = nx.approximation.one_exchange(graph, seed=1)
    classical_solution = [0] * graph.number_of_nodes()
    for node in partition[1]:
        classical_solution[node] = 1

    qaoa_batch = GraphPartitioningQAOA(
        problem="maxcut",
        graph=graph,
        n_layers=1,
        n_clusters=4,
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=5,
    )

    qaoa_batch.create_programs()
    qaoa_batch.run()
    quantum_solution = qaoa_batch.aggregate_results()

    analyze_results(classical_solution, quantum_solution)
