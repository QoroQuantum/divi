import random

import networkx as nx

from divi.qprog import GraphPartitioningQAOA, GraphProblem
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService


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


def analyze_results(quantum_solution, classical_cut_size):
    cut_edges = 0
    for u, v in graph.edges():
        if quantum_solution[u] != quantum_solution[v]:
            cut_edges += 1

    print(
        f"Quantum Cut Size to Classical Cut Size Ratio = {cut_edges / classical_cut_size}"
    )


if __name__ == "__main__":
    N_NODES = 15
    N_EDGES = 20

    graph = generate_random_graph(N_NODES, N_EDGES)

    classical_cut_size, classical_partition = nx.approximation.one_exchange(
        graph, seed=1
    )

    # q_service = QoroService("4497dcabd079bedbeeec9d16b3dcccb1344461b9")
    q_service = None

    qaoa_batch = GraphPartitioningQAOA(
        graph_problem=GraphProblem.MAXCUT,
        graph=graph,
        n_layers=1,
        n_clusters=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=20,
        qoro_service=q_service,
    )

    qaoa_batch.create_programs()
    qaoa_batch.run()
    quantum_solution = qaoa_batch.aggregate_results()

    print(f"Total circuits: {qaoa_batch.total_circuit_count}")
    analyze_results(quantum_solution, classical_cut_size)
