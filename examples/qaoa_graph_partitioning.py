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
    graph = nx.DiGraph()

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


if __name__ == "__main__":
    graph = generate_random_graph(15, 20)

    batch = GraphPartitioningQAOA(
        problem="maxcut",
        graph=graph,
        n_layers=1,
        n_clusters=3,
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=5,
    )

    batch.create_programs()
    batch.run()
    batch.aggregate_results()
