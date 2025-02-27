import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

if __name__ == "__main__":
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        "max_clique",
        G,
        n_layers=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=10,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    losses = qaoa_problem.losses[-1]
    print(f"Minimum Energy Achieved: {min(losses.values()):.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(f"Classical solution:{nx.algorithms.approximation.max_clique(G)}")
    print(f"Quantum Solution: {set(qaoa_problem.compute_final_solution())}")

    qaoa_problem.draw_solution()
