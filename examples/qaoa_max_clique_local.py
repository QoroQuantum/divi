import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers


if __name__ == "__main__":
    qaoa_problem = QAOA(
        "max_clique",
        nx.bull_graph(),
        n_layers=1,
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=10,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    losses = qaoa_problem.losses[-1]
    print(losses)
    print(f"Minimum Energy Achieved: {min(losses.values()):.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(
        f"Classical solution:{nx.algorithms.approximation.max_clique(nx.bull_graph())}"
    )
    print(f"Quantum Solution: {set(qaoa_problem.compute_final_solution())}")

    qaoa_problem.draw_solution()
