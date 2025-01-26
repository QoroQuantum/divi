import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

if __name__ == "__main__":
    qaoa_problem = QAOA(
        "max_clique",
        nx.krackhardt_kite_graph(),
        n_layers=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=10,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    losses = qaoa_problem.losses[qaoa_problem.current_iteration - 1]
    print(losses)
    print(f"Minimum Energy Achieved: {min(losses.values()):.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    qaoa_problem.draw_solution()
