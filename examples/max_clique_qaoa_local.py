from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

import networkx as nx

if __name__ == "__main__":
    qaoa_problem = QAOA(
        "max_clique",
        nx.octahedral_graph(),
        2,
        optimizer=Optimizers.MONTE_CARLO,
        is_constrained=True,
        qoro_service=None,
    )

    qaoa_problem.run()

    losses = qaoa_problem.losses[qaoa_problem.current_iteration - 1]
    print(losses)
    print(f"Minimum Energy Achieved: {min(losses.values()):.4f}")

    print(f"Total circuits: {len(qaoa_problem.circuits)}")

    # TODO: Print graph and solution
