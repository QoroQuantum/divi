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
