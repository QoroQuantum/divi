# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from divi.qprog import QAOA, GraphProblem
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from tutorials._backend import get_backend

if __name__ == "__main__":
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        problem=G,
        graph_problem=GraphProblem.MAX_CLIQUE,
        n_layers=2,
        optimizer=PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        max_iterations=5,
        is_constrained=True,
        backend=get_backend(),
    )

    qaoa_problem.run()
    print(f"Minimum Energy Achieved: {qaoa_problem.best_loss:.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(f"Classical solution:{nx.algorithms.approximation.max_clique(G)}")
    print(f"Quantum Solution: {set(qaoa_problem.solution)}")

    qaoa_problem.draw_solution()
