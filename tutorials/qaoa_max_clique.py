# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.qprog.problems import MaxCliqueProblem, draw_graph_solution_nodes
from tutorials._backend import get_backend

if __name__ == "__main__":
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        MaxCliqueProblem(G, is_constrained=True),
        n_layers=2,
        optimizer=PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        max_iterations=5,
        backend=get_backend(),
    )

    qaoa_problem.run()
    print(f"Minimum Energy Achieved: {qaoa_problem.best_loss:.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(f"Classical solution:{nx.algorithms.approximation.max_clique(G)}")
    print(f"Quantum Solution: {set(qaoa_problem.solution)}")

    draw_graph_solution_nodes(G, qaoa_problem.solution)
