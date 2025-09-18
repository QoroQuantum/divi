# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import QAOA, GraphProblem
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    G = nx.bull_graph()

    qaoa_problem = QAOA(
        problem=G,
        graph_problem=GraphProblem.MAX_CLIQUE,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=10,
        is_constrained=True,
        backend=ParallelSimulator(),
    )

    qaoa_problem.run()
    qaoa_problem.compute_final_solution()

    losses = qaoa_problem.losses[-1]
    print(f"Minimum Energy Achieved: {min(losses.values()):.4f}")

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(f"Classical solution:{nx.algorithms.approximation.max_clique(G)}")
    print(f"Quantum Solution: {set(qaoa_problem.solution)}")

    qaoa_problem.draw_solution()
