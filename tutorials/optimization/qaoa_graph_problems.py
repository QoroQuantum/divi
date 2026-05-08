# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOA on combinatorial graph problems.

Two short examples that share a common pattern — instantiate a problem class,
hand it to ``QAOA``, run, compare against a classical baseline.

1. Maximum clique with a constrained mixer (``is_constrained=True``).
2. Maximum-weight matching with the default penalty-based formulation.

For partitioning a single large graph problem across multiple QAOA programs,
see ``qaoa_partitioning.py``.
"""

import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import (
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import (
    MaxCliqueProblem,
    MaxWeightMatchingProblem,
    draw_graph_solution_nodes,
    is_valid_matching,
)
from tutorials._backend import get_backend

if __name__ == "__main__":
    backend = get_backend()

    # ── 1) Maximum clique (constrained mixer) ─────────────────────────
    print("Part 1 — Max Clique on the bull graph")
    print("-" * 40)

    G_clique = nx.bull_graph()

    qaoa_clique = QAOA(
        MaxCliqueProblem(G_clique, is_constrained=True),
        n_layers=2,
        optimizer=PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
        max_iterations=5,
        backend=backend,
    )
    qaoa_clique.run()

    print(f"Best loss: {qaoa_clique.best_loss:.4f}")
    print(f"Total circuits: {qaoa_clique.total_circuit_count}")
    print(f"Classical solution: {nx.algorithms.approximation.max_clique(G_clique)}")
    print(f"Quantum solution:   {set(qaoa_clique.solution)}")

    draw_graph_solution_nodes(G_clique, qaoa_clique.solution)

    # ── 2) Maximum-weight matching (penalty-based) ────────────────────
    print("\nPart 2 — Max Weight Matching on a small weighted graph")
    print("-" * 40)

    G_match = nx.Graph()
    G_match.add_weighted_edges_from(
        [
            (0, 1, 5.0),
            (1, 2, 1.0),
            (2, 3, 5.0),
            (3, 4, 3.0),
        ]
    )

    qaoa_match = QAOA(
        MaxWeightMatchingProblem(G_match, penalty_scale=10.0),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=20,
        backend=backend,
    )
    qaoa_match.run()

    classical = nx.max_weight_matching(G_match, maxcardinality=False)
    classical_weight = sum(G_match[u][v]["weight"] for u, v in classical)

    print(f"Quantum matching: {qaoa_match.solution}")
    print(f"Valid:            {is_valid_matching(qaoa_match.solution)}")
    print(f"Best loss:        {qaoa_match.best_loss:.4f}")
    print(f"Total circuits:   {qaoa_match.total_circuit_count}")
    print(f"Classical optimal weight: {classical_weight}")
