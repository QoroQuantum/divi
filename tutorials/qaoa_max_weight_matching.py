# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Maximum-weight matching with QAOA (standalone and partitioned)."""

import networkx as nx

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MaxWeightMatchingProblem, is_valid_matching
from divi.qprog.workflows import PartitioningProgramEnsemble
from tutorials._backend import get_backend

if __name__ == "__main__":
    # --- 1. Small graph: standalone QAOA ---

    G_small = nx.Graph()
    G_small.add_weighted_edges_from(
        [
            (0, 1, 5.0),
            (1, 2, 1.0),
            (2, 3, 5.0),
            (3, 4, 3.0),
        ]
    )

    problem = MaxWeightMatchingProblem(G_small, penalty_scale=10.0)
    backend = get_backend()

    qaoa = QAOA(
        problem,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=20,
        backend=backend,
    )
    qaoa.run()

    print(f"Quantum matching: {qaoa.solution}")
    print(f"Valid: {is_valid_matching(qaoa.solution)}")
    print(f"Best loss: {qaoa.best_loss:.4f}")
    print(f"Total circuits: {qaoa.total_circuit_count}")

    classical = nx.max_weight_matching(G_small, maxcardinality=False)
    classical_weight = sum(G_small[u][v]["weight"] for u, v in classical)
    print(f"Classical optimal weight: {classical_weight}")

    # --- 2. Larger graph: edge-based partitioning ---

    G_large = nx.gnm_random_graph(16, 30, seed=42)
    for u, v in G_large.edges():
        G_large[u][v]["weight"] = float(u + v + 1)

    problem_partitioned = MaxWeightMatchingProblem(
        G_large,
        penalty_scale=10.0,
        max_edges_per_partition=10,
        partition_algorithm="kernighan_lin",
        seed=42,
    )

    ensemble = PartitioningProgramEnsemble(
        problem=problem_partitioned,
        n_layers=1,
        backend=backend,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
    )

    ensemble.create_programs()
    print(f"\nCreated {len(ensemble.programs)} sub-programs")

    ensemble.run(blocking=True)
    matching, weight = ensemble.aggregate_results(beam_width=3)

    print(f"Quantum matching: {matching}")
    print(f"Weight: {weight}")
    print(f"Valid: {is_valid_matching(matching)}")
    print(f"Total circuits: {ensemble.total_circuit_count}")

    classical_large = nx.max_weight_matching(G_large, maxcardinality=False)
    classical_large_weight = sum(G_large[u][v]["weight"] for u, v in classical_large)
    print(f"Classical optimal weight: {classical_large_weight}")
    print(f"Approximation ratio: {weight / classical_large_weight:.2%}")
