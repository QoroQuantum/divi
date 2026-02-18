# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import hybrid
import numpy as np

from divi.qprog import QUBOPartitioningQAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

if __name__ == "__main__":
    bqm: dimod.BinaryQuadraticModel = dimod.generators.gnp_random_bqm(
        25,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )

    qubo_partition = QUBOPartitioningQAOA(
        qubo=bqm,
        decomposer=hybrid.EnergyImpactDecomposer(size=5),
        composer=hybrid.SplatComposer(),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=get_backend(),
    )

    qubo_partition.create_programs()
    qubo_partition.run().join()

    print(f"Total circuits: {qubo_partition.total_circuit_count}")

    # --- Greedy aggregation (default) ---
    greedy_solution, greedy_energy = qubo_partition.aggregate_results(beam_width=1)

    # --- Beam search aggregation ---
    # beam_width=3: keep top 3 partial solutions after each partition step
    # n_partition_candidates=5: consider 5 candidates from each partition
    beam_solution, beam_energy = qubo_partition.aggregate_results(
        beam_width=3, n_partition_candidates=5
    )

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.SimulatedAnnealingSampler().sample(bqm).lowest().record[0]
    )

    print(f"\nClassical Solution: {best_classical_bitstring}")
    print(f"Classical Energy: {best_classical_energy:.9f}")
    print(f"\n--- Greedy (beam_width=1) ---")
    print(f"Solution: {greedy_solution}")
    print(f"Energy:   {greedy_energy:.9f}")
    print(f"\n--- Beam Search (beam_width=3, n_partition_candidates=5) ---")
    print(f"Solution: {beam_solution}")
    print(f"Energy:   {beam_energy:.9f}")
