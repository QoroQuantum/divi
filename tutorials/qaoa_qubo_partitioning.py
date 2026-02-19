# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import hybrid
import numpy as np

from divi.qprog import EarlyStopping, QUBOPartitioningQAOA
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
        max_iterations=30,
        early_stopping=EarlyStopping(patience=5),
        backend=get_backend(),
    )

    qubo_partition.create_programs()
    qubo_partition.run().join()

    print(f"Total circuits: {qubo_partition.total_circuit_count}")

    # --- Greedy aggregation (default) ---
    greedy_solution, greedy_energy = qubo_partition.aggregate_results(
        beam_width=1, n_partition_candidates=5
    )

    # --- Beam search aggregation ---
    # beam_width=3: keep top 3 partial solutions after each partition step
    # n_partition_candidates=5: consider 5 candidates from each partition
    beam_solution, beam_energy = qubo_partition.aggregate_results(
        beam_width=3, n_partition_candidates=5
    )

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.SimulatedAnnealingSampler().sample(bqm).lowest().record[0]
    )

    # --- Print comparison table ---
    rows = [
        ("Classical (SA)", best_classical_energy, str(best_classical_bitstring), "-"),
        (
            "Greedy (beam=1)",
            greedy_energy,
            str(greedy_solution),
            qubo_partition.total_circuit_count,
        ),
        (
            "Beam Search (beam=3)",
            beam_energy,
            str(beam_solution),
            qubo_partition.total_circuit_count,
        ),
    ]
    pad = "  "
    col_m, col_e, col_c = 22, 16, 10
    sep_len = col_m + col_e + col_c + len(pad) * 2
    print("\n" + "-" * sep_len)
    print(f"{'Method':<{col_m}}{pad}{'Energy':>{col_e}}{pad}{'# Circuits':>{col_c}}")
    print("-" * sep_len)
    for method, energy, sol, circuits in rows:
        print(
            f"{method:<{col_m}}{pad}{energy:>{col_e}.6f}{pad}{str(circuits):>{col_c}}"
        )
        print(f"  Solution: {sol}")
    print("-" * sep_len)
