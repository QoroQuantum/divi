# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import hybrid
import numpy as np

from divi.backends import ParallelSimulator
from divi.qprog import QUBOPartitioningQAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

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
        backend=ParallelSimulator(),
    )

    qubo_partition.create_programs()
    qubo_partition.run().join()

    print(f"Total circuits: {qubo_partition.total_circuit_count}")

    quantum_solution, quantum_energy = qubo_partition.aggregate_results()

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.SimulatedAnnealingSampler().sample(bqm).lowest().record[0]
    )

    print(f"Classical Solution: {best_classical_bitstring}")
    print(f"Classical Energy: {best_classical_energy:.9f}")
    print(f"Quantum Solution: {quantum_solution}")
    print(f"Quantum Energy: {quantum_energy:.9f}")
