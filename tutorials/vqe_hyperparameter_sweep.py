# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import VQEAnsatz, VQEHyperparameterSweep
from divi.qprog.optimizers import Optimizers

if __name__ == "__main__":
    vqe_problem = VQEHyperparameterSweep(
        symbols=["H", "H"],
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        bond_lengths=list(np.linspace(0.1, 2.7, 5)),
        ansatze=[VQEAnsatz.HARTREE_FOCK],
        optimizer=Optimizers.MONTE_CARLO,
        max_iterations=3,
        backend=ParallelSimulator(shots=2000),
        grouping_strategy="wires",
    )

    t1 = time.time()

    vqe_problem.create_programs()
    vqe_problem.run()
    vqe_problem.aggregate_results()

    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")

    vqe_problem.visualize_results()
