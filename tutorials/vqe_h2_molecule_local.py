# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import VQE, VQEAnsatz
from divi.qprog.optimizers import Optimizer

if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        bond_length=0.5,
        ansatz=VQEAnsatz.HARTREE_FOCK,
        n_layers=1,
        optimizer=Optimizer.L_BFGS_B,
        max_iterations=3,
        backend=ParallelSimulator(),
    )

    t1 = time.time()

    vqe_problem.run()
    energies = vqe_problem.losses[-1]

    print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
