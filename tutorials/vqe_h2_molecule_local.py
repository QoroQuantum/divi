# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pennylane as qml

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import VQE, VQEAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    optim = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

    vqe_problem = VQE(
        molecule=mol,
        ansatz=VQEAnsatz.HARTREE_FOCK,
        n_layers=1,
        optimizer=optim,
        max_iterations=3,
        backend=ParallelSimulator(),
    )

    t1 = time.time()

    vqe_problem.run()
    energies = vqe_problem.losses[-1]

    print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
