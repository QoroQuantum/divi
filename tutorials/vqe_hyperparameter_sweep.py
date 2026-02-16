# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pennylane as qml

from divi.qprog import (
    HartreeFockAnsatz,
    MoleculeTransformer,
    UCCSDAnsatz,
    VQEHyperparameterSweep,
)
from divi.qprog.optimizers import MonteCarloOptimizer
from tutorials._backend import get_backend

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    transformer = MoleculeTransformer(
        base_molecule=mol, bond_modifiers=[-0.4, -0.25, 0, 0.25, 0.4]
    )

    optim = MonteCarloOptimizer(population_size=10, n_best_sets=3)

    vqe_problem = VQEHyperparameterSweep(
        molecule_transformer=transformer,
        ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()],
        optimizer=optim,
        max_iterations=3,
        backend=get_backend(shots=2000),
    )

    t1 = time.time()

    vqe_problem.create_programs()
    vqe_problem.run(blocking=True)
    vqe_problem.aggregate_results()

    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")

    vqe_problem.visualize_results("line")
