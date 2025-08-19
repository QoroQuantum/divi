# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pennylane as qml

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import VQEAnsatz, VQEHyperparameterSweep
from divi.qprog._vqe_sweep import MoleculeTransformer
from divi.qprog.optimizers import Optimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    transformer = MoleculeTransformer(
        base_molecule=mol, bond_modifiers=[-0.4, -0.25, 0, 0.25, 0.4]
    )

    vqe_problem = VQEHyperparameterSweep(
        molecule_transformer=transformer,
        ansatze=[VQEAnsatz.HARTREE_FOCK],
        optimizer=Optimizer.MONTE_CARLO,
        max_iterations=3,
        backend=ParallelSimulator(shots=2000),
        grouping_strategy="wires",
    )

    t1 = time.time()

    vqe_problem.create_programs()
    vqe_problem.run(blocking=True)
    vqe_problem.aggregate_results()

    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")

    vqe_problem.visualize_results()
