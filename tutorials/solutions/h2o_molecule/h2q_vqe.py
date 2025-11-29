# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
import itertools
import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import VQE, HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

# Sweep parameters for investigation
# 1. We sweep between the different bond lengths
# 2. We sweep over different Ansatz HartreeFockAnsatz, UCCSDAnsatz
# 3. We sweep over number of layers

sweep_params = {
    "bond_length": np.linspace(0.9, 1.5, 5),  # in bohr
    "ansatz": [HartreeFockAnsatz, UCCSDAnsatz],
    "n_layers": [1, 2, 3],
}

def run_vqe_sweep():
    optim = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
    results = []

    # Loop over all combinations of sweep parameters
    for bond_length, AnsatzClass, n_layers in itertools.product(
        sweep_params["bond_length"],
        sweep_params["ansatz"],
        sweep_params["n_layers"]
    ):
        mol = qml.qchem.Molecule(
            symbols=["H", "H"],
            coordinates=np.array([(0, 0, 0), (0, 0, bond_length)])
        )

        ansatz = AnsatzClass()

        vqe_problem = VQE(
            molecule=mol,
            ansatz=ansatz,
            n_layers=n_layers,
            optimizer=optim,
            max_iterations=3,
            backend=ParallelSimulator()
        )

        t_start = time.time()
        vqe_problem.run()
        t_end = time.time()

        result = {
            "bond_length": bond_length,
            "ansatz": AnsatzClass.__name__,
            "n_layers": n_layers,
            "energy": vqe_problem.best_loss,
            "eigenstate": vqe_problem.eigenstate,
            "total_circuits": vqe_problem.total_circuit_count,
            "time": round(t_end - t_start, 5)
        }
        results.append(result)

        print(
            f"Bond: {bond_length:.3f} | Ansatz: {AnsatzClass.__name__} | "
            f"Layers: {n_layers} | Energy: {vqe_problem.best_loss:.6f} | "
            f"Time: {round(t_end - t_start, 5)} s | Circuits: {vqe_problem.total_circuit_count}"
        )

    # Find the configuration with the lowest energy
    best_result = min(results, key=lambda x: x["energy"])
    print("\nBest Configuration Found:")
    print(best_result)

if __name__ == "__main__":
    run_vqe_sweep()
