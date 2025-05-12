import time

import numpy as np

from divi.qprog import VQEAnsatze, VQEHyperparameterSweep
from divi.qprog.optimizers import Optimizers

if __name__ == "__main__":
    vqe_problem = VQEHyperparameterSweep(
        symbols=["H", "H"],
        bond_lengths=list(np.linspace(0.1, 2.7, 5)),
        ansatze=[VQEAnsatze.HARTREE_FOCK],
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        max_iterations=3,
        optimizer=Optimizers.MONTE_CARLO,
        shots=2000,
        qoro_service=None,
    )

    t1 = time.time()

    vqe_problem.create_programs()
    vqe_problem.run()
    vqe_problem.aggregate_results()

    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")

    vqe_problem.visualize_results()
