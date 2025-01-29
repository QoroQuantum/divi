from divi.qprog import VQE, VQEHyperparameterSweep
from divi.qprog import VQEAnsatze
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService

import time
import numpy as np

# This is an API key that only works on my local deployment, no one else can use it
# q_service = QoroService("f634df7181c56dae4c7ba530ba0bfb2a0e6e3f4e")

# This one is live
q_service = QoroService("4497dcabd079bedbeeec9d16b3dcccb1344461b9")

# This uses local sim
# q_service = None


def run_program(prog):
    print('running')
    prog.run()


if __name__ == "__main__":
    # vqe_problem = VQE(
    #     symbols=["H", "H"],
    #     bond_length=0.5,
    #     coordinate_structure=[(0, 0, 0), (0, 0, 1)],
    #     ansatz=VQEAnsatze.HARTREE_FOCK,
    #     optimizer=Optimizers.MONTE_CARLO,
    #     shots=5000,
    #     max_iterations=3,
    #     qoro_service=q_service,  # Run through the local simulator
    # )
    # t1 = time.time()
    # vqe_problem.run()
    # energies = vqe_problem.energies[vqe_problem.current_iteration - 1]

    # print(energies)
    # print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    # print(f"Total circuits: {vqe_problem.total_circuit_count}")
    # print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    # np.linspace(0.1, 2.7, 2)

    vqe_problem = VQEHyperparameterSweep(
        symbols=["H", "H"],
        bond_lengths=list(np.linspace(0.1, 2.7, 10)),
        ansatze=[VQEAnsatze.HARTREE_FOCK],
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        max_iterations=3,
        optimizer=Optimizers.MONTE_CARLO,
        qoro_service=q_service,
        shots=2000
    )

    t1 = time.time()
    vqe_problem.create_programs()
    vqe_problem.run()
    vqe_problem.aggregate_results()
    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
    vqe_problem.visualize_results()
