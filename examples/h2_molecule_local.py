from divi.qprog import VQE, VQEAnsatze, VQEHyperparameterSweep
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService
import numpy as np

# This is an API key that only works on my local deployment, no one else can use it
# q_service = QoroService("4497dcabd079bedbeeec9d16b3dcccb1344461b9")
# q_service = QoroService("f634df7181c56dae4c7ba530ba0bfb2a0e6e3f4e")
q_service = None

if __name__ == "__main__":
    # vqe_problem = VQE(
    #     symbols=["H", "H"],
    #     bond_length=0.5,
    #     coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
    #     ansatz=VQEAnsatze.HARTREE_FOCK,
    #     optimizer=Optimizers.MONTE_CARLO,
    #     shots=5000,
    #     max_iterations=3,
    #     qoro_service=q_service,  # Run through the local simulator
    # )

    # vqe_problem.run()

    # energies = vqe_problem.energies[vqe_problem.current_iteration - 1]

    # print(energies)
    # print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    # print(f"Total circuits: {vqe_problem.total_circuit_count}")
    # np.linspace(0.1, 2.7, 2)

    vqe_problem = VQEHyperparameterSweep(
        symbols=["H", "H"],
        bond_lengths=np.linspace(0.2, 2.6, 10),
        ansatze=[VQEAnsatze.HARTREE_FOCK],
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        max_iterations=4,
        optimizer=Optimizers.MONTE_CARLO,
        qoro_service=q_service
    )
    vqe_problem.create_programs()
    for prog in vqe_problem.programs.values():
        print('running')
        prog.run()
    vqe_problem.visualize_results()
