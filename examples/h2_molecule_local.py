from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
        ansatz=VQEAnsatze.HARTREE_FOCK,
        optimizer=Optimizers.MONTE_CARLO,
        shots=5000,
        max_iterations=4,
        qoro_service=None,  # Run through the local simulator
    )

    vqe_problem.run()

    energies = vqe_problem.energies[vqe_problem.current_iteration - 1]

    print(energies)

    print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")

    # vqe_problem.visualize_results()

    # data = []
    # for energy in vqe_problem.energies:
    #     data.append(energy[Ansatze.HARTREE_FOCK][0])
