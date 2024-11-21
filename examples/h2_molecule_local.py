from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers

# q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_lengths=[0.5, 0.75, 1, 1.25],
        coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
        ansatze=[VQEAnsatze.HARTREE_FOCK, VQEAnsatze.RY],
        optimizer=Optimizers.MONTE_CARLO,
        shots=5000,
        max_iterations=4,
        qoro_service=None,  # Run through the local simulator
    )

    vqe_problem.run()
    energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
    ansatz = vqe_problem.ansatze[0]
    print(energies)
    for i in range(len(vqe_problem.bond_lengths)):
        print(f"Minimum Energy Achieved: {
              min(energies[i][ansatz].values()):.4f}")

    breakpoint()

    vqe_problem.visualize_results()

    # data = []
    # for energy in vqe_problem.energies:
    #     data.append(energy[Ansatze.HARTREE_FOCK][0])

    c = 0
    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            c += 1

    print(f"Total circuits: {c}")
