from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService

# This is an API key that only works on my local deployment, no one else can use it
q_service = QoroService("f634df7181c56dae4c7ba530ba0bfb2a0e6e3f4e")
# q_service = None
if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_lengths=[0.5, 0.75],
        coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
        ansatze=[VQEAnsatze.HARTREE_FOCK, VQEAnsatze.RY],
        optimizer=Optimizers.MONTE_CARLO,
        shots=5000,
        max_iterations=2,
        qoro_service=q_service,  # Run through the local simulator
    )

    vqe_problem.run()
    energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
    ansatz = vqe_problem.ansatze[0]
    print(energies)
    for i in range(len(vqe_problem.bond_lengths)):
        print(f"Minimum Energy Achieved: {
              min(energies[i][ansatz].values()):.4f}")

    vqe_problem.visualize_results()

    # data = []
    # for energy in vqe_problem.energies:
    #     data.append(energy[Ansatze.HARTREE_FOCK][0])

    c = 0
    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            c += 1

    print(f"Total circuits: {c}")
