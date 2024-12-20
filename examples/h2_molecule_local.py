from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService

# This is an API key that only works on my local deployment, no one else can use it
q_service = QoroService("f634df7181c56dae4c7ba530ba0bfb2a0e6e3f4e")
# q_service = None
if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
        ansatz=VQEAnsatze.HARTREE_FOCK,
        optimizer=Optimizers.MONTE_CARLO,
        shots=5000,
        max_iterations=2,
        qoro_service=q_service,  # Run through the local simulator
    )

    vqe_problem.run()

    energies = vqe_problem.energies[vqe_problem.current_iteration - 1]

    print(energies)

    print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
