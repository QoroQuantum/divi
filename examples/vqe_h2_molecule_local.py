import time

from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers
from divi.services import QoroService

# This one is live
# from divi.services import QoroService
q_service = QoroService("4497dcabd079bedbeeec9d16b3dcccb1344461b9")

# This uses local sim
# q_service = None


if __name__ == "__main__":
    vqe_problem = VQE(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        n_layers=1,
        ansatz=VQEAnsatze.HARTREE_FOCK,
        optimizer=Optimizers.L_BFGS_B,
        max_iterations=3,
        qoro_service=q_service,
        shots=5000,
    )

    t1 = time.time()

    vqe_problem.run()
    energies = vqe_problem.losses[-1]

    print(f"Minimum Energy Achieved: {min(energies.values()):.4f}")
    print(f"Total circuits: {vqe_problem.total_circuit_count}")
    print(f"Time taken: {round(time.time() - t1, 5)} seconds")
    print(f"Simulation time: {vqe_problem.total_run_time}")
