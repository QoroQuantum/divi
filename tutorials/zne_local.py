from functools import partial

from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random

from divi.parallel_simulator import ParallelSimulator
from divi.qem import ZNE
from divi.qprog import VQE, VQEAnsatze
from divi.qprog.optimizers import Optimizers

if __name__ == "__main__":
    args = dict(
        symbols=["H", "H"],
        bond_length=0.5,
        coordinate_structure=[(0, 0, 0), (0, 0, 1)],
        n_layers=1,
        ansatz=VQEAnsatze.HARTREE_FOCK,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=5,
        seed=1997,
    )

    vqe_problem_exact = VQE(backend=ParallelSimulator(n_processes=4), **args)
    vqe_problem_exact.run()

    print(
        f"Minimum Energy Achieved (Exact): {min(vqe_problem_exact.losses[-1].values()):.4f}"
    )
    print(f"Circuits Executed (Exact): {vqe_problem_exact.total_circuit_count}")

    vqe_problem_noisy = VQE(backend=ParallelSimulator(qiskit_backend="auto"), **args)
    vqe_problem_noisy.run()

    print(
        f"Minimum Energy Achieved (Noisy): {min(vqe_problem_noisy.losses[-1].values()):.4f}"
    )
    print(f"Circuits Executed (Noisy): {vqe_problem_noisy.total_circuit_count}")

    scale_factors = [1.0, 3.0, 5.0]

    vqe_problem_zne = VQE(
        backend=ParallelSimulator(qiskit_backend="auto"),
        qem_protocol=ZNE(
            scale_factors,
            partial(fold_gates_at_random),
            RichardsonFactory(scale_factors=scale_factors),
        ),
        **args,
    )
    vqe_problem_zne.run()

    print(
        f"Minimum Energy Achieved (Mitigated): {min(vqe_problem_zne.losses[-1].values()):.4f}"
    )
    print(f"Circuits Executed (Mitigated): {vqe_problem_zne.total_circuit_count}")
