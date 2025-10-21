# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import pennylane as qml
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random

from divi.backends import ParallelSimulator
from divi.circuits.qem import ZNE
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    args = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=5,
        seed=1997,
    )

    vqe_problem_exact = VQE(backend=ParallelSimulator(n_processes=4), **args)
    vqe_problem_exact.run()

    print(f"Lowest Energy Achieved (Exact): {vqe_problem_exact.best_loss:.4f}")
    print(f"Circuits Executed (Exact): {vqe_problem_exact.total_circuit_count}")

    vqe_problem_noisy = VQE(
        backend=ParallelSimulator(n_processes=4, qiskit_backend="auto"), **args
    )
    vqe_problem_noisy.run()

    print(f"Lowest Energy Achieved (Noisy): {vqe_problem_noisy.best_loss:.4f}")
    print(f"Circuits Executed (Noisy): {vqe_problem_noisy.total_circuit_count}")

    scale_factors = [1.0, 3.0, 5.0]

    vqe_problem_zne = VQE(
        backend=ParallelSimulator(n_processes=4, qiskit_backend="auto"),
        qem_protocol=ZNE(
            scale_factors,
            partial(fold_gates_at_random),
            RichardsonFactory(scale_factors=scale_factors),
        ),
        **args,
    )
    vqe_problem_zne.run()

    print(f"Lowest Energy Achieved (Mitigated): {vqe_problem_zne.best_loss:.4f}")
    print(f"Circuits Executed (Mitigated): {vqe_problem_zne.total_circuit_count}")
