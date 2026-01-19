# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
PCE-VQE for a QUBO problem.

Demonstrates Pauli Correlation Encoding (PCE) with a random symmetric QUBO.
"""

from functools import partial

import dimod
import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import PCE, GenericLayerAnsatz
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.qprog.typing import qubo_to_matrix


def create_optimizer(pop_size: int) -> PymooOptimizer:
    return PymooOptimizer(method=PymooMethod.DE, population_size=pop_size)


def main() -> None:
    pop_size = 10
    iters = 10
    layers = 2

    ansatz = GenericLayerAnsatz(
        gate_sequence=[qml.RY, qml.RZ],
        entangler=qml.CNOT,
        entangling_layout="all-to-all",
    )

    bqm = dimod.generators.gnp_random_bqm(
        16,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )
    qubo_mat = qubo_to_matrix(bqm)

    optimizer = create_optimizer(pop_size)
    backend = ParallelSimulator(shots=10_000)

    solver = PCE(
        qubo_matrix=qubo_mat,
        ansatz=ansatz,
        optimizer=optimizer,
        backend=backend,
        max_iterations=iters,
        n_layers=layers,
        alpha=1.0,
    )
    solver.run()
    solution = solver.solution
    energy = float(bqm.energy(solution))

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.ExactSolver().sample(bqm).lowest().record[0]
    )

    print(f"Total circuits: {solver.total_circuit_count}\n")
    print("Method     Energy        Solution")
    print("-------    -----------   --------------------------------")
    print(f"PCE        {energy:>11.6f}   {solution}")
    print(f"Classical  {best_classical_energy:>11.6f}   {best_classical_bitstring}")


if __name__ == "__main__":
    main()
