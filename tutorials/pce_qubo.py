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

from divi.hamiltonians import qubo_to_matrix
from divi.qprog import PCE, GenericLayerAnsatz
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from tutorials._backend import get_backend


def create_optimizer(pop_size: int) -> PymooOptimizer:
    return PymooOptimizer(method=PymooMethod.DE, population_size=pop_size)


def main():
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
    backend = get_backend(shots=10_000)

    solver = PCE(
        problem=qubo_mat,
        ansatz=ansatz,
        optimizer=optimizer,
        backend=backend,
        max_iterations=iters,
        n_layers=layers,
        alpha=1.0,
    )
    solver.run()

    best_classical_array, best_classical_energy, _ = (
        dimod.ExactSolver().sample(bqm).lowest().record[0]
    )
    best_classical_bitstring = "".join(str(int(b)) for b in best_classical_array)

    # Get top solutions sorted by energy (ascending)
    top_solutions = solver.get_top_solutions(n=5, min_prob=0.01, sort_by="energy")
    best_pce = top_solutions[0]

    print(f"Total circuits: {solver.total_circuit_count}\n")
    print("Method     Energy        Solution")
    print("-------    -----------   --------------------------------")
    print(f"PCE        {best_pce.energy:>11.6f}   {best_pce.bitstring}")
    print(f"Classical  {best_classical_energy:>11.6f}   {best_classical_bitstring}")

    # Demonstrate get_top_solutions - returns decoded QUBO solutions
    print("\n" + "=" * 80)
    print("Top 5 solutions from PCE (decoded QUBO variable assignments):")
    print("=" * 80)

    # Print table header
    print(f"{'Rank':<6} {'Bitstring':<20} {'Probability':<15} {'Energy':<15}")
    print("-" * 80)

    # Print table rows
    for i, sol in enumerate(top_solutions, 1):
        print(f"{i:<6} {sol.bitstring:<20} {sol.prob:>13.2%}  {sol.energy:>13.6f}")


if __name__ == "__main__":
    main()
