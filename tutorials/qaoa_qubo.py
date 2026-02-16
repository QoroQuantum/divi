# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import numpy as np

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def _bqm_to_numpy_matrix(bqm):
    ldata, (irow, icol, qdata), _ = bqm.to_numpy_vectors(range(bqm.num_variables))

    # make sure it's upper triangular
    idx = irow > icol
    if idx.any():
        irow[idx], icol[idx] = icol[idx], irow[idx]

    dense = np.zeros((bqm.num_variables, bqm.num_variables), dtype=bqm.dtype)
    dense[irow, icol] = qdata

    # set the linear
    np.fill_diagonal(dense, ldata)

    return dense


if __name__ == "__main__":
    bqm = dimod.generators.gnp_random_bqm(
        10,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )

    qubo_array = _bqm_to_numpy_matrix(bqm)

    qaoa_problem = QAOA(
        problem=qubo_array,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=get_backend(shots=10000),
    )

    qaoa_problem.run()

    quantum_solution = qaoa_problem.solution
    quantum_energy = bqm.energy(quantum_solution)
    best_classical_bitstring, best_classical_energy, _ = (
        dimod.ExactSolver().sample(bqm).lowest().record[0]
    )

    quantum_bitstring = "".join(str(bit) for bit in quantum_solution)
    classical_bitstring = "".join(str(bit) for bit in best_classical_bitstring)
    bitstring_width = max(len(quantum_bitstring), len(classical_bitstring))

    print(f"Total circuits: {qaoa_problem.total_circuit_count}\n")
    print("Method     Energy        Bitstring")
    print("-------    -----------   " + "-" * bitstring_width)
    print(
        f"QAOA       {quantum_energy:>11.6f}   {quantum_bitstring:<{bitstring_width}}"
    )
    print(
        f"Classical  {best_classical_energy:>11.6f}   "
        f"{classical_bitstring:<{bitstring_width}}"
    )

    # Demonstrate top-N solutions functionality
    print("\n" + "=" * 60)
    print("Top 5 Solutions by Probability:")
    print("=" * 60)
    top_solutions = qaoa_problem.get_top_solutions(n=5)
    for i, sol in enumerate(top_solutions, 1):
        # Convert bitstring to numpy array for energy calculation
        solution_array = np.array([int(bit) for bit in sol.bitstring])
        energy = bqm.energy(solution_array)
        print(
            f"{i}. Bitstring: {sol.bitstring} | "
            f"Probability: {sol.prob:0.2%} | "
            f"Energy: {energy:.4f}"
        )
