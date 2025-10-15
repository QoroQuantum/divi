# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import numpy as np

from divi.backends import ParallelSimulator
from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer


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
        backend=ParallelSimulator(shots=10000),
    )

    qaoa_problem.run()

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.ExactSolver().sample(bqm).lowest().record[0]
    )

    print(f"Classical Solution: {best_classical_bitstring}")
    print(f"Classical Energy: {best_classical_energy:.9f}")
    print(f"Quantum Solution: {qaoa_problem.solution}")
    print(f"Quantum Energy: {bqm.energy(qaoa_problem.solution):.9f}")
