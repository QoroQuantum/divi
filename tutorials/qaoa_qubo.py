from functools import partial

import numpy as np

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

try:
    import dimod
except ImportError:
    raise ImportError(
        "This tutorial requires the 'dimod' package. "
        "Please install it with:\n"
        "    pip install dimod"
    )

if __name__ == "__main__":
    bqm = dimod.generators.gnp_random_bqm(
        10,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )
    qubo_array = bqm.to_numpy_matrix()

    qaoa_problem = QAOA(
        problem=qubo_array,
        n_layers=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=10,
        backend=ParallelSimulator(shots=10000),
    )

    qaoa_problem.run()
    qaoa_problem.compute_final_solution()

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.ExactSolver().sample(bqm).lowest().record[0]
    )

    print(f"Classical Solution: {best_classical_bitstring}")
    print(f"Classical Energy: {best_classical_energy:.9f}")
    print(f"Quantum Solution: {qaoa_problem.solution}")
    print(f"Quantum Energy: {bqm.energy(qaoa_problem.solution):.9f}")
