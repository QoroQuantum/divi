# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
from dimod import ExactSolver

from divi.backends import ParallelSimulator
from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    # Create a BinaryQuadraticModel with more complexity
    bqm = dimod.BinaryQuadraticModel(
        {"w": 10, "x": -3, "y": 2, "z": -5, "a": 1, "b": -2},
        {
            ("w", "x"): -1,
            ("x", "y"): 1,
            ("y", "z"): -2,
            ("z", "a"): 3,
            ("a", "b"): -1,
            ("w", "z"): 2,
        },
        0.0,
        dimod.Vartype.BINARY,
    )

    qaoa_problem = QAOA(
        bqm,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=ParallelSimulator(shots=10000),
    )

    qaoa_problem.run()

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    # Get classical solution for comparison
    classical_samples = ExactSolver().sample(bqm)
    best_classical = classical_samples.first
    classical_solution = [best_classical.sample[v] for v in bqm.variables]

    # Get quantum solution (solution is a numpy array from _solution_bitstring)
    quantum_solution = qaoa_problem.solution
    solution_dict = {var: int(val) for var, val in zip(bqm.variables, quantum_solution)}
    quantum_energy = bqm.energy(solution_dict)

    # Print solutions side by side for easy comparison
    print(f"Classical Solution:\t[{''.join(map(str, classical_solution))}]")
    print(f"Quantum Solution:\t[{''.join(map(str, quantum_solution))}]")
    print(f"Classical Energy:\t{best_classical.energy:.9f}")
    print(f"Quantum Energy:\t\t{quantum_energy:.9f}")
