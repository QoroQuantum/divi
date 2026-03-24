# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
from dimod import ExactSolver

from divi.qprog import QAOA, BinaryOptimizationProblem
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

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
        BinaryOptimizationProblem(bqm),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=get_backend(shots=10000),
    )

    qaoa_problem.run()

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    # Get classical solution for comparison
    classical_samples = ExactSolver().sample(bqm)
    best_classical = classical_samples.first
    classical_solution = [best_classical.sample[v] for v in bqm.variables]

    # Get quantum solution (dict for named BQM vars, else array in variable order)
    sol = qaoa_problem.solution
    solution_dict = (
        {v: int(sol[v]) for v in bqm.variables}
        if isinstance(sol, dict)
        else dict(zip(bqm.variables, sol))
    )
    quantum_energy = bqm.energy(solution_dict)
    quantum_values = [solution_dict[v] for v in bqm.variables]

    # Print solutions side by side for easy comparison
    print(f"Classical Solution:\t[{''.join(map(str, classical_solution))}]")
    print(f"Quantum Solution:\t[{''.join(map(str, quantum_values))}]")
    print(f"Classical Energy:\t{best_classical.energy:.9f}")
    print(f"Quantum Energy:\t\t{quantum_energy:.9f}")
