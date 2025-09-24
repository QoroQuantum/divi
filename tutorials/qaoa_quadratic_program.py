# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from divi.backends import ParallelSimulator
from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    qp = QuadraticProgram()
    qp.binary_var("w")
    qp.binary_var("x")
    qp.binary_var("y")
    qp.integer_var(lowerbound=0, upperbound=7, name="z")

    qp.minimize(linear={"x": -3, "y": 2, "z": -1, "w": 10})

    qaoa_problem = QAOA(
        qp,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=ParallelSimulator(shots=10000),
    )

    qaoa_problem.run()
    qaoa_problem.compute_final_solution()

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    try:
        from qiskit_algorithms import NumPyMinimumEigensolver

        exact_solver = exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        sol = exact_solver.solve(qp)
        print(f"Classical Solution: {sol.raw_samples[0].x}")
        print(f"Classical Energy: {sol.raw_samples[0].fval:.9f}")
    except ImportError:
        pass

    print(f"Quantum Solution: {qaoa_problem.solution}")
    print(
        f"Quantum Energy: {qaoa_problem.problem.objective.evaluate(qaoa_problem.solution).item():.9f}"
    )
