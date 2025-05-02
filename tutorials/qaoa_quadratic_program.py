import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

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
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=10,
        shots=10000,
        qoro_service=None,
    )

    qaoa_problem.run()
    qaoa_problem.compute_final_solution()

    exact_solver = exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    sol = exact_solver.solve(qp)

    print(f"Total circuits: {qaoa_problem.total_circuit_count}")

    print(f"Classical Solution: {sol.raw_samples[0].x}")
    print(f"Classical Energy: {sol.raw_samples[0].fval:.9f}")
    print(f"Quantum Solution: {qaoa_problem.solution}")
    print(
        f"Quantum Energy: {qaoa_problem.problem.objective.evaluate(qaoa_problem.solution).item():.9f}"
    )
