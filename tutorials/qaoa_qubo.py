import numpy as np
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram

from divi.qprog import QAOA
from divi.qprog.optimizers import Optimizers

try:
    import dimod
except ImportError:
    raise ImportError(
        "This functionality requires the 'dimod' package. "
        "Please install it with:\n"
        "    pip install dimod"
    )

if __name__ == "__main__":
    bqm = dimod.generators.randint(5, vartype="BINARY", low=-10, high=10, seed=1997)
    qubo_array = bqm.to_numpy_matrix()

    qaoa_problem = QAOA(
        problem=qubo_array,
        n_layers=2,
        optimizer=Optimizers.NELDER_MEAD,
        max_iterations=5,
        shots=10000,
        qoro_service=None,
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
