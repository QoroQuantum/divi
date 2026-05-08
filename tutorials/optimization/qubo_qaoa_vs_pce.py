# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Solving a QUBO with QAOA and with PCE on the same problem.

Two ways to drive a QUBO through divi, side by side:

* ``QAOA(BinaryOptimizationProblem(bqm))`` — accepts a dimod
  :class:`BinaryQuadraticModel` directly.  The cost Hamiltonian is built
  from the BQM's quadratic and linear biases, and solutions are returned
  as a dict keyed by the BQM variable names.
* ``PCE(qubo_to_matrix(bqm))`` — Pauli Correlation Encoding takes the QUBO
  as a numpy matrix.  Solutions come back as bitstrings indexed by the
  BQM's variable order.

Using a hand-built BQM with named variables lets the tutorial demonstrate
the variable-name → bit branching that QAOA's ``.solution`` exposes.
"""

import dimod
import pennylane as qp
from dimod import ExactSolver

from divi.hamiltonians import qubo_to_matrix
from divi.qprog import PCE, QAOA, GenericLayerAnsatz
from divi.qprog.optimizers import (
    PymooMethod,
    PymooOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import BinaryOptimizationProblem
from tutorials._backend import get_backend


def _print_summary(
    qaoa_circuits: int,
    pce_circuits: int,
    variables: list,
    qaoa_energy: float,
    qaoa_bitstring: str,
    pce_energy: float,
    pce_bitstring: str,
    classical_energy: float,
    classical_bitstring: str,
) -> None:
    print(f"QAOA circuits: {qaoa_circuits}")
    print(f"PCE  circuits: {pce_circuits}\n")
    print(f"Variable order: {variables}\n")

    width = max(len(qaoa_bitstring), len(pce_bitstring), len(classical_bitstring))
    print(f"{'Method':<10} {'Energy':>12}   Bitstring")
    print("-" * (10 + 14 + width))
    print(f"{'QAOA':<10} {qaoa_energy:>12.6f}   {qaoa_bitstring:<{width}}")
    print(f"{'PCE':<10} {pce_energy:>12.6f}   {pce_bitstring:<{width}}")
    print(
        f"{'Classical':<10} {classical_energy:>12.6f}   {classical_bitstring:<{width}}"
    )


def _print_top(label: str, top, variables: list, bqm) -> None:
    print(f"\nTop-5 {label} solutions:")
    print(f"  {'Rank':<6} {'Bitstring':<10} {'Probability':>12} {'Energy':>14}")
    print("  " + "-" * 46)
    for i, sol in enumerate(top, 1):
        if hasattr(sol, "energy") and sol.energy is not None:
            energy = sol.energy
        else:
            sol_dict = dict(zip(variables, (int(b) for b in sol.bitstring)))
            energy = bqm.energy(sol_dict)
        print(f"  {i:<6} {sol.bitstring:<10} {sol.prob:>11.2%}  {energy:>14.6f}")


if __name__ == "__main__":
    # Hand-built BQM with named variables — same problem for both solvers.
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
    variables = list(bqm.variables)

    backend = get_backend(shots=10_000)

    # ── QAOA on the BQM directly ──────────────────────────────────────
    qaoa_problem = QAOA(
        BinaryOptimizationProblem(bqm),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=backend,
    )
    qaoa_problem.run()

    # ``solution`` may be either a dict (named-variable BQMs) or an array
    # in variable order; normalize to a dict for downstream code.
    sol = qaoa_problem.solution
    qaoa_solution = (
        {v: int(sol[v]) for v in variables}
        if isinstance(sol, dict)
        else dict(zip(variables, sol))
    )
    qaoa_bitstring = "".join(str(qaoa_solution[v]) for v in variables)
    qaoa_energy = bqm.energy(qaoa_solution)

    # ── PCE on the QUBO matrix form ───────────────────────────────────
    # qubo_to_matrix maps named variables to integer indices using
    # ``list(bqm.variables)`` order, so the bitstring positions line up
    # with ``variables`` above.
    pce_solver = PCE(
        problem=qubo_to_matrix(bqm),
        ansatz=GenericLayerAnsatz(
            gate_sequence=[qp.RY, qp.RZ],
            entangler=qp.CNOT,
            entangling_layout="all-to-all",
        ),
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=10),
        backend=backend,
        max_iterations=10,
        n_layers=2,
        alpha=1.0,
    )
    pce_solver.run()

    # ── Classical reference ───────────────────────────────────────────
    classical_sample, classical_energy, _ = ExactSolver().sample(bqm).first
    classical_bitstring = "".join(str(classical_sample[v]) for v in variables)

    # ── Output ────────────────────────────────────────────────────────
    pce_top = pce_solver.get_top_solutions(n=5, min_prob=0.01, sort_by="energy")
    qaoa_top = qaoa_problem.get_top_solutions(n=5)

    _print_summary(
        qaoa_problem.total_circuit_count,
        pce_solver.total_circuit_count,
        variables,
        qaoa_energy,
        qaoa_bitstring,
        pce_top[0].energy,
        pce_top[0].bitstring,
        classical_energy,
        classical_bitstring,
    )
    _print_top("QAOA", qaoa_top, variables, bqm)
    _print_top("PCE", pce_top, variables, bqm)
