# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Solve a higher-order binary optimization (HUBO) problem with QAOA.

Unlike QUBOs (quadratic), HUBOs can contain cubic or higher-order
interactions between variables.  QAOA supports HUBOs directly via two
Hamiltonian builders:

- ``"native"``       – maps each polynomial term to a multi-Z Ising
                       interaction (no ancilla qubits).
- ``"quadratized"``  – reduces the polynomial to quadratic form by
                       introducing ancilla qubits with a penalty strength.

The quadratized builder uses a penalty parameter (quadratization_strength)
to enforce the consistency of ancilla qubits with the original variables.
Larger values tighten the constraint but can make the landscape harder to
optimize; smaller values may yield invalid (inconsistent) solutions.
When comparing different strengths in the table below, a *lower* penalty
can sometimes yield a better solution: the optimizer then follows the
actual problem landscape rather than being dominated by penalty wells,
which can trap it in a consistent but suboptimal assignment.

This tutorial defines a small 4-variable cubic HUBO with string-labelled
variables, solves it with both builders at several penalty strengths, and
compares to the exact classical minimum.
"""

from rich.console import Console
from rich.table import Table

from divi.qprog import QAOA, BinaryOptimizationProblem
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def exact_minimum(hubo: dict, variables: list[str]) -> tuple[dict, float]:
    """Brute-force the minimum of a small HUBO."""
    n = len(variables)
    best_energy = float("inf")
    best_assignment = {}
    for mask in range(1 << n):
        assignment = {var: (mask >> i) & 1 for i, var in enumerate(variables)}
        energy = 0.0
        for term, coeff in hubo.items():
            prod = 1.0
            for var in term:
                prod *= assignment[var]
            energy += coeff * prod
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment
    return best_assignment, best_energy


if __name__ == "__main__":
    # ── Define a 4-variable cubic HUBO with string labels ────────────
    hubo = {
        ("a",): -2.0,
        ("b",): 1.0,
        ("c",): -3.0,
        ("d",): 0.5,
        ("a", "b"): 1.5,
        ("c", "d"): -1.0,
        ("a", "b", "c"): 2.0,
    }
    variables = sorted({var for term in hubo for var in term})

    classical_solution, classical_energy = exact_minimum(hubo, variables)

    # ── Solve with the native builder ────────────────────────────────
    qaoa_native = QAOA(
        BinaryOptimizationProblem(hubo, hamiltonian_builder="native"),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=15,
        backend=get_backend(shots=10000),
        seed=42,
    )
    qaoa_native.run()

    # ── Solve with the quadratized builder (several penalty strengths) ─
    quadratization_strengths = [1.0, 3.0, 5.0, 10.0]
    qaoa_quad_runs: list[tuple[float, QAOA]] = []
    for strength in quadratization_strengths:
        qaoa = QAOA(
            BinaryOptimizationProblem(
                hubo,
                hamiltonian_builder="quadratized",
                quadratization_strength=strength,
            ),
            n_layers=3,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=get_backend(shots=10000),
            seed=42,
        )
        qaoa.run()
        qaoa_quad_runs.append((strength, qaoa))

    # ── Compare results ──────────────────────────────────────────────
    # HUBO solutions are returned as dicts mapping variable names to values.
    # Note: Quadratized "Energy" is the optimizer's loss (full Hamiltonian).
    # A lower penalty (e.g. P=1) can outperform higher P because the landscape
    # stays dominated by the problem terms; high P can trap the optimizer in
    # a consistent but suboptimal assignment.
    native_solution = qaoa_native.solution

    def format_assignment(sol: dict) -> str:
        return "  ".join(f"{k}={v}" for k, v in sorted(sol.items()))

    console = Console()
    table = Table(
        title="HUBO Optimization Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Method")
    table.add_column("Energy", justify="right")
    table.add_column("Assignment")
    table.add_column("Circuits", justify="right")

    table.add_row(
        "Classical",
        f"{classical_energy:.4f}",
        format_assignment(classical_solution),
        "-",
    )
    table.add_row(
        "Native",
        f"{qaoa_native.best_loss:.4f}",
        format_assignment(native_solution),
        str(qaoa_native.total_circuit_count),
    )
    for strength, qaoa_quad in qaoa_quad_runs:
        quad_solution = qaoa_quad.solution
        table.add_row(
            f"Quadratized (P={strength})",
            f"{qaoa_quad.best_loss:.4f}",
            format_assignment(quad_solution),
            str(qaoa_quad.total_circuit_count),
        )

    console.print(table)
