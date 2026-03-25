# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Constraint-Enhanced QAOA on the Capacitated Vehicle Routing Problem.

Demonstrates CE-QAOA for CVRP with both one-hot and binary encodings:

1. One-hot encoding on a small 3-customer / 2-vehicle instance.
2. Binary encoding on the same instance — qubit savings comparison.
3. Qubit projections for QOBLIB-scale problems (n=20, k=4).
4. VRP file parser: loading a TSPLIB/CVRPLIB .vrp instance.
"""

import tempfile
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from divi.qprog import QAOA, MonteCarloOptimizer
from divi.qprog.problems import (
    CVRPProblem,
    binary_block_config,
    cvrp_block_structure,
    parse_vrp_file,
    parse_vrp_solution,
)
from tutorials._backend import get_backend

COST_MATRIX = np.array(
    [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 12], [20, 30, 12, 0]],
    dtype=float,
)
DEMANDS = np.array([0, 3, 4, 2], dtype=float)
CAPACITY = 6.0
N_VEHICLES = 2
DEPOT = 0

# A small CVRP instance in TSPLIB format (for parser demo)
SAMPLE_VRP = """\
NAME : DEMO-n3-k2-01
COMMENT : "Demo instance; Optimal cost: 67"
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 6
NODE_COORD_SECTION
1    0   0
2    10  0
3    0   15
4    10  12
DEMAND_SECTION
1    0
2    3
3    4
4    2
DEPOT_SECTION
1
-1
EOF
"""

SAMPLE_SOL = """\
Route #1: 2
Route #2: 3 4
Cost 67
"""


if __name__ == "__main__":
    console = Console()
    backend = get_backend(shots=5000)

    # ── 1) One-hot CE-QAOA ────────────────────────────────────────────
    console.rule("[bold]CE-QAOA: One-Hot Encoding (3 customers, 2 vehicles)")

    problem_oh = CVRPProblem(
        COST_MATRIX,
        demands=DEMANDS,
        capacity=CAPACITY,
        n_vehicles=N_VEHICLES,
        depot=DEPOT,
        encoding="one_hot",
    )
    ce_oh = QAOA(
        problem_oh,
        n_layers=1,
        backend=backend,
        optimizer=MonteCarloOptimizer(population_size=10, n_best_sets=3),
        max_iterations=5,
    )
    ce_oh.run()

    console.print(
        f"Qubits: {ce_oh.n_qubits}  "
        f"({ce_oh.initial_state.n_blocks} blocks × {ce_oh.initial_state.block_size})"
    )
    console.print(f"Circuits: {ce_oh.total_circuit_count}")
    console.print(f"Best loss: {ce_oh.best_loss:.4f}")

    oh_sols = ce_oh.get_top_solutions(n=3, feasibility="repair", include_decoded=True)
    table = Table(title="One-Hot Routes")
    table.add_column("#", style="dim")
    table.add_column("Routes")
    table.add_column("Cost", justify="right")
    table.add_column("Prob", justify="right")
    for i, sol in enumerate(oh_sols, 1):
        if sol.decoded:
            display = " | ".join(
                " → ".join(str(c) for c in r) for r in sol.decoded if len(r) > 2
            )
        else:
            display = "—"
        table.add_row(
            str(i),
            display,
            f"{sol.energy:.0f}" if sol.energy else "—",
            f"{sol.prob:.2%}",
        )
    console.print(table)

    # ── 2) Binary CE-QAOA on same instance ────────────────────────────
    console.rule("[bold]CE-QAOA: Binary Encoding (same instance)")

    problem_bin = CVRPProblem(
        COST_MATRIX,
        demands=DEMANDS,
        capacity=CAPACITY,
        n_vehicles=N_VEHICLES,
        depot=DEPOT,
        encoding="binary",
    )
    ce_bin = QAOA(
        problem_bin,
        n_layers=1,
        backend=backend,
        optimizer=MonteCarloOptimizer(population_size=10, n_best_sets=3),
        max_iterations=5,
    )
    ce_bin.run()

    cfg = problem_bin.binary_config
    console.print(
        f"Qubits: {ce_bin.n_qubits}  "
        f"({cfg.n_slots} slots × {cfg.bits_per_slot} bits)"
    )
    console.print(f"Circuits: {ce_bin.total_circuit_count}")
    console.print(f"Best loss: {ce_bin.best_loss:.4f}")

    # ── 3) Encoding comparison table ──────────────────────────────────
    console.rule("[bold]One-Hot vs Binary Comparison")

    comp = Table(title="Encoding Comparison")
    comp.add_column("Metric")
    comp.add_column("One-Hot", justify="right")
    comp.add_column("Binary", justify="right")
    comp.add_row("Qubits", str(ce_oh.n_qubits), str(ce_bin.n_qubits))
    comp.add_row(
        "Circuits", str(ce_oh.total_circuit_count), str(ce_bin.total_circuit_count)
    )
    comp.add_row("Best loss", f"{ce_oh.best_loss:.4f}", f"{ce_bin.best_loss:.4f}")
    console.print(comp)

    # ── 4) Qubit projections at scale ─────────────────────────────────
    console.rule("[bold]Qubit Projections")

    proj = Table(title="Qubits by Encoding and Scale")
    proj.add_column("N (customers)")
    proj.add_column("K (vehicles)")
    proj.add_column("One-Hot", justify="right")
    proj.add_column("Binary (full)", justify="right")
    proj.add_column("Binary (tight)", justify="right")
    proj.add_column("Savings", justify="right")

    for n_cust, k in [(3, 2), (10, 3), (20, 4), (50, 10)]:
        oh_q = cvrp_block_structure(n_cust, k)[0] * cvrp_block_structure(n_cust, k)[1]
        bin_full = binary_block_config(n_cust, k)
        tight_steps = max(2, (n_cust + k - 1) // k + 1)
        bin_tight = binary_block_config(n_cust, k, max_steps=tight_steps)
        proj.add_row(
            str(n_cust),
            str(k),
            f"{oh_q:,}",
            f"{bin_full.n_qubits:,}",
            f"{bin_tight.n_qubits:,}",
            f"{(1 - bin_tight.n_qubits / oh_q) * 100:.0f}%",
        )
    console.print(proj)
    console.print(
        "[dim]Paper's enhanced formula for QOBLIB (N=20, K=4): ~133 qubits[/dim]"
    )

    # ── 5) VRP file parser demo ───────────────────────────────────────
    console.rule("[bold]VRP File Parser")

    with tempfile.TemporaryDirectory() as tmp:
        vrp_path = Path(tmp) / "demo.vrp"
        sol_path = Path(tmp) / "demo.opt.sol"
        vrp_path.write_text(SAMPLE_VRP)
        sol_path.write_text(SAMPLE_SOL)

        inst = parse_vrp_file(vrp_path)
        opt_routes, opt_cost = parse_vrp_solution(sol_path)

    console.print(f"Instance: {inst.name}")
    console.print(
        f"Nodes: {inst.dimension} ({inst.n_customers} customers + depot), "
        f"Vehicles: {inst.n_vehicles}, Capacity: {inst.capacity}"
    )
    console.print(f"Optimal cost: {opt_cost}")
    for i, route in enumerate(opt_routes, 1):
        console.print(f"  Vehicle {i}: {' → '.join(str(n) for n in route)}")
