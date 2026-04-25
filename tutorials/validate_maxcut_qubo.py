# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""MaxCut QUBO — Validate & Solve
===================================

Demonstrates how the Divi Validation Service accelerates QAOA by
replacing expensive shot-based parameter optimization with a single
exact diagnostic call.

**Key insight**: QAOA performance hinges on finding good (γ, β)
parameters.  Traditional approaches burn hundreds of circuit
evaluations searching blindly.  The validator uses a 
simulator to sweep the full parameter landscape and
returns the optimal point — no shots, no noise, no guesswork.

Workflow:
  1. Build a MaxCut QUBO (10-qubit Petersen graph).
  2. **Validate** — exact (γ, β) sweep via the Qoro service.
  3. **Compare** — QAOA with the validator's params (1 evaluation)
     vs blind optimization (80 iterations × 5 random seeds).

Requirements:
    - ``QORO_API_KEY`` in ``.env`` or environment variable.
    - ``pip install divi networkx``
"""

import itertools

import networkx as nx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.validate import validate
from tutorials._backend import get_backend

console = Console()

# ──────────────────────────────────────────────────────────────────────
# 1. Build graph & QUBO
# ──────────────────────────────────────────────────────────────────────

G = nx.petersen_graph()
rng = np.random.default_rng(42)
for u, v in G.edges():
    G[u][v]["weight"] = round(rng.uniform(0.5, 3.0), 2)

n = G.number_of_nodes()
Q = np.zeros((n, n))

for u, v, data in G.edges(data=True):
    w = data.get("weight", 1.0)
    Q[u, v] = -w
    Q[v, u] = -w
    Q[u, u] += w
    Q[v, v] += w

problem = BinaryOptimizationProblem(Q)

console.print(f"\n[bold cyan]1. Graph & QUBO[/bold cyan]")
console.print(f"   Petersen graph: {n} nodes, {G.number_of_edges()} edges (weighted)")
console.print(f"   QUBO matrix: {Q.shape}, {np.count_nonzero(Q)} non-zero entries\n")


# ──────────────────────────────────────────────────────────────────────
# 2. Classical ground truth
# ──────────────────────────────────────────────────────────────────────


def maxcut_value(bitstring: str, graph: nx.Graph) -> float:
    """Compute the MaxCut objective for a given bitstring."""
    total = 0.0
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        if bitstring[u] != bitstring[v]:
            total += w
    return total


best_cut = -float("inf")
best_bitstrings = []
for bits in itertools.product("01", repeat=n):
    bs = "".join(bits)
    cut = maxcut_value(bs, G)
    if cut > best_cut:
        best_cut = cut
        best_bitstrings = [bs]
    elif cut == best_cut:
        best_bitstrings.append(bs)

table = Table(title="Exact Optimal Solutions (Brute Force)", border_style="green")
table.add_column("Bitstring", style="bold")
table.add_column("Cut Value", justify="right")
for bs in best_bitstrings[:4]:
    table.add_row(bs, f"{best_cut:.2f}")
console.print(table)
console.print(f"   Max cut: {best_cut:.2f}  ({len(best_bitstrings)} optimal solutions)\n")

# ──────────────────────────────────────────────────────────────────────
# 3. Validate — parameter sweep + diagnostic report
# ──────────────────────────────────────────────────────────────────────

console.print("[bold cyan]2. Validation Service — Parameter Sweep[/bold cyan]")
console.print("   Sweeping (γ, β)...\n")

ansatz_config = {"mixer": "x", "layers": 1}

sweep_result = validate(
    Q,
    target_states=best_bitstrings[:2],
    parameter_sweep=True,
    sensitivity=True,
    n_qubits=n,
    ansatz=ansatz_config,
)

bp = sweep_result.best_parameters
optimal_gamma = bp["gamma"]
optimal_beta = bp["beta"]
prob = bp.get("probability", 0)
uniform_prob = 1 / 2**n

console.print(f"   [green]Optimal: γ = {optimal_gamma:.4f}, β = {optimal_beta:.4f}[/green]")
console.print(f"   P(optimal cut) = {prob:.6f}  ({prob / uniform_prob:.1f}× above uniform)\n")

sweep_result.display()

# ──────────────────────────────────────────────────────────────────────
# 4. QAOA comparison
# ──────────────────────────────────────────────────────────────────────
# Approach A — "Blind" QAOA: optimizer searches from random starting
#   points using noisy shot-based evaluations (80 iters × 5 seeds).
#
# Approach B — "Validator-guided": we already have the optimal params,
#   so we just evaluate the circuit once (max_iterations=1).  No
#   optimization needed — the validator did the hard work.

SHOTS = 20000
backend = get_backend(shots=SHOTS)

console.print(f"\n[bold cyan]3. QAOA Comparison (p=1, {SHOTS} shots)[/bold cyan]\n")

# --- A. Blind QAOA: 5 random starting points ---
N_BLIND_ITERS = 80
console.print(f"   [bold]A) Blind QAOA[/bold]: 5 random seeds × {N_BLIND_ITERS} iterations")

random_results = []
for seed in [7, 21, 42, 77, 99]:
    qaoa = QAOA(
        problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=N_BLIND_ITERS,
        backend=backend,
        seed=seed,
    )
    qaoa.run()
    bs = "".join(str(b) for b in qaoa.solution)
    cut = maxcut_value(bs, G)
    # Also measure quality of top-5 solutions
    top5_cuts = [maxcut_value(s.bitstring, G) for s in qaoa.get_top_solutions(n=5)]
    best_top5 = max(top5_cuts)
    random_results.append({
        "seed": seed,
        "bitstring": bs,
        "cut": cut,
        "best_top5": best_top5,
        "loss": qaoa.best_loss,
        "circuits": qaoa.total_circuit_count,
    })
    console.print(
        f"      seed={seed:>2}  →  cut={cut:>5.2f}  "
        f"best-in-top5={best_top5:>5.2f}  ({qaoa.total_circuit_count} circuits)"
    )

best_random_cut = max(r["cut"] for r in random_results)
best_random_top5 = max(r["best_top5"] for r in random_results)
avg_random_cut = np.mean([r["cut"] for r in random_results])
total_random_circuits = sum(r["circuits"] for r in random_results)

# --- B. Validator-guided: one evaluation at optimal params ---
console.print(f"\n   [bold]B) Validator-guided[/bold]: 1 evaluation at (γ={optimal_gamma:.4f}, β={optimal_beta:.4f})")

qaoa_seeded = QAOA(
    problem,
    n_layers=1,
    optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
    max_iterations=1,
    backend=backend,
    seed=42,
)
initial_params = np.array([[optimal_gamma, optimal_beta]])
qaoa_seeded.run(initial_params=initial_params)

seeded_bs = "".join(str(b) for b in qaoa_seeded.solution)
seeded_cut = maxcut_value(seeded_bs, G)
seeded_top5 = [maxcut_value(s.bitstring, G) for s in qaoa_seeded.get_top_solutions(n=5)]
seeded_best_top5 = max(seeded_top5)
seeded_circuits = qaoa_seeded.total_circuit_count

console.print(
    f"      seeded  →  cut={seeded_cut:>5.2f}  "
    f"best-in-top5={seeded_best_top5:>5.2f}  ({seeded_circuits} circuits)\n"
)

# ──────────────────────────────────────────────────────────────────────
# 5. Results
# ──────────────────────────────────────────────────────────────────────

cmp = Table(title="QAOA Performance Comparison", border_style="cyan")
cmp.add_column("", style="bold")
cmp.add_column("Blind QAOA\n(best of 5)", justify="right")
cmp.add_column("Blind QAOA\n(average)", justify="right")
cmp.add_column("Validator-\nGuided", justify="right", style="green")

cmp.add_row("Solution Cut", f"{best_random_cut:.2f}", f"{avg_random_cut:.2f}", f"{seeded_cut:.2f}")
cmp.add_row("Best in Top-5", f"{best_random_top5:.2f}", "—", f"{seeded_best_top5:.2f}")
cmp.add_row(
    "Approx. Ratio",
    f"{best_random_cut/best_cut:.3f}",
    f"{avg_random_cut/best_cut:.3f}",
    f"{seeded_cut/best_cut:.3f}",
)
cmp.add_row("Total Circuits", str(total_random_circuits), str(total_random_circuits), str(seeded_circuits))
cmp.add_row("Optimizer Iters", f"5 × {N_BLIND_ITERS}", f"5 × {N_BLIND_ITERS}", "0 (exact)")
cmp.add_row("Optimal", f"{best_cut:.2f}", f"{best_cut:.2f}", f"{best_cut:.2f}")
console.print(cmp)

# Top-5 from seeded run
top = qaoa_seeded.get_top_solutions(n=10)
sol_table = Table(title="Top 10 States (Validator-Guided, Single Evaluation)", border_style="green")
sol_table.add_column("Bitstring", style="bold")
sol_table.add_column("Probability", justify="right")
sol_table.add_column("Cut Value", justify="right")
sol_table.add_column("Optimal?", justify="center")
for s in top:
    c = maxcut_value(s.bitstring, G)
    sol_table.add_row(
        s.bitstring,
        f"{s.prob:.4f}",
        f"{c:.2f}",
        "[green]✓[/green]" if s.bitstring in best_bitstrings else "",
    )
console.print(sol_table)

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

speedup = total_random_circuits / max(seeded_circuits, 1)
console.print(
    Panel(
        f"[bold]The Validation Service found the optimal QAOA parameters — zero circuit evaluations needed.[/bold]\n\n"
        f"• Validator-guided: [green]{seeded_circuits} circuits[/green] → "
        f"cut = {seeded_cut:.2f}, best-in-top5 = {seeded_best_top5:.2f}\n"
        f"• Blind QAOA:       [yellow]{total_random_circuits} circuits[/yellow] → "
        f"best cut = {best_random_cut:.2f} (across 5 attempts)\n\n"
        f"The validator achieves comparable solution quality with "
        f"[bold green]{speedup:.0f}× fewer circuit evaluations[/bold green] — "
        f"a direct reduction in quantum hardware cost.",
        title="Summary",
        border_style="cyan",
    )
)
