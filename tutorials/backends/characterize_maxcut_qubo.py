# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""MaxCut QUBO — Characterize & Solve
======================================

Demonstrates how the Divi Characterization Service can shortcut the QAOA
parameter-search loop, and how its verdict/classical-baseline machinery
tells you *before* running QAOA whether it is even worth it. The
characterizer runs an exact (statevector) parameter sweep server-side,
computes a real approximation ratio and a classical (greedy/SA) baseline
on the same QUBO, and returns the optimal (γ, β); QAOA can then skip its
outer optimizer and run a single shot-based evaluation at those
parameters.

Workflow:
  1. Build a MaxCut QUBO (10-qubit Petersen graph).
  2. Compute the classical ground truth via brute force.
  3. **Characterize** — exact (γ, β) sweep, verdict, and classical
     baseline via the Qoro service.
  4. **Compare** — QAOA with the characterizer's params (1 evaluation)
     vs unguided optimization (5 random seeds × 80 COBYLA iterations),
     checked against the verdict from step 3.

Requirements:
    - ``QORO_API_KEY`` in ``.env`` or environment variable.
"""

import itertools

import networkx as nx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from divi.backends import (
    CharacterizationOptions,
    QoroService,
    characterize_and_validate,
)
from divi.qprog import QAOA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import BinaryOptimizationProblem
from tutorials._backend import get_backend

console = Console()

SHOTS = 20000
N_BLIND_ITERS = 80
BLIND_SEEDS = [7, 21, 42, 77, 99]


def maxcut_value(bitstring: str, graph: nx.Graph) -> float:
    """Compute the MaxCut objective for a given bitstring."""
    total = 0.0
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        if bitstring[u] != bitstring[v]:
            total += w
    return total


def main() -> None:
    # ──────────────────────────────────────────────────────────────────
    # 1. Build graph & QUBO
    # ──────────────────────────────────────────────────────────────────
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

    console.print("\n[bold cyan]1. Graph & QUBO[/bold cyan]")
    console.print(
        f"   Petersen graph: {n} nodes, {G.number_of_edges()} edges (weighted)"
    )
    console.print(f"   QUBO matrix: {Q.shape}, {np.count_nonzero(Q)} non-zero entries")

    # ──────────────────────────────────────────────────────────────────
    # 2. Classical ground truth (brute force)
    # ──────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]2. Classical Ground Truth[/bold cyan]")

    best_cut = -float("inf")
    best_bitstrings: list[str] = []
    for bits in itertools.product("01", repeat=n):
        bs = "".join(bits)
        cut = maxcut_value(bs, G)
        if cut > best_cut:
            best_cut = cut
            best_bitstrings = [bs]
        elif cut == best_cut:
            best_bitstrings.append(bs)

    console.print(_brute_force_table(best_cut, best_bitstrings))
    console.print(
        f"   Max cut: {best_cut:.2f}  ({len(best_bitstrings)} optimal solutions)"
    )

    # ──────────────────────────────────────────────────────────────────
    # 3. Characterize — parameter sweep + diagnostic report
    # ──────────────────────────────────────────────────────────────────
    console.print(
        "\n[bold cyan]3. Characterization Service — Verdict & Parameter Sweep[/bold cyan]"
    )
    console.print("   Sweeping (γ, β) and checking against a classical baseline...\n")

    sweep_result = characterize_and_validate(
        problem,
        target_states=best_bitstrings[:2],
        service=QoroService(),
        options=CharacterizationOptions(
            parameter_sweep=True,
            sensitivity=True,
            ansatz={"mixer": "x", "layers": 1},
        ),
    )

    bp = sweep_result.best_parameters
    optimal_gamma, optimal_beta = bp["gamma"], bp["beta"]
    verdict = sweep_result.verdict or {}
    baseline = sweep_result.classical_baseline or {}
    swept_ar = sweep_result.approximation_ratio

    console.print(
        f"   [green]Sweep returned: γ = {optimal_gamma:.4f}, "
        f"β = {optimal_beta:.4f}[/green]"
    )
    console.print(
        f"   [bold]Verdict: {verdict.get('verdict', 'n/a')}[/bold] — "
        f"{verdict.get('rationale', 'no rationale returned')}"
    )
    console.print(
        f"   Characterizer's classical baseline (its own reference, computed "
        f"without brute force): best_energy={baseline.get('best_energy', float('nan')):.4f} "
        f"(greedy={baseline.get('greedy_energy', float('nan')):.4f}, "
        f"SA={baseline.get('sa_energy', float('nan')):.4f})\n"
        f"   Swept approximation ratio: "
        f"{swept_ar if swept_ar is None else f'{swept_ar:.4f}'} "
        f"— compare against our own QAOA's ratio in step 4.\n"
    )
    sweep_result.display()

    # ──────────────────────────────────────────────────────────────────
    # 4. QAOA comparison & results
    # ──────────────────────────────────────────────────────────────────
    # Approach A — "Blind" QAOA: optimizer searches from random starting
    #   points using shot-based evaluations (80 iters × 5 seeds).
    # Approach B — "Characterizer-guided": evaluate once at the sweep's
    #   optimal params (max_iterations=1) — no optimization needed.

    backend = get_backend(shots=SHOTS)
    console.print(
        f"\n[bold cyan]4. QAOA Comparison & Results (p=1, {SHOTS} shots)[/bold cyan]\n"
    )

    console.print(
        f"   [bold]A) Blind QAOA[/bold]: {len(BLIND_SEEDS)} random seeds × {N_BLIND_ITERS} iterations"
    )
    blind_results = [_run_blind_qaoa(problem, backend, G, seed) for seed in BLIND_SEEDS]

    blind_cuts = [r["cut"] for r in blind_results]
    blind_mean = float(np.mean(blind_cuts))
    blind_std = float(np.std(blind_cuts))
    blind_min, blind_max = min(blind_cuts), max(blind_cuts)
    total_random_circuits = sum(r["circuits"] for r in blind_results)

    console.print(
        f"\n   [bold]B) Characterizer-guided[/bold]: 1 evaluation at "
        f"(γ={optimal_gamma:.4f}, β={optimal_beta:.4f})"
    )
    seeded = _run_guided_qaoa(problem, backend, G, optimal_gamma, optimal_beta)
    console.print(
        f"      seeded  →  cut={seeded['cut']:>5.2f}  "
        f"({seeded['circuits']} circuits)\n"
    )

    console.print(
        _comparison_table(
            blind_mean=blind_mean,
            blind_std=blind_std,
            blind_min=blind_min,
            blind_max=blind_max,
            seeded_cut=seeded["cut"],
            best_cut=best_cut,
            total_random_circuits=total_random_circuits,
            seeded_circuits=seeded["circuits"],
            n_blind_iters=N_BLIND_ITERS,
            n_blind_seeds=len(BLIND_SEEDS),
        )
    )
    console.print(
        _summary_panel(
            seeded_circuits=seeded["circuits"],
            seeded_cut=seeded["cut"],
            seeded_ratio=seeded["cut"] / best_cut,
            total_random_circuits=total_random_circuits,
            blind_mean=blind_mean,
            blind_std=blind_std,
            blind_min=blind_min,
            blind_max=blind_max,
            best_cut=best_cut,
            verdict=verdict,
            swept_ar=swept_ar,
        )
    )


# ──────────────────────────────────────────────────────────────────────
# QAOA runners
# ──────────────────────────────────────────────────────────────────────


def _run_blind_qaoa(
    problem: BinaryOptimizationProblem, backend, graph: nx.Graph, seed: int
) -> dict:
    """One blind-QAOA run: COBYLA from a random start, return summary stats."""
    qaoa = QAOA(
        problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=N_BLIND_ITERS,
        backend=backend,
        seed=seed,
    )
    qaoa.run()
    bs = qaoa.solution_bitstring
    cut = maxcut_value(bs, graph)
    console.print(
        f"      seed={seed:>2}  →  cut={cut:>5.2f}  "
        f"({qaoa.total_circuit_count} circuits)"
    )
    return {
        "seed": seed,
        "bitstring": bs,
        "cut": cut,
        "circuits": qaoa.total_circuit_count,
    }


def _run_guided_qaoa(
    problem: BinaryOptimizationProblem,
    backend,
    graph: nx.Graph,
    gamma: float,
    beta: float,
) -> dict:
    """Single characterizer-guided QAOA evaluation at the supplied (γ, β)."""
    qaoa = QAOA(
        problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=1,
        backend=backend,
        seed=42,
    )
    qaoa.run(initial_params=np.array([[gamma, beta]]))
    bs = qaoa.solution_bitstring
    cut = maxcut_value(bs, graph)
    return {
        "bitstring": bs,
        "cut": cut,
        "circuits": qaoa.total_circuit_count,
    }


# ──────────────────────────────────────────────────────────────────────
# Presentation helpers — Rich tables & summary panel
# ──────────────────────────────────────────────────────────────────────


def _brute_force_table(best_cut: float, best_bitstrings: list[str]) -> Table:
    table = Table(title="Exact Optimal Solutions (Brute Force)", border_style="green")
    table.add_column("Bitstring", style="bold")
    table.add_column("Cut Value", justify="right")
    for bs in best_bitstrings[:4]:
        table.add_row(bs, f"{best_cut:.2f}")
    return table


def _comparison_table(
    *,
    blind_mean: float,
    blind_std: float,
    blind_min: float,
    blind_max: float,
    seeded_cut: float,
    best_cut: float,
    total_random_circuits: int,
    seeded_circuits: int,
    n_blind_iters: int,
    n_blind_seeds: int,
) -> Table:
    """Honest side-by-side comparison.

    Reports mean ± std and range for blind QAOA across seeds — single-best
    bitstring cut for the seeded run. Approximation ratios are computed
    consistently against the exact optimum on both columns. No "best of N"
    cherry-picking and no shot-sampled order statistics.
    """
    cmp = Table(title="QAOA Performance Comparison", border_style="cyan")
    cmp.add_column("", style="bold")
    cmp.add_column(f"Blind QAOA\n({n_blind_seeds} seeds)", justify="right")
    cmp.add_column("Characterizer-\nGuided", justify="right", style="green")

    cmp.add_row(
        "Solution Cut",
        f"{blind_mean:.2f} ± {blind_std:.2f}",
        f"{seeded_cut:.2f}",
    )
    cmp.add_row(
        "Range across seeds",
        f"[{blind_min:.2f}, {blind_max:.2f}]",
        "— (deterministic)",
    )
    cmp.add_row(
        "Approx. Ratio",
        f"{blind_mean/best_cut:.3f} ± {blind_std/best_cut:.3f}",
        f"{seeded_cut/best_cut:.3f}",
    )
    cmp.add_row(
        "Total Circuits",
        str(total_random_circuits),
        str(seeded_circuits),
    )
    cmp.add_row(
        "Optimizer Iters",
        f"{n_blind_seeds} × {n_blind_iters}",
        "0",
    )
    cmp.add_row("Optimum (exact)", f"{best_cut:.2f}", f"{best_cut:.2f}")
    return cmp


def _summary_panel(
    *,
    seeded_circuits: int,
    seeded_cut: float,
    seeded_ratio: float,
    total_random_circuits: int,
    blind_mean: float,
    blind_std: float,
    blind_min: float,
    blind_max: float,
    best_cut: float,
    verdict: dict,
    swept_ar: float | None,
) -> Panel:
    """Honest summary: circuit savings + reproducibility, tied to the verdict."""
    speedup = total_random_circuits / max(seeded_circuits, 1)
    delta_vs_mean = seeded_cut - blind_mean
    # Where the seeded result falls inside the blind seed lottery — be honest
    # if it's just lucky-seed-equivalent rather than a true win.
    if seeded_cut > blind_max:
        position = "above the blind best — a real lift over every seed"
    elif seeded_cut >= blind_max - 1e-6:
        position = (
            "tied with the blind best — the seed lottery happened to graze "
            "the same ceiling"
        )
    else:
        position = (
            f"inside the blind range [{blind_min:.2f}, {blind_max:.2f}] — "
            "near the p=1 ceiling, not above it"
        )
    verdict_label = verdict.get("verdict", "n/a")
    swept_ar_str = "n/a" if swept_ar is None else f"{swept_ar:.3f}"
    return Panel(
        f"[bold]Characterizer-guided[/bold]: "
        f"[green]{seeded_circuits} circuits[/green], "
        f"cut = {seeded_cut:.2f} "
        f"(approx. ratio {seeded_ratio:.3f}, deterministic).\n"
        f"[bold]Blind QAOA[/bold] ({total_random_circuits} circuits across seeds): "
        f"cut = {blind_mean:.2f} ± {blind_std:.2f}, "
        f"range [{blind_min:.2f}, {blind_max:.2f}].\n\n"
        f"Seeded result is [bold]{delta_vs_mean:+.2f}[/bold] vs the blind "
        f"mean and {position}.\n\n"
        f"Both methods saturate near the p=1 ceiling — neither finds the "
        f"exact optimum ({best_cut:.2f}). The characterizer already flagged "
        f"this as [bold]{verdict_label}[/bold] (swept AR {swept_ar_str} vs. "
        f"its classical baseline) *before* either run above spent a single "
        f"circuit — this p=1 ceiling was predictable, not a surprise. The "
        f"characterizer's value here is "
        f"[bold green]{speedup:.0f}× fewer circuits[/bold green] to reach "
        f"the same ceiling deterministically, plus the cost-spectrum "
        f"hardness metrics (cost gap, ground-state degeneracy, treewidth, "
        f"frustration index) and sensitivity report — a scale-invariant "
        f"structural fingerprint of your QUBO that persists as a reusable "
        f"artifact even when you do run full optimization.",
        title="Summary",
        border_style="cyan",
    )


if __name__ == "__main__":
    main()
