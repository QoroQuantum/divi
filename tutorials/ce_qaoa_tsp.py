# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Constraint-Enhanced QAOA on the Travelling Salesman Problem.

End-to-end walkthrough of CE-QAOA (arXiv:2511.14296) on a 3-city TSP:

1. Brute-force reference solution.
2. Grid search over (gamma, beta) with loss landscape visualization.
3. Parameter transfer: best grid angles → variational refinement.
4. Feasibility statistics and Hungarian-algorithm repair.
"""

from itertools import permutations

import numpy as np
from rich.console import Console
from rich.table import Table

from divi.qprog import QAOA, GridSearchOptimizer, MonteCarloOptimizer
from divi.qprog.problems import TSPProblem, is_valid_tsp_tour, tour_cost
from tutorials._backend import get_backend

COST_MATRIX = np.array(
    [[0, 10, 15], [10, 0, 20], [15, 20, 0]],
    dtype=float,
)
START_CITY = 0
N_CITIES = 3


def brute_force_tsp(cost_matrix, start_city):
    """Optimal tour by exhaustive enumeration."""
    cities = [c for c in range(len(cost_matrix)) if c != start_city]
    best_cost, best_tour = float("inf"), None
    for perm in permutations(cities):
        t = [start_city, *perm, start_city]
        c = tour_cost(t, cost_matrix)
        if c < best_cost:
            best_cost, best_tour = c, t
    return best_tour, best_cost


def _print_loss_landscape(losses, grid):
    table = Table(title="Loss Landscape (γ \\ β)")
    table.add_column("γ \\ β", style="bold")
    betas = sorted(set(grid[:, 1]))
    for b in betas:
        table.add_column(f"{b:.2f}", justify="right")
    for g in sorted(set(grid[:, 0])):
        row = [f"{g:.2f}"]
        for b in betas:
            mask = np.isclose(grid[:, 0], g) & np.isclose(grid[:, 1], b)
            idx = np.where(mask)[0]
            row.append(f"{losses[idx[0]]:.2f}" if len(idx) else "—")
        table.add_row(*row)
    Console().print(table)


def _print_top_tours(repaired, opt_cost):
    table = Table(title="Top 5 Tours")
    table.add_column("#", style="dim")
    table.add_column("Tour")
    table.add_column("Cost", justify="right")
    table.add_column("Prob", justify="right")
    table.add_column("Gap", justify="right")

    for i, sol in enumerate(repaired, 1):
        tour_str = " → ".join(str(c) for c in sol.decoded) if sol.decoded else "—"
        gap = f"{(sol.energy - opt_cost) / opt_cost * 100:.1f}%" if sol.energy else "—"
        table.add_row(
            str(i),
            tour_str,
            f"{sol.energy:.1f}" if sol.energy else "—",
            f"{sol.prob:.2%}",
            gap,
        )
    console = Console()
    console.print(table)
    console.print(f"\n[dim]Optimal cost: {opt_cost}[/dim]")


if __name__ == "__main__":
    console = Console()
    backend = get_backend(shots=5000)

    problem = TSPProblem(COST_MATRIX, start_city=START_CITY)

    # ── 1) Brute-force reference ──────────────────────────────────────
    opt_tour, opt_cost = brute_force_tsp(COST_MATRIX, START_CITY)
    console.rule("[bold]Brute-Force Optimal")
    console.print(f"Tour: {opt_tour}  Cost: {opt_cost}")

    # ── 2) Grid search with landscape ─────────────────────────────────
    console.rule("[bold]Grid Search (5x5)")

    grid_opt = GridSearchOptimizer(
        param_ranges=[(0, np.pi), (0, np.pi)],
        grid_points=5,
    )
    ce_grid = QAOA(
        problem,
        n_layers=1,
        backend=backend,
        optimizer=grid_opt,
        max_iterations=1,
    )
    ce_grid.run()

    console.print(f"Grid points: {grid_opt.n_param_sets}")
    console.print(f"Best loss: {ce_grid.best_loss:.4f}")
    console.print(
        f"Best (γ, β): ({ce_grid.best_params[0]:.3f}, {ce_grid.best_params[1]:.3f})"
    )

    _print_loss_landscape(grid_opt._all_losses, grid_opt._param_grid)

    # ── 3) Parameter transfer: grid → variational ─────────────────────
    console.rule("[bold]Parameter Transfer → Monte Carlo Refinement")

    pop_size = 10
    best_grid_params = ce_grid.best_params.copy()
    rng = np.random.default_rng(42)
    warm_start = best_grid_params + rng.normal(0, 0.1, size=(pop_size, 2))
    warm_start = warm_start % (2 * np.pi)

    console.print(
        f"Warm-start from grid: (γ, β) = "
        f"({best_grid_params[0]:.3f}, {best_grid_params[1]:.3f})"
    )

    ce_mc = QAOA(
        problem,
        n_layers=1,
        backend=backend,
        optimizer=MonteCarloOptimizer(population_size=pop_size, n_best_sets=3),
        max_iterations=5,
    )
    ce_mc.run(initial_params=warm_start)

    console.print(f"Best loss after refinement: {ce_mc.best_loss:.4f}")

    # ── 4) Feasibility statistics ─────────────────────────────────────
    console.rule("[bold]Feasibility Statistics")

    all_sols = ce_mc.get_top_solutions(n=0)
    n_feas = sum(1 for s in all_sols if is_valid_tsp_tour(s.bitstring, N_CITIES))
    prob_feas = sum(
        s.prob for s in all_sols if is_valid_tsp_tour(s.bitstring, N_CITIES)
    )
    console.print(
        f"Unique bitstrings: {len(all_sols)} "
        f"({n_feas} feasible, {len(all_sols) - n_feas} infeasible)"
    )
    console.print(f"Feasible probability mass: {prob_feas:.2%}")
    console.print(
        f"Hilbert space: 2^{(N_CITIES-1)**2} = {2**((N_CITIES-1)**2)} states, "
        f"feasible: {N_CITIES-1}! = {problem.feasible_dimension}"
    )

    # ── 5) Repair + top solutions ─────────────────────────────────────
    console.rule("[bold]Top Solutions (with repair)")

    repaired = ce_mc.get_top_solutions(n=5, feasibility="repair", include_decoded=True)
    _print_top_tours(repaired, opt_cost)
