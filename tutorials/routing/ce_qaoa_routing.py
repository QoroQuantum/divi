# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Constraint-Enhanced QAOA on routing problems (arXiv:2511.14296).

Two complementary walkthroughs sharing the CE-QAOA pattern:

Part 1 — TSP (4 cities, asymmetric edge weights)
    1. Brute-force reference solution.
    2. Grid search over (gamma, beta) with loss landscape visualization.
    3. Parameter transfer: best grid angles → variational refinement.
    4. Feasibility statistics and Hungarian-algorithm repair.

The 4-city instance is chosen so the (4-1)! = 6 feasible tours have
*different* costs.  At 3 cities the constraint-preserving mixer keeps
the state in a 2-d subspace where both tours have identical cost, so
the grid-search landscape is degenerate and parameter transfer carries
no signal — bumping to 4 cities makes both genuinely informative.

Part 2 — CVRP (3 customers, 2 vehicles)
    1. One-hot encoding on a small instance.
    2. Binary encoding on the same instance — qubit savings comparison.
    3. Qubit projections for QOBLIB-scale problems (n=20, k=4).
    4. VRP file parser: loading a TSPLIB/CVRPLIB ``.vrp`` instance.
"""

import tempfile
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from divi.qprog import QAOA, GridSearchOptimizer, MonteCarloOptimizer
from divi.qprog.problems import (
    CVRPProblem,
    TSPProblem,
    binary_block_config,
    cvrp_block_structure,
    is_valid_tsp_tour,
    parse_vrp_file,
    parse_vrp_solution,
    tour_cost,
)
from tutorials._backend import get_backend

# ── TSP fixture ─────────────────────────────────────────────────────────
# 4 cities with asymmetric edge weights so the 6 feasible tours have
# distinct costs (85, 90, 95) — required for the grid landscape to vary.
TSP_COST_MATRIX = np.array(
    [
        [0, 10, 20, 15],
        [10, 0, 35, 25],
        [20, 35, 0, 30],
        [15, 25, 30, 0],
    ],
    dtype=float,
)
TSP_START_CITY = 0
TSP_N_CITIES = 4

# ── CVRP fixture ────────────────────────────────────────────────────────
CVRP_COST_MATRIX = np.array(
    [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 12], [20, 30, 12, 0]],
    dtype=float,
)
CVRP_DEMANDS = np.array([0, 3, 4, 2], dtype=float)
CVRP_CAPACITY = 6.0
CVRP_N_VEHICLES = 2
CVRP_DEPOT = 0

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


def _brute_force_tsp(cost_matrix, start_city):
    """Optimal TSP tour by exhaustive enumeration."""
    cities = [c for c in range(len(cost_matrix)) if c != start_city]
    best_cost, best_tour = float("inf"), None
    for perm in permutations(cities):
        t = [start_city, *perm, start_city]
        c = tour_cost(t, cost_matrix)
        if c < best_cost:
            best_cost, best_tour = c, t
    return best_tour, best_cost


def _plot_loss_landscape(losses, grid):
    """Render the (γ, β) loss grid as an annotated heatmap."""
    gammas, betas = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = np.full((gammas.size, betas.size), np.nan)
    z[np.searchsorted(gammas, grid[:, 0]), np.searchsorted(betas, grid[:, 1])] = losses

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(z, cmap="viridis", aspect="auto")
    ax.set_xticks(range(betas.size), [f"{b:.2f}" for b in betas])
    ax.set_yticks(range(gammas.size), [f"{g:.2f}" for g in gammas])
    ax.set(xlabel="β", ylabel="γ", title="CE-QAOA Loss Landscape")

    midpoint = (np.nanmin(z) + np.nanmax(z)) / 2
    for (i, j), v in np.ndenumerate(z):
        ax.text(
            j,
            i,
            f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v < midpoint else "black",
        )

    fig.colorbar(im, ax=ax, label="loss")
    fig.tight_layout()
    plt.show()


def _print_qubit_projections():
    """Render the 6-column qubit-scaling table via rich.Table."""
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
    Console().print(proj)
    print("(Paper's enhanced formula for QOBLIB N=20, K=4: ~133 qubits.)")


def _print_top_tours(repaired, opt_cost):
    print(f"\n{'#':<4} {'Tour':<28} {'Cost':>8} {'Prob':>8} {'Gap':>8}")
    print("-" * 60)
    for i, sol in enumerate(repaired, 1):
        tour = " → ".join(str(c) for c in sol.decoded) if sol.decoded else "—"
        cost = f"{sol.energy:.1f}" if sol.energy else "—"
        gap = f"{(sol.energy - opt_cost) / opt_cost * 100:.1f}%" if sol.energy else "—"
        print(f"{i:<4} {tour:<28} {cost:>8} {sol.prob:>7.2%} {gap:>8}")
    print(f"\nOptimal cost: {opt_cost}")


def _print_one_hot_routes(oh_sols):
    print(f"\n{'#':<4} {'Routes':<32} {'Cost':>6} {'Prob':>8}")
    print("-" * 56)
    for i, sol in enumerate(oh_sols, 1):
        if sol.decoded:
            display = " | ".join(
                " → ".join(str(c) for c in r) for r in sol.decoded if len(r) > 2
            )
        else:
            display = "—"
        cost = f"{sol.energy:.0f}" if sol.energy else "—"
        print(f"{i:<4} {display:<32} {cost:>6} {sol.prob:>7.2%}")


def _print_encoding_comparison(ce_oh, ce_bin, bin_logical_qubits):
    print(f"\n{'Metric':<12} {'One-Hot':>10} {'Binary':>10}")
    print("-" * 36)
    print(f"{'Qubits':<12} {ce_oh.n_qubits:>10} {ce_bin.n_qubits:>10}")
    print(
        f"{'Circuits':<12} {ce_oh.total_circuit_count:>10} "
        f"{ce_bin.total_circuit_count:>10}"
    )
    print(f"{'Best loss':<12} {ce_oh.best_loss:>10.4f} {ce_bin.best_loss:>10.4f}")
    print(
        f"\n(Binary uses {bin_logical_qubits} logical qubits + "
        f"{ce_bin.n_qubits - bin_logical_qubits} ancillas after HUBO quadratization; "
        "savings only materialize at larger N — see projections table.)"
    )


def _print_parsed_instance(inst, opt_routes, opt_cost):
    print(f"Instance: {inst.name}")
    print(
        f"Nodes: {inst.dimension} ({inst.n_customers} customers + depot), "
        f"Vehicles: {inst.n_vehicles}, Capacity: {inst.capacity}"
    )
    print(f"Optimal cost: {opt_cost}")
    for i, route in enumerate(opt_routes, 1):
        print(f"  Vehicle {i}: {' → '.join(str(n) for n in route)}")


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def run_tsp_walkthrough(backend) -> None:
    problem = TSPProblem(TSP_COST_MATRIX, start_city=TSP_START_CITY)

    # ── 1) Brute-force reference ──────────────────────────────────────
    _section("TSP — Brute-Force Optimal")
    opt_tour, opt_cost = _brute_force_tsp(TSP_COST_MATRIX, TSP_START_CITY)
    print(f"Tour: {opt_tour}  Cost: {opt_cost}")

    # ── 2) Grid search with landscape ─────────────────────────────────
    _section("TSP — Grid Search (5x5)")

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

    print(f"Grid points: {grid_opt.n_param_sets}")
    print(f"Best loss: {ce_grid.best_loss:.4f}")
    print(f"Best (γ, β): ({ce_grid.best_params[0]:.3f}, {ce_grid.best_params[1]:.3f})")

    _plot_loss_landscape(grid_opt._all_losses, grid_opt._param_grid)

    # ── 3) Parameter transfer: grid → variational ─────────────────────
    _section("TSP — Parameter Transfer → Monte Carlo Refinement")

    pop_size = 10
    best_grid_params = ce_grid.best_params.copy()
    rng = np.random.default_rng(42)
    warm_start = best_grid_params + rng.normal(0, 0.1, size=(pop_size, 2))
    warm_start = warm_start % (2 * np.pi)

    print(
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

    print(f"Best loss after refinement: {ce_mc.best_loss:.4f}")

    # ── 4) Feasibility statistics ─────────────────────────────────────
    _section("TSP — Feasibility Statistics")

    all_sols = ce_mc.get_top_solutions(n=0)
    n_feas = sum(1 for s in all_sols if is_valid_tsp_tour(s.bitstring, TSP_N_CITIES))
    prob_feas = sum(
        s.prob for s in all_sols if is_valid_tsp_tour(s.bitstring, TSP_N_CITIES)
    )
    print(
        f"Unique bitstrings: {len(all_sols)} "
        f"({n_feas} feasible, {len(all_sols) - n_feas} infeasible)"
    )
    print(f"Feasible probability mass: {prob_feas:.2%}")
    print(
        f"Hilbert space: 2^{(TSP_N_CITIES-1)**2} = "
        f"{2**((TSP_N_CITIES-1)**2)} states, "
        f"feasible: {TSP_N_CITIES-1}! = {problem.feasible_dimension}"
    )

    # ── 5) Repair + top solutions ─────────────────────────────────────
    _section("TSP — Top Solutions (with repair)")
    repaired = ce_mc.get_top_solutions(n=5, feasibility="repair", include_decoded=True)
    _print_top_tours(repaired, opt_cost)


def run_cvrp_walkthrough(backend) -> None:
    # ── 1) One-hot CE-QAOA ────────────────────────────────────────────
    _section("CVRP — One-Hot Encoding (3 customers, 2 vehicles)")

    problem_oh = CVRPProblem(
        CVRP_COST_MATRIX,
        demands=CVRP_DEMANDS,
        capacity=CVRP_CAPACITY,
        n_vehicles=CVRP_N_VEHICLES,
        depot=CVRP_DEPOT,
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

    print(
        f"Qubits: {ce_oh.n_qubits}  "
        f"({ce_oh.initial_state.n_blocks} blocks × {ce_oh.initial_state.block_size})"
    )
    print(f"Circuits: {ce_oh.total_circuit_count}")
    print(f"Best loss: {ce_oh.best_loss:.4f}")

    _print_one_hot_routes(
        ce_oh.get_top_solutions(n=3, feasibility="repair", include_decoded=True)
    )

    # ── 2) Binary CE-QAOA on same instance ────────────────────────────
    _section("CVRP — Binary Encoding (same instance)")

    problem_bin = CVRPProblem(
        CVRP_COST_MATRIX,
        demands=CVRP_DEMANDS,
        capacity=CVRP_CAPACITY,
        n_vehicles=CVRP_N_VEHICLES,
        depot=CVRP_DEPOT,
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
    print(
        f"Qubits: {ce_bin.n_qubits}  "
        f"({cfg.n_slots} slots × {cfg.bits_per_slot} bits)"
    )
    print(f"Circuits: {ce_bin.total_circuit_count}")
    print(f"Best loss: {ce_bin.best_loss:.4f}")

    # ── 3) Encoding comparison table ──────────────────────────────────
    _section("CVRP — One-Hot vs Binary Comparison")
    _print_encoding_comparison(ce_oh, ce_bin, problem_bin.binary_config.n_qubits)

    # ── 4) Qubit projections at scale ─────────────────────────────────
    _section("CVRP — Qubit Projections")
    _print_qubit_projections()

    # ── 5) VRP file parser demo ───────────────────────────────────────
    _section("CVRP — VRP File Parser")

    with tempfile.TemporaryDirectory() as tmp:
        vrp_path = Path(tmp) / "demo.vrp"
        sol_path = Path(tmp) / "demo.opt.sol"
        vrp_path.write_text(SAMPLE_VRP)
        sol_path.write_text(SAMPLE_SOL)

        inst = parse_vrp_file(vrp_path)
        opt_routes, opt_cost = parse_vrp_solution(sol_path)

    _print_parsed_instance(inst, opt_routes, opt_cost)


if __name__ == "__main__":
    backend = get_backend(shots=5000)

    print("\n=== Part 1 — CE-QAOA on TSP ===")
    run_tsp_walkthrough(backend)

    print("\n=== Part 2 — CE-QAOA on CVRP ===")
    run_cvrp_walkthrough(backend)
