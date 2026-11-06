# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Travelling Salesman Problem via QAOA, partitioned QAOA and PCE.

Part A runs ``TSPProblem`` directly with constraint-enhanced QAOA. Part B
clusters the cities into sub-tours with ``PartitioningProgramEnsemble``. Part C
runs Pauli Correlation Encoding (PCE) on the assignment variables.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import CXGate, RYGate, RZGate

from divi.backends import JobConfig, MaestroSimulator, QoroService
from divi.qprog import EarlyStopping
from divi.qprog.algorithms import (
    PCE,
    QAOA,
    DenseEncoding,
    GenericLayerAnsatz,
    PolyEncoding,
    SoftObjective,
)
from divi.qprog.ensembles import BeamSearchStrategy, PartitioningProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer, PymooMethod, PymooOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    LinearConstraint,
    TSPProblem,
    compute_tour_cost,
    create_tsp_qubo,
)


def generate_cities(n_cities: int, seed: int = 42) -> np.ndarray:
    """Generate random 2D city coordinates in the unit square.

    Args:
        n_cities: Number of cities.
        seed:     Random seed for reproducibility.

    Returns:
        Array of shape (n_cities, 2) with (x, y) coordinates.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))


def compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix between all city pairs.

    Args:
        cities: Array of shape (n_cities, 2).

    Returns:
        Symmetric distance matrix of shape (n_cities, n_cities).
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def solve_brute_force(dist_matrix: np.ndarray) -> tuple[list[int], float]:
    """Find the shortest tour by exhaustive enumeration.

    Fixes city 0 as the tour start (symmetry reduction) and checks
    all (n−1)! permutations.

    Args:
        dist_matrix: Distance matrix of shape (n, n).

    Returns:
        (best_tour, best_distance), the tour returning to city 0.
    """
    n = len(dist_matrix)
    best_tour, best_dist = None, float("inf")

    for perm in itertools.permutations(range(1, n)):
        candidate = [0, *perm, 0]
        d = compute_tour_cost(candidate, dist_matrix)
        if d < best_dist:
            best_dist = d
            best_tour = candidate

    return best_tour, best_dist


def solve_with_qaoa(
    dist_matrix: np.ndarray,
    n_layers: int = 2,
    max_iterations: int = 15,
    population_size: int = 50,
    n_best_sets: int = 5,
    shots: int = 10_000,
    backend=None,
) -> QAOA:
    """Run constraint-enhanced QAOA on a TSP instance.

    The start city is fixed, so the register holds (n−1)² qubits. A W-state
    start and an XY mixer keep every city in exactly one time slot.

    Args:
        dist_matrix:      Distance matrix.
        n_layers:         Number of QAOA layers (circuit depth).
        max_iterations:   Optimiser iterations.
        population_size:  Monte Carlo population size.
        n_best_sets:      Number of elite parameter sets to keep.
        shots:            Measurement samples per circuit evaluation.
        backend:          Divi backend (defaults to MaestroSimulator).

    Returns:
        The QAOA instance after ``run()``.
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    qaoa = QAOA(
        problem=TSPProblem(dist_matrix),
        n_layers=n_layers,
        max_iterations=max_iterations,
        optimizer=MonteCarloOptimizer(
            population_size=population_size,
            n_best_sets=n_best_sets,
        ),
        backend=backend,
    )

    qaoa.run()
    return qaoa


def extract_best_tour(qaoa: QAOA, top_n: int = 100) -> tuple[list[int], float]:
    """Return the cheapest valid tour among the most probable QAOA bitstrings.

    Invalid bitstrings are repaired to the nearest valid tour with the
    Hungarian algorithm.

    Args:
        qaoa:  Solved QAOA instance.
        top_n: How many top bitstrings to inspect.

    Returns:
        (tour, distance), the tour returning to the start city.
    """
    n_feasible = len(
        qaoa.get_top_solutions(n=top_n, include_decoded=True, feasibility="filter")
    )
    print(
        f"\n   Inspecting top {top_n} QAOA bitstrings: "
        f"{n_feasible} exactly feasible, the rest repaired"
    )
    best = qaoa.get_top_solutions(n=top_n, include_decoded=True, feasibility="repair")[
        0
    ]
    return best.decoded, best.energy


def plot_cities(cities: np.ndarray, title: str = "City Locations"):
    """Plot city locations as a scatter plot.

    Args:
        cities: Array of shape (n, 2).
        title:  Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        cities[:, 0],
        cities[:, 1],
        s=120,
        c="#4FC3F7",
        edgecolors="#0D47A1",
        zorder=5,
        linewidths=1.5,
    )
    for i, (x, y) in enumerate(cities):
        ax.annotate(
            str(i), (x, y), fontsize=10, ha="center", va="center", fontweight="bold"
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_comparison(
    cities: np.ndarray,
    classical_tour: list[int],
    classical_dist: float,
    quantum_tour: list[int],
    quantum_dist: float,
    save_path: str | None = None,
):
    """Side-by-side comparison of classical and quantum tours.

    Args:
        cities:         Array of shape (n, 2).
        classical_tour: Best classical tour, returning to its start.
        classical_dist: Classical tour distance.
        quantum_tour:   Best quantum tour, returning to its start.
        quantum_dist:   Quantum tour distance.
        save_path:      If given, save the figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for ax, t, d, label, col in [
        (ax1, classical_tour, classical_dist, "Classical (Brute Force)", "#FF7043"),
        (ax2, quantum_tour, quantum_dist, "Quantum (Partitioned QAOA)", "#4FC3F7"),
    ]:
        for i, j in itertools.pairwise(t):
            ax.annotate(
                "",
                xy=cities[j],
                xytext=cities[i],
                arrowprops=dict(arrowstyle="->", color=col, lw=2),
            )
        ax.scatter(
            cities[:, 0],
            cities[:, 1],
            s=180,
            c="#E3F2FD",
            edgecolors="#0D47A1",
            zorder=5,
            linewidths=2,
        )
        for i, (x, y) in enumerate(cities):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=6,
            )
        ax.set_title(f"{label}\ndistance = {d:.4f}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(
        "Travelling Salesman — Classical vs. Quantum",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Comparison plot saved to {save_path}")
    plt.show()


def print_comparison(
    classical_tour: list[int],
    classical_dist: float,
    quantum_tour: list[int],
    quantum_dist: float,
):
    """Print a one-line comparison of classical vs quantum solutions."""
    ratio = quantum_dist / classical_dist
    print(
        f"  classical {classical_tour} d={classical_dist:.4f}  |  "
        f"quantum {quantum_tour} d={quantum_dist:.4f}  |  ratio={ratio:.3f}"
    )


def solve_partitioned_tsp(
    dist_matrix: np.ndarray,
    max_cities_per_partition: int = 4,
    n_layers: int = 2,
    max_iterations: int = 10,
    population_size: int = 30,
    n_best_sets: int = 5,
    shots: int = 10_000,
    backend=None,
    patience: int = 3,
) -> tuple[list[int], float]:
    """Run a partitioned TSP: cluster the cities and patch the sub-tours.

    ``TSPProblem`` clusters the cities into groups of at most
    ``max_cities_per_partition``. Each group runs as its own
    constraint-enhanced QAOA over a closed sub-tour, and beam-search
    aggregation patches the sub-tours into one tour by repeatedly merging the
    pair whose edge exchange adds the least distance.

    Args:
        dist_matrix: Distance matrix (n_cities × n_cities).
        max_cities_per_partition: Max cities per sub-problem (at least 3).
        n_layers: QAOA circuit depth per sub-problem.
        max_iterations: Optimiser iterations per sub-problem.
        population_size: MonteCarloOptimizer population.
        n_best_sets: Top parameter sets retained per iteration.
        shots: Measurement shots.
        backend: Divi backend (defaults to MaestroSimulator).
        patience: EarlyStopping patience per sub-problem.

    Returns:
        ``(tour, distance)``, the tour returning to city 0.
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    problem = TSPProblem(dist_matrix, max_cities_per_partition=max_cities_per_partition)
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=n_layers,
        optimizer=MonteCarloOptimizer(
            population_size=population_size,
            n_best_sets=n_best_sets,
        ),
        max_iterations=max_iterations,
        early_stopping=EarlyStopping(patience=patience),
        backend=backend,
    )
    ensemble.run()
    print(f"  Solved {len(ensemble.programs)} sub-programs")

    aggregated = ensemble.aggregate_results(
        BeamSearchStrategy(beam_width=3, n_partition_candidates=5)
    )
    if aggregated is None:
        raise RuntimeError("Aggregation produced no valid solution.")
    return aggregated


def solve_with_pce(
    dist_matrix: np.ndarray,
    n_layers: int = 3,
    max_iterations: int = 20,
    alpha: float = 3.0,
    penalty_weight: float | None = None,
    population_size: int = 30,
    shots: int = 10_000,
    encoding_type: str = "poly",
    backend=None,
) -> tuple[list[int], float, int]:
    """Solve TSP using PCE (Pauli Correlation Encoding).

    The objective is the tour length over (n−1)² assignment variables, with
    the start city fixed. Each city taking one slot and each slot taking one
    city are linear equality constraints, which add no qubits. PCE compresses
    the variables into fewer qubits with a polynomial or dense encoding.

    Args:
        dist_matrix:      Distance matrix.
        n_layers:         Ansatz circuit depth.
        max_iterations:   Optimiser iterations.
        alpha:            Binary activation sharpness (higher = harder).
        penalty_weight:   Constraint penalty weight; defaults to four times
                          the longest distance.
        population_size:  DE optimiser population.
        shots:            Measurement samples.
        encoding_type:    "poly" or "dense".
        backend:          Divi backend.

    Returns:
        (tour, distance, n_qubits_used), the tour returning to city 0.
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    m = len(dist_matrix) - 1
    if penalty_weight is None:
        penalty_weight = 4.0 * float(dist_matrix.max())

    # Variable i*m + t: city i+1 is visited at slot t.
    objective = create_tsp_qubo(dist_matrix, penalty_weight=0.0)
    constraints = [
        LinearConstraint({i * m + t: 1 for t in range(m)}, "==", 1) for i in range(m)
    ] + [LinearConstraint({i * m + t: 1 for i in range(m)}, "==", 1) for t in range(m)]
    problem = BinaryOptimizationProblem(
        objective, constraints=constraints, penalty_weight=penalty_weight
    )

    pce_solver = PCE(
        problem,
        ansatz=GenericLayerAnsatz(
            gate_sequence=[RYGate, RZGate],
            entangler=CXGate,
            entangling_layout="all-to-all",
        ),
        n_layers=n_layers,
        encoding=DenseEncoding() if encoding_type == "dense" else PolyEncoding(),
        optimizer=PymooOptimizer(
            method=PymooMethod.DE, population_size=population_size
        ),
        max_iterations=max_iterations,
        objective=SoftObjective(alpha=alpha),
        backend=backend,
    )

    n_qubits = pce_solver.n_qubits
    print(f"   PCE qubits: {n_qubits}  ({encoding_type} encoding of {m * m} variables)")

    pce_solver.run()

    tsp = TSPProblem(dist_matrix)
    top_solutions = pce_solver.get_top_solutions(n=50)
    n_feasible = sum(tsp.is_feasible(sol.bitstring) for sol in top_solutions)
    print(
        f"   Found {n_feasible} exactly feasible solutions out of {len(top_solutions)}"
    )

    _, tour, distance = min(
        (tsp.repair_infeasible_bitstring(sol.bitstring) for sol in top_solutions),
        key=lambda repaired: repaired[2],
    )
    return tour, distance, n_qubits


if __name__ == "__main__":
    USE_CLOUD = False
    SHOTS = 20_000
    SEED = 42
    N_CITIES_SMALL = 4
    N_CITIES_LARGE = 9

    backend = (
        QoroService(job_config=JobConfig(shots=SHOTS))
        if USE_CLOUD
        else MaestroSimulator(shots=SHOTS)
    )
    print(
        f"Backend: {'QoroService' if USE_CLOUD else 'MaestroSimulator'} (shots={SHOTS})"
    )

    # Part A: Direct QAOA
    print(
        f"\n=== Part A: Direct QAOA ({N_CITIES_SMALL} cities, "
        f"{(N_CITIES_SMALL - 1) ** 2} qubits) ==="
    )
    cities_small = generate_cities(N_CITIES_SMALL, seed=SEED)
    dist_small = compute_distance_matrix(cities_small)
    classical_tour_small, classical_dist_small = solve_brute_force(dist_small)

    qaoa = solve_with_qaoa(
        dist_small, n_layers=3, max_iterations=10, shots=SHOTS, backend=backend
    )
    q_tour_small, q_dist_small = extract_best_tour(qaoa)
    print_comparison(
        classical_tour_small, classical_dist_small, q_tour_small, q_dist_small
    )
    plot_cities(cities_small, title=f"TSP — {N_CITIES_SMALL} Cities (Direct QAOA)")

    # Part B: Partitioned QAOA on a larger instance
    # 64 qubits unpartitioned; the cities are clustered into sub-problems.
    print(
        f"\n=== Part B: Partitioned QAOA ({N_CITIES_LARGE} cities, "
        f"{(N_CITIES_LARGE - 1) ** 2} qubits unpartitioned) ==="
    )
    cities_large = generate_cities(N_CITIES_LARGE, seed=SEED)
    dist_large = compute_distance_matrix(cities_large)
    classical_tour_large, classical_dist_large = solve_brute_force(dist_large)

    q_tour_large, q_dist_large = solve_partitioned_tsp(
        dist_large,
        max_cities_per_partition=4,
        n_layers=2,
        max_iterations=10,
        shots=SHOTS,
        backend=backend,
    )
    print_comparison(
        classical_tour_large, classical_dist_large, q_tour_large, q_dist_large
    )
    plot_comparison(
        cities_large,
        classical_tour_large,
        classical_dist_large,
        q_tour_large,
        q_dist_large,
        save_path="tsp_partitioned_comparison.png",
    )

    # Part C: PCE compression of the Part A instance
    print(
        f"\n=== Part C: PCE compression (same {N_CITIES_SMALL}-city "
        f"instance, qubit-compressed) ==="
    )
    pce_tour, pce_dist, pce_qubits = solve_with_pce(
        dist_small,
        n_layers=3,
        max_iterations=20,
        shots=SHOTS,
        backend=backend,
    )
    print(
        f"  PCE qubits: {pce_qubits} (vs {(N_CITIES_SMALL - 1) ** 2} for direct QAOA)"
    )
    print_comparison(classical_tour_small, classical_dist_small, pce_tour, pce_dist)

    # Summary
    print("\n=== Summary ===")
    rows = [
        ("Classical (brute force)", N_CITIES_SMALL, "—", classical_dist_small, 1.000),
        (
            "A: Direct QAOA",
            N_CITIES_SMALL,
            str((N_CITIES_SMALL - 1) ** 2),
            q_dist_small,
            q_dist_small / classical_dist_small,
        ),
        (
            "C: PCE (poly encoding)",
            N_CITIES_SMALL,
            str(pce_qubits),
            pce_dist,
            pce_dist / classical_dist_small,
        ),
        ("Classical (brute force)", N_CITIES_LARGE, "—", classical_dist_large, 1.000),
        (
            "B: Partitioned QAOA",
            N_CITIES_LARGE,
            "≤9",
            q_dist_large,
            q_dist_large / classical_dist_large,
        ),
    ]
    print(f"  {'Method':<28} {'Cities':>6} {'Qubits':>7} {'Distance':>10} {'Ratio':>7}")
    for name, nc, nq, dist, ratio in rows:
        print(f"  {name:<28} {nc:>6} {nq:>7} {dist:>10.4f} {ratio:>7.3f}")
