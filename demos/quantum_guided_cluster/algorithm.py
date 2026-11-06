# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum-guided cluster algorithm for Max-Cut (arXiv:2508.10656).

Graph generation and cost functions, two-point correlation extraction (QAOA or
PCE) into a ``CorrelationResult``, the correlation-guided cluster Monte Carlo,
and the simulated-annealing and coupling-constant runs.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

from divi.backends import MaestroSimulator
from divi.hamiltonians.trotterization import QDrift
from divi.qprog import EarlyStopping
from divi.qprog.algorithms import PCE, QAOA, DenseEncoding, PolyEncoding, SoftObjective
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.qprog.problems import BinaryOptimizationProblem


def generate_random_maxcut_graph(n: int, degree: int, seed: int = 42) -> nx.Graph:
    """Generate a random regular graph with ±1 edge weights (Ising spin glass).

    Weights are drawn uniformly from {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(degree, n, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.choice([-1.0, 1.0])
    return G


def compute_ising_energy(G: nx.Graph, config: np.ndarray) -> float:
    """Compute Ising energy H = -Σ_{(i,j)} J_ij x_i x_j."""
    energy = 0.0
    for u, v, w in G.edges(data="weight", default=1.0):
        energy -= w * config[u] * config[v]
    return energy


@dataclass
class CorrelationResult:
    """Return type shared by the QAOA and PCE correlation extractors."""

    Z: np.ndarray
    label: str
    n_qubits: int
    total_circuit_count: int


def _compute_correlations(distribution, n_vars: int) -> np.ndarray:
    """Compute Z_ij = ⟨Z_i Z_j⟩ = Σ_x p(x) · spin(x)_i · spin(x)_j.

    ``distribution`` is an iterable of ``(bitstring, prob)`` pairs where each
    bitstring is the length-``n_vars`` 0/1 assignment in variable order.
    """
    Z = np.zeros((n_vars, n_vars))
    for bitstring, prob in distribution:
        spins = np.fromiter(
            (1 - 2 * int(b) for b in bitstring), dtype=float, count=n_vars
        )
        Z += prob * np.outer(spins, spins)
    np.fill_diagonal(Z, 1.0)
    return Z


def extract_qaoa_correlations(
    G: nx.Graph,
    n_layers: int = 2,
    *,
    max_iterations: int = 10,
    patience: int = 3,
    shots: int = 10_000,
    use_qdrift: bool = False,
    backend=None,
) -> CorrelationResult:
    """Run QAOA on a Max-Cut instance and extract the ZZ correlation matrix.

    The optimised state is sampled and Z_ij = Σ_x p(x) · σ_i^z(x) · σ_j^z(x)
    is computed classically from bitstring frequencies.

    Args:
        G: The graph (Max-Cut instance).
        n_layers: Number of QAOA layers (circuit depth p).
        max_iterations: Optimizer iterations (cap; early-stopping may halt sooner).
        patience: EarlyStopping patience.
        shots: Number of measurement shots.
        use_qdrift: If True, use QDrift trotterisation.
        backend: Divi backend. Defaults to MaestroSimulator.

    Returns:
        CorrelationResult — Z matrix plus metadata for plots/tables.
    """
    n = G.number_of_nodes()
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    qaoa_kwargs = dict(
        problem=BinaryOptimizationProblem(build_maxcut_qubo(G)),
        n_layers=n_layers,
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=20),
        max_iterations=max_iterations,
        early_stopping=EarlyStopping(patience=patience),
        backend=backend,
    )
    if use_qdrift:
        qaoa_kwargs["trotterization_strategy"] = QDrift(
            keep_fraction=0.3,
            sampling_budget=8,
            n_hamiltonians_per_iteration=3,
            sampling_strategy="weighted",
            seed=42,
        )

    qaoa = QAOA(**qaoa_kwargs)
    label = f"QAOA p={n_layers}"
    print(f"  Running {label} on {n} qubits...")
    t0 = time.time()
    qaoa.run()
    elapsed = time.time() - t0
    print(
        f"  done in {elapsed:.1f}s  |  "
        f"circuits={qaoa.total_circuit_count}  |  "
        f"best_loss={qaoa.best_loss:.4f}"
    )

    return CorrelationResult(
        Z=qaoa.get_correlations(),
        label=label,
        n_qubits=n,
        total_circuit_count=qaoa.total_circuit_count,
    )


@dataclass
class ClusterAlgoResult:
    """Result container for the cluster algorithm."""

    best_config: np.ndarray
    best_energy: float
    energy_history: list[float] = field(default_factory=list)
    acceptance_rate: float = 0.0


def estimate_percolation_threshold(G: nx.Graph) -> float:
    """Estimate the bond percolation threshold of a graph.

    p_c ≈ (⟨d⟩ - 1) / (⟨d²⟩ - ⟨d⟩)
    where ⟨d⟩ and ⟨d²⟩ are the first and second moments of the degree distribution.
    """
    degrees = np.array([d for _, d in G.degree()])
    d_mean = degrees.mean()
    d2_mean = (degrees**2).mean()
    if d2_mean - d_mean < 1e-10:
        return 0.5
    return (d_mean - 1.0) / (d2_mean - d_mean)


def create_cluster(
    G: nx.Graph,
    seed_node: int,
    config: np.ndarray,
    Z: np.ndarray,
    lambda_scale: float,
    p_c: float,
    mean_abs_z: float,
    rng: random.Random,
) -> set[int]:
    """Build a cluster starting from seed_node using the correlation matrix Z.

    Grows a cluster via graph contraction, adding neighbours with probability
    p_link proportional to the correlation strength between a cluster node
    and its neighbour.

    Args:
        G: The graph.
        seed_node: Starting node for cluster growth.
        config: Current spin configuration (±1).
        Z: Correlation matrix.
        lambda_scale: Scaling hyperparameter λ_scale.
        p_c: Estimated percolation threshold.
        mean_abs_z: Mean absolute value of nonzero off-diagonal correlations.
        rng: Random number generator.

    Returns:
        Set of node indices comprising the cluster.
    """
    cluster = {seed_node}
    # frontier stores (parent_in_cluster, neighbor_candidate) pairs
    frontier = [(seed_node, nb) for nb in G.neighbors(seed_node)]
    rng.shuffle(frontier)
    visited = {seed_node}

    while frontier:
        parent, neighbor = frontier.pop()

        if neighbor in visited:
            continue
        visited.add(neighbor)

        # Link probability from the correlation between the parent cluster node
        # and the candidate
        z_ij = Z[parent, neighbor]

        # Skip candidates whose correlation sign differs from the spin agreement
        agreement = config[parent] * config[neighbor]
        effective_z = z_ij * agreement

        if effective_z <= 0:
            continue

        # p_link = lambda_scale * p_c * |Z_ij| / (2 * E[|Z|])
        if mean_abs_z < 1e-10:
            p_link = p_c
        else:
            p_link = lambda_scale * p_c * abs(z_ij) / (2.0 * mean_abs_z)

        p_link = min(p_link, 1.0)

        if rng.random() < p_link:
            cluster.add(neighbor)

            # Expand frontier from newly added node (graph contraction)
            new_edges = [
                (neighbor, nn) for nn in G.neighbors(neighbor) if nn not in visited
            ]
            rng.shuffle(new_edges)
            frontier.extend(new_edges)

    return cluster


def run_cluster_algorithm(
    G: nx.Graph,
    Z: np.ndarray,
    n_iterations_factor: int = 500,
    n_repetitions: int = 10,
    lambda_scale: float = 6,
    beta_f: float = 8.0,
    seed: int = 42,
) -> ClusterAlgoResult:
    """Run the correlation-guided cluster algorithm.

    Clusters are formed from the correlation matrix Z and flipped as a unit
    under a Metropolis criterion with a linear inverse-temperature schedule.

    Args:
        G: The Max-Cut graph.
        Z: (n x n) two-point correlation matrix from QAOA or other source.
        n_iterations_factor: Total iterations = n_iterations_factor * n.
        n_repetitions: Number of independent random restarts.
        lambda_scale: Scaling factor for link probability.
        beta_f: Final inverse temperature.
        seed: Random seed.

    Returns:
        ClusterAlgoResult with best configuration found.
    """
    n = G.number_of_nodes()
    total_iters = n_iterations_factor * n
    rng = random.Random(seed)

    p_c = estimate_percolation_threshold(G)

    # Mean absolute value of nonzero off-diagonal Z entries
    mask = ~np.eye(n, dtype=bool)
    nonzero_z = np.abs(Z[mask])
    nonzero_z = nonzero_z[nonzero_z > 1e-10]
    mean_abs_z = nonzero_z.mean() if len(nonzero_z) > 0 else 1.0

    overall_best_config = None
    overall_best_energy = np.inf
    all_energies = []

    n_accepted = 0
    n_total_proposals = 0

    for _ in range(n_repetitions):
        config = np.array([rng.choice([-1, 1]) for _ in range(n)])
        current_energy = compute_ising_energy(G, config)

        best_config = config.copy()
        best_energy = current_energy

        iteration = 0
        while iteration < total_iters:
            # Inverse temperature schedule (linear)
            beta = 1.0 + (beta_f - 1.0) * iteration / total_iters

            seed_node = rng.randint(0, n - 1)
            cluster = create_cluster(
                G, seed_node, config, Z, lambda_scale, p_c, mean_abs_z, rng
            )
            cluster_size = len(cluster)

            delta_e = 0.0
            for node in cluster:
                for neighbor in G.neighbors(node):
                    w = G[node][neighbor].get("weight", 1.0)
                    if neighbor not in cluster:
                        # External edge: energy change = 2 * J_ij * x_i * x_j
                        delta_e += 2.0 * w * config[node] * config[neighbor]

            # Metropolis acceptance criterion
            n_total_proposals += 1
            if delta_e <= 0 or rng.random() < math.exp(-beta * delta_e):
                for node in cluster:
                    config[node] *= -1
                current_energy += delta_e
                n_accepted += 1

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_config = config.copy()

            # Each cluster flip of size k counts as k iterations
            iteration += cluster_size

        all_energies.append(best_energy)

        if best_energy < overall_best_energy:
            overall_best_energy = best_energy
            overall_best_config = best_config.copy()

    acceptance_rate = n_accepted / max(n_total_proposals, 1)

    return ClusterAlgoResult(
        best_config=overall_best_config,
        best_energy=overall_best_energy,
        energy_history=all_energies,
        acceptance_rate=acceptance_rate,
    )


def run_simulated_annealing(
    G: nx.Graph,
    n_iterations_factor: int = 500,
    n_repetitions: int = 10,
    beta_f: float = 8.0,
    seed: int = 42,
) -> ClusterAlgoResult:
    """Single-spin-flip simulated annealing with a linear inverse-temperature schedule."""
    n = G.number_of_nodes()
    total_iters = n_iterations_factor * n
    rng = random.Random(seed)

    overall_best_config = None
    overall_best_energy = np.inf
    all_energies = []

    for _ in range(n_repetitions):
        config = np.array([rng.choice([-1, 1]) for _ in range(n)])
        current_energy = compute_ising_energy(G, config)
        best_config = config.copy()
        best_energy = current_energy

        for iteration in range(total_iters):
            beta = 1.0 + (beta_f - 1.0) * iteration / total_iters

            node = rng.randint(0, n - 1)
            delta_e = 0.0
            for neighbor in G.neighbors(node):
                w = G[node][neighbor].get("weight", 1.0)
                delta_e += 2.0 * w * config[node] * config[neighbor]

            if delta_e <= 0 or rng.random() < math.exp(-beta * delta_e):
                config[node] *= -1
                current_energy += delta_e

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_config = config.copy()

        all_energies.append(best_energy)

        if best_energy < overall_best_energy:
            overall_best_energy = best_energy
            overall_best_config = best_config.copy()

    return ClusterAlgoResult(
        best_config=overall_best_config,
        best_energy=overall_best_energy,
        energy_history=all_energies,
    )


def build_coupling_correlations(G: nx.Graph) -> np.ndarray:
    """Build Z from the coupling constants J_ij, with ones on the diagonal."""
    n = G.number_of_nodes()
    Z = np.zeros((n, n))
    for u, v, w in G.edges(data="weight", default=1.0):
        Z[u, v] = w
        Z[v, u] = w
    np.fill_diagonal(Z, 1.0)
    return Z


def build_maxcut_qubo(G: nx.Graph) -> np.ndarray:
    """Convert the spin-glass energy to a QUBO matrix.

    With x_i = (1 - s_i) / 2 the energy H = -Σ_{(i,j)} J_ij s_i s_j equals
    2·C(x) - Σ J_ij, where the weighted cut is
        C(x) = Σ_{(i,j)∈E} w_ij * (x_i + x_j - 2 x_i x_j)

    Minimising x^T Q x therefore finds the ground state of H:
        Q_ij = -w_ij  (off-diagonal, for each edge)
        Q_ii = Σ_j w_ij  (diagonal = weighted degree)
    """
    n = G.number_of_nodes()
    Q = np.zeros((n, n))
    for u, v, w in G.edges(data="weight", default=1.0):
        Q[u, v] -= w
        Q[v, u] -= w
        Q[u, u] += w
        Q[v, v] += w
    return Q


def extract_pce_correlations(
    G: nx.Graph,
    encoding: Literal["dense", "poly"] = "dense",
    *,
    n_layers: int = 2,
    max_iterations: int = 15,
    alpha: float = 2.0,
    shots: int = 10_000,
    backend=None,
) -> CorrelationResult:
    """Run PCE on a Max-Cut QUBO and extract variable-variable correlations.

    PCE maps each variable to the parity of a fixed qubit subset. The
    optimised state is sampled in the encoded space and decoded: each encoded
    bitstring gives an N-vector of ±1 spins, from which Z_ij is computed as
    for QAOA.

    Args:
        G: The Max-Cut graph.
        encoding: ``"dense"`` or ``"poly"`` (``DenseEncoding`` /
            ``PolyEncoding``).
        n_layers: PCE ansatz depth.
        max_iterations: Optimizer iterations.
        alpha: Sharpness of the ``SoftObjective`` relaxation.
        shots: Number of measurement shots.
        backend: Divi backend. Defaults to MaestroSimulator.

    Returns:
        CorrelationResult — Z matrix plus metadata.
    """
    n = G.number_of_nodes()
    Q = build_maxcut_qubo(G)
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    pce = PCE(
        BinaryOptimizationProblem(Q),
        encoding=DenseEncoding() if encoding == "dense" else PolyEncoding(),
        objective=SoftObjective(alpha=alpha),
        n_layers=n_layers,
        max_iterations=max_iterations,
        backend=backend,
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=20),
    )

    label = f"PCE {encoding}"
    print(f"  Running {label} ({pce.n_qubits} qubits for {n} variables)...")
    t0 = time.time()
    pce.run()
    elapsed = time.time() - t0
    print(
        f"  done in {elapsed:.1f}s  |  "
        f"circuits={pce.total_circuit_count}  |  "
        f"best_loss={pce.best_loss:.4f}"
    )

    # ``bitstring`` is the decoded length-N assignment, not the encoded qubits.
    distribution = (
        (entry.bitstring, entry.prob)
        for entry in pce.get_top_solutions(n=0, sort_by="prob")
    )
    Z = _compute_correlations(distribution, n_vars=n)
    return CorrelationResult(
        Z=Z,
        label=label,
        n_qubits=pce.n_qubits,
        total_circuit_count=pce.total_circuit_count,
    )
