# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
QAOA with QDrift Trotterization.

This tutorial demonstrates QDrift, a randomized Trotterization strategy that
approximates the cost Hamiltonian by sampling terms. QDrift is useful for large
Hamiltonians where exact Trotterization is expensive.

Key concepts:
- keep_fraction: Deterministically keep the top fraction of terms by coefficient magnitude
- sampling_budget: Number of terms to sample from the remaining Hamiltonian
- n_hamiltonians_per_iteration: Multiple samples per cost evaluation; losses are averaged
- sampling_strategy: "uniform" or "weighted" (by coefficient magnitude)
"""

import random
import warnings

import networkx as nx

from divi.backends import ParallelSimulator
from divi.qprog import QAOA, GraphProblem, QDrift
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer


def _generate_random_graph(
    n_nodes: int, n_edges: int, seed: int | None = None
) -> nx.Graph:
    """Generate a random connected graph with n_nodes and n_edges."""
    if seed is not None:
        random.seed(seed)
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        warnings.warn(
            f"Requested {n_edges} edges, but max for {n_nodes} nodes is {max_edges}. Capping."
        )
        n_edges = max_edges
    graph = nx.random_labeled_tree(n_nodes)
    current_edges = set(graph.edges())
    while len(current_edges) < n_edges:
        u, v = random.sample(range(n_nodes), 2)
        edge = tuple(sorted((u, v)))
        if edge not in current_edges:
            graph.add_edge(*edge)
            current_edges.add(edge)
    return graph


def _cut_value(graph: nx.Graph, partition: list) -> int:
    """Number of edges crossing the cut (partition vs complement)."""
    p0 = set(graph.nodes()) - set(partition)
    return sum(1 for u, v in graph.edges() if (u in p0) != (v in p0))


if __name__ == "__main__":
    # Use a larger graph so QDrift can show a circuit-count advantage: with many Hamiltonian
    # terms, sampling a subset uses fewer circuits per cost evaluation than exact Trotterization.
    N_NODES, N_EDGES = 12, 25
    G = _generate_random_graph(N_NODES, N_EDGES, seed=1997)
    max_cut_val, _ = nx.algorithms.approximation.maxcut.one_exchange(G)

    # Single backend with depth tracking; reset tracker between runs to compare depths
    backend = ParallelSimulator(shots=1000, track_depth=True)

    common = dict(
        problem=G,
        graph_problem=GraphProblem.MAXCUT,
        n_layers=1,
        max_iterations=5,
        backend=backend,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        seed=1997,
    )

    # Baseline: default ExactTrotterization (full Hamiltonian, all terms)
    qaoa_exact = QAOA(**common)
    qaoa_exact.run()
    depth_exact = f"{backend.average_depth():.1f} ± {backend.std_depth():.1f}"
    backend.clear_depth_history()

    # QDrift: keep 20% of terms, sample 2 from the rest, 3 Hamiltonian samples per iteration.
    # Fewer terms per sample than exact -> shallower circuits (fewer evolution layers).
    qdrift_strategy = QDrift(
        keep_fraction=0.2,
        sampling_budget=2,
        n_hamiltonians_per_iteration=3,
        sampling_strategy="weighted",
        seed=1997,
    )
    qaoa_qdrift = QAOA(
        trotterization_strategy=qdrift_strategy,
        **common,
    )
    qaoa_qdrift.run()
    depth_qdrift = f"{backend.average_depth():.1f} ± {backend.std_depth():.1f}"

    # --- Print comparison table ---
    cut_exact = _cut_value(G, qaoa_exact.solution) if qaoa_exact.solution else 0
    cut_qdrift = _cut_value(G, qaoa_qdrift.solution) if qaoa_qdrift.solution else 0

    rows = [
        (
            "Exact Trotterization",
            qaoa_exact.best_loss,
            qaoa_exact.total_circuit_count,
            depth_exact,
            qaoa_exact.solution,
            cut_exact,
        ),
        (
            "QDrift",
            qaoa_qdrift.best_loss,
            qaoa_qdrift.total_circuit_count,
            depth_qdrift,
            qaoa_qdrift.solution,
            cut_qdrift,
        ),
    ]
    pad = "  "
    pad_light = " "
    col_m, col_l, col_c, col_d, col_s, col_k = 24, 12, 10, 23, 26, 8
    sep_len = (
        col_m + col_l + col_c + col_d + col_s + col_k + len(pad) * 4 + len(pad_light)
    )
    print("\n" + "-" * sep_len)
    print(
        f"{'Method':<{col_m}}{pad}{'Best loss':>{col_l}}{pad}{'# Circuits':>{col_c}}{pad}"
        f"{'Circuit Depth (avg±std)':<{col_d}}{pad}{'Solution Nodes':<{col_s}}{pad_light}{'Cut Size':>{col_k}}"
    )
    print("-" * sep_len)
    for method, loss, circuits, depth, sol, cut in rows:
        sol_str = str(sol)
        if len(sol_str) > col_s - 2:
            cut_at = sol_str[: col_s - 5].rfind(", ")
            sol_str = (
                sol_str[: cut_at + 1] if cut_at > 0 else sol_str[: col_s - 5]
            ) + "..."
        print(
            f"{method:<{col_m}}{pad}{loss:>{col_l}.4f}{pad}{circuits:>{col_c}}{pad}"
            f"{depth:<{col_d}}{pad}{sol_str:<{col_s}}{pad_light}{cut:>{col_k}}"
        )
    print("-" * sep_len)
    print(f"Optimal MAXCUT value: {max_cut_val}")
    print(
        "\nNote: QDrift yields shallower circuits (fewer evolution layers per circuit) "
        "at the cost of more circuits per iteration (n_hamiltonians_per_iteration samples). "
        "On noisy hardware, lower depth can improve fidelity despite the higher circuit count."
    )
