# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Partitioning a large problem into many quantum sub-programs.

``PartitioningProgramEnsemble`` decomposes one big problem into smaller
sub-problems, runs a quantum routine on each, and aggregates the results.
This tutorial shows three distinct entry points:

1. **Graph-level partitioning** — split a graph by topology (METIS), solve
   MaxCut on each cluster, then aggregate via greedy/beam search.
2. **QUBO-level partitioning** — use a decomposer/composer from D-Wave
   Ocean's ``hybrid`` workflow library to split a random BQM and compare
   two quantum routines (QAOA vs PCE) on the same partitioned problem.
3. **Edge-based partitioning** — use ``MaxWeightMatchingProblem``'s edge
   budget plus a Kernighan-Lin partitioner to chunk a weighted graph into
   matching-sized sub-instances.

All three flows use the same ensemble class and aggregation API; the
difference is in the problem class and decomposition strategy.
"""

import random
import warnings
from functools import partial

import dimod
import hybrid
import networkx as nx
import numpy as np
from qiskit.circuit.library import RYGate, RZGate

from divi.qprog import EarlyStopping
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    GraphPartitioningConfig,
    MaxCutProblem,
    MaxWeightMatchingProblem,
    is_valid_matching,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tutorials._backend import get_backend


def _generate_random_graph(n_nodes: int, n_edges: int) -> nx.Graph:
    """Connected random graph with the requested number of nodes/edges."""
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        warnings.warn(
            f"Requested {n_edges} edges, but max for {n_nodes} nodes is "
            f"{max_edges}. Capping to max."
        )
        n_edges = max_edges

    graph = nx.random_labeled_tree(n_nodes)
    current_edges = set(graph.edges())

    while len(current_edges) < n_edges:
        u, v = random.sample(range(n_nodes), 2)
        edge = tuple(sorted((u, v)))
        if edge not in current_edges:
            graph.add_edge(*edge, weight=round(random.uniform(0.1, 1.0), 1))
            current_edges.add(edge)

    for u, v in graph.edges():
        if "weight" not in graph[u][v]:
            graph[u][v]["weight"] = round(random.uniform(0.1, 1.0), 1)

    return graph


def _cut_ratio(quantum_solution, classical_cut_size: int, graph: nx.Graph) -> float:
    cut_edges = sum(
        1
        for u, v in graph.edges()
        if (u in quantum_solution) != (v in quantum_solution)
    )
    return cut_edges / classical_cut_size


def _run_qubo_partitioning(
    bqm: dimod.BinaryQuadraticModel, engine: str, engine_kwargs: dict
) -> dict:
    """Run one quantum routine on a hybrid-decomposed BQM."""
    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=hybrid.EnergyImpactDecomposer(size=5),
        composer=hybrid.SplatComposer(),
    )

    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine=engine,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=30,
        early_stopping=EarlyStopping(patience=5),
        backend=get_backend(),
        **engine_kwargs,
    )

    ensemble.create_programs()
    ensemble.run().join()

    greedy_solution, greedy_energy = ensemble.aggregate_results(
        beam_width=1, n_partition_candidates=5
    )
    beam_solution, beam_energy = ensemble.aggregate_results(
        beam_width=3, n_partition_candidates=5
    )
    top_solutions = ensemble.get_top_solutions(
        n=5, beam_width=5, n_partition_candidates=5
    )

    return {
        "greedy_energy": greedy_energy,
        "greedy_solution": greedy_solution,
        "beam_energy": beam_energy,
        "beam_solution": beam_solution,
        "top_solutions": top_solutions,
        "total_circuits": ensemble.total_circuit_count,
    }


def _print_maxcut_results(
    total_circuits: int,
    greedy_sol,
    beam_sol,
    top_solutions,
    classical_cut_size: int,
    graph: nx.Graph,
) -> None:
    print(f"Total circuits: {total_circuits}")

    greedy_ratio = _cut_ratio(greedy_sol, classical_cut_size, graph)
    beam_ratio = _cut_ratio(beam_sol, classical_cut_size, graph)
    print(f"\n{'Method':<32} {'Cut/Classical':>14}")
    print("-" * 48)
    print(f"{'Greedy (beam_width=1, n=5)':<32} {greedy_ratio:>14.6f}")
    print(f"{'Beam (beam_width=3, n=5)':<32} {beam_ratio:>14.6f}")

    print("\nTop-5 solutions:")
    for i, sol in enumerate(top_solutions, 1):
        ratio = _cut_ratio(sol, classical_cut_size, graph)
        print(f"  {i}. cut/classical={ratio:.6f}  nodes={sol}")


def _print_qubo_comparison(
    classical_energy: float,
    classical_bitstring: str,
    results_by_engine: dict,
) -> None:
    rows = [
        ("Classical (SA)", classical_energy, classical_bitstring, "-"),
        (
            "QAOA Greedy (beam=1)",
            results_by_engine["qaoa"]["greedy_energy"],
            str(results_by_engine["qaoa"]["greedy_solution"]),
            str(results_by_engine["qaoa"]["total_circuits"]),
        ),
        (
            "QAOA Beam (beam=3)",
            results_by_engine["qaoa"]["beam_energy"],
            str(results_by_engine["qaoa"]["beam_solution"]),
            str(results_by_engine["qaoa"]["total_circuits"]),
        ),
        (
            "PCE Greedy (beam=1)",
            results_by_engine["pce"]["greedy_energy"],
            str(results_by_engine["pce"]["greedy_solution"]),
            str(results_by_engine["pce"]["total_circuits"]),
        ),
        (
            "PCE Beam (beam=3)",
            results_by_engine["pce"]["beam_energy"],
            str(results_by_engine["pce"]["beam_solution"]),
            str(results_by_engine["pce"]["total_circuits"]),
        ),
    ]

    print(f"\n{'Method':<22} {'Energy':>14} {'# Circuits':>12}")
    print("-" * 50)
    for method, energy, _, circuits in rows:
        print(f"{method:<22} {energy:>14.6f} {circuits:>12}")

    print("\nSolutions:")
    for method, _, sol, _ in rows:
        print(f"  {method:<22} {sol}")

    for engine_name, res in results_by_engine.items():
        print(f"\nTop-5 solutions ({engine_name.upper()}):")
        for rank, (sol, energy) in enumerate(res["top_solutions"], 1):
            print(f"  {rank}. energy={energy:>12.6f}  {sol}")


def _print_matching_summary(
    weight: float,
    classical_weight: float,
    matching,
    total_circuits: int,
) -> None:
    print(f"\n{'Metric':<28} {'Value':>10}")
    print("-" * 40)
    print(f"{'Quantum matching weight':<28} {weight:>10.4f}")
    print(f"{'Classical optimal weight':<28} {classical_weight:>10.4f}")
    print(f"{'Approximation ratio':<28} {weight / classical_weight:>9.2%}")
    print(f"{'Valid matching':<28} {str(is_valid_matching(matching)):>10}")
    print(f"{'Total circuits':<28} {total_circuits:>10}")


if __name__ == "__main__":
    # ── 1) Graph-level partitioning of MaxCut ─────────────────────────
    print("\n=== Part 1 — Graph partitioning (MaxCut) ===")

    N_NODES = 30
    N_EDGES = 40
    graph = _generate_random_graph(N_NODES, N_EDGES)

    maxcut_problem = MaxCutProblem(
        graph,
        config=GraphPartitioningConfig(
            max_n_nodes_per_cluster=10,
            partitioning_algorithm="metis",
        ),
    )

    maxcut_ensemble = PartitioningProgramEnsemble(
        problem=maxcut_problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=20,
        backend=get_backend(),
    )

    maxcut_ensemble.create_programs()
    maxcut_ensemble.run(blocking=True)

    classical_cut_size, _ = nx.approximation.one_exchange(graph, seed=1)

    greedy_sol = maxcut_ensemble.aggregate_results(
        beam_width=1, n_partition_candidates=5
    )
    beam_sol = maxcut_ensemble.aggregate_results(beam_width=3, n_partition_candidates=5)
    top_solutions = maxcut_ensemble.get_top_solutions(
        n=5, beam_width=5, n_partition_candidates=5
    )

    _print_maxcut_results(
        maxcut_ensemble.total_circuit_count,
        greedy_sol,
        beam_sol,
        top_solutions,
        classical_cut_size,
        graph,
    )

    # ── 2) QUBO-level partitioning: QAOA vs PCE ───────────────────────
    print("\n=== Part 2 — QUBO partitioning (QAOA vs PCE) ===")

    bqm: dimod.BinaryQuadraticModel = dimod.generators.gnp_random_bqm(
        25,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )

    results_by_engine = {
        "qaoa": _run_qubo_partitioning(bqm=bqm, engine="qaoa", engine_kwargs={}),
        "pce": _run_qubo_partitioning(
            bqm=bqm,
            engine="pce",
            engine_kwargs={
                "ansatz": GenericLayerAnsatz([RYGate, RZGate]),
                "encoding_type": "dense",
                "alpha": 2.0,
            },
        ),
    }

    classical_bitstring, classical_energy, _ = (
        dimod.SimulatedAnnealingSampler().sample(bqm).lowest().record[0]
    )

    _print_qubo_comparison(
        classical_energy, str(classical_bitstring), results_by_engine
    )

    # ── 3) Edge-based partitioning of MaxWeightMatching ───────────────
    print("\n=== Part 3 — Edge-based partitioning (MaxWeightMatching) ===")

    G_match = nx.gnm_random_graph(16, 30, seed=42)
    for u, v in G_match.edges():
        G_match[u][v]["weight"] = float(u + v + 1)

    matching_problem = MaxWeightMatchingProblem(
        G_match,
        penalty_scale=10.0,
        max_edges_per_partition=10,
        partition_algorithm="kernighan_lin",
        seed=42,
    )

    matching_ensemble = PartitioningProgramEnsemble(
        problem=matching_problem,
        n_layers=1,
        backend=get_backend(),
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
    )

    matching_ensemble.create_programs()
    print(f"Created {len(matching_ensemble.programs)} sub-programs")

    matching_ensemble.run(blocking=True)
    matching, weight = matching_ensemble.aggregate_results(beam_width=3)
    strict_top = matching_ensemble.get_top_solutions(
        n=5,
        beam_width=None,
        n_partition_candidates=5,
        strict=True,
    )

    classical_match = nx.max_weight_matching(G_match, maxcardinality=False)
    classical_match_weight = sum(G_match[u][v]["weight"] for u, v in classical_match)

    _print_matching_summary(
        weight, classical_match_weight, matching, matching_ensemble.total_circuit_count
    )
    print(f"{'Strict valid candidates':<28} {len(strict_top):>10}")
