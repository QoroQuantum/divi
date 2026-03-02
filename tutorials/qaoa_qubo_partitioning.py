# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dimod
import hybrid
import numpy as np
import pennylane as qml

from divi.qprog import EarlyStopping, QUBOPartitioningQAOA
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def _run_partitioning_engine(
    bqm: dimod.BinaryQuadraticModel, engine: str, engine_kwargs: dict
) -> dict:
    """Run one partitioning engine and return comparable metrics."""
    qubo_partition = QUBOPartitioningQAOA(
        qubo=bqm,
        decomposer=hybrid.EnergyImpactDecomposer(size=5),
        composer=hybrid.SplatComposer(),
        engine=engine,
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=30,
        early_stopping=EarlyStopping(patience=5),
        backend=get_backend(),
        **engine_kwargs,
    )

    qubo_partition.create_programs()
    qubo_partition.run().join()

    greedy_solution, greedy_energy = qubo_partition.aggregate_results(
        beam_width=1, n_partition_candidates=5
    )
    beam_solution, beam_energy = qubo_partition.aggregate_results(
        beam_width=3, n_partition_candidates=5
    )

    # Retrieve multiple ranked solutions via beam search
    top_solutions = qubo_partition.get_top_solutions(
        n=5, beam_width=5, n_partition_candidates=5
    )

    return {
        "greedy_energy": greedy_energy,
        "greedy_solution": greedy_solution,
        "beam_energy": beam_energy,
        "beam_solution": beam_solution,
        "top_solutions": top_solutions,
        "total_circuits": qubo_partition.total_circuit_count,
    }


if __name__ == "__main__":
    bqm: dimod.BinaryQuadraticModel = dimod.generators.gnp_random_bqm(
        25,
        p=0.5,
        vartype="BINARY",
        random_state=1997,
        bias_generator=partial(np.random.default_rng().uniform, -5, 5),
    )

    results_by_engine = {
        "qaoa": _run_partitioning_engine(bqm=bqm, engine="qaoa", engine_kwargs={}),
        "pce": _run_partitioning_engine(
            bqm=bqm,
            engine="pce",
            engine_kwargs={
                "ansatz": GenericLayerAnsatz([qml.RY, qml.RZ]),
                "encoding_type": "dense",
                "alpha": 2.0,
            },
        ),
    }

    best_classical_bitstring, best_classical_energy, _ = (
        dimod.SimulatedAnnealingSampler().sample(bqm).lowest().record[0]
    )

    # --- Print side-by-side comparison table ---
    rows = [
        ("Classical (SA)", best_classical_energy, str(best_classical_bitstring), "-"),
        (
            "QAOA Greedy (beam=1)",
            results_by_engine["qaoa"]["greedy_energy"],
            str(results_by_engine["qaoa"]["greedy_solution"]),
            results_by_engine["qaoa"]["total_circuits"],
        ),
        (
            "QAOA Beam (beam=3)",
            results_by_engine["qaoa"]["beam_energy"],
            str(results_by_engine["qaoa"]["beam_solution"]),
            results_by_engine["qaoa"]["total_circuits"],
        ),
        (
            "PCE Greedy (beam=1)",
            results_by_engine["pce"]["greedy_energy"],
            str(results_by_engine["pce"]["greedy_solution"]),
            results_by_engine["pce"]["total_circuits"],
        ),
        (
            "PCE Beam (beam=3)",
            results_by_engine["pce"]["beam_energy"],
            str(results_by_engine["pce"]["beam_solution"]),
            results_by_engine["pce"]["total_circuits"],
        ),
    ]
    pad = "  "
    col_m, col_e, col_c = 22, 16, 10
    sep_len = col_m + col_e + col_c + len(pad) * 2
    print("\n" + "-" * sep_len)
    print(f"{'Method':<{col_m}}{pad}{'Energy':>{col_e}}{pad}{'# Circuits':>{col_c}}")
    print("-" * sep_len)
    for method, energy, sol, circuits in rows:
        print(
            f"{method:<{col_m}}{pad}{energy:>{col_e}.6f}{pad}{str(circuits):>{col_c}}"
        )
        print(f"  Solution: {sol}")
    print("-" * sep_len)

    # --- Print top-N solutions per engine ---
    for engine_name, res in results_by_engine.items():
        print(f"\nTop-5 solutions ({engine_name.upper()}):")
        print(f"  {'Rank':<6}{pad}{'Energy':>12}{pad}Solution")
        print(f"  {'-' * 50}")
        for rank, (sol, energy) in enumerate(res["top_solutions"], 1):
            print(f"  {rank:<6}{pad}{energy:>12.6f}{pad}{sol}")
