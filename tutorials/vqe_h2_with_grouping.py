# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml

from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    # Create backend with deterministic execution enabled for debugging
    backend = get_backend(
        simulation_seed=1997, shots=500, _deterministic_execution=True
    )

    vqe_input = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=1,
        seed=2000,
        backend=backend,
    )

    vqe_problem_no_grouping = VQE(
        **vqe_input,
        grouping_strategy=None,
    )
    vqe_problem_wire_grouping = VQE(
        **vqe_input,
        grouping_strategy="wires",
    )
    vqe_problem_qwc_grouping = VQE(
        **vqe_input,
        grouping_strategy="qwc",
    )

    vqe_problem_no_grouping.run()
    vqe_problem_wire_grouping.run()
    vqe_problem_qwc_grouping.run()

    no_grouping_measurement_groups = vqe_problem_no_grouping.meta_circuit_factories[
        "cost_circuit"
    ].measurement_groups
    wire_grouping_measurement_groups = vqe_problem_wire_grouping.meta_circuit_factories[
        "cost_circuit"
    ].measurement_groups
    qwc_grouping_measurement_groups = vqe_problem_qwc_grouping.meta_circuit_factories[
        "cost_circuit"
    ].measurement_groups

    strategies = [
        ("None", no_grouping_measurement_groups, vqe_problem_no_grouping),
        ("Wires", wire_grouping_measurement_groups, vqe_problem_wire_grouping),
        ("QWC", qwc_grouping_measurement_groups, vqe_problem_qwc_grouping),
    ]

    header = f"{'Strategy':<10} {'Groups':>8} {'Loss':>14} {'Circuits':>10}"
    print(header)
    print("-" * len(header))
    for name, groups, problem in strategies:
        print(
            f"{name:<10} {len(groups):>8} {problem.best_loss:>14.6f} {problem.total_circuit_count:>10}"
        )

    losses = [p.best_loss for _, _, p in strategies]
    max_diff = max(losses) - min(losses)
    print(f"\nMax loss difference: {max_diff:.6f}")
    assert max_diff < 0.5, f"Losses diverged too much: {losses}"
