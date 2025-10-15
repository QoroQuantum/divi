# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    # Create backend with deterministic execution enabled for debugging
    backend = ParallelSimulator(
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
    no_grouping_measurement_groups = vqe_problem_no_grouping.meta_circuits[
        "cost_circuit"
    ].measurement_groups

    vqe_problem_wire_grouping = VQE(
        **vqe_input,
        grouping_strategy="wires",
    )
    wire_grouping_measurement_groups = vqe_problem_wire_grouping.meta_circuits[
        "cost_circuit"
    ].measurement_groups

    vqe_problem_qwc_grouping = VQE(
        **vqe_input,
        grouping_strategy="qwc",
    )
    qwc_grouping_measurement_groups = vqe_problem_qwc_grouping.meta_circuits[
        "cost_circuit"
    ].measurement_groups

    print(
        f"Number of measurement groups without grouping: {len(no_grouping_measurement_groups)}"
    )
    print(
        f"Number of measurement groups with wires grouping: {len(wire_grouping_measurement_groups)}"
    )
    print(
        f"Number of measurement groups with qwc grouping: {len(qwc_grouping_measurement_groups)}"
    )

    print("-" * 20)
    vqe_problem_no_grouping.run()
    vqe_problem_wire_grouping.run()
    vqe_problem_qwc_grouping.run()
    print("-" * 20)

    no_grouping_loss = vqe_problem_no_grouping.losses[-1][0].item()
    wire_grouping_loss = vqe_problem_wire_grouping.losses[-1][0].item()
    qwc_grouping_loss = vqe_problem_qwc_grouping.losses[-1][0].item()

    print(f"Final Loss (no grouping): {no_grouping_loss}")
    print(f"Final Loss (wires grouping): {wire_grouping_loss}")
    print(f"Final Loss (qwc grouping): {qwc_grouping_loss}")
    all_equal = no_grouping_loss == wire_grouping_loss == qwc_grouping_loss
    print(f"All losses equal? {'Yes' if all_equal else 'No'}")
    assert all_equal

    print("-" * 20)

    print(f"Circuits Run (no grouping): {vqe_problem_no_grouping.total_circuit_count}")
    print(
        f"Circuits Run (wires grouping): {vqe_problem_wire_grouping.total_circuit_count}"
    )
    print(
        f"Circuits Run (qwc grouping): {vqe_problem_qwc_grouping.total_circuit_count}"
    )
