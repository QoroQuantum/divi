# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Single-run error-mitigation comparison: ZNE vs QuEPP.

The script runs four VQE configurations with the same optimization budget
and fixed seeds:

  1. Noiseless baseline
  2. Noisy (shot-based with depolarizing noise)
  3. ZNE-mitigated (global folding + Richardson extrapolation)
  4. QuEPP-mitigated (CPT + Pauli twirling + rescaling)

All values come from one deterministic run per method.
"""

import numpy as np
import pennylane as qml
from qiskit_aer.noise import NoiseModel, depolarizing_error
from rich.console import Console
from rich.table import Table

from divi.backends import QiskitSimulator
from divi.circuits.qem import ZNE, RichardsonExtrapolator
from divi.circuits.quepp import QuEPP
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
    )

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.005, 1), ["sx", "x", "rz"]
    )
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "ecr"])

    optimizer_seed = 1997
    noiseless_simulator_seed = 4242
    noisy_simulator_seed = 4343
    zne_simulator_seed = 4444
    quepp_simulator_seed = 4545
    max_iterations = 20

    # --- Shared VQE settings across methods ---
    common = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=max_iterations,
        seed=optimizer_seed,
    )

    vqe_noiseless = VQE(
        backend=QiskitSimulator(
            n_processes=1,
            simulation_seed=noiseless_simulator_seed,
        ),
        **common,
    )
    vqe_noisy = VQE(
        backend=QiskitSimulator(
            n_processes=1,
            simulation_seed=noisy_simulator_seed,
            noise_model=noise_model,
        ),
        **common,
    )
    vqe_zne = VQE(
        backend=QiskitSimulator(
            n_processes=1,
            simulation_seed=zne_simulator_seed,
            noise_model=noise_model,
        ),
        qem_protocol=ZNE(
            scale_factors=[1.0, 3.0, 5.0],
            extrapolator=RichardsonExtrapolator(),
        ),
        **common,
    )
    vqe_quepp = VQE(
        backend=QiskitSimulator(
            n_processes=1,
            simulation_seed=quepp_simulator_seed,
            noise_model=noise_model,
        ),
        qem_protocol=QuEPP(truncation_order=1, n_twirls=3),
        **common,
    )

    runs = [
        ("Noiseless baseline", vqe_noiseless),
        ("Noisy (no mitigation)", vqe_noisy),
        ("ZNE-mitigated", vqe_zne),
        ("QuEPP-mitigated", vqe_quepp),
    ]
    for _, vqe in runs:
        vqe.run()

    reference_energy = vqe_noiseless.best_loss
    print(f"Noiseless baseline energy: {reference_energy:.6f}")

    # --- Print comparison table ---
    table = Table(title="Error Mitigation Comparison (single deterministic runs)")
    table.add_column("Method", style="bold")
    table.add_column("Energy", justify="right")
    table.add_column("|Error vs baseline|", justify="right")
    table.add_column("# Circuits", justify="right")

    for name, vqe in runs:
        energy = vqe.best_loss
        error = abs(energy - reference_energy)
        table.add_row(
            name,
            f"{energy:.6f}",
            f"{error:.6f}",
            str(vqe.total_circuit_count),
        )

    Console().print(table)
