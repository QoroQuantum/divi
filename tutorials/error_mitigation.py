# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Error mitigation comparison: ZNE vs QuEPP.

First optimizes H₂ ground-state energy on a noiseless backend, then
evaluates the optimal parameters under four configurations to isolate
mitigation quality from optimizer convergence:

  1. Exact (statevector, noiseless)
  2. Noisy (shot-based with depolarizing noise)
  3. ZNE-mitigated (noise folding + Richardson extrapolation)
  4. QuEPP-mitigated (CPT + Pauli twirling + rescaling)
"""

from functools import partial

import numpy as np
import pennylane as qml
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random
from qiskit_aer.noise import NoiseModel, depolarizing_error
from rich.console import Console
from rich.table import Table

from divi.backends import QiskitSimulator
from divi.circuits.qem import ZNE
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

    # --- 1. Optimize on the noiseless backend to find ground-state params ---
    vqe_exact = VQE(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=20,
        seed=1997,
        backend=QiskitSimulator(),
    )
    vqe_exact.run()
    optimal_params = vqe_exact.best_params.reshape(1, -1)
    exact_energy = vqe_exact.best_loss
    print(f"Exact ground-state energy: {exact_energy:.6f}")

    # --- Evaluate at the same optimal params under different noise/mitigation ---
    # max_iterations=1 so the optimizer returns after a single cost evaluation
    # at the supplied initial_params (no further optimization steps).
    common = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=1,
        initial_params=optimal_params,
    )

    # --- 2. Noisy (0.5% depolarizing) ---
    vqe_noisy = VQE(
        backend=QiskitSimulator(n_processes=8, noise_model=noise_model), **common
    )
    vqe_noisy.run()

    # --- 3. ZNE-mitigated ---
    scale_factors = [1.0, 3.0, 5.0]
    vqe_zne = VQE(
        backend=QiskitSimulator(n_processes=8, noise_model=noise_model),
        qem_protocol=ZNE(
            scale_factors,
            partial(fold_gates_at_random),
            RichardsonFactory(scale_factors=scale_factors),
        ),
        **common,
    )
    vqe_zne.run()

    # --- 4. QuEPP-mitigated (with Pauli twirling) ---
    vqe_quepp = VQE(
        backend=QiskitSimulator(n_processes=8, noise_model=noise_model),
        qem_protocol=QuEPP(truncation_order=1, n_twirls=3),
        **common,
    )

    # Dry run shows the per-stage circuit fan-out including QEM + twirling
    vqe_quepp.dry_run()

    vqe_quepp.run()

    # --- Print comparison table ---
    table = Table(title="Error Mitigation Comparison (H₂ ground state)")
    table.add_column("Method", style="bold")
    table.add_column("Energy", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("# Circuits", justify="right")

    for name, vqe in [
        ("Exact (statevector)", vqe_exact),
        ("Noisy (no mitigation)", vqe_noisy),
        ("ZNE-Mitigated", vqe_zne),
        ("QuEPP-Mitigated", vqe_quepp),
    ]:
        energy = vqe.best_loss
        error = abs(energy - exact_energy)
        table.add_row(
            name,
            f"{energy:.6f}",
            f"{error:.6f}",
            str(vqe.total_circuit_count),
        )

    Console().print(table)
