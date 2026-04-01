# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Error mitigation comparison: ZNE vs QuEPP.

Runs the same VQE problem under four configurations and prints a
comparison table:

  1. Exact (statevector, noiseless)
  2. Noisy (shot-based with FakeTorino noise)
  3. ZNE-mitigated (noise folding + Richardson extrapolation)
  4. QuEPP-mitigated (CPT + Pauli twirling + rescaling)
"""

from functools import partial

import numpy as np
import pennylane as qml
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random
from qiskit_ibm_runtime.fake_provider import FakeTorino
from rich.console import Console
from rich.table import Table

from divi.backends import QiskitSimulator
from divi.circuits.qem import ZNE
from divi.circuits.quepp import QuEPP
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    noisy_backend = FakeTorino()

    common = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=10,
        seed=1997,
    )

    # --- 1. Exact (statevector) ---
    vqe_exact = VQE(backend=QiskitSimulator(), **common)
    vqe_exact.run()

    # --- 2. Noisy (FakeTorino — IBM Heron-class noise) ---
    common["optimizer"].reset()
    vqe_noisy = VQE(
        backend=QiskitSimulator(n_processes=8, qiskit_backend=noisy_backend), **common
    )
    vqe_noisy.run()

    # --- 3. ZNE-mitigated ---
    common["optimizer"].reset()
    scale_factors = [1.0, 3.0, 5.0]
    vqe_zne = VQE(
        backend=QiskitSimulator(n_processes=8, qiskit_backend=noisy_backend),
        qem_protocol=ZNE(
            scale_factors,
            partial(fold_gates_at_random),
            RichardsonFactory(scale_factors=scale_factors),
        ),
        **common,
    )
    vqe_zne.run()

    # --- 4. QuEPP-mitigated (with Pauli twirling) ---
    common["optimizer"].reset()
    vqe_quepp = VQE(
        backend=QiskitSimulator(n_processes=8, qiskit_backend=noisy_backend),
        qem_protocol=QuEPP(truncation_order=1, n_twirls=10),
        **common,
    )

    # Dry run shows the per-stage circuit fan-out including QEM + twirling
    vqe_quepp.dry_run()

    vqe_quepp.run()

    # --- Print comparison table ---
    table = Table(title="Error Mitigation Comparison")
    table.add_column("Method", style="bold")
    table.add_column("Best Energy", justify="right")
    table.add_column("# Circuits", justify="right")

    for name, loss, count in [
        ("Exact (statevector)", vqe_exact.best_loss, vqe_exact.total_circuit_count),
        ("Noisy", vqe_noisy.best_loss, vqe_noisy.total_circuit_count),
        ("ZNE-Mitigated", vqe_zne.best_loss, vqe_zne.total_circuit_count),
        ("QuEPP-Mitigated", vqe_quepp.best_loss, vqe_quepp.total_circuit_count),
    ]:
        table.add_row(name, f"{loss:.4f}", str(count))

    Console().print(table)
