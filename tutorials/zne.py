# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import pennylane as qml
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random

from divi.backends import QiskitSimulator
from divi.circuits.qem import ZNE
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import PymooMethod, PymooOptimizer

if __name__ == "__main__":
    mol = qml.qchem.Molecule(
        symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
    )

    common = dict(
        molecule=mol,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        max_iterations=10,
        seed=1997,
    )

    # --- Exact (statevector) ---
    vqe_exact = VQE(
        backend=QiskitSimulator(n_processes=4),
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=10),
        **common,
    )
    vqe_exact.run()

    # --- Noisy (shot-based with noise model) ---
    vqe_noisy = VQE(
        backend=QiskitSimulator(n_processes=4, qiskit_backend="auto"),
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=10),
        **common,
    )
    vqe_noisy.run()

    # --- ZNE-mitigated (shot-based + zero-noise extrapolation) ---
    scale_factors = [1.0, 3.0, 5.0]
    vqe_zne = VQE(
        backend=QiskitSimulator(n_processes=4, qiskit_backend="auto"),
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=10),
        qem_protocol=ZNE(
            scale_factors,
            partial(fold_gates_at_random),
            RichardsonFactory(scale_factors=scale_factors),
        ),
        **common,
    )
    vqe_zne.run()

    # --- Print comparison table ---
    rows = [
        ("Exact (statevector)", vqe_exact.best_loss, vqe_exact.total_circuit_count),
        ("Noisy", vqe_noisy.best_loss, vqe_noisy.total_circuit_count),
        ("ZNE-Mitigated", vqe_zne.best_loss, vqe_zne.total_circuit_count),
    ]
    pad = "  "
    col_m, col_e, col_c = 22, 14, 12
    sep_len = col_m + col_e + col_c + len(pad) * 2
    print("\n" + "-" * sep_len)
    print(
        f"{'Method':<{col_m}}{pad}{'Best Energy':>{col_e}}{pad}{'# Circuits':>{col_c}}"
    )
    print("-" * sep_len)
    for method, energy, circuits in rows:
        print(f"{method:<{col_m}}{pad}{energy:>{col_e}.4f}{pad}{circuits:>{col_c}}")
    print("-" * sep_len)
    print(
        "\nNote: ZNE uses additional circuit executions (one per scale factor) "
        "to extrapolate towards the zero-noise limit, trading circuit count "
        "for improved energy estimates on noisy hardware."
    )
