# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Single-run error-mitigation comparison: ZNE vs QuEPP.

The script runs four VQE configurations with the same optimization budget
and fixed seeds:

  1. Noiseless baseline
  2. Noisy (depolarizing gate noise)
  3. ZNE-mitigated (global folding + Richardson extrapolation)
  4. QuEPP-mitigated (CPT + Pauli twirling + rescaling)

All values come from one deterministic run per method.
"""

import maestro
import numpy as np
import pennylane as qp
from rich.console import Console
from rich.table import Table

from divi.backends import MaestroConfig, MaestroSimulator
from divi.circuits.quepp import QuEPP
from divi.circuits.zne import ZNE, RichardsonExtrapolator
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem


def _print_mitigation_comparison(runs, reference_energy):
    print(f"Noiseless baseline energy: {reference_energy:.6f}")

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


if __name__ == "__main__":
    mol = qp.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
    )
    problem = MolecularProblem.from_molecule(mol)

    n_qubits = problem.hamiltonian.num_qubits
    noise_model = maestro.NoiseModel()
    noise_model.set_all_1q_gate_depolarizing(n_qubits, 0.005)
    noise_model.set_all_2q_gate_depolarizing(n_qubits, 0.01)

    optimizer_seed = 1997
    noiseless_simulator_seed = 4242
    noisy_simulator_seed = 4343
    zne_simulator_seed = 4444
    quepp_simulator_seed = 4545
    max_iterations = 20

    # --- Shared VQE settings across methods ---
    common = dict(
        problem=problem,
        n_layers=1,
        ansatz=HartreeFockAnsatz(),
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=max_iterations,
        seed=optimizer_seed,
    )

    def noisy_simulator(seed):
        return MaestroSimulator(
            maestro_config=MaestroConfig(
                seed=seed,
                simulation_type="DensityMatrix",
                noise_model=noise_model,
            )
        )

    vqe_noiseless = VQE(
        backend=MaestroSimulator(
            maestro_config=MaestroConfig(seed=noiseless_simulator_seed)
        ),
        **common,
    )
    vqe_noisy = VQE(backend=noisy_simulator(noisy_simulator_seed), **common)
    vqe_zne = VQE(
        backend=noisy_simulator(zne_simulator_seed),
        qem_protocol=ZNE(
            scale_factors=[1.0, 3.0, 5.0],
            extrapolator=RichardsonExtrapolator(),
        ),
        **common,
    )
    vqe_quepp = VQE(
        backend=noisy_simulator(quepp_simulator_seed),
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

    _print_mitigation_comparison(runs, vqe_noiseless.best_loss)
