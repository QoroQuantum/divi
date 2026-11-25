# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QSCI on LiH: sample a seed subspace, then let the Hamiltonian grow it.

:class:`~divi.qprog.ensembles.qsci.QSCI` evolves the Hartree-Fock state under
qDRIFT, keeps the most frequently measured bitstrings as a seed subspace, then
grows it by second-order (PT2) contribution and diagonalises the Hamiltonian in
it classically. LiH in STO-3G (12 qubits), against exact FCI:

1. Each record's subspace size, error and PT2 correction, the first being the
   sampled seed alone.
2. The final energy with its PT2 correction.
"""

import math

from pyscf import fci, gto, scf

from divi.hamiltonians.trotterization import QDrift
from divi.qprog.algorithms import TimeEvolution
from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.qsci import QSCI
from divi.qprog.initial_states import CustomPerQubitState
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

if __name__ == "__main__":
    mean_field = scf.RHF(
        gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    ).run()
    exact = fci.FCI(mean_field).kernel()[0]
    problem = MolecularProblem.from_molecule(mean_field)
    hamiltonian = problem.hamiltonian
    # Qubit 2p is spatial orbital p's alpha spin-orbital and 2p + 1 its beta one.
    hartree_fock = "1" * problem.n_electrons + "0" * (
        hamiltonian.num_qubits - problem.n_electrons
    )

    def is_in_sector(config: str) -> bool:
        return (
            config[0::2].count("1") == problem.n_alpha
            and config[1::2].count("1") == problem.n_beta
        )

    def build_programs(backend):
        return {
            f"t={k}": TimeEvolution(
                hamiltonian=hamiltonian,
                trotterization_strategy=QDrift(
                    sampling_budget=60,
                    sampling_strategy="weighted",
                    n_hamiltonians_per_iteration=10,
                    seed=k,
                ),
                time=2 * math.pi / 5 * k,
                initial_state=CustomPerQubitState(hartree_fock),
                backend=backend,
            )
            for k in (1, 2, 3)
        }

    qsci = QSCI(
        hamiltonian,
        build_programs,
        backend=get_backend(shots=2000),
        config_filter=is_in_sector,
        max_seed_configs=50,
        expansion_width=20,
        reporting_level=ReportingLevel.OFF,
    )
    state = qsci.run().aggregate_results()

    print("Part 1: expansion records")
    print(f"{'Step':>4} {'Subspace':>9} {'Error (mHa)':>12} {'PT2 (mHa)':>11}")
    for step, record in enumerate(state.records):
        print(
            f"{step:>4} {record.subspace_size:>9} "
            f"{1e3 * (record.energy - exact):>12.4f} "
            f"{1e3 * record.pt2_correction:>11.4f}"
        )

    print("\nPart 2: final energy with its PT2 correction")
    print(f"  {1e3 * (state.pt2_energy - exact):+.4f} mHa")
