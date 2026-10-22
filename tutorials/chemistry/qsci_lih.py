# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QSCI on LiH: sample a seed subspace, then let the Hamiltonian grow it.

:class:`~divi.qprog.QSCI` (quantum-selected configuration interaction) uses the
quantum device only to *choose determinants*. It evolves the Hartree-Fock state
under the Hamiltonian, measures it, and keeps the most frequently observed
bitstrings as a seed subspace. Everything after that is classical: the
Hamiltonian is projected onto the subspace and diagonalised exactly, then the
subspace is grown by the determinants the Hamiltonian connects to it most
strongly (ranked by their Epstein-Nesbet second-order contribution), and the
projection is diagonalised again.

Because the energy comes from a classical diagonalisation, it is a variational
upper bound, and the register width is not limited by a statevector: the same
workflow accepts any Hermitian ``SparsePauliOp``, however wide.

This tutorial runs LiH in the STO-3G basis (12 qubits) against exact FCI and
reads the result off ``state.records``, one entry per expansion step:

* the first record is the **sampled seed alone**, which is plain QSCI;
* each later record is one expansion step, and ``pt2_correction`` estimates what
  the subspace still misses;
* ``extrapolate_pt2_to_zero`` fits the energy against that correction to
  estimate the limit.
"""

import math
import time

from pyscf import fci, gto, scf

from divi.hamiltonians import QDrift
from divi.qprog import (
    QSCI,
    CustomPerQubitState,
    ReportingLevel,
    TimeEvolution,
    extrapolate_pt2_to_zero,
)
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

MILLIHARTREE = 1e3


def main() -> None:
    lih = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(lih).run(verbose=0)
    exact_energy = fci.FCI(mean_field).kernel()[0]

    problem = MolecularProblem.from_molecule(mean_field)
    hamiltonian = problem.hamiltonian
    n_qubits = hamiltonian.num_qubits
    # Qubit 2p is the alpha spin-orbital of spatial orbital p, qubit 2p + 1 its
    # beta partner, so Hartree-Fock fills the first n_electrons qubits.
    hartree_fock = "1" * problem.n_electrons + "0" * (n_qubits - problem.n_electrons)

    def in_sector(config: str) -> bool:
        """Keep sampled bitstrings with the right alpha and beta electron counts."""
        return (
            config[0::2].count("1") == problem.n_alpha
            and config[1::2].count("1") == problem.n_beta
        )

    def sampling_programs(backend):
        """One qDRIFT evolution of Hartree-Fock per time point, each with its
        own seed so the points draw independent circuits."""
        times = [2 * math.pi / 5 * k for k in (1, 2, 3)]
        return {
            f"t={t:.3f}": TimeEvolution(
                hamiltonian=hamiltonian,
                trotterization_strategy=QDrift(
                    sampling_budget=60,
                    sampling_strategy="weighted",
                    n_hamiltonians_per_iteration=10,
                    seed=seed,
                ),
                time=t,
                initial_state=CustomPerQubitState(hartree_fock),
                backend=backend,
            )
            for seed, t in enumerate(times)
        }

    print(f"LiH / STO-3G: {n_qubits} qubits, {hamiltonian.size} Pauli terms")
    print(f"RHF energy:  {mean_field.e_tot:.6f} Ha")
    print(f"FCI energy:  {exact_energy:.6f} Ha\n")

    qsci = QSCI(
        hamiltonian,
        sampling_programs,
        backend=get_backend(shots=2000),
        config_filter=in_sector,
        max_seed_configs=50,
        n_expansions=10,
        expansion_width=20,
        reporting_level=ReportingLevel.OFF,
    )
    start = time.time()
    state = qsci.run().aggregate_results()
    elapsed = time.time() - start

    print(f"{'step':>4} {'subspace':>9} {'error (mHa)':>12} {'PT2 (mHa)':>11}")
    for step, record in enumerate(state.records):
        error = (record.energy - exact_energy) * MILLIHARTREE
        pt2 = record.pt2_correction * MILLIHARTREE
        print(f"{step:>4} {record.subspace_size:>9} {error:>12.4f} {pt2:>11.4f}")

    seed_error = (state.records[0].energy - exact_energy) * MILLIHARTREE
    final_error = (state.energy - exact_energy) * MILLIHARTREE
    print(
        f"\nSeed subspace alone ({state.records[0].subspace_size} determinants) "
        f"sits {seed_error:.3f} mHa above FCI."
    )
    print(
        f"After {len(state.records) - 1} expansion steps "
        f"({len(state.configs)} determinants) it sits {final_error:.4f} mHa above."
    )
    print(
        f"With the PT2 correction: "
        f"{(state.pt2_energy - exact_energy) * MILLIHARTREE:+.4f} mHa "
        "(not variational, so it can land below FCI)."
    )
    if len(state.records) > 2:
        extrapolated, _, r_squared = extrapolate_pt2_to_zero(state.records)
        print(
            f"Extrapolated to zero PT2: "
            f"{(extrapolated - exact_energy) * MILLIHARTREE:+.4f} mHa "
            f"(fit r^2 = {r_squared:.3f})."
        )
    print(f"\nFull run, sampling included: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
