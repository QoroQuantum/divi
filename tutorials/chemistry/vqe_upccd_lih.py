# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Paired-doubles UpCCD against UCCSD on LiH.

LiH in STO-3G at 1.6 Å and 3.0 Å, with its four electrons in the four σ
orbitals (8 qubits).

1. Parameter count, energy error against CASCI and fidelity with the exact
   ground state for :class:`~divi.qprog.algorithms.UCCSDAnsatz` and
   :class:`~divi.qprog.algorithms.UpCCDAnsatz`, beside the seniority-zero
   floor: the lowest energy of any state whose orbitals are all doubly occupied
   or empty.
2. UpCCD started from CCSD pair amplitudes through
   :meth:`~divi.qprog.algorithms.UpCCDAnsatz.params_from_amplitudes`, against
   a start at Hartree-Fock.
"""

import numpy as np
from pyscf import cc, gto, mcscf, scf

from divi.backends import CircuitRunner
from divi.qprog.algorithms import (
    VQE,
    Ansatz,
    ElectronCounts,
    UCCSDAnsatz,
    UpCCDAnsatz,
)
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

N_ORBITALS, N_ELECTRONS = 4, 4
N_QUBITS = 2 * N_ORBITALS
REFERENCE = ElectronCounts(N_ELECTRONS)
SIGMA_ORBITALS = [0, 1, 2, 5]
PI_ORBITALS = [3, 4]


def build_sector_masks() -> tuple[np.ndarray, np.ndarray]:
    """Basis states with two electrons of each spin, and those of them whose
    orbitals are all doubly occupied or empty; qubit 0 is the least significant
    bit, alpha on even qubits."""
    basis = np.arange(2**N_QUBITS)
    occupied = np.array([(basis >> qubit) & 1 for qubit in range(N_QUBITS)])
    alpha, beta = occupied[0::2], occupied[1::2]
    right_counts = (alpha.sum(axis=0) == 2) & (beta.sum(axis=0) == 2)
    return right_counts, right_counts & np.all(alpha == beta, axis=0)


def find_lowest_eigenpair(
    problem: MolecularProblem, mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """The lowest eigenpair of the problem's Hamiltonian among the basis states
    in ``mask``."""
    matrix = problem.hamiltonian.to_matrix(sparse=True)
    states = np.flatnonzero(mask)
    values, vectors = np.linalg.eigh(matrix[states][:, states].toarray().real)
    ground = np.zeros(matrix.shape[0])
    ground[states] = vectors[:, 0]
    return float(values[0]), ground


def make_vqe(problem: MolecularProblem, ansatz: Ansatz, backend: CircuitRunner) -> VQE:
    return VQE(
        problem,
        ansatz=ansatz,
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        max_iterations=200,
        backend=backend,
    )


if __name__ == "__main__":
    backend = get_backend()
    right_counts, paired = build_sector_masks()
    systems = []

    print("Part 1: UCCSD against UpCCD, LiH/STO-3G σ CAS(4,4), 8 qubits")
    for bond_length in (1.6, 3.0):
        mean_field = scf.RHF(
            gto.M(atom=f"Li 0 0 0; H 0 0 {bond_length}", basis="sto-3g", verbose=0)
        ).run()
        casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
        orbitals = casci.sort_mo([index + 1 for index in SIGMA_ORBITALS])
        problem = MolecularProblem.from_casci(
            mean_field, N_ORBITALS, N_ELECTRONS, mo_coeff=orbitals
        )
        exact_energy, exact_state = find_lowest_eigenpair(problem, right_counts)
        floor, _ = find_lowest_eigenpair(problem, paired)
        systems.append((bond_length, mean_field, problem, exact_energy))

        print(
            f"\n  R = {bond_length} Å, seniority-zero floor "
            f"{1e3 * (floor - exact_energy):.3f} mHa above CASCI"
        )
        print(f"  {'ansatz':<8} {'params':>6} {'error':>12} {'fidelity':>10}")
        for name, ansatz in {"UCCSD": UCCSDAnsatz(), "UpCCD": UpCCDAnsatz()}.items():
            vqe = make_vqe(problem, ansatz, backend)
            vqe.run(perform_final_computation=False)
            state = backend.statevector(
                ansatz.build(vqe.best_params, N_QUBITS, vqe.n_layers, REFERENCE)
            )
            fidelity = abs(np.vdot(exact_state, state)) ** 2
            print(
                f"  {name:<8} {vqe.n_params:>6} "
                f"{1e3 * (vqe.best_loss - exact_energy):>8.3f} mHa {fidelity:>10.6f}"
            )

    print("\nPart 2: UpCCD from CCSD pair amplitudes")
    for bond_length, mean_field, problem, exact_energy in systems:
        coupled_cluster = cc.CCSD(mean_field, frozen=PI_ORBITALS).run(verbose=0)
        t1, t2 = coupled_cluster.t1, coupled_cluster.t2
        same_spin = t2 - t2.transpose(0, 1, 3, 2)
        seed = UpCCDAnsatz().params_from_amplitudes(
            (t1, t1), (same_spin, t2, same_spin), N_QUBITS, REFERENCE
        )
        for label, initial_params in (("Hartree-Fock", None), ("CCSD seed", seed)):
            vqe = make_vqe(problem, UpCCDAnsatz(), backend)
            vqe.run(initial_params=initial_params, perform_final_computation=False)
            print(
                f"  R = {bond_length} Å from {label:<12}: "
                f"{1e3 * (vqe.best_loss - exact_energy):.3f} mHa after "
                f"{len(vqe.losses_history)} iterations"
            )
