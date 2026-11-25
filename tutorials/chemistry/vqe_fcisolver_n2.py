# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""VQE as PySCF's active-space solver, on N2 dissociation.

:class:`~divi.qprog.algorithms.VQEFCISolver` replaces PySCF's exact CI solver:
assigned to ``casci.fcisolver``, it solves each active space with a VQE. The
active space is N2's six electrons in six orbitals (12 qubits).

1. Dissociation curve from 1.0 to 2.5 Å: Hartree-Fock, exact CASCI and VQE with
   :class:`~divi.qprog.algorithms.UCCSDAnsatz`, against chemical accuracy
   (1.6 mHa), and whether RHF-based CCSD converges at 2.5 Å.
2. Natural occupations from the solver's ``make_rdm1`` beside the exact ones,
   near equilibrium and stretched.
3. One against two UCCSD layers at 2.0 Å, each from Hartree-Fock with the same
   iteration budget.
"""

import numpy as np
from pyscf import cc, gto, mcscf, scf

from divi.backends import CircuitRunner
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

CHEMICAL_ACCURACY = 1.6e-3  # Hartree
N_ORBITALS, N_ELECTRONS = 6, 6
BOND_LENGTHS = (1.0, 1.1, 1.2, 1.5, 2.0, 2.5)


def run_mean_field(bond_length: float) -> scf.hf.RHF:
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {bond_length}", basis="sto-3g", verbose=0)
    return scf.RHF(mol).run()


def make_solver(
    backend: CircuitRunner, n_layers: int = 1, max_iterations: int = 200
) -> VQEFCISolver:
    return VQEFCISolver(
        backend=backend,
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=max_iterations,
        n_layers=n_layers,
    )


def run_casci(mean_field: scf.hf.RHF, solver=None) -> mcscf.casci.CASCI:
    casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
    if solver is not None:
        casci.fcisolver = solver
    casci.kernel()
    return casci


def compute_natural_occupations(casci: mcscf.casci.CASCI) -> np.ndarray:
    density = casci.fcisolver.make_rdm1(casci.ci, casci.ncas, casci.nelecas)
    return np.sort(np.linalg.eigvalsh(density))[::-1]


if __name__ == "__main__":
    backend = get_backend()

    print("Part 1: N2 dissociation, CAS(6,6), 12 qubits")
    header = f"{'R (Å)':>6} {'HF':>12} {'CASCI':>12} {'VQE':>12} {'VQE - CASCI':>14}"
    print(header)
    print("-" * len(header))

    solver = make_solver(backend)
    results = {}
    for bond_length in BOND_LENGTHS:
        mean_field = run_mean_field(bond_length)
        exact = run_casci(mean_field)
        vqe = run_casci(mean_field, solver)
        error = vqe.e_tot - exact.e_tot
        flag = (
            ""
            if abs(error) < CHEMICAL_ACCURACY
            else f"  above {1e3 * CHEMICAL_ACCURACY:.1f} mHa"
        )
        print(
            f"{bond_length:>6.1f} {mean_field.e_tot:>12.6f} {exact.e_tot:>12.6f} "
            f"{vqe.e_tot:>12.6f} {1e3 * error:>10.2f} mHa{flag}"
        )
        results[bond_length] = (mean_field, exact, vqe)

    coupled_cluster = cc.CCSD(results[2.5][0]).run(verbose=0)
    print(f"\nRHF-based CCSD at 2.5 Å converged: {coupled_cluster.converged}")

    print("\nPart 2: natural occupations")
    for bond_length in (1.1, 2.5):
        _, exact, vqe = results[bond_length]
        for label, casci in (("VQE  ", vqe), ("exact", exact)):
            values = " ".join(f"{n:.3f}" for n in compute_natural_occupations(casci))
            print(f"  R = {bond_length} Å, {label}: {values}")

    print("\nPart 3: one against two UCCSD layers at 2.0 Å, 30 iterations each")
    mean_field, exact, _ = results[2.0]
    for n_layers in (1, 2):
        casci = run_casci(mean_field, make_solver(backend, n_layers, max_iterations=30))
        print(
            f"  {n_layers} layer(s): {1e3 * (casci.e_tot - exact.e_tot):.2f} mHa after "
            f"{len(casci.ci.losses_history)} iterations"
        )
