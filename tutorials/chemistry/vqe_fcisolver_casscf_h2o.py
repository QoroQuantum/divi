# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""CASSCF and molecular properties with VQE as PySCF's active-space solver.

Water in 6-31G with four electrons in four orbitals (8 qubits), solved by
:class:`~divi.qprog.algorithms.VQEFCISolver` inside PySCF's CASCI and CASSCF.

1. CASCI against CASSCF with ``natorb=True``, each against PySCF's exact
   solver.
2. Dipole moment and natural occupations through PySCF's ``make_rdm1`` and
   ``dip_moment``.
3. RMS error of the energy and dipole from density matrices sampled at 1k, 10k
   and 100k shots, over five repeats each.
"""

import numpy as np
from pyscf import gto, mcscf, scf

from divi.backends import CircuitRunner
from divi.hamiltonians import sum_spin_rdms
from divi.qprog import EarlyStopping
from divi.qprog.algorithms import VQE, UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import RotosolveOptimizer
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

N_ORBITALS, N_ELECTRONS = 4, 4
N_REPEATS = 5


def make_solver(backend: CircuitRunner) -> VQEFCISolver:
    return VQEFCISolver(
        backend=backend,
        optimizer=RotosolveOptimizer(),
        ansatz=UCCSDAnsatz(),
        max_iterations=30,
        early_stopping=EarlyStopping(patience=1, min_delta=1e-7),
    )


def compute_dipole_debye(mean_field: scf.hf.RHF, density_ao: np.ndarray) -> float:
    dipole = mean_field.dip_moment(dm=density_ao, unit="Debye", verbose=0)
    return float(np.linalg.norm(dipole))


def compute_energy_and_dipole(
    rdms,
    problem: MolecularProblem,
    mean_field: scf.hf.RHF,
    casscf: mcscf.mc1step.CASSCF,
) -> tuple[float, float]:
    """Energy and dipole from spin-resolved RDMs in the CASSCF's active orbitals."""
    dm1, dm2 = sum_spin_rdms(rdms)
    energy = (
        problem.constant
        + np.einsum("pq,pq", problem.one_body, dm1)
        + 0.5 * np.einsum("pqrs,pqrs", problem.two_body, dm2)
    )
    core = casscf.mo_coeff[:, : casscf.ncore]
    active = casscf.mo_coeff[:, casscf.ncore : casscf.ncore + casscf.ncas]
    density_ao = 2 * core @ core.T + active @ dm1 @ active.T
    return float(energy), compute_dipole_debye(mean_field, density_ao)


if __name__ == "__main__":
    backend = get_backend()
    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="6-31g", verbose=0
    )
    mean_field = scf.RHF(mol).run()

    print("Part 1: CASCI against CASSCF, H2O/6-31G CAS(4,4), 8 qubits")
    exact_casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS).run()
    exact_casscf = mcscf.CASSCF(mean_field, N_ORBITALS, N_ELECTRONS).run()

    casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
    casci.fcisolver = make_solver(backend)
    casci.kernel()

    casscf = mcscf.CASSCF(mean_field, N_ORBITALS, N_ELECTRONS)
    casscf.fcisolver = make_solver(backend)
    casscf.natorb = True
    casscf.kernel()

    print(f"  Hartree-Fock      {mean_field.e_tot:.6f} Ha")
    for label, vqe, exact in (
        ("CASCI ", casci, exact_casci),
        ("CASSCF", casscf, exact_casscf),
    ):
        print(
            f"  {label} with VQE {vqe.e_tot:.6f} Ha, "
            f"{1e3 * (vqe.e_tot - exact.e_tot):+.3f} mHa from PySCF's exact solver, "
            f"{1e3 * (mean_field.e_tot - vqe.e_tot):.1f} mHa below Hartree-Fock"
        )
    print(f"  CASSCF converged: {casscf.converged}")

    print("\nPart 2: properties through PySCF")
    densities = {
        "Hartree-Fock": mean_field.make_rdm1(),
        "VQE CASSCF": casscf.make_rdm1(),
        "exact CASSCF": exact_casscf.make_rdm1(),
    }
    for label, density in densities.items():
        print(f"  {label:<13} dipole {compute_dipole_debye(mean_field, density):.4f} D")
    occupations = casscf.mo_occ[casscf.ncore : casscf.ncore + casscf.ncas]
    print(f"  Natural occupations: {np.round(occupations, 4)}")

    print("\nPart 3: density matrices from sampled measurements")
    problem = MolecularProblem.from_casci(
        mean_field, N_ORBITALS, N_ELECTRONS, mo_coeff=casscf.mo_coeff
    )
    params = casscf.ci.best_params
    exact_energy, exact_dipole = compute_energy_and_dipole(
        casscf.ci.measure_rdms(), problem, mean_field, casscf
    )
    print(f"  {'shots':>7} {'RMS energy error':>18} {'RMS dipole error':>18}")
    for shots in (1_000, 10_000, 100_000):
        sampling_backend = get_backend(shots=shots, force_sampling=True)
        sampler = VQE(
            problem,
            ansatz=UCCSDAnsatz(),
            optimizer=RotosolveOptimizer(),
            backend=sampling_backend,
        )
        errors = np.array(
            [
                np.subtract(
                    compute_energy_and_dipole(
                        sampler.measure_rdms(params), problem, mean_field, casscf
                    ),
                    (exact_energy, exact_dipole),
                )
                for _ in range(N_REPEATS)
            ]
        )
        energy_rms, dipole_rms = np.sqrt(np.mean(errors**2, axis=0))
        print(
            f"  {sampling_backend.shots:>7} {1e3 * energy_rms:>14.3f} mHa "
            f"{dipole_rms:>16.4f} D"
        )
