# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Ethylene's π→π* excitation with single-fragment LASSQD.

Ethylene in 6-31+G*, with the π and π* orbitals as the active space (4 qubits).
At this size the sampled subspace spans the whole active space.

1. The active orbitals, picked by irreducible representation: in a diffuse
   basis the lowest virtual orbital is not π*.
2. The ground and first excited singlet with state-averaged orbitals: excitation
   energy, wavelength and oscillator strength, against PySCF's state-averaged
   CASSCF.
3. NEVPT2 corrections to both states, beside the experimental band maximum.
"""

from pyscf import fci, gto, mcscf, scf, symm
from pyscf.data.nist import HARTREE2EV, HARTREE2WAVENUMBER

from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.lassqd import (
    LASSQD,
    FragmentationConfig,
    FragmentSpec,
    SQDConfig,
    StateAveragedOrbitalUpdate,
)
from divi.qprog.problems import MolecularProblem
from tutorials._backend import get_backend

EXPERIMENT_EV = 7.66


def convert_to_nanometres(energy_hartree: float) -> float:
    return 1e7 / (energy_hartree * HARTREE2WAVENUMBER)


if __name__ == "__main__":
    backend = get_backend(shots=2000)
    mol = gto.M(
        atom=(
            "C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; "
            "H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321"
        ),
        basis="6-31+g*",
        symmetry=True,
        verbose=0,
    )
    mean_field = scf.RHF(mol).run()

    print("Part 1: active orbitals")
    irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mean_field.mo_coeff)
    homo = mol.nelectron // 2 - 1
    pi_star = next(i for i in range(homo + 1, len(irreps)) if irreps[i] == "B2g")
    active = [homo, pi_star]
    print(f"  HOMO {homo} ({irreps[homo]}), LUMO {homo + 1} ({irreps[homo + 1]})")
    print(f"  Active: π = {homo} ({irreps[homo]}), π* = {pi_star} ({irreps[pi_star]})")

    print("\nPart 2: S0 and S1 with state-averaged orbitals")
    ensemble = LASSQD(
        MolecularProblem.from_molecule(mean_field),
        fragmentation=FragmentationConfig(
            active_spaces=[FragmentSpec(orbitals=tuple(active), n_alpha=1, n_beta=1)]
        ),
        sqd=SQDConfig(states={1: 2}),
        orbital_update=StateAveragedOrbitalUpdate(),
        seed=0,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
    )
    ensemble.run(max_rounds=10)
    states = ensemble.electronic_states
    ground, excited = states[(1, 0)], states[(1, 1)]
    gap = excited.energy - ground.energy
    strength = ensemble.transition_properties()[((1, 0), (1, 1))].oscillator_strength

    casscf = mcscf.CASSCF(mean_field, 2, 2)
    casscf.fcisolver = fci.direct_spin1.FCI(mol)
    reference = casscf.state_average_([0.5, 0.5])
    reference.fix_spin_(ss=0)
    reference.kernel(reference.sort_mo([index + 1 for index in active]))
    reference_gap = reference.e_states[1] - reference.e_states[0]

    print(
        f"  LASSQD: {gap * HARTREE2EV:.4f} eV "
        f"({convert_to_nanometres(gap):.1f} nm), f = {strength:.3f}"
    )
    print(f"  PySCF state-averaged CASSCF: {reference_gap * HARTREE2EV:.4f} eV")

    print("\nPart 3: NEVPT2")
    corrections = ensemble.nevpt2()
    corrected = gap + corrections[(1, 1)] - corrections[(1, 0)]
    print(
        f"  LASSQD + NEVPT2: {corrected * HARTREE2EV:.4f} eV "
        f"({convert_to_nanometres(corrected):.1f} nm)"
    )
    experiment = EXPERIMENT_EV / HARTREE2EV
    print(
        f"  Experiment, band maximum: {EXPERIMENT_EV} eV "
        f"({convert_to_nanometres(experiment):.1f} nm)"
    )
