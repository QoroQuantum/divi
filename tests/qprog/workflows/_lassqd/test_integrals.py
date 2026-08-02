# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD integral machinery and the raw-integral Hamiltonian builder."""

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from pyscf import ao2mo, fci, mcscf, scf

from divi.hamiltonians._chem import (
    _spo_from_integrals,
    molecular_hamiltonian_from_pyscf,
)
from divi.qprog.workflows._lassqd import _integrals as _integrals_module
from divi.qprog.workflows._lassqd._integrals import (
    ORBITAL_MINIMIZE_OPTIONS,
    MOIntegrals,
    assemble_active_rdms,
    build_active_permutation,
    build_energy_rdms,
    cached_ao_eri,
    cached_h_ao,
    fragment_effective_integrals,
    optimize_orbitals,
    rotation_energy_gradient_fn,
    total_energy,
    transform_integrals,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec, FragmentState
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    dense_fci_energy,
    h2_molecule,
    h4_chain,
    orbital_rotation_case,
)


def test_spo_from_integrals_matches_whole_molecule_builder():
    """Handed full-molecule integrals, the new builder must reproduce the old one."""
    mol = h2_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb = mo_coeff.shape[1]

    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orb)

    expected, _ = molecular_hamiltonian_from_pyscf(mean_field)
    actual = _spo_from_integrals(one_body, two_body, float(mol.energy_nuc()))

    assert actual.num_qubits == expected.num_qubits
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(actual.to_matrix())),
        np.sort(np.linalg.eigvalsh(expected.to_matrix())),
        atol=1e-10,
    )


def test_spo_from_integrals_ground_state_matches_fci():
    """The FCI energy for the (1, 1) sector must appear in the operator's spectrum."""
    mol = h2_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb = mo_coeff.shape[1]

    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orb)
    constant = float(mol.energy_nuc())

    spo = _spo_from_integrals(one_body, two_body, constant)
    expected = dense_fci_energy(one_body, two_body, 1, 1, constant)

    # The operator spans every particle-number sector, so only assert the
    # FCI energy is somewhere in the spectrum, not that it is the minimum.
    eigenvalues = np.linalg.eigvalsh(spo.to_matrix())
    assert np.any(
        np.isclose(eigenvalues, expected, atol=1e-8)
    ), f"FCI energy {expected} not found in the operator's spectrum"


def test_spo_from_integrals_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="two_body"):
        _spo_from_integrals(np.zeros((2, 2)), np.zeros((3, 3, 3, 3)), 0.0)


def test_build_active_permutation_orders_core_active_virtual():
    specs = [
        FragmentSpec(orbitals=(5, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(3,), n_alpha=1, n_beta=1),
    ]
    permutation = build_active_permutation(specs, n_core=0, n_orbitals_total=6)

    # Active orbitals come first (no core), in spec order, then the rest.
    assert list(permutation[:3]) == [5, 1, 3]
    assert sorted(permutation) == list(range(6))


def test_build_active_permutation_places_core_first():
    specs = [FragmentSpec(orbitals=(4,), n_alpha=1, n_beta=1)]
    permutation = build_active_permutation(specs, n_core=2, n_orbitals_total=6)

    assert permutation[2] == 4
    assert sorted(permutation) == list(range(6))
    # Core slots must not contain the active orbital.
    assert 4 not in permutation[:2]


def test_single_fragment_effective_integrals_are_the_bare_block():
    """With one fragment and no core, h_eff is just the MO block."""
    mol = h2_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    integrals = transform_integrals(mol, mo_coeff, n_core=0)

    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    state = FragmentState(spec=spec, rdm1=np.zeros((2, 2)), rdm2=np.zeros((2,) * 4))

    h_eff, g_frag = fragment_effective_integrals(integrals, [state], 0)

    np.testing.assert_allclose(h_eff, integrals.h_mo[:2, :2], atol=1e-12)
    np.testing.assert_allclose(g_frag, integrals.g_mo[:2, :2, :2, :2], atol=1e-12)


def test_two_fragment_effective_integrals_include_the_other_mean_field():
    """The cross term must equal an explicitly contracted reference."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    integrals = transform_integrals(mol, mo_coeff, n_core=0)

    rdm_other = np.array([[1.4, 0.1], [0.1, 0.6]])
    states = [
        FragmentState(
            spec=FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            rdm1=np.zeros((2, 2)),
            rdm2=np.zeros((2,) * 4),
        ),
        FragmentState(
            spec=FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
            rdm1=rdm_other,
            rdm2=np.zeros((2,) * 4),
        ),
    ]

    h_eff, _ = fragment_effective_integrals(integrals, states, 0)

    expected = integrals.h_mo[:2, :2].copy()
    for p in range(2):
        for q in range(2):
            for r_idx, r in enumerate((2, 3)):
                for s_idx, s in enumerate((2, 3)):
                    expected[p, q] += rdm_other[r_idx, s_idx] * (
                        2.0 * integrals.g_mo[p, q, r, s] - integrals.g_mo[p, r, s, q]
                    )

    np.testing.assert_allclose(h_eff, expected, atol=1e-12)


def test_fragment_hamiltonian_ground_state_matches_fci():
    """End-to-end: effective integrals -> SparsePauliOp -> lowest eigenvalue."""
    mol = h2_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    integrals = transform_integrals(mol, mo_coeff, n_core=0)

    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    state = FragmentState(spec=spec, rdm1=np.zeros((2, 2)), rdm2=np.zeros((2,) * 4))
    h_eff, g_frag = fragment_effective_integrals(integrals, [state], 0)

    spo = _spo_from_integrals(h_eff, g_frag, 0.0)
    expected = dense_fci_energy(h_eff, g_frag, 1, 1, 0.0)

    # The operator spans every particle-number sector, and for effective
    # fragment integrals the (1, 1) ground state is not necessarily the global
    # minimum. Assert the FCI energy is in the spectrum rather than that it is
    # the lowest eigenvalue.
    eigenvalues = np.linalg.eigvalsh(spo.to_matrix())
    assert np.any(
        np.isclose(eigenvalues, expected, atol=1e-8)
    ), f"FCI energy {expected} not found in the fragment operator's spectrum"


def test_assemble_active_rdms_block_diagonalizes():
    states = [
        FragmentState(
            spec=FragmentSpec(orbitals=(0,), n_alpha=1, n_beta=1),
            rdm1=np.array([[2.0]]),
            rdm2=np.ones((1, 1, 1, 1)),
        ),
        FragmentState(
            spec=FragmentSpec(orbitals=(1,), n_alpha=1, n_beta=1),
            rdm1=np.array([[1.5]]),
            rdm2=np.full((1, 1, 1, 1), 3.0),
        ),
    ]
    rdm1, rdm2 = assemble_active_rdms(states)

    np.testing.assert_allclose(rdm1, np.diag([2.0, 1.5]))
    assert rdm2[0, 0, 0, 0] == pytest.approx(1.0)
    assert rdm2[1, 1, 1, 1] == pytest.approx(3.0)
    # No cross-fragment 2-RDM elements are populated.
    assert rdm2[0, 0, 1, 1] == pytest.approx(0.0)


def test_total_energy_matches_fci_for_full_active_space():
    """With n_core=0 and the exact FCI RDMs, total_energy must equal the FCI energy."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb = mo_coeff.shape[1]

    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orb)

    n_alpha, n_beta = 2, 2
    electronic_energy, civec = fci.direct_spin1.kernel(
        one_body, two_body, n_orb, (n_alpha, n_beta)
    )
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, n_orb, (n_alpha, n_beta))
    expected = electronic_energy + mol.energy_nuc()

    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    energy = total_energy(mol, mo_coeff, 0, rdm1, rdm2, ao_eri, h_ao)

    assert energy == pytest.approx(expected, abs=1e-10)


def test_fragment_effective_integrals_matches_casci_h1eff_with_frozen_core():
    """With a real n_core=1 frozen core, h_eff must equal CASCI's active Fock."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)

    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.mo_coeff = mo_coeff
    h1eff, _ = mc.get_h1eff()

    integrals = transform_integrals(mol, mo_coeff, n_core=1)
    spec = FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1)
    state = FragmentState(spec=spec, rdm1=np.zeros((2, 2)), rdm2=np.zeros((2,) * 4))

    h_eff, _ = fragment_effective_integrals(integrals, [state], 0)

    np.testing.assert_allclose(h_eff, h1eff, atol=1e-10)


def test_total_energy_matches_casci_with_frozen_core():
    """total_energy must reproduce a CASCI total energy through a real frozen core."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)

    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.kernel()

    mo_coeff = np.asarray(mc.mo_coeff)
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)

    energy = total_energy(mol, mo_coeff, mc.ncore, rdm1, rdm2, ao_eri, h_ao)

    assert energy == pytest.approx(mc.e_tot, abs=1e-8)


def test_fragment_effective_integrals_honors_noncontiguous_permutation():
    """A non-identity, non-contiguous permutation must still index the caller's orbitals."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)

    specs = [
        FragmentSpec(orbitals=(0, 3), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1),
    ]
    permutation = build_active_permutation(specs, n_core=0, n_orbitals_total=4)
    permuted_mo_coeff = mo_coeff[:, permutation]
    integrals = transform_integrals(mol, permuted_mo_coeff, n_core=0)

    rdm_other = np.array([[1.3, 0.2], [0.2, 0.7]])
    states = [
        FragmentState(spec=specs[0], rdm1=np.zeros((2, 2)), rdm2=np.zeros((2,) * 4)),
        FragmentState(spec=specs[1], rdm1=rdm_other, rdm2=np.zeros((2,) * 4)),
    ]

    h_eff, g_frag = fragment_effective_integrals(integrals, states, 0)

    # Oracle built directly from the unpermuted MO integrals, indexed at the
    # caller's requested orbitals (0, 3) and (1, 2), not at positions 0..3.
    unpermuted = transform_integrals(mol, mo_coeff, n_core=0)
    h_mo, g_mo = unpermuted.h_mo, unpermuted.g_mo
    target_orbitals = (0, 3)
    other_orbitals = (1, 2)

    expected_h = np.zeros((2, 2))
    expected_g = np.zeros((2, 2, 2, 2))
    for p_idx, p in enumerate(target_orbitals):
        for q_idx, q in enumerate(target_orbitals):
            expected_h[p_idx, q_idx] = h_mo[p, q]
            for r_idx, r in enumerate(other_orbitals):
                for s_idx, s in enumerate(other_orbitals):
                    expected_h[p_idx, q_idx] += rdm_other[r_idx, s_idx] * (
                        2.0 * g_mo[p, q, r, s] - g_mo[p, r, s, q]
                    )
            for r_idx, r in enumerate(target_orbitals):
                for s_idx, s in enumerate(target_orbitals):
                    expected_g[p_idx, q_idx, r_idx, s_idx] = g_mo[p, q, r, s]

    np.testing.assert_allclose(h_eff, expected_h, atol=1e-10)
    np.testing.assert_allclose(g_frag, expected_g, atol=1e-10)


def test_fragment_effective_integrals_rejects_inconsistent_n_core():
    """A fragment set that overruns the MO register signals its n_core mismatch."""
    integrals = MOIntegrals(
        h_mo=np.zeros((2, 2)),
        g_mo=np.zeros((2, 2, 2, 2)),
        j_core=np.zeros((2, 2)),
        k_core=np.zeros((2, 2)),
        n_core=1,
    )
    state = FragmentState(
        spec=FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        rdm1=np.zeros((2, 2)),
        rdm2=np.zeros((2,) * 4),
    )

    with pytest.raises(ValueError, match="n_core"):
        fragment_effective_integrals(integrals, [state], 0)


def _diagonal_active_rdms(n_act):
    """A simple idempotent-like diagonal RDM guess, for optimize_orbitals tests
    that only need a well-defined energy functional, not a physical one."""
    rdm1_active = np.eye(n_act)
    rdm2_active = np.zeros((n_act, n_act, n_act, n_act))
    for p in range(n_act):
        for q in range(n_act):
            rdm2_active[p, p, q, q] = rdm1_active[p, p] * rdm1_active[q, q]
    return rdm1_active, rdm2_active


def test_optimize_orbitals_spans_all_four_rotation_categories(mocker):
    """With only two 2-orbital fragments and no frozen core or virtuals, the
    active-active fixtures elsewhere in this test module never populate the
    core-active, core-virtual, or active-virtual rotation categories. Build a
    fragmentation with a real frozen core and leftover virtual orbitals so all
    four categories are exercised, and confirm both the exact rotation count
    and that the resulting orbitals stay orthonormal under the AO overlap --
    the property a non-skew-symmetric generator would violate."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb_total = mo_coeff.shape[1]

    specs = [
        FragmentSpec(orbitals=(1,), n_alpha=1, n_beta=0),
        FragmentSpec(orbitals=(2,), n_alpha=0, n_beta=1),
    ]
    n_core = 1
    n_act = sum(spec.n_orbitals for spec in specs)
    n_vir = n_orb_total - n_core - n_act

    permutation = build_active_permutation(specs, n_core, n_orb_total)
    permuted_mo_coeff = mo_coeff[:, permutation]
    rdm1_active, rdm2_active = _diagonal_active_rdms(n_act)
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)

    spy = mocker.spy(_integrals_module, "minimize")
    rotated_mo_coeff, _ = optimize_orbitals(
        mol, permuted_mo_coeff, n_core, specs, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    expected_n_rot = (
        n_core * n_act  # core-active
        + n_core * n_vir  # core-virtual
        + n_act * n_vir  # active-virtual
        + sum(  # cross-fragment active-active
            specs[i].n_orbitals * specs[j].n_orbitals
            for i in range(len(specs))
            for j in range(len(specs))
            if i < j
        )
    )
    actual_n_rot = len(spy.call_args.args[1])
    assert actual_n_rot == expected_n_rot == 6

    overlap = mol.intor("int1e_ovlp")
    gram = rotated_mo_coeff.T @ overlap @ rotated_mo_coeff
    np.testing.assert_allclose(gram, np.eye(n_orb_total), atol=1e-10)


def test_optimize_orbitals_reports_the_real_energy_with_no_rotation_freedom():
    """A single fragment spanning the whole active space, with no frozen core
    and no virtuals, leaves zero rotation pairs. ``scipy.optimize.minimize``
    called with a zero-length ``x0`` never evaluates the objective and
    reports a spurious ``fun=0.0``; ``optimize_orbitals`` must not surface
    that value as the energy, and must leave the coefficients unrotated."""
    mol = h2_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)

    energy_fci, civec = fci.FCI(mean_field).kernel()
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, 2, (1, 1))
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    rotated_mo_coeff, energy = optimize_orbitals(
        mol, mo_coeff, 0, [spec], rdm1, rdm2, ao_eri, h_ao
    )

    unrotated_energy = total_energy(mol, mo_coeff, 0, rdm1, rdm2, ao_eri, h_ao)
    assert energy == pytest.approx(unrotated_energy)
    assert energy == pytest.approx(energy_fci, abs=1e-10)
    np.testing.assert_allclose(rotated_mo_coeff, mo_coeff, atol=1e-12)


def test_optimize_orbitals_discards_a_scipy_result_worse_than_baseline(mocker):
    """If ``minimize`` returns a result whose energy is worse than doing no
    rotation at all, that result must be discarded in favor of the baseline,
    keeping the routine monotone regardless of what scipy reports."""
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_act = 4
    rdm1_active, rdm2_active = _diagonal_active_rdms(n_act)
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
    ]

    baseline_energy = total_energy(
        mol, mo_coeff, 0, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    fake_result = mocker.Mock()
    fake_result.fun = baseline_energy + 1.0
    fake_result.x = np.full(4, 0.5)
    mocker.patch.object(_integrals_module, "minimize", return_value=fake_result)

    rotated_mo_coeff, energy = optimize_orbitals(
        mol, mo_coeff, 0, specs, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    assert energy == pytest.approx(baseline_energy)
    np.testing.assert_allclose(rotated_mo_coeff, mo_coeff, atol=1e-12)


def _rotated_mo_coeff(mo_coeff, rotation_pairs, rotation_params):
    """``mo_coeff`` under the skew generator the rotation angles parameterize."""
    n_orb = mo_coeff.shape[1]
    generator = np.zeros((n_orb, n_orb))
    for idx, (p, q) in enumerate(rotation_pairs):
        generator[p, q] = rotation_params[idx]
        generator[q, p] = -rotation_params[idx]
    return mo_coeff @ scipy.linalg.expm(generator)


def test_energy_rdms_reconstruct_total_energy(orbital_rotation_case):
    """The contracted ``E_nuc + sum(h D) + 0.5 * sum(d g)`` form the analytic
    gradient is derived from must reproduce ``total_energy``'s explicit loops
    exactly, which is what pins the core-core, core-active and core-active
    exchange blocks of the two-particle density."""
    mol, mo_coeff, n_core, _, rdm1_active, rdm2_active, ao_eri, h_ao = (
        orbital_rotation_case
    )
    n_orb = mo_coeff.shape[1]

    one_rdm, two_rdm = build_energy_rdms(n_orb, n_core, rdm1_active, rdm2_active)
    h_mo = mo_coeff.T @ h_ao @ mo_coeff
    g_mo = ao2mo.restore(1, ao2mo.incore.full(ao_eri, mo_coeff), n_orb)

    reconstructed = (
        mol.energy_nuc()
        + np.einsum("mn,mn->", h_mo, one_rdm)
        + 0.5 * np.einsum("mnop,mnop->", two_rdm, g_mo)
    )
    expected = total_energy(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    assert reconstructed == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("at_origin", [True, False])
def test_rotation_gradient_matches_central_differences(
    orbital_rotation_case, at_origin
):
    """The analytic gradient must match central differences of ``total_energy``
    itself, both at zero rotation and away from it. Away from zero is the
    discriminating case: the derivative of the matrix exponential is a Frechet
    pullback, and the naive commutator form it is easily confused with agrees
    with it only at the origin."""
    mol, mo_coeff, n_core, _, rdm1_active, rdm2_active, ao_eri, h_ao = (
        orbital_rotation_case
    )
    rotation_pairs, energy_and_gradient = rotation_energy_gradient_fn(
        *orbital_rotation_case
    )
    n_rot = len(rotation_pairs)
    assert n_rot == 61

    rotation_params = (
        np.zeros(n_rot)
        if at_origin
        else 0.15 * np.random.default_rng(20250801).standard_normal(n_rot)
    )

    def energy_at(params):
        rotated = _rotated_mo_coeff(mo_coeff, rotation_pairs, params)
        return total_energy(
            mol, rotated, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
        )

    step = 1e-5
    numerical = np.empty(n_rot)
    for idx in range(n_rot):
        forward = rotation_params.copy()
        forward[idx] += step
        backward = rotation_params.copy()
        backward[idx] -= step
        numerical[idx] = (energy_at(forward) - energy_at(backward)) / (2.0 * step)

    energy, analytic = energy_and_gradient(rotation_params)

    assert energy == pytest.approx(energy_at(rotation_params), abs=1e-10)
    np.testing.assert_allclose(analytic, numerical, atol=1e-6)


def test_optimize_orbitals_improves_strictly_on_the_baseline(orbital_rotation_case):
    """The routine returns ``min(baseline, minimize_result)``, so an
    inverted-sign gradient would silently return the unrotated orbitals at the
    baseline energy. Assert a strict improvement, and that the reported energy
    is the one the returned coefficients actually produce."""
    mol, mo_coeff, n_core, _, rdm1_active, rdm2_active, ao_eri, h_ao = (
        orbital_rotation_case
    )
    baseline_energy = total_energy(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    rotated_mo_coeff, energy = optimize_orbitals(*orbital_rotation_case)

    assert energy < baseline_energy - 1e-8
    assert energy == pytest.approx(
        total_energy(
            mol, rotated_mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
        ),
        abs=1e-8,
    )

    overlap = mol.intor("int1e_ovlp")
    gram = rotated_mo_coeff.T @ overlap @ rotated_mo_coeff
    np.testing.assert_allclose(gram, np.eye(mo_coeff.shape[1]), atol=1e-10)


def test_orbital_minimize_options_reach_a_small_gradient(orbital_rotation_case):
    """A ``tol=1e-6`` shorthand halts with the gradient still around 2e-2,
    since ``ftol`` is relative to ``|E|``. Asserted on ``minimize``'s own
    Jacobian: the rotation-pair set is not closed under the Fréchet pullback,
    so the local gradient at the returned coefficients is a different quantity.
    """
    mol, mo_coeff, n_core, specs, rdm1_active, rdm2_active, ao_eri, h_ao = (
        orbital_rotation_case
    )
    rotation_pairs, energy_and_gradient = rotation_energy_gradient_fn(
        mol, mo_coeff, n_core, specs, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    result = scipy.optimize.minimize(
        energy_and_gradient,
        np.zeros(len(rotation_pairs)),
        method="L-BFGS-B",
        jac=True,
        options=ORBITAL_MINIMIZE_OPTIONS,
    )

    assert result.status == 0
    assert np.abs(result.jac).max() < 1e-3
