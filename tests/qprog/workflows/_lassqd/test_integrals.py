# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD integral machinery and the raw-integral Hamiltonian builder."""

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from pyscf import ao2mo, fci, mcscf, scf
from pyscf.fci import cistring

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


def test_two_fragment_effective_integrals_match_pyscfs_embedding_potential():
    """A doubly occupied other fragment is indistinguishable from frozen core,
    so PySCF's ``CASCI.get_h1eff()`` is an independent oracle for the
    embedding potential.

    This replaces a version whose ``expected`` re-implemented the same
    contraction as the source. That passed for *any* scale factor on the
    density term, and the term was in fact 2x too large: ``rdm1`` is
    spin-traced, so contracting it against ``2J - K`` -- coefficients that
    already assume double occupancy -- double-counted exactly.
    """
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    integrals = transform_integrals(mol, mo_coeff, n_core=0)

    states = [
        FragmentState(
            spec=FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            rdm1=2.0 * np.eye(2),
            rdm2=np.zeros((2,) * 4),
        ),
        FragmentState(
            spec=FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
            rdm1=np.zeros((2, 2)),
            rdm2=np.zeros((2,) * 4),
        ),
    ]

    h_eff, _ = fragment_effective_integrals(integrals, states, 1)

    # ncas=2 with zero active electrons forces ncore=2, so orbitals 0 and 1 are
    # the core and 2, 3 the active block -- the same partition as above.
    casci = mcscf.CASCI(mean_field, 2, 0)
    casci.mo_coeff = mo_coeff
    assert casci.ncore == 2
    expected, _ = casci.get_h1eff()

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


def test_assemble_active_rdms_places_blocks_and_cross_fragment_terms():
    """Intra-fragment blocks sit on the diagonal; the cross-fragment 2-RDM
    carries the product-state Coulomb and exchange terms.

    The 1-RDM stays strictly block-diagonal -- a product of fragment
    wavefunctions has no inter-fragment coherence.
    """
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
    assert rdm1[0, 1] == pytest.approx(0.0)
    assert rdm2[0, 0, 0, 0] == pytest.approx(1.0)
    assert rdm2[1, 1, 1, 1] == pytest.approx(3.0)

    # Coulomb: gamma_A[0,0] * gamma_B[0,0], both orderings.
    assert rdm2[0, 0, 1, 1] == pytest.approx(3.0)
    assert rdm2[1, 1, 0, 0] == pytest.approx(3.0)

    # Exchange: -(alpha_A alpha_B + beta_A beta_B); with no per-spin RDMs
    # supplied each half is gamma/2, giving -(1.0 * 0.75 + 1.0 * 0.75).
    assert rdm2[0, 1, 1, 0] == pytest.approx(-1.5)
    assert rdm2[1, 0, 0, 1] == pytest.approx(-1.5)


@pytest.mark.parametrize(
    "particles_a,particles_b",
    [((1, 1), (1, 1)), ((1, 0), (1, 2))],
    ids=["closed-shell", "spin-polarized"],
)
def test_assemble_active_rdms_matches_an_explicit_product_state(
    particles_a, particles_b
):
    """Index-level oracle against PySCF.

    Two exactly-solved 2-orbital fragments, combined into an explicit product
    CI vector over the full 4-orbital space, whose RDMs PySCF then computes
    independently. 2x2 blocks are the point: with 1x1 blocks every index
    permutation of the Coulomb and exchange einsums is numerically identical, so
    a transposed exchange term or a swapped target slice passes unnoticed.
    """
    rng = np.random.default_rng(5)
    fragments = []
    civecs = []
    for particles in (particles_a, particles_b):
        one_body = rng.normal(size=(2, 2))
        one_body = one_body + one_body.T
        two_body = np.zeros((2,) * 4)
        _, civec = fci.direct_spin1.kernel(one_body, two_body, 2, particles)
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, 2, particles)
        alpha, beta = fci.direct_spin1.make_rdm1s(civec, 2, particles)
        civecs.append((civec, particles))
        fragments.append(
            FragmentState(
                spec=FragmentSpec(
                    orbitals=(0, 1) if not fragments else (2, 3),
                    n_alpha=particles[0],
                    n_beta=particles[1],
                ),
                rdm1=rdm1,
                rdm2=rdm2,
                rdm1_alpha=alpha,
                rdm1_beta=beta,
            )
        )

    rdm1, rdm2 = assemble_active_rdms(fragments)

    # The product state, written out over the combined 4-orbital space. A's
    # orbitals (0, 1) precede B's (2, 3), so the string-combination sign is +1.
    total = (
        particles_a[0] + particles_b[0],
        particles_a[1] + particles_b[1],
    )
    product = np.zeros(
        (
            cistring.num_strings(4, total[0]),
            cistring.num_strings(4, total[1]),
        )
    )
    for (vec_a, part_a), (vec_b, part_b) in [tuple(civecs)]:
        for ia, sa in enumerate(cistring.make_strings(range(2), part_a[0])):
            for ib, sb in enumerate(cistring.make_strings(range(2), part_a[1])):
                for ja, ta in enumerate(cistring.make_strings(range(2, 4), part_b[0])):
                    for jb, tb in enumerate(
                        cistring.make_strings(range(2, 4), part_b[1])
                    ):
                        addr_a = cistring.str2addr(4, total[0], sa | ta)
                        addr_b = cistring.str2addr(4, total[1], sb | tb)
                        product[addr_a, addr_b] += vec_a[ia, ib] * vec_b[ja, jb]

    expected1, expected2 = fci.direct_spin1.make_rdm12(product, 4, total)
    np.testing.assert_allclose(rdm1, expected1, atol=1e-12)
    np.testing.assert_allclose(rdm2, expected2, atol=1e-12)


def test_assemble_active_rdms_exchange_uses_per_spin_densities():
    """A spin-polarized pair exchanges only within a spin channel, so supplying
    the alpha/beta halves must give a different answer from the closed-shell
    ``gamma / 2`` fallback -- an all-alpha and an all-beta fragment have no
    same-spin overlap to exchange at all."""
    alpha_only = FragmentState(
        spec=FragmentSpec(orbitals=(0,), n_alpha=1, n_beta=0),
        rdm1=np.array([[1.0]]),
        rdm2=np.zeros((1, 1, 1, 1)),
        rdm1_alpha=np.array([[1.0]]),
        rdm1_beta=np.zeros((1, 1)),
    )
    beta_only = FragmentState(
        spec=FragmentSpec(orbitals=(1,), n_alpha=0, n_beta=1),
        rdm1=np.array([[1.0]]),
        rdm2=np.zeros((1, 1, 1, 1)),
        rdm1_alpha=np.zeros((1, 1)),
        rdm1_beta=np.array([[1.0]]),
    )
    _, rdm2 = assemble_active_rdms([alpha_only, beta_only])

    assert rdm2[0, 0, 1, 1] == pytest.approx(1.0)
    assert rdm2[0, 1, 1, 0] == pytest.approx(0.0)

    # The closed-shell fallback would wrongly predict -0.5 here.
    _, rdm2_traced = assemble_active_rdms(
        [
            FragmentState(spec=s.spec, rdm1=s.rdm1, rdm2=s.rdm2)
            for s in (alpha_only, beta_only)
        ]
    )
    assert rdm2_traced[0, 1, 1, 0] == pytest.approx(-0.5)


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
    """A non-identity, non-contiguous permutation must still index the caller's
    orbitals.

    The oracle here reproduces the source's contraction, so it pins the
    *indexing* and not the coefficients; those are pinned independently against
    PySCF in ``test_two_fragment_effective_integrals_match_pyscfs_embedding_potential``.
    """
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
                        g_mo[p, q, r, s] - 0.5 * g_mo[p, r, s, q]
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
