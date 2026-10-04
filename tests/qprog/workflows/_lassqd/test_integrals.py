# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LASSQD integral machinery and the raw-integral Hamiltonian builder."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import cistring

from divi.hamiltonians import molecular_hamiltonian_from_pyscf
from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog.workflows._lassqd import _integrals as _integrals_module
from divi.qprog.workflows._lassqd._config import (
    FullOrbitalSolve,
    SecondOrderOrbitalSolve,
)
from divi.qprog.workflows._lassqd._integrals import (
    MOIntegrals,
    _total_energy,
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    ciah_orbital_solve,
    fragment_effective_integrals,
    optimize_orbitals,
    rotation_energy_gradient_fn,
    transform_integrals,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec, FragmentState
from tests._helpers import exact_match
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    build_energy_rdms,
    capped_ciah_orbital_solve,
    dense_fci_energy,
    h2_mean_field,
    h4_chain_mean_field,
    mo_integrals,
    orbital_rotation_case,
    orbital_solver,
)


def _idle_fragment(orbitals, n_alpha=1, n_beta=1):
    """A fragment holding no density, so it contributes no embedding."""
    n_orb = len(orbitals)
    return FragmentState(
        spec=FragmentSpec(orbitals=orbitals, n_alpha=n_alpha, n_beta=n_beta),
        rdm1=np.zeros((n_orb, n_orb)),
        rdm2=np.zeros((n_orb,) * 4),
    )


def test_spo_from_integrals_matches_whole_molecule_builder(h2_mean_field):
    """Handed full-molecule integrals, the new builder must reproduce the old one."""
    one_body, two_body, _, constant = mo_integrals(h2_mean_field)

    expected, _ = molecular_hamiltonian_from_pyscf(h2_mean_field)
    actual = _spo_from_integrals(one_body, two_body, constant)

    assert actual.num_qubits == expected.num_qubits
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(actual.to_matrix())),
        np.sort(np.linalg.eigvalsh(expected.to_matrix())),
        atol=1e-10,
    )


def test_spo_from_integrals_ground_state_matches_fci(h2_mean_field):
    """The FCI energy for the (1, 1) sector must appear in the operator's spectrum."""
    one_body, two_body, _, constant = mo_integrals(h2_mean_field)

    spo = _spo_from_integrals(one_body, two_body, constant)
    expected = dense_fci_energy(one_body, two_body, 1, 1, constant)

    # The operator spans every particle-number sector, so only assert the
    # FCI energy is somewhere in the spectrum, not that it is the minimum.
    eigenvalues = np.linalg.eigvalsh(spo.to_matrix())
    assert np.any(
        np.isclose(eigenvalues, expected, atol=1e-8)
    ), f"FCI energy {expected} not found in the operator's spectrum"


@pytest.mark.parametrize(
    "one_body, two_body, one_body_beta, message",
    [
        (
            np.zeros((2, 2)),
            np.zeros((3,) * 4),
            None,
            "two_body must have shape (2, 2, 2, 2); got (3, 3, 3, 3).",
        ),
        (
            np.zeros((2, 3)),
            np.zeros((2,) * 4),
            None,
            "one_body must be square; got (2, 3).",
        ),
        (
            np.zeros((2, 2)),
            np.zeros((2,) * 4),
            np.zeros((3, 3)),
            "one_body_beta must have shape (2, 2); got (3, 3).",
        ),
    ],
    ids=["mismatched_two_body", "non_square_one_body", "mismatched_beta"],
)
def test_spo_from_integrals_rejects_bad_shapes(
    one_body, two_body, one_body_beta, message
):
    with pytest.raises(ValueError, match=exact_match(message)):
        _spo_from_integrals(one_body, two_body, 0.0, one_body_beta=one_body_beta)


def test_spo_from_integrals_keeps_idle_orbitals_in_the_register():
    spo = _spo_from_integrals(np.diag([-1.0, 0.0]), np.zeros((2,) * 4), 0.0)
    assert spo.num_qubits == 4


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


def test_single_fragment_effective_integrals_are_the_bare_block(h2_mean_field):
    """With one fragment and no core, h_eff is just the MO block."""
    integrals = transform_integrals(
        h2_mean_field.mol, h2_mean_field.mo_coeff, n_core=0, n_act=2
    )

    h_eff, _, g_frag = fragment_effective_integrals(
        integrals, [_idle_fragment((0, 1))], 0
    )

    np.testing.assert_allclose(h_eff, integrals.h_act, atol=1e-12)
    np.testing.assert_allclose(g_frag, integrals.g_act, atol=1e-12)


def test_two_fragment_effective_integrals_match_pyscfs_embedding_potential(
    h4_chain_mean_field,
):
    """A doubly occupied other fragment is indistinguishable from frozen core,
    so PySCF's ``CASCI.get_h1eff()`` is an independent oracle for the
    embedding potential.

    This replaces a version whose ``expected`` re-implemented the same
    contraction as the source. That passed for *any* scale factor on the
    density term, and the term was in fact 2x too large: ``rdm1`` is
    spin-traced, so contracting it against ``2J - K`` -- coefficients that
    already assume double occupancy -- double-counted exactly.
    """
    mean_field = h4_chain_mean_field
    mo_coeff = np.asarray(mean_field.mo_coeff)
    integrals = transform_integrals(mean_field.mol, mo_coeff, n_core=0, n_act=4)

    states = [
        FragmentState(
            spec=FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            rdm1=2.0 * np.eye(2),
            rdm2=np.zeros((2,) * 4),
        ),
        _idle_fragment((2, 3)),
    ]

    h_eff, h_beta, _ = fragment_effective_integrals(integrals, states, 1)

    # No spin density in the neighbour, so the spin-resolved path reduces to
    # the spin-traced one.
    np.testing.assert_allclose(h_eff, h_beta, atol=1e-14)

    # ncas=2 with zero active electrons forces ncore=2, so orbitals 0 and 1 are
    # the core and 2, 3 the active block -- the same partition as above.
    casci = mcscf.CASCI(mean_field, 2, 0)
    casci.mo_coeff = mo_coeff
    assert casci.ncore == 2
    expected, _ = casci.get_h1eff()

    np.testing.assert_allclose(h_eff, expected, atol=1e-12)


def test_spo_beta_one_body_touches_only_beta_spin_orbitals():
    """``one_body_beta`` must land on the beta spin-orbitals, not the alpha ones.

    ``spinorb_from_spatial`` interleaves with alpha on even indices, so the beta
    block is written at ``[1::2, 1::2]``. Nothing downstream would notice a swap:
    particle number, Sz and every symmetry the other tests assert are preserved
    under exchanging the two channels, while an antiferromagnetic fragment would
    silently get its neighbour's beta potential applied to its alpha electrons.

    Anchored structurally rather than by an energy: the operator built with a
    distinct beta channel, minus the one built without, must act non-trivially
    only on odd (beta) qubits.
    """
    n_orb = 2
    h_alpha = np.diag([-0.9, 0.4])
    h_beta = np.diag([-0.2, -0.6])
    two_body = np.zeros((n_orb,) * 4)

    with_beta = _spo_from_integrals(h_alpha, two_body, 0.0, one_body_beta=h_beta)
    alpha_only = _spo_from_integrals(h_alpha, two_body, 0.0)
    difference = (with_beta - alpha_only).simplify()

    acted_on = set()
    for pauli in difference.paulis:
        acted_on.update(np.flatnonzero(pauli.x | pauli.z).tolist())
    # Only the beta channel changed, and it did change.
    assert acted_on, "supplying one_body_beta changed nothing"
    assert all(
        qubit % 2 == 1 for qubit in acted_on
    ), f"beta one-body reached qubits {sorted(acted_on)}; even qubits are alpha"


# sqrt of LASSQD's default energy_tol, the convergence threshold it passes.
_GRADIENT_TOL = 1e-3


@pytest.fixture(scope="module")
def converged_orbital_solve(orbital_rotation_case):
    return optimize_orbitals(*orbital_rotation_case, gradient_tol=_GRADIENT_TOL)


@pytest.fixture(scope="module")
def second_order_solve(orbital_rotation_case):
    return capped_ciah_orbital_solve(*orbital_rotation_case, gradient_tol=_GRADIENT_TOL)


_EACH_SOLVE = pytest.mark.parametrize(
    "solver, solve_fixture",
    [
        (optimize_orbitals, "converged_orbital_solve"),
        (capped_ciah_orbital_solve, "second_order_solve"),
    ],
    ids=["l-bfgs-b", "ciah"],
)


_CAP_REASON = {
    optimize_orbitals: "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
    capped_ciah_orbital_solve: "macro-iteration limit",
}


def _unconverged_warnings(record, solve, reason, gradient_tol):
    """Assert ``record`` holds exactly the non-convergence warning for ``solve``."""
    expected = (
        "Orbital optimisation ended without converging after "
        f"{solve.n_iterations} iterations and {solve.n_evaluations} evaluations "
        f"({reason}): its orbital-gradient norm {solve.gradient_norm:.2e} "
        f"exceeds {gradient_tol:.2e}. The returned orbitals are the best seen, so "
        "the energy is still an upper bound, but this round is not a stationary "
        "point -- a small round-to-round energy change here means the optimizer "
        "gave up, not that the macro-cycle converged."
    )
    assert [str(warning.message) for warning in record] == [expected]


def _energy_and_gradient_at(case, mo_coeff):
    """Energy and orbital gradient of ``case``'s functional at ``mo_coeff``."""
    mol, _, *rest = case
    pairs, energy_and_gradient = rotation_energy_gradient_fn(mol, mo_coeff, *rest)
    return energy_and_gradient(np.zeros(len(pairs)))


def _random_rotation(n_orbitals, scale, seed):
    generator = np.random.default_rng(seed).normal(scale=scale, size=(n_orbitals,) * 2)
    return scipy.linalg.expm(generator - generator.T)


@_EACH_SOLVE
def test_an_orbital_solve_reports_whether_it_converged(
    orbital_rotation_case, solver, solve_fixture, request
):
    """A solve capped before convergence is flagged, so its barely moving
    energy is not mistaken for a converged round."""
    converged_solve = request.getfixturevalue(solve_fixture)
    assert converged_solve.converged is True

    with pytest.warns(UserWarning) as record:
        starved_solve = solver(
            *orbital_rotation_case, max_iterations=1, gradient_tol=_GRADIENT_TOL
        )

    _unconverged_warnings(record, starved_solve, _CAP_REASON[solver], _GRADIENT_TOL)
    assert starved_solve.converged is False
    assert starved_solve.n_iterations == 1
    assert starved_solve.n_iterations < converged_solve.n_iterations
    assert starved_solve.energy >= converged_solve.energy - 1e-12


@pytest.mark.parametrize(
    "capped_solve",
    [
        partial(optimize_orbitals, max_iterations=1),
        partial(ciah_orbital_solve, max_iterations=1),
        FullOrbitalSolve(max_iterations=1)._solve,
        SecondOrderOrbitalSolve(max_iterations=1)._solve,
    ],
    ids=["l-bfgs-b", "ciah", "full-orbital-solve", "second-order-orbital-solve"],
)
def test_the_non_convergence_warning_points_at_the_caller(
    orbital_rotation_case, capped_solve
):
    with pytest.warns(UserWarning, match="ended without converging") as record:
        capped_solve(*orbital_rotation_case, gradient_tol=_GRADIENT_TOL)

    assert [warning.filename for warning in record] == [__file__]


@_EACH_SOLVE
def test_an_orbital_solve_reports_the_energy_and_gradient_at_its_orbitals(
    orbital_rotation_case, solver, solve_fixture, request
):
    """The reported energy and gradient are those at the returned orbitals."""
    solve = request.getfixturevalue(solve_fixture)
    energy, gradient = _energy_and_gradient_at(orbital_rotation_case, solve.mo_coeff)

    assert solve.energy == pytest.approx(energy, abs=1e-10)
    assert solve.gradient_norm == pytest.approx(np.linalg.norm(gradient), rel=1e-6)


def test_the_second_order_solve_converges_below_its_start(
    orbital_rotation_case, second_order_solve
):
    start_energy, _ = _energy_and_gradient_at(
        orbital_rotation_case, orbital_rotation_case[1]
    )

    assert second_order_solve.converged
    assert second_order_solve.gradient_norm <= _GRADIENT_TOL
    assert second_order_solve.energy < start_energy


def test_the_second_order_solve_never_ends_above_its_start(orbital_rotation_case):
    """Far from the minimum the Hessian is indefinite; capped at one
    macro-iteration the solve must still hand back the lower orbitals its step
    reached, not its start."""
    mol, mo_coeff, *rest = orbital_rotation_case
    case = (mol, mo_coeff @ _random_rotation(mo_coeff.shape[1], 0.6, 5), *rest)
    start_energy, _ = _energy_and_gradient_at(case, case[1])

    with pytest.warns(UserWarning, match="without converging"):
        solve = ciah_orbital_solve(*case, gradient_tol=1e-12, max_iterations=1)

    assert solve.energy < start_energy


def test_the_second_order_solve_leaves_a_stationary_point_alone(
    orbital_rotation_case, converged_orbital_solve
):
    mol, _, *rest = orbital_rotation_case

    solve = capped_ciah_orbital_solve(
        mol, converged_orbital_solve.mo_coeff, *rest, gradient_tol=_GRADIENT_TOL
    )

    assert solve.converged
    assert solve.n_iterations == 0
    np.testing.assert_array_equal(solve.mo_coeff, converged_orbital_solve.mo_coeff)


def test_the_second_order_solve_counts_every_orbital_transform(
    orbital_rotation_case, mocker
):
    transforms = mocker.spy(_integrals_module, "_energy_fock_and_potentials")

    solve = capped_ciah_orbital_solve(
        *orbital_rotation_case, gradient_tol=_GRADIENT_TOL
    )

    assert solve.n_evaluations == transforms.call_count


def test_a_capped_second_order_solve_builds_no_step_it_discards(
    orbital_rotation_case, mocker
):
    hessian_builds = mocker.spy(_integrals_module._LASOrbitalHessian, "gen_g_hop")

    with pytest.warns(UserWarning, match="without converging"):
        ciah_orbital_solve(*orbital_rotation_case, gradient_tol=1e-12, max_iterations=1)

    assert hessian_builds.call_count == 1


def _scripted_steps(mocker, *scripts):
    """Replace pyscf's step generator; each call yields the next script's steps."""
    remaining = iter(scripts)

    def rotate_orb_cc(hessian, u0, *args, **kwargs):
        for step in next(remaining):
            yield step, None, None

    return mocker.patch.object(
        _integrals_module.ciah, "rotate_orb_cc", side_effect=rotate_orb_cc
    )


def test_a_second_order_solve_with_no_step_left_reports_a_stall(
    orbital_rotation_case, mocker
):
    identity = np.eye(orbital_rotation_case[1].shape[1])
    steps = _scripted_steps(mocker, [identity] * 3)

    with pytest.warns(UserWarning) as record:
        solve = capped_ciah_orbital_solve(
            *orbital_rotation_case, gradient_tol=_GRADIENT_TOL
        )

    _unconverged_warnings(record, solve, "stalled", _GRADIENT_TOL)
    assert solve.n_iterations == 0
    assert steps.call_count == 1


def test_a_zero_step_restarts_the_second_order_solve(orbital_rotation_case, mocker):
    """pyscf seeds each augmented-Hessian solve with the previous step, so a
    vanishing step would repeat forever; a fresh solve seeds from the gradient."""
    n_orbitals = orbital_rotation_case[1].shape[1]
    step = _random_rotation(n_orbitals, 1e-3, 3)
    steps = _scripted_steps(mocker, [step, np.eye(n_orbitals)], [step])

    with pytest.warns(UserWarning, match="without converging"):
        solve = ciah_orbital_solve(
            *orbital_rotation_case, gradient_tol=1e-12, max_iterations=2
        )

    assert solve.n_iterations == 2
    assert steps.call_count == 2


def test_the_second_order_solve_prefers_a_converged_point_level_with_its_best(
    orbital_rotation_case, mocker
):
    """A converged point that rounding leaves a hair above an earlier,
    unconverged one is the better answer."""
    n_orbitals = orbital_rotation_case[1].shape[1]
    n_pairs = len(rotation_energy_gradient_fn(*orbital_rotation_case)[0])
    _scripted_steps(mocker, [_random_rotation(n_orbitals, 1e-3, 3)] * 2)
    mocker.patch.object(
        _integrals_module._LASOrbitalHessian,
        "energy_and_gradient_at",
        side_effect=[
            (0.0, np.ones(n_pairs)),
            (-1.0, np.ones(n_pairs)),
            (-1.0 + 1e-12, np.zeros(n_pairs)),
        ],
    )

    solve = capped_ciah_orbital_solve(
        *orbital_rotation_case, gradient_tol=_GRADIENT_TOL
    )

    assert solve.converged
    assert solve.n_iterations == 2


def test_the_second_order_solve_keeps_the_lower_of_two_unconverged_points(
    orbital_rotation_case, mocker
):
    n_orbitals = orbital_rotation_case[1].shape[1]
    n_pairs = len(rotation_energy_gradient_fn(*orbital_rotation_case)[0])
    _scripted_steps(mocker, [_random_rotation(n_orbitals, 1e-3, 3)] * 2)
    mocker.patch.object(
        _integrals_module._LASOrbitalHessian,
        "energy_and_gradient_at",
        side_effect=[
            (0.0, np.ones(n_pairs)),
            (-1.0, np.ones(n_pairs)),
            (-1.0 + 1e-12, np.ones(n_pairs)),
        ],
    )

    with pytest.warns(UserWarning, match="without converging"):
        solve = ciah_orbital_solve(
            *orbital_rotation_case, gradient_tol=_GRADIENT_TOL, max_iterations=2
        )

    assert solve.energy == -1.0


def test_an_optimizer_stop_above_the_gradient_tolerance_is_not_converged(
    orbital_rotation_case, converged_orbital_solve
):
    """L-BFGS-B also reports success when the relative energy change stalls.
    Convergence is the gradient norm against the caller's tolerance, not
    scipy's status, so a tolerance below L-BFGS-B's reach is not met."""
    tolerance = 1e-6 * converged_orbital_solve.gradient_norm

    with pytest.warns(UserWarning, match="orbital-gradient norm"):
        solve = optimize_orbitals(*orbital_rotation_case, gradient_tol=tolerance)

    assert solve.converged is False
    assert solve.gradient_norm > tolerance


def test_polarized_neighbour_splits_the_embedding_by_spin(h4_chain_mean_field):
    """A spin-polarized neighbour must give the two spin channels different
    one-body potentials, matching AO-basis Coulomb/exchange builds.

    The oracle contracts the neighbour's alpha and beta densities in the AO
    basis via ``dot_eri_dm``, independent of the source's MO-basis einsum. This
    is the term a spin-averaged embedding erases: without it each fragment's
    solver cannot see the sign of its neighbour's local moment, which is what
    generates inter-fragment magnetic coupling.
    """
    mean_field = h4_chain_mean_field
    mol = mean_field.mol
    mo_coeff = np.asarray(mean_field.mo_coeff)
    ao_eri = cached_ao_eri(mol)
    integrals = transform_integrals(mol, mo_coeff, n_core=0, n_act=4, ao_eri=ao_eri)

    alpha_other = np.diag([0.9, 0.1])
    beta_other = np.diag([0.2, 0.6])
    states = [
        _idle_fragment((0, 1)),
        FragmentState(
            spec=FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
            rdm1=alpha_other + beta_other,
            rdm2=np.zeros((2,) * 4),
            rdm1_alpha=alpha_other,
            rdm1_beta=beta_other,
        ),
    ]

    h_alpha, h_beta, _ = fragment_effective_integrals(integrals, states, 0)

    target = mo_coeff[:, :2]
    other = mo_coeff[:, 2:4]
    dm_alpha = other @ alpha_other @ other.T
    dm_beta = other @ beta_other @ other.T
    vj_alpha, vk_alpha = scf.hf.dot_eri_dm(ao_eri, dm_alpha, hermi=1)
    vj_beta, vk_beta = scf.hf.dot_eri_dm(ao_eri, dm_beta, hermi=1)
    bare = target.T @ mean_field.get_hcore() @ target
    coulomb = target.T @ (vj_alpha + vj_beta) @ target
    expected_alpha = bare + coulomb - target.T @ vk_alpha @ target
    expected_beta = bare + coulomb - target.T @ vk_beta @ target

    np.testing.assert_allclose(h_alpha, expected_alpha, atol=1e-10)
    np.testing.assert_allclose(h_beta, expected_beta, atol=1e-10)
    # The channels genuinely differ, so a spin-averaged embedding would fail.
    assert np.abs(h_alpha - h_beta).max() > 1e-3
    # Their average is the spin-traced potential the previous version returned.
    np.testing.assert_allclose(
        0.5 * (h_alpha + h_beta),
        bare + coulomb - 0.5 * target.T @ (vk_alpha + vk_beta) @ target,
        atol=1e-10,
    )


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
    """A spin-polarised pair exchanges only within a spin channel: an all-alpha
    and an all-beta fragment have no same-spin overlap to exchange at all."""
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


def test_total_energy_matches_fci_for_full_active_space(h4_chain_mean_field):
    """With n_core=0 and the exact FCI RDMs, _total_energy must equal the FCI energy."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    one_body, two_body, n_orb, constant = mo_integrals(h4_chain_mean_field)

    n_alpha, n_beta = 2, 2
    electronic_energy, civec = fci.direct_spin1.kernel(
        one_body, two_body, n_orb, (n_alpha, n_beta)
    )
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, n_orb, (n_alpha, n_beta))
    expected = electronic_energy + constant

    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    energy = _total_energy(mol, mo_coeff, 0, rdm1, rdm2, ao_eri, h_ao)

    assert energy == pytest.approx(expected, abs=1e-10)


def test_fragment_effective_integrals_matches_casci_h1eff_with_frozen_core(
    h4_chain_mean_field,
):
    """With a real n_core=1 frozen core, h_eff must equal CASCI's active Fock."""
    mean_field = h4_chain_mean_field
    mo_coeff = np.asarray(mean_field.mo_coeff)

    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.mo_coeff = mo_coeff
    h1eff, _ = mc.get_h1eff()

    integrals = transform_integrals(mean_field.mol, mo_coeff, n_core=1, n_act=2)

    h_eff, _, _ = fragment_effective_integrals(integrals, [_idle_fragment((1, 2))], 0)

    np.testing.assert_allclose(h_eff, h1eff, atol=1e-10)


def test_total_energy_matches_casci_with_frozen_core(h4_chain_mean_field):
    """_total_energy must reproduce a CASCI total energy through a real frozen core."""
    mol = h4_chain_mean_field.mol

    mc = mcscf.CASCI(h4_chain_mean_field, 2, 2)
    mc.kernel()

    mo_coeff = np.asarray(mc.mo_coeff)
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)

    energy = _total_energy(mol, mo_coeff, mc.ncore, rdm1, rdm2, ao_eri, h_ao)

    assert energy == pytest.approx(mc.e_tot, abs=1e-8)


def test_fragment_effective_integrals_honors_noncontiguous_permutation(
    h4_chain_mean_field,
):
    """A non-identity, non-contiguous permutation must still index the caller's
    orbitals.

    The oracle here reproduces the source's contraction, so it pins the
    *indexing* and not the coefficients; those are pinned independently against
    PySCF in ``test_two_fragment_effective_integrals_match_pyscfs_embedding_potential``.
    """
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)

    specs = [
        FragmentSpec(orbitals=(0, 3), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1),
    ]
    permutation = build_active_permutation(specs, n_core=0, n_orbitals_total=4)
    permuted_mo_coeff = mo_coeff[:, permutation]
    integrals = transform_integrals(mol, permuted_mo_coeff, n_core=0, n_act=4)

    rdm_other = np.array([[1.3, 0.2], [0.2, 0.7]])
    states = [
        _idle_fragment(specs[0].orbitals),
        FragmentState(spec=specs[1], rdm1=rdm_other, rdm2=np.zeros((2,) * 4)),
    ]

    h_eff, _, g_frag = fragment_effective_integrals(integrals, states, 0)

    # Oracle built directly from the unpermuted MO integrals, indexed at the
    # caller's requested orbitals (0, 3) and (1, 2), not at positions 0..3.
    unpermuted = transform_integrals(mol, mo_coeff, n_core=0, n_act=4)
    h_mo, g_mo = unpermuted.h_act, unpermuted.g_act
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


def test_fragment_effective_integrals_rejects_mismatched_active_space():
    """Fragments covering a different orbital count than the integrals span
    means the two were built from different specs -- which would otherwise
    silently read the wrong block."""
    integrals = MOIntegrals(
        h_act=np.zeros((2, 2)),
        g_act=np.zeros((2, 2, 2, 2)),
        j_core=np.zeros((2, 2)),
        k_core=np.zeros((2, 2)),
    )
    states = [_idle_fragment((0, 1))] * 2

    with pytest.raises(
        ValueError,
        match=exact_match(
            "Fragments cover 4 orbitals but the active-space integrals span 2. "
            "`integrals` and `fragments` were built from different fragment specs."
        ),
    ):
        fragment_effective_integrals(integrals, states, 0)


def _diagonal_active_rdms(n_act):
    """A simple idempotent-like diagonal RDM guess, for optimize_orbitals tests
    that only need a well-defined energy functional, not a physical one."""
    rdm1_active = np.eye(n_act)
    rdm2_active = np.zeros((n_act, n_act, n_act, n_act))
    for p in range(n_act):
        for q in range(n_act):
            rdm2_active[p, p, q, q] = rdm1_active[p, p] * rdm1_active[q, q]
    return rdm1_active, rdm2_active


def test_an_orbital_solve_spans_all_four_rotation_categories(
    h4_chain_mean_field, orbital_solver
):
    """A frozen core, two single-orbital fragments and leftover virtuals give
    every rotation category; the rotated orbitals stay orthonormal under the AO
    overlap."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
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

    solve = orbital_solver(
        mol,
        permuted_mo_coeff,
        n_core,
        specs,
        rdm1_active,
        rdm2_active,
        ao_eri,
        h_ao,
        gradient_tol=_GRADIENT_TOL,
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
    assert solve.n_rotation_pairs == expected_n_rot == 6

    overlap = mol.intor("int1e_ovlp")
    gram = solve.mo_coeff.T @ overlap @ solve.mo_coeff
    np.testing.assert_allclose(gram, np.eye(n_orb_total), atol=1e-10)


def test_an_orbital_solve_reports_the_real_energy_with_no_rotation_freedom(
    h2_mean_field, orbital_solver
):
    """One fragment spanning every orbital leaves zero rotation pairs; the solve
    reports the energy of the unrotated orbitals, not a spurious ``0.0``."""
    mol = h2_mean_field.mol
    mo_coeff = np.asarray(h2_mean_field.mo_coeff)

    energy_fci, civec = fci.FCI(h2_mean_field).kernel()
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, 2, (1, 1))
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    solve = orbital_solver(
        mol, mo_coeff, 0, [spec], rdm1, rdm2, ao_eri, h_ao, gradient_tol=_GRADIENT_TOL
    )

    unrotated_energy = _total_energy(mol, mo_coeff, 0, rdm1, rdm2, ao_eri, h_ao)
    assert solve.energy == pytest.approx(unrotated_energy)
    assert solve.energy == pytest.approx(energy_fci, abs=1e-10)
    np.testing.assert_allclose(solve.mo_coeff, mo_coeff, atol=1e-12)
    assert solve.n_rotation_pairs == 0
    assert solve.n_iterations == 0
    assert solve.converged is True
    assert solve.n_evaluations == 1
    assert solve.gradient_norm == 0.0


@pytest.mark.filterwarnings("ignore:Orbital optimisation ended without converging")
@pytest.mark.parametrize(
    "energy_offset, accepted",
    [(1.0, False), (0.0, False), (-1.0, True)],
    ids=["worse", "tie", "better"],
)
def test_optimize_orbitals_keeps_scipy_result_only_when_strictly_lower(
    mocker, h4_chain_mean_field, energy_offset, accepted
):
    """``minimize``'s result replaces the zero-rotation baseline only when its
    energy is strictly lower, keeping the routine monotone regardless of what
    scipy reports; the gradient and orbitals follow whichever point is kept."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    rdm1_active, rdm2_active = _diagonal_active_rdms(4)
    ao_eri = cached_ao_eri(mol)
    h_ao = cached_h_ao(mol)
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
    ]

    rotation_pairs, energy_and_gradient = rotation_energy_gradient_fn(
        mol, mo_coeff, 0, specs, rdm1_active, rdm2_active, ao_eri, h_ao
    )
    baseline_energy, baseline_gradient = energy_and_gradient(
        np.zeros(len(rotation_pairs))
    )

    fake_result = scipy.optimize.OptimizeResult(
        fun=baseline_energy + energy_offset,
        x=np.array([0.5, -0.2, 0.1, 0.3]),
        jac=np.array([0.1, -7.0, 0.2, 0.3]),
        nit=3,
        nfev=4,
        success=True,
        message="",
    )
    no_progress = scipy.optimize.OptimizeResult(
        fun=np.inf, x=np.zeros(4), nit=0, nfev=fake_result.nfev, message=""
    )
    # A restart from the accepted orbitals finds nothing lower, ending the solve.
    minimize = mocker.patch.object(
        _integrals_module, "minimize", side_effect=[fake_result, no_progress]
    )

    solve = optimize_orbitals(
        mol,
        mo_coeff,
        0,
        specs,
        rdm1_active,
        rdm2_active,
        ao_eri,
        h_ao,
        gradient_tol=0.0,
    )

    spent = 1 + fake_result.nfev * minimize.call_count
    if accepted:
        rotated = _rotated_mo_coeff(mo_coeff, rotation_pairs, fake_result.x)
        rotated_energy, local_gradient = _energy_and_gradient_at(
            (mol, mo_coeff, 0, specs, rdm1_active, rdm2_active, ao_eri, h_ao), rotated
        )
        assert minimize.call_count == 2
        assert solve.n_evaluations == spent + 1
        assert solve.energy == pytest.approx(rotated_energy)
        assert solve.gradient_norm == pytest.approx(np.linalg.norm(local_gradient))
        np.testing.assert_allclose(solve.mo_coeff, rotated, atol=1e-12)
    else:
        assert minimize.call_count == 1
        assert solve.n_evaluations == spent
        assert solve.energy == baseline_energy
        assert solve.gradient_norm == pytest.approx(np.linalg.norm(baseline_gradient))
        np.testing.assert_allclose(solve.mo_coeff, mo_coeff, atol=1e-12)


def test_transform_integrals_reuses_a_supplied_ao_eri(mocker, h4_chain_mean_field):
    """A supplied AO ERI is used as-is, giving the same integrals as building it."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    built = transform_integrals(mol, mo_coeff, n_core=1, n_act=2)

    ao_eri = cached_ao_eri(mol)
    spy = mocker.spy(_integrals_module, "cached_ao_eri")
    reused = transform_integrals(mol, mo_coeff, n_core=1, n_act=2, ao_eri=ao_eri)

    spy.assert_not_called()
    for name in ("h_act", "g_act", "j_core", "k_core"):
        np.testing.assert_array_equal(getattr(reused, name), getattr(built, name))
    np.testing.assert_allclose(
        cached_h_ao(mol), h4_chain_mean_field.get_hcore(), atol=1e-14
    )


def test_cached_h_ao_includes_the_effective_core_potential():
    """An ECP belongs to the one-electron Hamiltonian; leaving it out shifts a
    heavy atom's energy by tens of Hartree while every other term stays
    consistent."""
    mol = gto.M(
        atom="I 0 0 0; H 0 0 1.61",
        basis="def2-svp",
        ecp={"I": "def2-svp"},
        verbose=0,
    )
    expected = mol.intor("int1e_kin") + mol.intor("int1e_nuc") + mol.intor("ECPscalar")

    np.testing.assert_allclose(cached_h_ao(mol), expected, atol=1e-12)


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
    gradient is derived from must reproduce ``_total_energy``'s explicit loops
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
    expected = _total_energy(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )

    assert reconstructed == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("at_origin", [True, False])
def test_rotation_gradient_matches_central_differences(
    orbital_rotation_case, at_origin
):
    """The analytic gradient must match central differences of ``_total_energy``
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
        return _total_energy(
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


def test_optimize_orbitals_improves_strictly_on_the_baseline(
    orbital_rotation_case, converged_orbital_solve
):
    """The routine returns ``min(baseline, minimize_result)``, so an
    inverted-sign gradient would silently return the unrotated orbitals at the
    baseline energy. Assert a strict improvement, and that the reported energy
    is the one the returned coefficients actually produce."""
    mol, mo_coeff, n_core, _, rdm1_active, rdm2_active, ao_eri, h_ao = (
        orbital_rotation_case
    )
    baseline_energy = _total_energy(
        mol, mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
    )
    solve = converged_orbital_solve

    assert solve.energy < baseline_energy - 1e-8
    assert solve.energy == pytest.approx(
        _total_energy(
            mol, solve.mo_coeff, n_core, rdm1_active, rdm2_active, ao_eri, h_ao
        ),
        abs=1e-8,
    )

    overlap = mol.intor("int1e_ovlp")
    gram = solve.mo_coeff.T @ overlap @ solve.mo_coeff
    np.testing.assert_allclose(gram, np.eye(mo_coeff.shape[1]), atol=1e-10)


def test_orbital_minimize_options_reach_a_small_gradient(converged_orbital_solve):
    """A ``tol=1e-6`` shorthand halts with the gradient still around 2e-2,
    since ``ftol`` is relative to ``|E|``. ``gradient_norm`` is ``minimize``'s
    own Jacobian: the rotation-pair set is not closed under the Fréchet
    pullback, so the local gradient at the returned coefficients is a different
    quantity.
    """
    assert converged_orbital_solve.converged is True
    assert converged_orbital_solve.gradient_norm < 1e-3


def test_an_orbital_solve_rotates_a_single_pair(h2_mean_field, orbital_solver):
    """A doubly occupied one-orbital fragment against one virtual leaves a single
    rotation pair; from rotated orbitals the solve returns to the mean field."""
    mol = h2_mean_field.mol
    mixing = scipy.linalg.expm(np.array([[0.0, 0.3], [-0.3, 0.0]]))
    mo_coeff = np.asarray(h2_mean_field.mo_coeff) @ mixing
    spec = FragmentSpec(orbitals=(0,), n_alpha=1, n_beta=1)

    solve = orbital_solver(
        mol,
        mo_coeff,
        0,
        [spec],
        np.array([[2.0]]),
        np.full((1, 1, 1, 1), 2.0),
        cached_ao_eri(mol),
        cached_h_ao(mol),
        gradient_tol=_GRADIENT_TOL,
    )

    assert solve.n_rotation_pairs == 1
    assert solve.converged is True
    assert solve.energy == pytest.approx(h2_mean_field.e_tot, abs=1e-8)


def test_optimize_orbitals_spends_one_iteration_budget_across_restarts(
    orbital_rotation_case, mocker
):
    """A run that lowers the energy restarts; the restart gets only the
    iterations the first run left over."""
    budgets = []
    # Lower than any real energy, then no progress, ending the solve.
    final_energies = iter([-1e3, np.inf])

    def two_iterations(fun, x0, *, method, jac, options, callback):
        budgets.append(options["maxiter"])
        for _ in range(2):
            callback(scipy.optimize.OptimizeResult(fun=0.0))
        return scipy.optimize.OptimizeResult(
            fun=next(final_energies), x=np.zeros_like(x0), nfev=1, message=""
        )

    mocker.patch.object(_integrals_module, "minimize", side_effect=two_iterations)

    with pytest.warns(UserWarning, match="without converging"):
        solve = optimize_orbitals(
            *orbital_rotation_case, gradient_tol=1e-12, max_iterations=5
        )

    assert budgets == [5, 3]
    assert solve.n_iterations == 4


@pytest.fixture
def orbital_hessian(orbital_rotation_case):
    return _integrals_module._LASOrbitalHessian(*orbital_rotation_case)


def test_the_orbital_hessian_packs_what_it_unpacks(orbital_hessian):
    angles = np.random.default_rng(7).standard_normal(orbital_hessian.pdim)

    packed = orbital_hessian.pack_uniq_var(orbital_hessian.unpack_uniq_var(angles))

    np.testing.assert_array_equal(packed, angles)


def test_the_orbital_hessian_evaluates_each_rotation_once(orbital_hessian):
    rotation = np.eye(orbital_hessian.norb)

    orbital_hessian.get_grad(rotation)
    orbital_hessian.gen_g_hop(rotation)

    assert orbital_hessian.evaluations == 1


def test_the_hessian_vector_product_is_linear(orbital_hessian):
    """Zero maps to zero, a unit vector does not, and scaling commutes."""
    pdim = orbital_hessian.pdim
    _, hessian_vector, _ = orbital_hessian.gen_g_hop(np.eye(orbital_hessian.norb))
    unit = np.zeros(pdim)
    unit[0] = 1.0
    vector = np.random.default_rng(11).standard_normal(pdim)

    at_zero = hessian_vector(np.zeros(pdim))

    assert at_zero.shape == (pdim,)
    assert not at_zero.any()
    assert np.linalg.norm(hessian_vector(unit)) > 0.0
    np.testing.assert_allclose(
        hessian_vector(2.0 * vector), 2.0 * hessian_vector(vector), rtol=1e-9
    )
