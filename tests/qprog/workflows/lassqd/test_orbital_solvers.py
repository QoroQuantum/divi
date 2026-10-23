# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's orbital solvers: L-BFGS-B and second-order solves, the
orbital Hessian, rotation gradients and saddle escape."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from pyscf import fci, gto, scf

from divi.qprog.workflows.lassqd import _integrals as _integrals_module
from divi.qprog.workflows.lassqd import _orbital_solvers as _orbital_solvers_module
from divi.qprog.workflows.lassqd._config import (
    LBFGSOrbitalUpdate,
    SecondOrderOrbitalUpdate,
)
from divi.qprog.workflows.lassqd._integrals import (
    AOIntegrals,
    _total_energy,
    assemble_active_rdms,
    build_active_permutation,
)
from divi.qprog.workflows.lassqd._orbital_solvers import (
    ciah_orbital_solve,
    escape_orbital_saddle,
    optimize_orbitals,
    rotation_energy_gradient_fn,
)
from divi.qprog.workflows.lassqd._state import (
    FragmentSpec,
    FragmentState,
    _OrbitalRound,
)
from divi.qprog.workflows.lassqd._workflow import _diagonal_rdm_guess
from tests.qprog.workflows.lassqd._helpers import (  # noqa: F401
    capped_ciah_orbital_solve,
    h2_mean_field,
    h4_chain_mean_field,
    orbital_rotation_case,
    orbital_solver,
)

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
        "Orbital optimisation stopped after "
        f"{solve.n_iterations} iterations and {solve.n_evaluations} evaluations "
        f"({reason}) with orbital-gradient norm {solve.gradient_norm:.2e} "
        f"above {gradient_tol:.2e}. The energy remains an upper bound, but "
        "this round is not a stationary point, so a small energy change from "
        "the last round may mean the optimiser stopped rather than that the "
        "macro-cycle converged."
    )
    assert [str(warning.message) for warning in record] == [expected]


def _energy_and_gradient_at(case, mo_coeff):
    """Energy and orbital gradient of ``case``'s functional at ``mo_coeff``."""
    ao, _, *rest = case
    pairs, energy_and_gradient = rotation_energy_gradient_fn(ao, mo_coeff, *rest)
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


def _round_solve(update):
    """``update``'s solve on a round carrying the case's arguments."""

    def solve(*case, gradient_tol):
        return update._solve(_OrbitalRound(*case, gradient_tol=gradient_tol))[0]

    return solve


@pytest.mark.parametrize(
    "capped_solve",
    [
        partial(optimize_orbitals, max_iterations=1),
        partial(ciah_orbital_solve, max_iterations=1),
        _round_solve(LBFGSOrbitalUpdate(max_iterations=1)),
        _round_solve(SecondOrderOrbitalUpdate(max_iterations=1)),
    ],
    ids=["l-bfgs-b", "ciah", "full-orbital-solve", "second-order-orbital-solve"],
)
def test_the_non_convergence_warning_points_at_the_caller(
    orbital_rotation_case, capped_solve
):
    with pytest.warns(UserWarning, match="stopped after") as record:
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
    ao, mo_coeff, *rest = orbital_rotation_case
    case = (ao, mo_coeff @ _random_rotation(mo_coeff.shape[1], 0.6, 5), *rest)
    start_energy, _ = _energy_and_gradient_at(case, case[1])

    with pytest.warns(UserWarning, match="stopped after"):
        solve = ciah_orbital_solve(*case, gradient_tol=1e-12, max_iterations=1)

    assert solve.energy < start_energy


def test_the_second_order_solve_takes_one_tight_step_from_within_tolerance(
    orbital_rotation_case, converged_orbital_solve
):
    """A gradient already under the tolerance can still sit on a long, shallow
    descent, so the solve steps once rather than stopping where it starts."""
    ao, _, *rest = orbital_rotation_case

    solve = capped_ciah_orbital_solve(
        ao, converged_orbital_solve.mo_coeff, *rest, gradient_tol=_GRADIENT_TOL
    )

    assert solve.converged
    assert solve.n_iterations == 1
    assert solve.energy <= converged_orbital_solve.energy
    assert solve.gradient_norm < converged_orbital_solve.gradient_norm


def test_the_second_order_solve_counts_every_orbital_transform(
    orbital_rotation_case, mocker
):
    moves = mocker.spy(_orbital_solvers_module, "_energy_fock_and_potentials")
    products = mocker.spy(_integrals_module, "_energy_fock_and_potentials")

    solve = capped_ciah_orbital_solve(
        *orbital_rotation_case, gradient_tol=_GRADIENT_TOL
    )

    assert solve.n_evaluations == moves.call_count + products.call_count


def test_a_capped_second_order_solve_builds_no_step_it_discards(
    orbital_rotation_case, mocker
):
    hessian_builds = mocker.spy(_orbital_solvers_module._LASOrbitalHessian, "gen_g_hop")

    with pytest.warns(UserWarning, match="stopped after"):
        ciah_orbital_solve(*orbital_rotation_case, gradient_tol=1e-12, max_iterations=1)

    assert hessian_builds.call_count == 1


def _scripted_steps(mocker, *scripts):
    """Replace pyscf's step generator; each call yields the next script's steps."""
    remaining = iter(scripts)

    def rotate_orb_cc(hessian, u0, *args, **kwargs):
        for step in next(remaining):
            yield step, None, None

    return mocker.patch.object(
        _orbital_solvers_module.ciah, "rotate_orb_cc", side_effect=rotate_orb_cc
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


def _gradient_step(case, scale):
    """The rotation ``scale`` times the orbital gradient at ``case``'s orbitals:
    downhill when negative, uphill when positive."""
    hessian = _orbital_solvers_module._LASOrbitalHessian(*case)
    _, gradient = hessian.energy_and_gradient_at(np.eye(case[1].shape[1]))
    return scipy.linalg.expm(hessian.unpack_uniq_var(scale * gradient))


def test_a_zero_step_restarts_the_second_order_solve(orbital_rotation_case, mocker):
    """pyscf seeds each augmented-Hessian solve with the previous step, so a
    vanishing step would repeat forever; a fresh solve seeds from the gradient."""
    n_orbitals = orbital_rotation_case[1].shape[1]
    step = _gradient_step(orbital_rotation_case, -1e-4)
    steps = _scripted_steps(mocker, [step, np.eye(n_orbitals)], [step])

    with pytest.warns(UserWarning, match="stopped after"):
        solve = ciah_orbital_solve(
            *orbital_rotation_case, gradient_tol=1e-12, max_iterations=2
        )

    assert solve.n_iterations == 2
    assert steps.call_count == 2


def test_an_uphill_second_order_step_hands_over_to_the_descent_solve(
    orbital_rotation_case, mocker
):
    """An augmented-Hessian step can climb towards a saddle; the rest of the
    budget goes to L-BFGS-B, which only descends."""
    _scripted_steps(mocker, [_gradient_step(orbital_rotation_case, 1e-3)])
    descent = mocker.spy(_orbital_solvers_module, "optimize_orbitals")
    ao, mo_coeff, n_core, _, rdm1, rdm2 = orbital_rotation_case
    start = _total_energy(ao, mo_coeff, n_core, rdm1, rdm2)

    # The capped budget leaves L-BFGS-B unconverged.
    with pytest.warns(UserWarning, match="stopped after"):
        solve = capped_ciah_orbital_solve(
            *orbital_rotation_case, gradient_tol=_GRADIENT_TOL
        )

    assert descent.call_count == 1
    assert solve.energy < start


def test_the_second_order_solve_prefers_a_converged_point_level_with_its_best(
    orbital_rotation_case, mocker
):
    """A converged point that rounding leaves a hair above an earlier,
    unconverged one is the better answer."""
    n_orbitals = orbital_rotation_case[1].shape[1]
    n_pairs = len(rotation_energy_gradient_fn(*orbital_rotation_case)[0])
    _scripted_steps(mocker, [_random_rotation(n_orbitals, 1e-3, 3)] * 2)
    mocker.patch.object(
        _orbital_solvers_module._LASOrbitalHessian,
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
        _orbital_solvers_module._LASOrbitalHessian,
        "energy_and_gradient_at",
        side_effect=[
            (0.0, np.ones(n_pairs)),
            (-1.0, np.ones(n_pairs)),
            (-1.0 + 1e-12, np.ones(n_pairs)),
        ],
    )

    with pytest.warns(UserWarning, match="stopped after"):
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

    solve = orbital_solver(
        AOIntegrals.of(mol),
        permuted_mo_coeff,
        n_core,
        specs,
        rdm1_active,
        rdm2_active,
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
    ao = AOIntegrals.of(mol)
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    solve = orbital_solver(
        ao, mo_coeff, 0, [spec], rdm1, rdm2, gradient_tol=_GRADIENT_TOL
    )

    unrotated_energy = _total_energy(ao, mo_coeff, 0, rdm1, rdm2)
    assert solve.energy == pytest.approx(unrotated_energy)
    assert solve.energy == pytest.approx(energy_fci, abs=1e-10)
    np.testing.assert_allclose(solve.mo_coeff, mo_coeff, atol=1e-12)
    assert solve.n_rotation_pairs == 0
    assert solve.n_iterations == 0
    assert solve.converged is True
    assert solve.n_evaluations == 1
    assert solve.gradient_norm == 0.0


@pytest.mark.filterwarnings("ignore:Orbital optimisation stopped after")
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
    ao = AOIntegrals.of(mol)
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
    ]

    rotation_pairs, energy_and_gradient = rotation_energy_gradient_fn(
        ao, mo_coeff, 0, specs, rdm1_active, rdm2_active
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
        _orbital_solvers_module, "minimize", side_effect=[fake_result, no_progress]
    )

    solve = optimize_orbitals(
        ao,
        mo_coeff,
        0,
        specs,
        rdm1_active,
        rdm2_active,
        gradient_tol=0.0,
    )

    spent = 1 + fake_result.nfev * minimize.call_count
    if accepted:
        rotated = _rotated_mo_coeff(mo_coeff, rotation_pairs, fake_result.x)
        rotated_energy, local_gradient = _energy_and_gradient_at(
            (ao, mo_coeff, 0, specs, rdm1_active, rdm2_active), rotated
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


def _rotated_mo_coeff(mo_coeff, rotation_pairs, rotation_params):
    """``mo_coeff`` under the skew generator the rotation angles parameterize."""
    n_orb = mo_coeff.shape[1]
    generator = np.zeros((n_orb, n_orb))
    for idx, (p, q) in enumerate(rotation_pairs):
        generator[p, q] = rotation_params[idx]
        generator[q, p] = -rotation_params[idx]
    return mo_coeff @ scipy.linalg.expm(generator)


@pytest.mark.parametrize("at_origin", [True, False])
def test_rotation_gradient_matches_central_differences(
    orbital_rotation_case, at_origin
):
    """The analytic gradient must match central differences of ``_total_energy``
    itself, both at zero rotation and away from it. Away from zero is the
    discriminating case: the derivative of the matrix exponential is a Frechet
    pullback, and the naive commutator form it is easily confused with agrees
    with it only at the origin."""
    ao, mo_coeff, n_core, _, rdm1_active, rdm2_active = orbital_rotation_case
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
        return _total_energy(ao, rotated, n_core, rdm1_active, rdm2_active)

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
    ao, mo_coeff, n_core, _, rdm1_active, rdm2_active = orbital_rotation_case
    baseline_energy = _total_energy(ao, mo_coeff, n_core, rdm1_active, rdm2_active)
    solve = converged_orbital_solve

    assert solve.energy < baseline_energy - 1e-8
    assert solve.energy == pytest.approx(
        _total_energy(ao, solve.mo_coeff, n_core, rdm1_active, rdm2_active),
        abs=1e-8,
    )

    overlap = ao.mol.intor("int1e_ovlp")
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
        AOIntegrals.of(mol),
        mo_coeff,
        0,
        [spec],
        np.array([[2.0]]),
        np.full((1, 1, 1, 1), 2.0),
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

    mocker.patch.object(_orbital_solvers_module, "minimize", side_effect=two_iterations)

    with pytest.warns(UserWarning, match="stopped after"):
        solve = optimize_orbitals(
            *orbital_rotation_case, gradient_tol=1e-12, max_iterations=5
        )

    assert budgets == [5, 3]
    assert solve.n_iterations == 4


@pytest.fixture
def orbital_hessian(orbital_rotation_case):
    return _orbital_solvers_module._LASOrbitalHessian(*orbital_rotation_case)


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


def _h4_split_by_occupation():
    """``optimize_orbitals`` arguments for H4 with the occupied and the virtual
    orbitals in separate fragments: a doubly excited reference, whose orbital
    Hessian has a negative mode."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 2.5; H 0 0 3.24", basis="sto-3g", verbose=0
    )
    specs = [FragmentSpec((0, 1), 1, 1), FragmentSpec((2, 3), 1, 1)]
    fragments = []
    for spec in specs:
        rdm1, rdm2, rdm1_alpha, rdm1_beta = _diagonal_rdm_guess(spec)
        fragments.append(
            FragmentState(
                spec=spec,
                rdm1=rdm1,
                rdm2=rdm2,
                rdm1_alpha=rdm1_alpha,
                rdm1_beta=rdm1_beta,
            )
        )
    return (
        AOIntegrals.of(mol),
        np.asarray(scf.RHF(mol).run().mo_coeff),
        0,
        specs,
        *assemble_active_rdms(fragments),
    )


def test_a_negative_orbital_mode_is_escaped():
    case = _h4_split_by_occupation()
    ao, mo_coeff, n_core, _, rdm1, rdm2 = case
    start = _total_energy(ao, mo_coeff, n_core, rdm1, rdm2)

    escape, conclusive = escape_orbital_saddle(*case, min_energy_drop=1e-5)

    assert conclusive
    assert escape is not None
    escaped_mo_coeff, energy = escape
    assert energy < start - 1e-5
    assert energy == pytest.approx(
        _total_energy(ao, escaped_mo_coeff, n_core, rdm1, rdm2),
        abs=1e-10,
    )


def test_a_minimum_is_not_escaped(orbital_rotation_case):
    ao, _, *rest = orbital_rotation_case
    minimum = optimize_orbitals(*orbital_rotation_case, gradient_tol=1e-4)

    assert minimum.converged
    assert escape_orbital_saddle(ao, minimum.mo_coeff, *rest, min_energy_drop=1e-5) == (
        None,
        True,
    )


def test_an_unconverged_saddle_check_is_inconclusive(
    orbital_rotation_case, monkeypatch
):
    ao, _, *rest = orbital_rotation_case
    minimum = optimize_orbitals(*orbital_rotation_case, gradient_tol=1e-4)
    monkeypatch.setattr(
        "divi.qprog.workflows.lassqd._orbital_solvers._DAVIDSON_TOL", 0.0
    )

    assert escape_orbital_saddle(ao, minimum.mo_coeff, *rest, min_energy_drop=1e-5) == (
        None,
        False,
    )
