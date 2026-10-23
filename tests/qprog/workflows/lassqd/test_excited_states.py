# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's excited states: tracking, warnings, wavefunctions,
transitions, NEVPT2 and state-averaged orbitals."""

import warnings

import numpy as np
import pytest
from pyscf import fci, gto, mcscf, mrpt, scf
from pyscf.fci import cistring

from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows.lassqd import (
    ElectronicState,
    FixedOrbitals,
    LBFGSOrbitalUpdate,
    SecondOrderOrbitalUpdate,
    StateAveragedOrbitalUpdate,
    StateWavefunction,
    _config,
    _excited_states,
    _sqd,
    _workflow,
)
from divi.qprog.workflows.lassqd._integrals import AOIntegrals
from divi.qprog.workflows.lassqd._orbital_solvers import (
    orbital_irreps,
    state_averaged_orbital_solve,
)
from divi.qprog.workflows.lassqd._sqd import SQDResult, SQDState
from divi.qprog.workflows.lassqd._state import FragmentSpec
from divi.qprog.workflows.lassqd._subspace import state_transitions
from tests._helpers import exact_match
from tests.qprog.workflows.lassqd._helpers import (  # noqa: F401
    _ONE_DETERMINANT_WARNING,
    H4_WHOLE_SPACE,
    _dense_casci_states,
    _h2o_states_lassqd,
    _lassqd,
    assert_roots_match,
    build_exact_sampler_lassqd,
    exact_sampler_lassqd,
    exported_h2o_states,
    fci_roots_by_multiplicity,
    h2o_molecule,
    h4_chain_mean_field,
    h8_chain,
    h8_frontier_lassqd,
    mo_integrals,
    nh2_radical,
)


def test_states_take_at_most_two_fragments(dummy_expval_backend):
    message = (
        "SQDConfig.states takes one or two fragments; got 3. Put the active "
        "space in one FragmentSpec, or split it in two."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        _lassqd(
            dummy_expval_backend,
            problem=MolecularProblem.from_molecule(h8_chain()),
            active_spaces=[
                FragmentSpec(orbitals=(0, 4), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(1, 5), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(2, 3, 6, 7), n_alpha=2, n_beta=2),
            ],
            states={1: 2},
        )


def test_automatic_fragmentation_into_two_fragments_solves_each_for_its_lowest_state(
    dummy_expval_backend,
):
    """The requested states span both fragments, so each fragment's own SQD
    solve targets its lowest state, not them."""
    ensemble = h8_frontier_lassqd(
        dummy_expval_backend, states={1: 2}, orbital_update=FixedOrbitals()
    )

    ensemble.initial_state()

    assert ensemble._fragment_sqd().states is None


@pytest.mark.parametrize(
    "orbital_update", [SecondOrderOrbitalUpdate(), LBFGSOrbitalUpdate()]
)
def test_excited_states_on_ground_state_orbitals_warn_once(
    dummy_expval_backend, orbital_update
):
    message = (
        f"{type(orbital_update).__name__} optimises the orbitals for the lowest "
        "state only, so the others miss their orbital relaxation and gaps to "
        "them tend to come out too large. Use StateAveragedOrbitalUpdate, or FixedOrbitals "
        "to compare with CASCI on the same orbitals."
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _lassqd(
            dummy_expval_backend,
            active_spaces=[H4_WHOLE_SPACE],
            states={1: 2},
            orbital_update=orbital_update,
        ).initial_state()

    assert [str(warning.message) for warning in record] == [message]


@pytest.mark.parametrize("states", [{1: 1}, {3: 1}], ids=["ground", "triplet"])
def test_a_single_state_does_not_warn_about_orbital_bias(dummy_expval_backend, states):
    """Orbitals optimised for the one requested state are its own, whichever
    state that is."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _lassqd(dummy_expval_backend, active_spaces=[H4_WHOLE_SPACE], states=states)


def _states_lassqd(backend, mocker, states, **overrides):
    """Exact-sampled H4 in one fragment on fixed orbitals, solving for
    ``states``."""
    return build_exact_sampler_lassqd(
        backend,
        mocker,
        active_spaces=[H4_WHOLE_SPACE],
        states=states,
        orbital_update=FixedOrbitals(),
        **overrides,
    )


def test_requested_states_match_fci_on_a_complete_subspace(
    dummy_expval_backend, mocker, h4_chain_mean_field
):
    """Each state's total energy is the exact FCI root of its multiplicity once
    the subspace holds every determinant, and the orbitals do not move."""
    states = {1: 2, 3: 1}
    ensemble, state = _states_lassqd(
        dummy_expval_backend,
        mocker,
        states,
        lambda_penalty=1.0,
        n_batches=1,
        batch_size=64,
        n_recovery_iterations=1,
    )
    one_body, two_body, _, constant = mo_integrals(h4_chain_mean_field)
    expected = fci_roots_by_multiplicity(one_body, two_body, 2, 2, constant)

    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_state = ensemble.update_state(state)

    report = ensemble.round_reports[-1]
    assert report.subspace_sizes == (36,)
    np.testing.assert_array_equal(new_state.mo_coeff, state.mo_coeff)
    assert report.orbital_converged and np.isnan(report.orbital_gradient_norm)
    assert "orbitals: none rotated;" in report.summary()
    reported = ensemble.electronic_states
    assert list(reported.values()) == list(report.states)
    assert list(reported) == [(1, 0), (3, 0), (1, 1)]
    assert new_state.energy == report.energy
    assert reported[(1, 0)].energy == pytest.approx(report.energy, abs=1e-10)
    assert report.averaged_energy is None
    assert_roots_match(reported.values(), expected, states)


def test_a_root_of_another_spin_is_not_labelled_as_the_requested_one(
    dummy_expval_backend, mocker, h4_chain_mean_field
):
    """At the default penalty the singlet ground state lies below several
    triplets once penalised; on a complete subspace every reported triplet must
    still be one."""
    states = {3: 6}
    ensemble, state = _states_lassqd(
        dummy_expval_backend,
        mocker,
        states,
        n_batches=1,
        batch_size=64,
        n_recovery_iterations=1,
    )
    one_body, two_body, _, constant = mo_integrals(h4_chain_mean_field)
    expected = fci_roots_by_multiplicity(one_body, two_body, 2, 2, constant)

    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ensemble.update_state(state)

    assert ensemble.round_reports[-1].subspace_sizes == (36,)
    assert_roots_match(ensemble.electronic_states.values(), expected, states)


@pytest.mark.parametrize(
    "triplets, split", [(2, True), (3, False)], ids=["half-pair", "whole-pair"]
)
def test_requesting_part_of_a_degenerate_set_warns(
    dummy_expval_backend, mocker, triplets, split
):
    """N2's second and third triplets are a degenerate pair; state averaging
    over one of them breaks its symmetry."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.2", basis="6-31g", verbose=0)
    ensemble, state = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(scf.RHF(mol).run()),
        active_spaces=[FragmentSpec(orbitals=(4, 5, 6, 7, 8, 9), n_alpha=3, n_beta=3)],
        states={1: 1, 3: triplets},
        orbital_update=StateAveragedOrbitalUpdate(),
        n_batches=1,
        batch_size=64,
        n_recovery_iterations=1,
    )

    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ensemble.update_state(state)

    assert (
        any(
            text.startswith("State (3, 1) lies within")
            and "of (3, 2), which SQDConfig.states does not request" in text
            for text in map(str, (warning.message for warning in record))
        )
        is split
    )


def test_requested_states_are_eigenstates_at_the_rotated_orbitals(
    dummy_expval_backend, mocker, h4_chain_mean_field
):
    """The CI is solved before the orbital update moves the orbitals; every
    state must still be reported as an eigenstate at the orbitals the round
    returns, not its pre-rotation CI evaluated there."""
    states = {1: 2, 3: 1}
    with pytest.warns(UserWarning, match="optimises the orbitals for the lowest"):
        ensemble, state = build_exact_sampler_lassqd(
            dummy_expval_backend,
            mocker,
            active_spaces=[FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1)],
            states=states,
            lambda_penalty=1.0,
        )

    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    new_state = ensemble.update_state(state)

    assert not np.allclose(new_state.mo_coeff, state.mo_coeff)
    casci = mcscf.CASCI(h4_chain_mean_field, 2, (1, 1))
    casci.ncore = 1
    one_body, constant = casci.get_h1eff(new_state.mo_coeff)
    expected = fci_roots_by_multiplicity(
        one_body, casci.get_h2eff(new_state.mo_coeff), 1, 1, constant
    )
    assert_roots_match(ensemble.electronic_states.values(), expected, states)


@pytest.fixture
def unresolved_state_warnings(dummy_expval_backend, mocker):
    """The warnings a round raises whose one-determinant subspace holds two
    singlets, the second off its spin and degenerate with the first."""
    ensemble, state = _states_lassqd(
        dummy_expval_backend, mocker, {1: 2}, extension=None
    )
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    reference = "1100"
    mocker.patch(
        "divi.qprog.workflows.lassqd._workflow.SQDSolver.solve",
        return_value=SQDResult(
            states=tuple(
                SQDState(
                    multiplicity=1,
                    index=index,
                    penalised_energy=0.0,
                    spin_square=spin_square,
                    amplitudes=np.array([[1.0]]),
                )
                for index, spin_square in enumerate((0.0, 0.5))
            ),
            strings_alpha=(reference,),
            strings_beta=(reference,),
        ),
    )
    mocker.patch(
        "divi.qprog.workflows.lassqd._excited_states.subspace_states_at",
        return_value=(
            ElectronicState(multiplicity=1, index=0, energy=0.0, spin_square=0.0),
            ElectronicState(multiplicity=1, index=1, energy=0.0, spin_square=0.5),
        ),
    )

    with pytest.warns(UserWarning) as record:
        ensemble.update_state(state)
    return [str(warning.message) for warning in record]


@pytest.mark.parametrize(
    "message",
    [
        _ONE_DETERMINANT_WARNING.format(
            fragment="fragment_0", extension=", SQDConfig.extension='singles',"
        ),
        "fragment_0's subspace holds 1 of 36 determinants. The circuit samples "
        "the ground state, so excited states can come out too high in an "
        "incomplete subspace. Set SQDConfig.extension to add the determinants it "
        "misses.",
        "State (1, 1) has <S^2> = 0.5000, not 0.0000, so the subspace does not "
        "resolve it as that spin. Raise lambda_penalty or the sampling budget "
        "(batch_size, n_recovery_iterations).",
        "States (1, 0) and (1, 1) lie within 0.0e+00 Ha of each other, so their "
        "order can swap between rounds.",
    ],
    ids=["one-determinant", "incomplete-subspace", "off-spin", "degenerate"],
)
def test_unresolved_states_warn(unresolved_state_warnings, message):
    """Each must be reported, not passed silently."""
    assert message in unresolved_state_warnings


def test_near_degenerate_states_keep_their_root_order():
    """Float noise below the degeneracy tolerance must not reorder states."""
    states = [
        ElectronicState(multiplicity=1, index=0, energy=-2.0, spin_square=0.0),
        ElectronicState(multiplicity=3, index=0, energy=-1.0 - 1e-14, spin_square=2.0),
        ElectronicState(multiplicity=1, index=1, energy=-1.0, spin_square=0.0),
    ]

    ordered = _excited_states._in_energy_order(states)

    assert [(state.multiplicity, state.index) for state in ordered] == [
        (1, 0),
        (1, 1),
        (3, 0),
    ]


@pytest.mark.parametrize(
    "states, weights, message",
    [
        (
            None,
            None,
            "StateAveragedOrbitalUpdate averages over SQDConfig.states, which is "
            "unset. Request the states to average, e.g. SQDConfig(states={1: 2}).",
        ),
        (
            {1: 2},
            {(1, 0): 0.5, (3, 0): 0.5},
            "StateAveragedOrbitalUpdate.weights must name exactly the states "
            "SQDConfig.states requests, [(1, 0), (1, 1)]; got [(1, 0), (3, 0)].",
        ),
    ],
    ids=["no-states", "weights-mismatch"],
)
def test_state_averaging_needs_matching_states(
    dummy_expval_backend, states, weights, message
):
    with pytest.raises(ValueError, match=exact_match(message)):
        _lassqd(
            dummy_expval_backend,
            active_spaces=[H4_WHOLE_SPACE],
            states=states,
            orbital_update=StateAveragedOrbitalUpdate(weights=weights),
        )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "weights, density_fitted",
    [(None, False), ({(1, 0): 0.7, (1, 1): 0.3}, False), (None, True)],
    ids=["equal-weights", "unequal-weights", "density-fitted-mean-field"],
)
def test_state_averaged_orbitals_reproduce_sa_casscf(
    dummy_expval_backend, mocker, weights, density_fitted
):
    """With a complete subspace every round, state-averaged LASSQD on one
    fragment of a symmetric molecule is exact-integral symmetry-adapted
    SA-CASSCF: the same average and per-state energies, not a saddle point above
    them, even when the mean field supplied was density-fitted."""
    mean_field = scf.RHF(h2o_molecule(symmetry=True)).run(verbose=0)
    supplied = mean_field.density_fit().run(verbose=0) if density_fitted else mean_field
    spec = FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2)
    orbital_update = StateAveragedOrbitalUpdate(weights=weights)
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(supplied),
        active_spaces=[spec],
        states={1: 2},
        extension="doubles",
        lambda_penalty=1.0,
        n_batches=1,
        n_recovery_iterations=1,
        orbital_update=orbital_update,
        energy_tol=1e-10,
    )
    ensemble.run(max_rounds=30)

    reference = mcscf.CASSCF(mean_field, 4, 4)
    reference.conv_tol = 1e-11
    # Dense singlet CI: a Davidson start can miss a root of another symmetry.
    reference.fcisolver = fci.direct_spin0.FCI()
    reference.fcisolver.davidson_only = False
    average = orbital_update._weights_for({1: 2})
    reference.state_average_([average[(1, 0)], average[(1, 1)]])
    reference.kernel()

    report = ensemble.round_reports[-1]
    states = ensemble.electronic_states
    assert ensemble.stop_reason.name == "COMPLETE"
    assert report.averaged_energy == pytest.approx(reference.e_tot, abs=1e-7)
    assert ensemble.workflow_state.energy == report.averaged_energy
    # Only the average is stationary, so each state carries first-order
    # residual orbital error.
    np.testing.assert_allclose(
        [state.energy for state in states.values()],
        sorted(reference.e_states),
        rtol=0.0,
        atol=1e-6,
    )
    assert ensemble.energy == report.energy == states[(1, 0)].energy


def test_state_averaging_over_half_a_degenerate_pair_warns():
    """CAS(6,6) N2 averaged over S0 and one member of the degenerate S1 pair:
    the average is flat along the rotation mixing the pair, so the solve never
    settles although its gradient is small."""
    mean_field = scf.RHF(
        gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    ).run()
    strings = tuple(
        format(string, "06b")[::-1] for string in cistring.make_strings(range(6), 3)
    )
    message = (
        "The state-averaged solve stopped at max_iterations=3 before settling, "
        "though its orbital gradient is below tolerance. Unless the cap is "
        "deliberately low, a common cause is a requested state degenerate with "
        "one left out, which leaves the averaged energy a flat direction. Request "
        "every component of the degenerate set."
    )
    with pytest.warns(UserWarning, match=exact_match(message)):
        solve, _ = state_averaged_orbital_solve(
            AOIntegrals.of(mean_field.mol),
            mean_field.mo_coeff,
            4,
            FragmentSpec(orbitals=(4, 5, 6, 7, 8, 9), n_alpha=3, n_beta=3),
            (strings, strings),
            {(1, 0): 0.5, (1, 1): 0.5},
            1.0,
            gradient_tol=1e-3,
            max_iterations=3,
        )

    assert solve.converged
    assert solve.n_iterations == 3


def _h2o_state_averaged_solve(symmetry, report=None):
    """H2O CAS(4,4) averaged over S0 and S1 from its RHF orbitals, whose
    symmetric point is a saddle of the average."""
    mol = h2o_molecule(symmetry=symmetry)
    mean_field = scf.RHF(mol).run(verbose=0)
    strings = tuple(
        format(string, "04b")[::-1] for string in cistring.make_strings(range(4), 2)
    )
    solve, _ = state_averaged_orbital_solve(
        AOIntegrals.of(mol),
        mean_field.mo_coeff,
        3,
        FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2),
        (strings, strings),
        {(1, 0): 0.5, (1, 1): 0.5},
        1.0,
        gradient_tol=1e-5,
        max_iterations=50,
        report=report,
    )
    return mol, solve


def test_a_state_averaged_solve_keeps_a_symmetric_molecules_symmetry():
    mol, solve = _h2o_state_averaged_solve(symmetry=True)

    assert orbital_irreps(mol, solve.mo_coeff) is not None


def test_a_state_averaged_solve_reports_once_per_macro_iteration():
    lines = []

    _, solve = _h2o_state_averaged_solve(symmetry=True, report=lines.append)

    assert [line.split(",")[0] for line in lines] == [
        f"State-averaged orbital solve: iteration {number}"
        for number in range(1, solve.n_iterations + 1)
    ]


def test_a_state_averaged_solve_that_breaks_symmetry_warns():
    message = (
        "The state-averaged solve broke the molecule's C2v symmetry, mixing states "
        "of different symmetry; which such solution a run reaches can depend on "
        "rounding and the starting orbitals. Build the molecule with symmetry=True "
        "to keep it."
    )

    with pytest.warns(UserWarning, match=exact_match(message)):
        _h2o_state_averaged_solve(symmetry=False)


def test_a_linear_molecule_keeps_its_symmetry_through_a_nudged_solve():
    """N2's 6-31G pi orbitals, nudged within each irrep, are no longer paired
    by energy, which PySCF's linear groups need."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.098", basis="6-31g", symmetry=True, verbose=0)
    mo_coeff = scf.RHF(mol).run(verbose=0).mo_coeff
    nudged = mo_coeff @ _workflow._orbital_nudge(
        mo_coeff.shape[1], 0, orbital_irreps(mol, mo_coeff)
    )
    strings = tuple(
        format(string, "06b")[::-1] for string in cistring.make_strings(range(6), 3)
    )

    solve, _ = state_averaged_orbital_solve(
        AOIntegrals.of(mol),
        nudged,
        4,
        FragmentSpec(orbitals=(4, 5, 6, 7, 8, 9), n_alpha=3, n_beta=3),
        (strings, strings),
        {(1, 0): 1 / 3, (1, 1): 1 / 3, (1, 2): 1 / 3},
        1.0,
        gradient_tol=1e-5,
        max_iterations=50,
    )

    assert orbital_irreps(mol, solve.mo_coeff) is not None


def _degenerate_pair(lower_index):
    """States (1, 1) and (1, 2), 1e-14 Ha apart, ``lower_index`` the lower."""
    return [
        ElectronicState(
            multiplicity=1,
            index=index,
            energy=-1.0 + (0.0 if index == lower_index else 1e-14),
            spin_square=0.0,
        )
        for index in (1, 2)
    ]


def test_degenerate_states_are_warned_about_once_whatever_their_energy_order():
    forward = dict(_excited_states._state_warnings(_degenerate_pair(1)))
    backward = dict(_excited_states._state_warnings(_degenerate_pair(2)))

    assert forward.keys() == backward.keys()
    assert all("States (1, 1) and (1, 2)" in message for message in forward.values())


def test_a_symmetric_orbital_nudge_keeps_every_orbitals_irrep():
    mol = h2o_molecule(symmetry=True)
    mo_coeff = scf.RHF(mol).run(verbose=0).mo_coeff
    irreps = orbital_irreps(mol, mo_coeff)

    nudged = mo_coeff @ _workflow._orbital_nudge(mo_coeff.shape[1], 0, irreps)

    np.testing.assert_array_equal(orbital_irreps(mol, nudged), irreps)
    assert not np.allclose(nudged, mo_coeff)


def test_state_wavefunctions_reproduce_dense_casci(exported_h2o_states):
    """On a complete subspace each exported state is the exact CASCI root at the
    round's final orbitals: energy, spin and both RDMs."""
    ensemble, reference = exported_h2o_states

    wavefunctions = ensemble.state_wavefunctions()

    assert list(wavefunctions) == list(ensemble.electronic_states)
    for label, wavefunction in wavefunctions.items():
        energy, vector = reference[label]
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(vector, 4, (2, 2))
        np.testing.assert_allclose(
            np.hstack([wavefunction.core_orbitals, wavefunction.active_orbitals]),
            ensemble.workflow_state.mo_coeff[:, :7],
            atol=1e-12,
        )
        assert wavefunction.state.energy == pytest.approx(energy, abs=1e-8)
        assert wavefunction.state.spin_square == pytest.approx(
            0.25 * (label[0] ** 2 - 1), abs=1e-8
        )
        np.testing.assert_allclose(wavefunction.rdm1, rdm1, atol=1e-8)
        np.testing.assert_allclose(wavefunction.rdm2, rdm2, atol=1e-8)


def test_transition_properties_reproduce_dense_casci(exported_h2o_states):
    """Only the singlet pair is reported; its oscillator strength, dipole
    length and natural-transition-orbital weights match the dense CASCI
    transition density. Phases are arbitrary, so the dipole's sign is not
    compared."""
    ensemble, reference = exported_h2o_states
    orbitals = ensemble.workflow_state.mo_coeff[:, 3:7]
    (energy_0, vector_0), (energy_1, vector_1) = reference[(1, 0)], reference[(1, 1)]
    expected_rdm1 = fci.direct_spin1.trans_rdm1(vector_1, vector_0, 4, (2, 2))
    dipole_mo = np.einsum(
        "xab,ap,bq->xpq",
        ensemble._mol.intor_symmetric("int1e_r"),
        orbitals,
        orbitals,
    )
    expected_dipole = np.linalg.norm(np.einsum("xpq,pq->x", dipole_mo, expected_rdm1))
    singular_values = np.linalg.svd(expected_rdm1, compute_uv=False)

    transitions = ensemble.transition_properties()

    assert list(transitions) == [((1, 0), (1, 1))]
    transition = transitions[((1, 0), (1, 1))]
    assert np.linalg.norm(transition.transition_dipole) == pytest.approx(
        expected_dipole, abs=1e-8
    )
    assert transition.oscillator_strength == pytest.approx(
        2.0 / 3.0 * (energy_1 - energy_0) * expected_dipole**2, abs=1e-8
    )
    np.testing.assert_allclose(
        transition.nto_weights,
        singular_values**2 / np.sum(singular_values**2),
        atol=1e-8,
    )


def test_a_degenerate_pair_transition_puts_the_lower_label_first(
    dummy_expval_backend, mocker
):
    """N2's two pi_g components are exactly degenerate at its symmetric RHF
    orbitals, so energy cannot order them."""
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(
            gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", symmetry=True, verbose=0)
        ),
        active_spaces=[FragmentSpec(orbitals=(4, 5, 6, 7, 8, 9), n_alpha=3, n_beta=3)],
        states={1: 3},
        extension="doubles",
        n_batches=1,
        n_recovery_iterations=1,
        orbital_update=FixedOrbitals(),
    )
    with pytest.warns(UserWarning, match="States \\(1, 1\\) and \\(1, 2\\) lie within"):
        ensemble.run(max_rounds=1)
    states = ensemble.electronic_states

    transitions = ensemble.transition_properties()

    assert states[(1, 1)].energy == pytest.approx(states[(1, 2)].energy, abs=1e-12)
    assert list(transitions) == [((1, 0), (1, 1)), ((1, 0), (1, 2)), ((1, 1), (1, 2))]


def test_natural_transition_orbitals_rebuild_the_transition_density(
    exported_h2o_states,
):
    """``sum_k s_k |hole_k><particle_k|`` over the active orbitals is the
    transition 1-RDM, and each set of orbitals is orthonormal."""
    ensemble, _ = exported_h2o_states
    transition = ensemble.transition_properties()[((1, 0), (1, 1))]
    overlap = ensemble._mol.intor_symmetric("int1e_ovlp")
    orbitals = ensemble.workflow_state.mo_coeff[:, 3:7]
    particles = orbitals.T @ overlap @ transition.particle_orbitals
    holes = orbitals.T @ overlap @ transition.hole_orbitals
    scale = np.sqrt(transition.nto_weights * np.sum(transition.transition_rdm1**2))

    np.testing.assert_allclose(
        holes @ np.diag(scale) @ particles.T, transition.transition_rdm1, atol=1e-10
    )
    for set_ in (particles, holes):
        np.testing.assert_allclose(set_.T @ set_, np.eye(4), atol=1e-10)


def test_the_hole_is_the_orbital_the_electron_leaves():
    """From both electrons in orbital 0 to the singlet with one in orbital 1,
    the hole is orbital 0 and the particle orbital 1."""
    mol = gto.M(atom="H 0 0 0; H 0 0 1.5", basis="sto-3g", verbose=0)
    orbitals = np.eye(2)
    strings = ("10", "01")

    def wavefunction(index, amplitudes):
        return StateWavefunction(
            state=ElectronicState(1, index, float(index), 0.0),
            core_orbitals=np.zeros((2, 0)),
            active_orbitals=orbitals,
            strings_alpha=strings,
            strings_beta=strings,
            amplitudes=np.asarray(amplitudes),
            rdm1=np.zeros((2, 2)),
            rdm2=np.zeros((2,) * 4),
        )

    ground = wavefunction(0, [[1.0, 0.0], [0.0, 0.0]])
    excited = wavefunction(1, np.array([[0.0, 1.0], [1.0, 0.0]]) / np.sqrt(2))

    (transition,) = state_transitions(mol, [ground, excited])

    np.testing.assert_allclose(abs(transition.hole_orbitals[:, 0]), [1.0, 0.0])
    np.testing.assert_allclose(abs(transition.particle_orbitals[:, 0]), [0.0, 1.0])


def _pyscf_nevpt2(mean_field, mo_coeff, n_active, nelec, vector):
    """PySCF's NEVPT2 correction for a dense active-space CI vector."""
    casci = mcscf.CASCI(mean_field, n_active, nelec)
    casci.verbose = 0
    casci.mo_coeff = mo_coeff
    casci.ci = vector
    return mrpt.NEVPT(casci).kernel()


def test_nevpt2_matches_pyscf_on_each_dense_casci_root(exported_h2o_states):
    ensemble, reference = exported_h2o_states
    mo_coeff = ensemble.workflow_state.mo_coeff

    corrections = ensemble.nevpt2()

    assert list(corrections) == list(ensemble.electronic_states)
    for label, correction in corrections.items():
        assert correction == pytest.approx(
            _pyscf_nevpt2(
                scf.RHF(ensemble._mol), mo_coeff, 4, (2, 2), reference[label][1]
            ),
            abs=1e-8,
        )


def test_nevpt2_corrects_the_doublets_of_an_open_shell_molecule(
    nh2_radical, dummy_expval_backend, mocker
):
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(nh2_radical),
        active_spaces=[FragmentSpec(orbitals=(3, 4, 5), n_alpha=2, n_beta=1)],
        states={2: 2},
        n_batches=1,
        batch_size=8,
        n_recovery_iterations=1,
        orbital_update=FixedOrbitals(),
    )
    ensemble.run(max_rounds=1)
    mo_coeff = ensemble.workflow_state.mo_coeff
    reference = _dense_casci_states(nh2_radical, mo_coeff, 3, (2, 1))

    corrections = ensemble.nevpt2()

    assert list(corrections) == [(2, 0), (2, 1)]
    for label, correction in corrections.items():
        assert correction == pytest.approx(
            _pyscf_nevpt2(nh2_radical, mo_coeff, 3, (2, 1), reference[label][1]),
            abs=1e-8,
        )


def test_a_state_export_after_a_run_without_states_says_to_run_again(
    exact_sampler_lassqd,
):
    ensemble, _ = exact_sampler_lassqd
    ensemble.run(max_rounds=1)
    message = (
        "state_wavefunctions needs a completed LASSQD round with SQDConfig.states "
        "set; this run requested none. Set them, e.g. SQDConfig(states={1: 2}), "
        "and run again."
    )

    with pytest.raises(RuntimeError, match=exact_match(message)):
        ensemble.state_wavefunctions()


def test_a_state_export_before_a_run_suggests_states_of_the_molecules_spin(
    nh2_radical, dummy_expval_backend, mocker
):
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(nh2_radical),
        active_spaces=[FragmentSpec(orbitals=(3, 4, 5), n_alpha=2, n_beta=1)],
    )
    message = (
        "nevpt2 needs a completed LASSQD round with SQDConfig.states set. Request "
        "the states, e.g. SQDConfig(states={2: 2}), and call run() first."
    )

    with pytest.raises(RuntimeError, match=exact_match(message)):
        ensemble.nevpt2()


def test_nevpt2_refuses_a_full_space_vector_beyond_max_memory(exported_h2o_states):
    ensemble, _ = exported_h2o_states
    ensemble._mol.max_memory = 1e-4
    message = (
        "NEVPT2 needs the full-space CI vector of CAS(4,4), 0.000288 MB, above "
        "mol.max_memory=0.0001 MB. Raise mol.max_memory."
    )

    with pytest.raises(MemoryError, match=exact_match(message)):
        ensemble.nevpt2()


@pytest.mark.parametrize(
    "method", ["state_wavefunctions", "transition_properties", "nevpt2"]
)
def test_state_exports_need_a_completed_states_round(
    dummy_expval_backend, mocker, method
):
    ensemble, _ = _h2o_states_lassqd(dummy_expval_backend, mocker, FixedOrbitals())
    message = (
        f"{method} needs a completed LASSQD round with SQDConfig.states set. "
        "Request the states, e.g. SQDConfig(states={1: 2}), and call run() first."
    )

    with pytest.raises(RuntimeError, match=exact_match(message)):
        getattr(ensemble, method)()


@pytest.fixture
def two_orbital_wavefunction():
    return StateWavefunction(
        state=ElectronicState(multiplicity=1, index=0, energy=0.0, spin_square=0.0),
        core_orbitals=np.zeros((2, 0)),
        active_orbitals=np.eye(2),
        strings_alpha=("10", "01"),
        strings_beta=("10", "01"),
        amplitudes=np.array([[0.1, -0.7], [0.7, 0.0]]),
        rdm1=np.zeros((2, 2)),
        rdm2=np.zeros((2,) * 4),
    )


def test_leading_determinants_rank_by_weight(two_orbital_wavefunction):
    assert two_orbital_wavefunction.leading_determinants(2) == (
        ("10", "01", -0.7),
        ("01", "10", 0.7),
    )
    assert len(two_orbital_wavefunction.leading_determinants(10)) == 4


@pytest.mark.parametrize("count", [0, -1])
def test_leading_determinants_reject_a_count_below_one(two_orbital_wavefunction, count):
    with pytest.raises(
        ValueError, match=exact_match(f"count must be at least 1; got {count}.")
    ):
        two_orbital_wavefunction.leading_determinants(count)


def test_state_averaging_works_on_a_symmetry_adapted_molecule(
    dummy_expval_backend, mocker
):
    """The starting-orbital nudge breaks point-group symmetry, which PySCF's
    symmetry-adapted CASSCF would reject."""
    mol = gto.M(
        atom="C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; "
        "H 0 0.923 -1.237; H 0 -0.923 -1.237",
        basis="sto-3g",
        symmetry=True,
        verbose=0,
    )
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(scf.RHF(mol).run(verbose=0)),
        active_spaces=[FragmentSpec(orbitals=(7, 8), n_alpha=1, n_beta=1)],
        states={1: 2, 3: 1},
        orbital_update=StateAveragedOrbitalUpdate(),
    )

    ensemble.run(max_rounds=1)

    assert set(ensemble.electronic_states) == {(1, 0), (1, 1), (3, 0)}


@pytest.mark.filterwarnings("ignore:Orbital optimisation stopped after")
def test_state_averaged_energy_keeps_its_weights_through_a_crossing(
    dummy_expval_backend, mocker
):
    """Symmetric rectangular H4's two upper singlets swap order as the
    orbitals relax, three solve iterations a round.
    The solve weights roots by energy rank, so each round's average must still
    be the label-weighted sum of that round's states."""
    weights = {(1, 0): 0.05, (1, 1): 0.05, (1, 2): 0.9}
    mean_field = scf.RHF(
        gto.M(
            atom="H 0 0 0; H 0 0 1.1; H 1.0 0 0; H 1.0 0 1.1",
            basis="6-31g",
            symmetry=True,
            verbose=0,
        )
    ).run(verbose=0)
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(mean_field),
        active_spaces=[FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)],
        states={1: 3},
        n_batches=1,
        batch_size=16,
        n_recovery_iterations=1,
        orbital_update=StateAveragedOrbitalUpdate(weights=weights, max_iterations=3),
    )

    ensemble.run(max_rounds=3)

    orders = [
        [state.index for state in report.states] for report in ensemble.round_reports
    ]
    assert [0, 1, 2] in orders and [0, 2, 1] in orders
    for report in ensemble.round_reports:
        assert report.averaged_energy == pytest.approx(
            sum(
                weights[(state.multiplicity, state.index)] * state.energy
                for state in report.states
            ),
            abs=1e-10,
        )


def test_state_averaging_weights_follow_the_tracked_states():
    weights = {(1, 0): 0.7, (1, 1): 0.3}

    assert _config._by_root(weights, {(1, 0): (1, 1), (1, 1): (1, 0)}) == {
        (1, 0): 0.3,
        (1, 1): 0.7,
    }
    assert _config._by_root(weights, {(1, 0): (1, 1)}) == {(1, 0): 0.3, (1, 1): 0.7}
    assert _config._by_root(weights, {}) == weights


def test_a_subspace_short_of_one_spin_leaves_the_missing_states_out(
    dummy_expval_backend, mocker
):
    """A root of another spin is not reported in the requested one's place, so
    no state appears twice and no transition crosses spins."""
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        active_spaces=[H4_WHOLE_SPACE],
        states={1: 4, 3: 2},
        n_batches=1,
        batch_size=2,
        n_recovery_iterations=1,
        extension=None,
        carryover_cutoff=None,
        orbital_update=FixedOrbitals(),
    )

    with pytest.warns(UserWarning) as record:
        ensemble.run(max_rounds=1)

    states = ensemble.electronic_states.values()
    resolved = {
        multiplicity: sorted(s.index for s in states if s.multiplicity == multiplicity)
        for multiplicity in (1, 3)
    }
    assert sum(map(len, resolved.values())) < 6
    assert all(indices == list(range(len(indices))) for indices in resolved.values())
    assert all(_sqd.resolves_spin(s.multiplicity, s.spin_square) for s in states)
    assert any(
        "requested states of multiplicity" in str(warning.message) for warning in record
    )
    assert all(
        initial[0] == final[0] for initial, final in ensemble.transition_properties()
    )


def test_electronic_states_are_empty_without_states(exact_sampler_lassqd):
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    ensemble.update_state(state)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ensemble.electronic_states == {}
