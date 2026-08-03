# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD workflow state, lifecycle wiring, and reference parity.

``MaestroSimulator.set_seed`` is a documented no-op, so shot noise on that
backend is unseedable. Any test whose assertion depends on SQD's subspace
recovery capturing a specific determinant inherits a coin flip from that
noise: too small a sampling budget (``n_batches`` / ``batch_size``) makes
capture unreliable and the test intermittent, even though the workflow's own
``seed`` is fixed. Do not lower an e2e test's sampling budget to match
another test's without checking that the determinant it depends on is still
reliably captured.
"""

import dataclasses

import numpy as np
import pytest
from pyscf import cc, fci, gto, mcscf, scf
from pyscf.cc import addons as cc_addons
from qiskit.quantum_info import SparsePauliOp, Statevector

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog import LASSQD, ReportingLevel, WorkflowStatus
from divi.qprog.algorithms import LUCJAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import _uccsd_excitations
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.workflows._lassqd import _workflow
from divi.qprog.workflows._lassqd._sqd import SQDResult
from divi.qprog.workflows._lassqd._state import (
    FragmentSpec,
    FragmentState,
    validate_fragment_specs,
)
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    PRODUCT_STATE_ENERGY,
    PRODUCT_STATE_MO_TRACE,
    _build_exact_sampler_program,
    build_exact_sampler_lassqd,
    embedded_fragment_ccsd,
    exact_sampler_lassqd,
    h2_molecule,
    h4_chain,
    h4_chain_mean_field,
    uniform_full_space_probs,
)


def test_fragment_spec_normalizes_orbitals_to_a_tuple():
    spec = FragmentSpec(orbitals=[2, 3], n_alpha=1, n_beta=1)
    assert spec.orbitals == (2, 3)
    assert spec.n_orbitals == 2


def test_fragment_spec_is_frozen():
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.n_alpha = 2


@pytest.mark.parametrize(
    "orbitals, n_alpha, n_beta, match",
    [
        ((), 0, 0, "at least one orbital"),
        ((0, 0), 1, 1, "duplicate"),
        ((0, 1), 3, 1, "n_alpha"),
        ((0, 1), 1, -1, "n_beta"),
    ],
)
def test_fragment_spec_rejects_invalid_input(orbitals, n_alpha, n_beta, match):
    with pytest.raises(ValueError, match=match):
        FragmentSpec(orbitals=orbitals, n_alpha=n_alpha, n_beta=n_beta)


def test_validate_fragment_specs_rejects_overlap():
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1),
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_rejects_out_of_range():
    specs = [FragmentSpec(orbitals=(0, 9), n_alpha=1, n_beta=1)]
    with pytest.raises(ValueError, match="out of range"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_accepts_disjoint_in_range():
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
    ]
    validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_rejects_electron_count_mismatch():
    """Fragments must account for exactly the electrons occupying the active
    orbitals they cover. A shortfall does not raise anywhere downstream; it
    silently yields an energy for the wrong number of electrons."""
    specs = [FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=1)]
    with pytest.raises(ValueError, match="electron"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_accepts_consistent_electron_count():
    specs = [
        FragmentSpec(orbitals=(0, 3), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1),
    ]
    validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_accepts_spin_imbalanced_fragments():
    """Spin-imbalanced fragments are the antiferromagnetic case a localized
    active space exists to describe, so they must be accepted as long as the
    fragments' electrons still add up.

    Both fragments here keep an excitation available in at least one spin
    channel, which is what makes them runnable rather than merely valid.
    """
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=0),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=2),
    ]
    validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_rejects_a_spin_saturated_fragment():
    """Dropping the spin-balance rule exposed a case the old spin-traced
    fully-occupied guard missed: ``(2a, 0b)`` on two orbitals fills the alpha
    channel and empties the beta one, so UCCSD has zero parameters and
    ``create_programs`` died with an error naming neither the fragment nor the
    cause."""
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=0),
        FragmentSpec(orbitals=(2, 3), n_alpha=0, n_beta=2),
    ]
    with pytest.raises(ValueError, match="no excitation available"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_rejects_a_nonzero_total_sz():
    """Relaxing the per-fragment balance rule left the *total* Sz unchecked, so
    an Sz=1 fragmentation of a closed-shell molecule ran to ``COMPLETE`` and
    reported the wrong spin sector (measured -1.662 against a singlet FCI of
    -2.252). The electron count alone does not catch it: 4 electrons split
    3-alpha/1-beta still sums to 4."""
    specs = [
        FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=1),
        FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=0),
    ]
    with pytest.raises(ValueError, match="Sz"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_still_rejects_inconsistent_electron_totals():
    """Relaxing the spin-balance rule must not relax the electron count: a lone
    spin-polarized fragment leaves the molecule's electrons unaccounted for."""
    specs = [FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=0)]
    with pytest.raises(ValueError, match="declare"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def test_validate_fragment_specs_rejects_a_fully_occupied_fragment():
    """A fragment with every spin-orbital occupied has no correlation to
    capture and is physically impossible as an active-space fragment; it
    must be rejected at construction rather than reaching ``run()`` and
    raising a bare ``ValueError`` with an empty ``round_history``."""
    specs = [FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=2)]
    with pytest.raises(ValueError, match="no excitation available"):
        validate_fragment_specs(specs, n_orbitals_total=4, n_occupied=2)


def _lassqd(backend, **overrides):
    kwargs = dict(
        active_spaces=[
            FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
        ],
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        max_iterations=3,
        n_batches=2,
        batch_size=8,
        n_sqd_iterations=2,
        seed=42,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
    )
    kwargs.update(overrides)
    return LASSQD(h4_chain(), **kwargs)


def test_rejects_both_explicit_and_automatic_fragments(dummy_expval_backend):
    with pytest.raises(ValueError, match="exactly one"):
        _lassqd(dummy_expval_backend, n_active_orbitals=4)


def test_rejects_no_fragment_specification(dummy_expval_backend):
    with pytest.raises(ValueError, match="exactly one"):
        _lassqd(dummy_expval_backend, active_spaces=None)


def test_rejects_overlapping_fragments(dummy_expval_backend):
    with pytest.raises(ValueError, match="overlap"):
        _lassqd(
            dummy_expval_backend,
            active_spaces=[
                FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1),
            ],
        )


@pytest.mark.parametrize(
    "override, match",
    [
        ({"n_batches": 0}, "n_batches"),
        ({"batch_size": 0}, "batch_size"),
        ({"n_sqd_iterations": 0}, "n_sqd_iterations"),
        ({"energy_tol": 0.0}, "energy_tol"),
        ({"coupling_threshold": -1e-3}, "coupling_threshold"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"lambda_penalty": -0.1}, "lambda_penalty"),
    ],
)
def test_rejects_invalid_sqd_sizing_arguments(dummy_expval_backend, override, match):
    """These are validated eagerly in the constructor: without this, an
    invalid value (e.g. n_batches=0) would dispatch a full round of paid
    circuits before SQDSolver ever raises, and batch_size=0 was never
    validated at all (silently clamped to a one-determinant pool).
    ``max_iterations=0`` is included for the same reason: unvalidated, it
    reaches the optimizer and raises a bare ``StopIteration``, the least
    actionable exception in Python, instead of a clear ``ValueError`` here."""
    with pytest.raises(ValueError, match=match):
        _lassqd(dummy_expval_backend, **override)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"n_active_orbitals": 0}, "n_active_orbitals"),
        ({"n_active_orbitals": -2}, "n_active_orbitals"),
        ({"energy_window": -0.1}, "energy_window"),
    ],
)
def test_rejects_invalid_automatic_fragmentation_arguments(
    dummy_expval_backend, override, match
):
    """``n_active_orbitals``/``energy_window`` used to only be validated
    lazily, inside ``select_frontier_orbitals`` during ``initial_state()``,
    unlike every sibling sizing argument above, which is validated eagerly in
    the constructor."""
    with pytest.raises(ValueError, match=match):
        _lassqd(dummy_expval_backend, active_spaces=None, **override)


def test_rejects_non_ansatz_instance(dummy_expval_backend):
    with pytest.raises(TypeError, match="ansatz"):
        _lassqd(dummy_expval_backend, ansatz="UCCSD")


def test_rejects_open_shell_molecules(dummy_expval_backend):
    triplet = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        LASSQD(
            triplet,
            active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            backend=dummy_expval_backend,
        )


def test_initial_state_seeds_diagonal_rdms(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()

    assert len(state.fragments) == 2
    for fragment in state.fragments:
        # Diagonal guess: 1.0 alpha + 1.0 beta on the lowest orbital.
        assert fragment.rdm1[0, 0] == pytest.approx(2.0)
        assert fragment.params is None
        assert fragment.rdm1_alpha is not None
        assert fragment.rdm1_beta is not None
    assert state.energy == float("inf")


def test_initial_state_diagonal_guess_splits_spin_for_a_polarized_fragment():
    """``_diagonal_rdm_guess`` must place alpha and beta separately. A
    closed-shell fragment cannot show this -- there the halves equal ``rdm1 / 2``
    and match ``spin_rdm1s()``'s fallback, so a spin-traced guess would pass."""
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=2, n_beta=1)
    rdm1, _, rdm1_alpha, rdm1_beta = _workflow._diagonal_rdm_guess(spec)

    assert not np.allclose(rdm1_alpha, rdm1_beta)
    np.testing.assert_allclose(rdm1_alpha + rdm1_beta, rdm1, atol=1e-12)
    np.testing.assert_allclose(np.diag(rdm1_alpha), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(np.diag(rdm1_beta), [1.0, 0.0, 0.0])


def test_initial_state_2rdm_only_populates_diagonal_blocks(dummy_expval_backend):
    ensemble = _lassqd(
        dummy_expval_backend,
        active_spaces=[FragmentSpec(orbitals=(0, 1, 2), n_alpha=2, n_beta=2)],
    )
    state = ensemble.initial_state()

    fragment = state.fragments[0]
    rdm1, rdm2 = fragment.rdm1, fragment.rdm2
    n_orb = fragment.spec.n_orbitals
    for p in range(n_orb):
        for q in range(n_orb):
            assert rdm2[p, p, q, q] == pytest.approx(rdm1[p, p] * rdm1[q, q])
    assert rdm2[0, 1, 0, 1] == pytest.approx(0.0)


def test_initial_state_automatic_path_covers_the_active_orbitals(
    dummy_expval_backend,
):
    ensemble = _lassqd(
        dummy_expval_backend,
        active_spaces=None,
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )
    state = ensemble.initial_state()

    assert sum(fragment.spec.n_orbitals for fragment in state.fragments) == 4
    assert sum(fragment.spec.n_alpha for fragment in state.fragments) == 2
    assert sum(fragment.spec.n_beta for fragment in state.fragments) == 2
    assert state.mo_coeff.shape[1] == 4


def test_initial_state_forwards_the_workflow_rng_to_auto_fragment_specs(
    dummy_expval_backend, mocker
):
    ensemble = _lassqd(
        dummy_expval_backend,
        active_spaces=None,
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )
    spy = mocker.spy(_workflow, "auto_fragment_specs")

    ensemble.initial_state()

    assert spy.call_args.args[4] is ensemble._rng


def test_create_programs_makes_one_vqe_per_fragment(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())

    assert len(ensemble.programs) == 2
    for program in ensemble.programs.values():
        # 2 spatial orbitals per fragment -> 4 qubits.
        assert program.n_qubits == 4


def test_create_programs_uses_the_configured_ansatz(dummy_expval_backend):
    """A non-default ansatz must be the one that actually reaches the
    programs: asserting only the default ansatz's type cannot tell a
    configured ansatz apart from one that is hard-coded."""
    ensemble = _lassqd(dummy_expval_backend, ansatz=LUCJAnsatz())
    ensemble.create_programs(ensemble.initial_state())
    for program in ensemble.programs.values():
        assert isinstance(program.ansatz, LUCJAnsatz)


def test_create_programs_defaults_to_uccsd(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())
    for program in ensemble.programs.values():
        assert isinstance(program.ansatz, UCCSDAnsatz)


def test_fragment_programs_get_distinct_seeds(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())
    seeds = {program._seed for program in ensemble.programs.values()}
    assert len(seeds) == 2


def test_create_programs_derives_n_core_from_an_externally_built_state(
    dummy_expval_backend,
):
    """create_programs must not depend on initial_state() having already run
    on this instance: a state built by a fresh LASSQD targeting the same
    molecule must work just as well."""
    ensemble = _lassqd(dummy_expval_backend)
    state = _lassqd(dummy_expval_backend).initial_state()

    ensemble.create_programs(state)

    assert len(ensemble.programs) == 2


def test_missing_backend_raises_type_error():
    with pytest.raises(TypeError, match="backend"):
        LASSQD(
            h4_chain(),
            active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )


def test_extra_kwargs_are_forwarded_to_each_fragment_vqe(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, n_layers=2)
    ensemble.create_programs(ensemble.initial_state())
    for program in ensemble.programs.values():
        assert program.n_layers == 2


def test_unknown_kwargs_raise_when_creating_programs(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, bogus_kwarg=123)
    with pytest.raises(TypeError, match="bogus_kwarg"):
        ensemble.create_programs(ensemble.initial_state())


def test_fragment_vqe_rejects_mismatched_seed_params_length(dummy_expval_backend):
    hamiltonian = SparsePauliOp(["ZZ"], [1.0])
    baseline = _workflow._FragmentVQE(
        hamiltonian=hamiltonian,
        n_electrons=2,
        ansatz=LUCJAnsatz(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=dummy_expval_backend,
    )
    bad_params = np.ones(baseline.n_params + 1)

    with pytest.raises(ValueError, match="seed_params"):
        _workflow._FragmentVQE(
            hamiltonian=hamiltonian,
            n_electrons=2,
            ansatz=LUCJAnsatz(),
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            backend=dummy_expval_backend,
            seed_params=bad_params,
        )


def test_is_complete_is_false_before_any_round(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    assert ensemble.is_complete(ensemble.initial_state()) is False


def test_is_complete_when_energy_change_is_below_tolerance(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, energy_tol=1e-5)
    state = dataclasses.replace(
        ensemble.initial_state(), energy=-2.0, previous_energy=-2.0 + 1e-9
    )
    assert ensemble.is_complete(state) is True


def test_is_complete_false_when_energy_still_moving(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, energy_tol=1e-5)
    state = dataclasses.replace(
        ensemble.initial_state(), energy=-2.0, previous_energy=-1.0
    )
    assert ensemble.is_complete(state) is False


def test_update_state_does_not_mutate_the_input(exact_sampler_lassqd):
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    # Deep-copy the values: dataclasses.replace() is shallow, so comparing a
    # replaced copy against the original would compare arrays to themselves.
    mo_before = state.mo_coeff.copy()
    rdm1_before = [fragment.rdm1.copy() for fragment in state.fragments]
    energy_before = state.energy

    with pytest.warns(UserWarning, match="no correlation"):
        new_state = ensemble.update_state(state)

    assert new_state is not state
    np.testing.assert_array_equal(state.mo_coeff, mo_before)
    for fragment, expected in zip(state.fragments, rdm1_before):
        np.testing.assert_array_equal(fragment.rdm1, expected)
    assert state.energy == energy_before


def test_update_state_populates_rdms_and_energy(exact_sampler_lassqd):
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)

    with pytest.warns(UserWarning, match="no correlation"):
        new_state = ensemble.update_state(state)

    assert np.isfinite(new_state.energy)
    assert new_state.previous_energy == state.energy
    for fragment in new_state.fragments:
        assert np.trace(fragment.rdm1) == pytest.approx(2.0, abs=1e-6)
        # Explicit, not the ``spin_rdm1s()`` fallback: that returns rdm1 / 2
        # twice, whose sum equals rdm1 identically, so a sum check alone would
        # pass even if update_state stopped populating the halves.
        assert fragment.rdm1_alpha is not None
        assert fragment.rdm1_beta is not None
        np.testing.assert_allclose(
            fragment.rdm1_alpha + fragment.rdm1_beta, fragment.rdm1, atol=1e-12
        )
        # Round-trips program.best_params into the next round's seed_params.
        np.testing.assert_allclose(fragment.params, [0.11, 0.22, 0.33])


def test_update_state_warns_when_a_fragment_collapses_to_one_determinant(
    exact_sampler_lassqd, mocker
):
    """``stop_reason`` reads ``COMPLETE`` whether a round converged or simply
    captured no correlation for a fragment. A recovered subspace of exactly
    one determinant is the recognizable signature of the latter, so it must
    surface as a warning rather than pass silently."""
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    spec = state.fragments[0].spec
    # Blocked "alpha + beta" bitstring: n_alpha electrons on the lowest
    # alpha orbitals, n_beta on the lowest beta orbitals.
    alpha_part = "1" * spec.n_alpha + "0" * (spec.n_orbitals - spec.n_alpha)
    beta_part = "1" * spec.n_beta + "0" * (spec.n_orbitals - spec.n_beta)
    collapsed_bitstring = alpha_part + beta_part
    mocker.patch(
        "divi.qprog.workflows._lassqd._workflow.SQDSolver.solve",
        return_value=SQDResult(
            energy=0.0, eigenvector=np.array([1.0]), subspace=[collapsed_bitstring]
        ),
    )

    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.update_state(state)


def test_symmetry_failure_names_the_fragment(exact_sampler_lassqd, mocker):
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    mocker.patch(
        "divi.qprog.workflows._lassqd._workflow.SQDSolver.solve",
        side_effect=ValueError(
            "No valid configurations found matching particle symmetry!"
        ),
    )
    with pytest.raises(ValueError) as exc_info:
        ensemble.update_state(state)
    assert "fragment_0" in str(exc_info.value)
    assert "particle symmetry" in str(exc_info.value)


def test_update_state_is_seed_reproducible(dummy_expval_backend):
    # ExactSamplerVQE's ground state is dominated by a single determinant, so
    # this needs a genuinely RNG-sensitive setup instead: a 9-determinant
    # symmetry sector, a batch far smaller than that (so which determinants
    # land in a batch's subspace is a real coin flip), and a non-degenerate
    # one-body spectrum so different subspaces give visibly different
    # energies. Two solves under the same workflow seed must still agree.
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=1, n_beta=2)
    probs = uniform_full_space_probs(spec.n_orbitals, spec.n_alpha, spec.n_beta)
    one_body = np.diag([-5.0, -1.0, 0.3])
    two_body = np.zeros((spec.n_orbitals,) * 4)

    def solve_once():
        ensemble = _lassqd(
            dummy_expval_backend,
            seed=11,
            n_batches=1,
            batch_size=2,
            n_sqd_iterations=1,
        )
        solver = ensemble._solver_for(0, spec)
        return solver.solve(probs, one_body, two_body).energy

    assert solve_once() == pytest.approx(solve_once(), abs=1e-12)


def test_solver_for_gives_each_fragment_an_independent_rng_stream(
    dummy_expval_backend,
):
    ensemble = _lassqd(dummy_expval_backend, seed=3)
    state = ensemble.initial_state()

    solver_0 = ensemble._solver_for(0, state.fragments[0].spec)
    solver_1 = ensemble._solver_for(1, state.fragments[1].spec)

    assert solver_0._rng is not solver_1._rng
    assert solver_0._rng.bit_generator.state != solver_1._rng.bit_generator.state


def test_lambda_penalty_defaults_to_the_solvers_own_default(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    assert solver.lambda_penalty == pytest.approx(0.2)


def test_lambda_penalty_is_threaded_to_the_solver(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, lambda_penalty=5.0)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    assert solver.lambda_penalty == pytest.approx(5.0)


def test_lassqd_state_and_fragment_state_compare_by_identity(dummy_expval_backend):
    """Numpy fields make a value-based ``__eq__`` raise; both dataclasses
    must fall back to identity comparison instead."""
    ensemble = _lassqd(dummy_expval_backend)
    state_a = ensemble.initial_state()
    state_b = ensemble.initial_state()

    assert state_a == state_a
    assert state_a != state_b
    assert state_a.fragments[0] == state_a.fragments[0]
    assert state_a.fragments[0] != state_b.fragments[0]


def test_energy_property_reflects_workflow_state(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    assert ensemble.energy == float("inf")

    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=1)

    assert np.isfinite(ensemble.energy)
    assert ensemble.energy != float("inf")


def test_aggregate_results_matches_energy_after_one_round(exact_sampler_lassqd):
    """``aggregate_results`` used to return the state that *built* the last
    round's programs (stale by one round) rather than the state
    ``update_state`` produced, so after exactly one round it returned the
    initial state's ``inf`` energy instead of the round's real, finite
    energy -- silently handing back garbage from the guide's headline
    ``ensemble.run().aggregate_results()`` idiom."""
    ensemble, _ = exact_sampler_lassqd
    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=1)

    result = ensemble.aggregate_results()

    assert result.energy == ensemble.energy
    assert result.energy != float("inf")
    assert result is ensemble.workflow_state


POLARIZED_SPECS = [
    FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=1),
    FragmentSpec(orbitals=(2, 3), n_alpha=0, n_beta=1),
]


def test_polarized_fragments_reach_their_vqe_programs(default_test_simulator):
    """Each fragment's own ``n_alpha``/``n_beta`` must reach its VQE program.

    The spin counts are asymmetric per fragment and mirror-imaged between them,
    so a swap at either forwarding site -- ``n_params_per_layer`` or the
    ``_FragmentVQE`` constructor -- prepares the wrong Sz sector. Parameter
    counts cannot catch that, since ``(2, 1)`` and ``(1, 2)`` are mirror images
    with identical excitation counts, so this asserts the occupied set of the
    reference determinant instead.
    """
    ensemble = _lassqd(default_test_simulator, active_spaces=POLARIZED_SPECS)
    state = ensemble.initial_state()
    n_occupied = ensemble._mol.nelectron // 2
    n_core = _workflow._compute_n_core(
        [fragment.spec for fragment in state.fragments], n_occupied
    )
    integrals = _workflow.transform_integrals(ensemble._mol, state.mo_coeff, n_core)

    for index, fragment in enumerate(state.fragments):
        h_eff, g_frag = _workflow.fragment_effective_integrals(
            integrals, state.fragments, index
        )
        with pytest.warns(UserWarning, match="CCSD"):
            program = ensemble._build_fragment_program(
                fragment, h_eff, g_frag, f"fragment_{index}", seed=0
            )

        program.sample_solution(params=np.zeros(program.n_params))

        probs = next(iter(program.best_probs.values()))
        assert len(probs) == 1
        spec = fragment.spec
        occupied = {
            position for position, bit in enumerate(next(iter(probs))) if bit == "1"
        }
        assert occupied == {2 * p for p in range(spec.n_alpha)} | {
            2 * p + 1 for p in range(spec.n_beta)
        }


def test_ccsd_seed_params_is_deterministic_and_correctly_sized(dummy_expval_backend):
    """Exercises the positional-heuristic path (no ``ansatz`` argument)."""
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()
    n_occupied = ensemble._mol.nelectron // 2
    n_core = _workflow._compute_n_core(
        [fragment.spec for fragment in state.fragments], n_occupied
    )
    integrals = _workflow.transform_integrals(ensemble._mol, state.mo_coeff, n_core)
    h_eff, g_frag = _workflow.fragment_effective_integrals(
        integrals, state.fragments, 0
    )
    spec = state.fragments[0].spec
    n_params = LUCJAnsatz.n_params_per_layer(2 * spec.n_orbitals)

    first = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params)
    second = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params)

    assert first is not None
    assert second is not None
    assert first.shape == (n_params,)
    assert np.any(first != 0.0)
    np.testing.assert_allclose(first, second)


def test_ccsd_seed_params_skips_spin_imbalanced_fragments():
    """Restricted CCSD cannot represent a polarized fragment, so seeding warns
    and defers to the optimizer's own initialization rather than failing the
    round. Such fragments are supported as long as the fragments together sum
    to ``Sz = 0``, so this path is reachable."""
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=0)
    h_eff = np.eye(2)
    g_frag = np.zeros((2, 2, 2, 2))

    with pytest.warns(UserWarning, match="CCSD"):
        result = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params=4)

    assert result is None


def test_ccsd_seed_params_uses_amplitude_correspondence_for_uccsd():
    """The seed's exact-statevector energy must land near the fragment's own
    CCSD energy, and clearly beats a permutation of the same values.

    A fragment with only one double excitation (e.g. a minimal H2 fragment)
    cannot exercise ordering, sign, or magnitude errors: with a single value
    there is nothing to permute, no crossed-spin case, and no scale to get
    wrong. This uses a single 4-orbital fragment (8 singles, 18 doubles of
    differing sign) so a scrambled index map, a wrong spin pairing, or an
    un-doubled amplitude all produce a detectably worse energy.
    """
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    spec = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)
    integrals = _workflow.transform_integrals(mol, mo_coeff, n_core=0)
    placeholder = FragmentState(
        spec=spec, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, g_frag = _workflow.fragment_effective_integrals(integrals, [placeholder], 0)
    n_qubits = 2 * spec.n_orbitals
    n_electrons = spec.n_alpha + spec.n_beta
    n_params = UCCSDAnsatz.n_params_per_layer(n_qubits, n_electrons=n_electrons)

    seed = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, UCCSDAnsatz())

    assert seed is not None
    hamiltonian_matrix = _spo_from_integrals(h_eff, g_frag, 0.0).to_matrix()
    ansatz = UCCSDAnsatz()

    def energy_at(params):
        circuit = ansatz.build(
            params, n_qubits=n_qubits, n_layers=1, n_electrons=n_electrons
        )
        state_vector = np.asarray(Statevector.from_instruction(circuit))
        return float(np.real(state_vector.conj() @ hamiltonian_matrix @ state_vector))

    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.kernel()
    ccsd_electronic_energy = coupled_cluster.e_tot - mol.energy_nuc()

    seed_energy = energy_at(seed)
    permuted_energy = energy_at(np.random.default_rng(0).permutation(seed))

    assert seed_energy == pytest.approx(ccsd_electronic_energy, abs=1e-3)
    assert seed_energy < permuted_energy - 0.05


def test_uccsd_seed_singles_match_the_ccsd_t1_amplitudes():
    """The energy assertion above cannot cover the singles, so pin their values.

    ``_ccsd_seed_params`` runs its own RHF on the fragment's effective
    integrals, so CCSD always sees canonical orbitals and ``t1`` is
    Brillouin-suppressed: measured ``max|t1|`` is 1.25e-03 against ``max|t2|``
    of 8.2e-02. Zeroing every single moves the seeded energy by 7.5e-06 and
    sign-flipping them by 2.4e-05 -- both far inside that test's 1e-3 tolerance,
    so the whole singles block could be scrambled or deleted unnoticed. The
    doubles are caught there with 40x margin.
    """
    mol = h4_chain()
    mean_field = scf.RHF(mol).run(verbose=0)
    spec = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)
    integrals = _workflow.transform_integrals(
        mol, np.asarray(mean_field.mo_coeff), n_core=0
    )
    placeholder = FragmentState(
        spec=spec, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, g_frag = _workflow.fragment_effective_integrals(integrals, [placeholder], 0)
    n_params = UCCSDAnsatz.n_params_per_layer(8, n_electrons=4)

    seed = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, UCCSDAnsatz())
    assert seed is not None

    embedded = embedded_fragment_ccsd(h_eff, g_frag, spec)
    t1_spin = cc_addons.spatial2spin(embedded.t1)

    n_spatial, n_occupied = spec.n_orbitals, spec.n_alpha
    singles_seen = 0
    for index, (occupied, unoccupied) in enumerate(
        _uccsd_excitations(n_spatial, (n_occupied, n_occupied))
    ):
        if len(occupied) != 1:
            continue
        spin_o, spatial_o = divmod(occupied[0], n_spatial)
        spin_u, spatial_u = divmod(unoccupied[0], n_spatial)
        expected = -t1_spin[
            2 * spatial_o + spin_o, 2 * (spatial_u - n_occupied) + spin_u
        ]
        assert seed[index] == pytest.approx(expected, abs=1e-12)
        singles_seen += 1

    assert singles_seen == 8
    # Guards against the whole block being zero, which the loop above would
    # otherwise accept if t1 itself came back empty.
    assert np.abs(t1_spin).max() > 1e-6


def test_create_programs_seeds_fresh_fragments_from_ccsd(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())

    for program in ensemble.programs.values():
        assert program._seed_params is not None
        assert program._seed_params.shape == (program.n_params,)
        assert np.any(program._seed_params != 0.0)


def test_create_programs_ccsd_seed_length_scales_with_n_layers(dummy_expval_backend):
    # n_layers=1 makes n_layers * n_params_per_layer indistinguishable from
    # n_params_per_layer alone, so this must use n_layers > 1 to actually
    # discriminate a regression that drops the n_layers factor.
    ensemble = _lassqd(dummy_expval_backend, n_layers=2)
    ensemble.create_programs(ensemble.initial_state())

    for program in ensemble.programs.values():
        expected_length = 2 * UCCSDAnsatz.n_params_per_layer(
            program.n_qubits, n_electrons=program.n_electrons
        )
        assert program._seed_params.shape == (expected_length,)


def test_create_programs_ccsd_seeding_is_deterministic_across_ensembles(
    dummy_expval_backend,
):
    first_ensemble = _lassqd(dummy_expval_backend)
    first_ensemble.create_programs(first_ensemble.initial_state())

    second_ensemble = _lassqd(dummy_expval_backend)
    second_ensemble.create_programs(second_ensemble.initial_state())

    for program_id, program in first_ensemble.programs.items():
        np.testing.assert_allclose(
            program._seed_params,
            second_ensemble.programs[program_id]._seed_params,
        )


def test_create_programs_warm_starts_without_calling_ccsd(dummy_expval_backend, mocker):
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()
    spy = mocker.spy(_workflow, "_ccsd_seed_params")

    spec = state.fragments[0].spec
    warm_params = np.full(
        UCCSDAnsatz.n_params_per_layer(
            2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
        ),
        0.5,
    )
    warm_state = dataclasses.replace(
        state,
        fragments=tuple(
            dataclasses.replace(fragment, params=warm_params)
            for fragment in state.fragments
        ),
    )

    ensemble.create_programs(warm_state)

    spy.assert_not_called()
    for program in ensemble.programs.values():
        np.testing.assert_allclose(program._seed_params, warm_params)


def test_ccsd_failure_falls_back_to_none_with_a_warning(dummy_expval_backend, mocker):
    ensemble = _lassqd(dummy_expval_backend)
    mocker.patch(
        "divi.qprog.workflows._lassqd._workflow.cc.CCSD",
        side_effect=RuntimeError("no convergence"),
    )

    with pytest.warns(UserWarning, match="CCSD"):
        ensemble.create_programs(ensemble.initial_state())

    for program in ensemble.programs.values():
        assert program._seed_params is None


def test_converged_energy_is_pinned_and_above_fci(
    exact_sampler_lassqd, h4_chain_mean_field
):
    """Golden pin on the converged macro-cycle, plus the bound that makes it
    physical.

    The product-state RDM is N-representable, so the energy is a real
    expectation value and cannot dip below full-space FCI. The zeroed-block
    functional this replaced returned -2.5236, i.e. 0.27 Ha *under* FCI, so this
    bound fails on exactly the defect that motivated filling the cross-fragment
    blocks.

    At least one fragment's recovered subspace collapses to a single
    determinant over these rounds -- the warning asserted below -- so what this
    pins is the deterministic pipeline: integral transform, orbital
    permutation, effective-integral construction, RDM assembly and orbital
    re-optimization, rather than correlation recovery.
    """
    ensemble, _ = exact_sampler_lassqd
    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=4)

    assert ensemble.energy == pytest.approx(PRODUCT_STATE_ENERGY, abs=1e-9)
    assert np.trace(ensemble.workflow_state.mo_coeff) == pytest.approx(
        PRODUCT_STATE_MO_TRACE, abs=1e-6
    )

    exact = fci.FCI(h4_chain_mean_field).kernel()[0]
    assert ensemble.energy > exact - 1e-8


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_polarized_fragments_run_the_full_macro_cycle(
    default_test_simulator, mocker, h4_chain_mean_field
):
    """A full run on fragments that are individually polarized but sum to
    ``Sz = 0``, covering the macro-cycle end to end rather than the pieces.

    The assembled RDM stays N-representable regardless of polarization, so the
    energy remains a variational upper bound on full-space FCI.
    """
    ensemble, _ = build_exact_sampler_lassqd(
        default_test_simulator, mocker, active_spaces=POLARIZED_SPECS
    )

    ensemble.run(max_rounds=2)

    assert ensemble.stop_reason in (
        WorkflowStatus.COMPLETE,
        WorkflowStatus.MAX_ROUNDS,
    )
    assert len(ensemble.round_history) >= 1
    exact = fci.FCI(h4_chain_mean_field).kernel()[0]
    assert np.isfinite(ensemble.best_energy)
    assert ensemble.best_energy > exact - 1e-8


def test_best_energy_tracks_the_lowest_round_not_the_last(exact_sampler_lassqd, mocker):
    """The macro-cycle is not guaranteed monotone, and ``energy`` reports the
    last round. Since every round's energy is a variational upper bound, the
    lowest is the tightest bound the run established."""
    ensemble, _ = exact_sampler_lassqd
    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=3)

    assert len(ensemble.energy_history) == len(ensemble.round_history)
    assert ensemble.best_energy == min(ensemble.energy_history)
    assert ensemble.best_energy <= ensemble.energy

    # A non-monotone history must report the minimum, not the final value.
    mocker.patch.object(ensemble, "_energy_history", [-1.5, -2.5, -2.0])
    assert ensemble.best_energy == pytest.approx(-2.5)


def test_best_energy_is_infinite_before_the_first_round(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    assert ensemble.best_energy == float("inf")
    assert ensemble.energy_history == ()


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_round_history_length_matches_macro_cycles(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    ensemble.run(max_rounds=3)

    # How many rounds this fixture needs is data-dependent, so assert the
    # contract rather than a count: capped runs fill max_rounds, converged ones
    # stop short.
    assert 1 <= len(ensemble.round_history) <= 3
    if ensemble.stop_reason is WorkflowStatus.MAX_ROUNDS:
        assert len(ensemble.round_history) == 3
    else:
        assert len(ensemble.round_history) < 3
    # Round numbers are 1-based and contiguous.
    assert [record.number for record in ensemble.round_history] == list(
        range(1, len(ensemble.round_history) + 1)
    )
    # Every recorded round dispatched both fragments. circuit_count and
    # status aren't asserted: ExactSamplerVQE never calls the backend (always
    # 0 circuits), and every reachable record here is unconditionally
    # COMPLETE (a failed round raises instead of leaving a FAILED record).
    for record in ensemble.round_history:
        assert record.program_count == 2


def test_stop_reason_is_max_rounds_when_capped(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=1)
    assert ensemble.stop_reason is WorkflowStatus.MAX_ROUNDS


def test_stop_reason_is_complete_when_converged(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=12)
    assert ensemble.stop_reason is WorkflowStatus.COMPLETE


def test_energy_is_monotonically_non_increasing(exact_sampler_lassqd):
    """Catches sign and orbital-indexing errors that still converge.

    This exact-sampler fixture converges within 2 macro-cycles under
    ``run()``'s own ``is_complete`` gate, which would leave only a single
    pair to compare -- and that pair is already forced to agree within
    ``energy_tol`` by the gate itself, so it proves nothing about
    monotonicity beyond it. Driving macro-cycles directly (bypassing
    ``is_complete``) forces 4 rounds regardless of convergence, giving 3
    independent comparisons.
    """
    ensemble, state = exact_sampler_lassqd
    energies = []
    for _ in range(4):
        ensemble.create_programs(state)
        ensemble.run_one_round(blocking=True)
        with pytest.warns(UserWarning, match="no correlation"):
            state = ensemble.update_state(state)
        energies.append(state.energy)
        ensemble._clear_completed_round()

    for earlier, later in zip(energies, energies[1:]):
        assert later <= earlier + 1e-8


def test_explicit_orbital_indices_are_honored(dummy_expval_backend):
    """The reference discards requested orbital indices; divi must not."""
    pairs = [
        [
            FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
        ],
        [
            FragmentSpec(orbitals=(0, 2), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(1, 3), n_alpha=1, n_beta=1),
        ],
    ]
    permutations = []
    for specs in pairs:
        ensemble = _lassqd(dummy_expval_backend, active_spaces=specs)
        state = ensemble.initial_state()
        permutations.append(state.mo_coeff.copy())

    assert not np.allclose(permutations[0], permutations[1])


def test_automatic_mode_reproduces_the_explicit_fragments(dummy_expval_backend):
    """Regression guard on the automatic partition itself, not just its
    shape: on this molecule/seed, automatic fragmentation's coupling-based
    clustering produces orbitals {0, 2} and {1, 3} -- an interleaved split,
    not the contiguous {0, 1} / {2, 3} blocks used by this module's
    hand-picked explicit fragments. A regression that silently changed
    which orbitals get grouped together (while keeping the same fragment
    sizes and electron counts) would otherwise pass unnoticed."""
    ensemble = _lassqd(
        dummy_expval_backend,
        active_spaces=None,
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )
    state = ensemble.initial_state()
    specs = [fragment.spec for fragment in state.fragments]

    assert len(specs) == 2
    assert all(spec.n_orbitals == 2 for spec in specs)
    assert all(spec.n_alpha == spec.n_beta == 1 for spec in specs)
    auto_orbital_sets = {frozenset(spec.orbitals) for spec in specs}
    assert auto_orbital_sets == {frozenset((0, 2)), frozenset((1, 3))}


def test_repeated_runs_start_from_a_clean_state(dummy_expval_backend, mocker):
    """A second ``run()`` on the same instance must reproduce the first: the
    workflow's own RNG and per-fragment SQD solvers must not carry state
    across runs.

    ``exact_sampler_lassqd`` cannot detect a regression here: it uses
    explicit ``active_spaces``, so it never reaches the automatic
    fragmentation's localization draw, which pulls its restarts straight
    from the workflow's RNG (see ``auto_fragment_specs``). This test uses
    automatic fragmentation instead, so an unreset RNG on the second run
    would localize into a different orbital basis (and very likely a
    different energy) rather than reproducing the first run.
    """
    ensemble = LASSQD(
        h4_chain(),
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        n_batches=2,
        batch_size=8,
        n_sqd_iterations=2,
        seed=0,
        backend=dummy_expval_backend,
        reporting_level=ReportingLevel.OFF,
    )
    mocker.patch.object(LASSQD, "_build_fragment_program", _build_exact_sampler_program)

    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=2)
    first_energy = ensemble.energy
    first_mo_coeff = ensemble.workflow_state.mo_coeff.copy()
    first_rounds = len(ensemble.round_history)

    with pytest.warns(UserWarning, match="no correlation"):
        ensemble.run(max_rounds=2)

    assert len(ensemble.round_history) == first_rounds
    assert ensemble.energy == pytest.approx(first_energy, abs=1e-9)
    np.testing.assert_allclose(ensemble.workflow_state.mo_coeff, first_mo_coeff)


@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_single_fragment_h2_reaches_chemical_accuracy(default_test_simulator):
    """One fragment covering the whole space degenerates to plain SQD, so FCI
    is an exact variational bound (no cross-fragment 2-RDM blocks are zeroed)
    rather than an approximation: a value below FCI would be a genuine bug.

    This fragment needs a larger sampling budget (``n_batches=12,
    batch_size=32``) than the two-fragment case below for SQD to reliably
    capture the correlated determinant; see the module docstring.

    ``seed`` cannot make this deterministic: ``MaestroSimulator.set_seed`` is a
    no-op (maestro does not expose seeding from C++), so shot outcomes depend on
    how many circuits the *process* has already simulated. An earlier version of
    this test pinned ``seed=7`` at ``n_batches=12, batch_size=32`` and passed in
    isolation while failing when other tests ran first -- it landed on the
    mean-field energy instead, the same collapse its budget note described for
    four of five seeds. The budget below is raised until the correlated
    determinant is captured regardless of where the sampler happens to be.

    ``best_energy`` rather than ``energy``, since the macro-cycle need not be
    monotone and every round is a valid upper bound.
    """
    mean_field = scf.RHF(h2_molecule()).run(verbose=0)
    exact = fci.FCI(mean_field).kernel()[0]

    ensemble = LASSQD(
        h2_molecule(),
        active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        max_iterations=200,
        n_batches=24,
        batch_size=256,
        n_sqd_iterations=3,
        seed=7,
        backend=default_test_simulator,
        reporting_level=ReportingLevel.OFF,
    )
    ensemble.run(max_rounds=5)

    assert ensemble.best_energy == pytest.approx(exact, abs=1e-6)
    assert ensemble.best_energy >= exact - 1e-8


@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_two_fragment_h4_lands_on_the_product_state_energy(
    default_test_simulator,
):
    """The two-fragment energy is bracketed: it cannot beat CASCI, and cannot do
    worse than the uncorrelated product state.

    Zeroing the cross-fragment 2-RDM blocks put this 1.47 Ha *below* CASCI,
    because the energy was then a truncated RDM contracted against untruncated
    integrals and not an expectation value at all. The lower bound is what
    catches a regression to that.

    RHF is asserted as a ceiling, not a target. These fragments capture no
    intra-fragment correlation at this sampling budget (measured: within 4.6e-07
    of RHF), but a 2-orbital ``(1a, 1b)`` fragment does have a double excitation
    available -- capturing it would correctly put the energy *below* RHF, so
    pinning equality would read a better result as a regression.
    """
    ensemble = LASSQD(
        h4_chain(),
        active_spaces=[
            FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
        ],
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        max_iterations=60,
        n_batches=6,
        batch_size=16,
        n_sqd_iterations=3,
        seed=7,
        backend=default_test_simulator,
        reporting_level=ReportingLevel.OFF,
    )
    ensemble.run(max_rounds=5)

    mean_field = scf.RHF(h4_chain()).run(verbose=0)
    casci = mcscf.CASCI(mean_field, 4, 4).kernel()[0]

    assert ensemble.energy > casci, "a product state cannot beat CASCI"
    assert ensemble.energy <= mean_field.e_tot + 1e-5
