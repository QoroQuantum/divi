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
import logging

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import cc, fci, gto, mcscf, scf
from pyscf.cc import addons as cc_addons
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog import (
    LASSQD,
    FragmentationConfig,
    ReportingLevel,
    WorkflowStatus,
)
from divi.qprog.algorithms import LUCJAnsatz, QCCAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import _uccsd_excitations
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.workflows._lassqd import _workflow
from divi.qprog.workflows._lassqd._sqd import SQDResult
from divi.qprog.workflows._lassqd._state import (
    FragmentSpec,
    FragmentState,
    LASSQDState,
    validate_fragment_specs,
)
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    PRODUCT_STATE_ENERGY,
    PRODUCT_STATE_MO_TRACE,
    _build_exact_sampler_program,
    ansatz_energy,
    build_exact_sampler_lassqd,
    embedded_fragment_ccsd,
    exact_sampler_lassqd,
    fragment_integrals,
    h2_molecule,
    h4_chain,
    h4_chain_mean_field,
    h8_chain,
    h8_frontier_lassqd,
    lassqd_kwargs,
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
        max_iterations=3,
        n_batches=2,
        batch_size=8,
        n_recovery_iterations=2,
        seed=42,
    )
    kwargs.update(overrides)
    return LASSQD(
        h4_chain(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(**kwargs),
    )


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
        ({"n_recovery_iterations": 0}, "n_recovery_iterations"),
        ({"energy_tol": 0.0}, "energy_tol"),
        ({"coupling_threshold": -1e-3}, "coupling_threshold"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"lambda_penalty": -0.1}, "lambda_penalty"),
        ({"carryover_cutoff": 0.0}, "carryover_cutoff must be positive"),
        ({"carryover_cutoff": None, "max_carryover": 4}, "needs carryover_cutoff"),
        ({"max_carryover": 0}, "max_carryover must be at least 1"),
    ],
)
def test_rejects_invalid_sqd_sizing_arguments(dummy_expval_backend, override, match):
    """These are validated eagerly in the constructor: without this, an
    invalid value (e.g. n_batches=0) would dispatch a full round of paid
    circuits before SQDSolver ever raises, and batch_size=0 was never
    validated at all (silently clamped to a one-determinant pool).
    ``max_iterations=0`` is included for the same reason: unvalidated, it
    reaches the optimizer and raises a bare ``StopIteration``, the least
    actionable exception in Python, instead of a clear ``ValueError`` here.
    The carryover arguments reach ``SQDSolver`` only in ``update_state``, so
    unvalidated they raise after a round's fragment VQEs have already run."""
    with pytest.raises(ValueError, match=match):
        _lassqd(dummy_expval_backend, **override)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"n_active_orbitals": 0}, "n_active_orbitals"),
        ({"n_active_orbitals": -2}, "n_active_orbitals"),
        ({"active_orbitals": [0, 0, 1]}, "duplicates"),
        ({"active_orbitals": [0, 999]}, "out of range"),
        ({"active_orbitals": [0, 1]}, "at least one occupied and one virtual"),
        ({"n_active_orbitals": 4, "fragment_atoms": [[0], [0, 1]]}, "disjoint"),
        ({"n_active_orbitals": 4, "fragment_atoms": [[0], [99]]}, "out of range"),
        (
            {"n_active_orbitals": 4, "local_spins": [0, 0]},
            "local_spins requires fragment_atoms",
        ),
        (
            {
                "n_active_orbitals": 4,
                "fragment_atoms": [[0, 1], [2, 3]],
                "local_spins": [0],
            },
            "local_spins has 1 entries",
        ),
    ],
)
def test_rejects_invalid_automatic_fragmentation_arguments(
    dummy_expval_backend, override, match
):
    """Every automatic-fragmentation argument is validated in the constructor,
    like the sizing arguments above. Deferring to ``initial_state()`` meant a
    bad index surfaced only after an SCF had already run, and an argument
    consumed twice (once by ``auto_fragment_specs``, once by the workflow) would
    arrive empty the second time if given as a generator."""
    with pytest.raises(ValueError, match=match):
        _lassqd(dummy_expval_backend, active_spaces=None, **override)


def test_active_orbitals_accepts_an_exhaustible_iterable():
    """Materialized by the config, so a generator is not consumed by the first
    of the two readers."""
    config = FragmentationConfig(active_orbitals=(o for o in [0, 1, 2, 3]))
    assert config.active_orbitals == (0, 1, 2, 3)


def test_rejects_non_ansatz_instance(dummy_expval_backend):
    with pytest.raises(TypeError, match="ansatz"):
        _lassqd(dummy_expval_backend, ansatz="UCCSD")


def test_rejects_open_shell_molecules(dummy_expval_backend):
    triplet = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        LASSQD(
            triplet,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            backend=dummy_expval_backend,
            **lassqd_kwargs(
                active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)]
            ),
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

    assert spy.call_args.args[3] is ensemble._rng


def test_create_programs_makes_one_vqe_per_fragment(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())

    assert len(ensemble.programs) == 2
    for program in ensemble.programs.values():
        # 2 spatial orbitals per fragment -> 4 qubits.
        assert program.n_qubits == 4


@pytest.mark.filterwarnings("ignore:.*only UCCSDAnsatz")
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
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            **lassqd_kwargs(
                active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)]
            ),
        )


def test_extra_kwargs_are_forwarded_to_each_fragment_vqe(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, n_layers=2)
    ensemble.create_programs(ensemble.initial_state())
    for program in ensemble.programs.values():
        assert program.n_layers == 2


def test_sampling_backend_stays_at_the_ensemble_boundary(
    dummy_expval_backend, make_dummy_simulator
):
    """LASSQD coordinates sampling instead of leaking the backend to fragments."""
    sampling_backend = make_dummy_simulator(100, seed=7)
    ensemble = _lassqd(
        dummy_expval_backend,
        sampling_backend=sampling_backend,
    )
    ensemble.create_programs(ensemble.initial_state())

    assert ensemble.sampling_backend is sampling_backend
    assert all(
        program.sampling_backend is None for program in ensemble.programs.values()
    )


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
    mocker.patch(
        "divi.qprog.workflows._lassqd._workflow.SQDSolver.solve",
        return_value=SQDResult(
            energy=0.0,
            amplitudes=np.array([[1.0]]),
            strings_alpha=(alpha_part,),
            strings_beta=(beta_part,),
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
            n_recovery_iterations=1,
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


def test_carryover_is_on_by_default(dummy_expval_backend):
    """Conventional SQD oscillates across macro-cycles where sampling covers
    only a fraction of the determinant space, so retention is the default."""
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    assert solver.carryover_cutoff == pytest.approx(1e-5)
    assert solver.max_carryover is None


def test_carryover_can_be_turned_off(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend, carryover_cutoff=None)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    assert solver.carryover_cutoff is None


def test_carryover_settings_are_threaded_to_the_solver(dummy_expval_backend):
    """An option users cannot reach from ``LASSQD`` is not an option."""
    ensemble = _lassqd(dummy_expval_backend, carryover_cutoff=1e-4, max_carryover=32)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    assert solver.carryover_cutoff == pytest.approx(1e-4)
    assert solver.max_carryover == 32


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

    ensemble.run(max_rounds=1)

    assert np.isfinite(ensemble.energy)
    assert ensemble.energy != float("inf")


def test_aggregate_results_matches_energy_after_one_round(exact_sampler_lassqd):
    """``aggregate_results`` returns the state ``update_state`` produced, not
    the one that built the round's programs, so after a single round its energy
    is the round's finite energy rather than the initial state's ``inf``."""
    ensemble, _ = exact_sampler_lassqd
    ensemble.run(max_rounds=1)

    result = ensemble.aggregate_results()

    assert result.energy == ensemble.energy
    assert result.energy != float("inf")
    assert result is ensemble.workflow_state


@pytest.mark.parametrize("name", ["local_spins", "fragment_atoms"])
def test_rejects_automatic_only_arguments_with_explicit_active_spaces(
    dummy_expval_backend, name
):
    override = {"fragment_atoms": [[0, 1], [2, 3]]}
    if name == "local_spins":
        override["local_spins"] = [0, 0]
    with pytest.raises(ValueError, match=f"{name} applies to automatic"):
        _lassqd(dummy_expval_backend, **{name: override[name]})


def _h8_lassqd(backend, local_spins=None):
    """``LASSQD`` on H8 fragmented one half-chain per fragment -- the smallest
    layout where a polarized split still leaves an excitation available."""
    return LASSQD(
        h8_chain(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(
            n_active_orbitals=8,
            fragment_atoms=([0, 1, 2, 3], [4, 5, 6, 7]),
            local_spins=local_spins,
            seed=42,
        ),
    )


def test_local_spins_polarize_automatically_built_fragments(dummy_expval_backend):
    """``local_spins`` must survive the remap from localized-column indices to
    register positions, which is where the automatic path rewrites every spec."""
    ensemble = _h8_lassqd(dummy_expval_backend, local_spins=[2, -2])

    state = ensemble.initial_state()

    assert [(f.spec.n_alpha, f.spec.n_beta) for f in state.fragments] == [
        (3, 1),
        (1, 3),
    ]


def test_fragment_atoms_alone_stays_closed_shell(dummy_expval_backend):
    ensemble = _h8_lassqd(dummy_expval_backend)

    state = ensemble.initial_state()

    assert [(f.spec.n_alpha, f.spec.n_beta) for f in state.fragments] == [
        (2, 2),
        (2, 2),
    ]


def test_local_spins_must_sum_to_zero_sz(dummy_expval_backend):
    """A per-fragment override can break the global Sz = 0 invariant, which
    ``auto_fragment_specs`` cannot see on its own since it checks each
    fragment's electron count independently."""
    ensemble = _h8_lassqd(dummy_expval_backend, local_spins=[2, 2])

    with pytest.raises(ValueError, match="total Sz"):
        ensemble.initial_state()


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
    n_act = sum(fragment.spec.n_orbitals for fragment in state.fragments)
    integrals = _workflow.transform_integrals(
        ensemble._mol, state.mo_coeff, n_core, n_act
    )

    for index, fragment in enumerate(state.fragments):
        h_alpha, h_beta, g_frag = _workflow.fragment_effective_integrals(
            integrals, state.fragments, index
        )
        with pytest.warns(UserWarning, match="CCSD"):
            program = ensemble._build_fragment_program(
                fragment, h_alpha, h_beta, g_frag, f"fragment_{index}", seed=0
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
    """Repeated seeding of the same fragment must agree exactly.

    Replaces a version that exercised a positional fallback for non-UCCSD
    ansaetze -- concatenating ``t1``/``t2`` and truncating to the parameter
    count. That correspondence was not physical (its own docstring said so), so
    a LUCJ caller received arbitrary seeds; seeding now warns and skips instead.
    """
    ensemble = _lassqd(dummy_expval_backend)
    state = ensemble.initial_state()
    h_eff, _, g_frag = fragment_integrals(ensemble, state.mo_coeff, state.fragments, 0)
    spec = state.fragments[0].spec
    ansatz = UCCSDAnsatz()
    n_params = ansatz.n_params_per_layer(
        2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
    )

    first = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, ansatz)
    second = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, ansatz)

    assert first is not None
    assert second is not None
    assert first.shape == (n_params,)
    assert np.any(first != 0.0)
    np.testing.assert_allclose(first, second)


def test_a_stalled_orbital_optimizer_is_not_reported_as_converged(
    dummy_expval_backend,
):
    """A round whose inner solve gave up must not count as a fixed point.

    ``optimize_orbitals`` is monotone -- it falls back to the unrotated orbitals
    rather than returning something worse -- so a stalled optimizer yields a
    round whose energy barely moves, which is exactly what convergence looks
    like. Every FeFe run before this was declared COMPLETE on that signature
    while the orbital optimization still had progress left: one round later
    reached a lower energy than a previous three-round "converged" result.
    """
    ensemble = _lassqd(dummy_expval_backend)
    converged = LASSQDState(
        mo_coeff=np.eye(4),
        fragments=(),
        energy=-1.0,
        previous_energy=-1.0 + 1e-12,
    )

    assert ensemble.is_complete(converged)

    stalled = dataclasses.replace(converged, orbitals_converged=False)
    with pytest.warns(UserWarning, match="did not converge"):
        assert not ensemble.is_complete(stalled)


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seeding_warns_when_the_embedding_is_spin_asymmetric(default_test_simulator):
    """A *balanced* fragment next to a polarized neighbour still gets a
    spin-asymmetric embedding, which restricted CCSD seeding has to average.

    The spin-imbalance skip keys on the fragment's own ``n_alpha != n_beta``, so
    it does not fire here -- the asymmetry comes from the neighbour. Without this
    warning the averaging is silent in exactly the regime ``local_spins`` exists
    for.
    """
    ensemble = _lassqd(default_test_simulator, ansatz=UCCSDAnsatz())
    state = ensemble.initial_state()
    polarized = dataclasses.replace(
        state.fragments[1],
        rdm1_alpha=np.diag([0.9, 0.1]),
        rdm1_beta=np.diag([0.2, 0.8]),
        rdm1=np.diag([1.1, 0.9]),
    )
    fragments = (state.fragments[0], polarized)
    h_alpha, h_beta, g_frag = fragment_integrals(ensemble, state.mo_coeff, fragments, 0)
    assert np.abs(h_alpha - h_beta).max() > 1e-6

    with pytest.warns(UserWarning, match="spin channels differ"):
        ensemble._build_fragment_program(
            fragments[0], h_alpha, h_beta, g_frag, "fragment_0", seed=0
        )


def test_ccsd_seed_params_skips_an_ansatz_with_no_correspondence(dummy_expval_backend):
    """UCCSD reads amplitudes off ``t1``/``t2`` directly and LUCJ goes through the
    double factorization, but neither route means anything for an ansatz whose
    parameters are unrelated to coupled-cluster amplitudes. The old fallback
    handed such a caller a truncated concatenation of the two."""
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    h_eff = np.eye(2)
    g_frag = np.zeros((2,) * 4)

    with pytest.warns(UserWarning, match="no correspondence is defined"):
        result = _workflow._ccsd_seed_params(
            h_eff, g_frag, spec, n_params=6, ansatz=QCCAnsatz()
        )

    assert result is None


def test_ccsd_seed_params_routes_lucj_through_the_factorization(mocker):
    """LUCJ is seeded, via the double factorization rather than by reading
    amplitudes off ``t1``/``t2`` -- its parameters are rotation and Coulomb
    angles, not excitation amplitudes."""
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)
    spy = mocker.patch.object(_workflow, "_lucj_seed_params", return_value=None)

    _workflow._ccsd_seed_params(
        np.eye(2), np.zeros((2,) * 4), spec, n_params=6, ansatz=LUCJAnsatz()
    )

    spy.assert_called_once()


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
    integrals = _workflow.transform_integrals(mol, mo_coeff, n_core=0, n_act=4)
    placeholder = FragmentState(
        spec=spec, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, _, g_frag = _workflow.fragment_effective_integrals(
        integrals, [placeholder], 0
    )
    n_params = UCCSDAnsatz.n_params_per_layer(
        2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
    )

    seed = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, UCCSDAnsatz())

    assert seed is not None
    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.kernel()
    ccsd_electronic_energy = coupled_cluster.e_tot - mol.energy_nuc()

    seed_energy = ansatz_energy(seed, h_eff, g_frag, spec)
    permuted_energy = ansatz_energy(
        np.random.default_rng(0).permutation(seed), h_eff, g_frag, spec
    )

    assert seed_energy == pytest.approx(ccsd_electronic_energy, abs=1e-3)
    assert seed_energy < permuted_energy - 0.05


def test_uccsd_seed_beats_the_reference_determinant_in_a_localized_basis(
    dummy_expval_backend,
):
    """The seed must recover correlation in the basis the fragments actually use.

    The test above uses canonical MOs, where the fragment basis and the basis an
    SCF on the fragment's integrals converges to coincide. Real fragments are
    localized: an SCF rotates within their occupied and virtual blocks, leaving
    the reference determinant untouched while permuting which amplitude belongs
    to which orbital pair. Amplitudes read off that rotated solution seeded a
    state *above* the reference determinant -- worse than starting from zeros.
    """
    ensemble = h8_frontier_lassqd(dummy_expval_backend)
    state = ensemble.initial_state()

    for index, fragment in enumerate(state.fragments):
        spec = fragment.spec
        h_alpha, h_beta, g_frag = fragment_integrals(
            ensemble, state.mo_coeff, state.fragments, index
        )
        n_params = UCCSDAnsatz.n_params_per_layer(
            2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
        )
        exact = fci.direct_spin1.kernel(
            h_alpha, g_frag, spec.n_orbitals, (spec.n_alpha, spec.n_beta)
        )[0]
        seed = _workflow._ccsd_seed_params(
            0.5 * (h_alpha + h_beta), g_frag, spec, n_params, UCCSDAnsatz()
        )
        assert seed is not None

        reference = ansatz_energy(np.zeros(n_params), h_alpha, g_frag, spec)
        seeded = ansatz_energy(seed, h_alpha, g_frag, spec)

        assert seeded > exact - 1e-8
        assert (reference - seeded) / (reference - exact) > 0.9


def test_lucj_seed_beats_the_reference_determinant(dummy_expval_backend):
    """The double factorization has to leave LUCJ below Hartree-Fock.

    Every link in that chain is a sign convention -- the Jastrow's factor of
    ``i`` absorbed into one sector's rotation, the conjugate transpose the
    sandwiched block needs, the ``RZZ(-J / 2)`` the pair term maps to -- and
    getting any one of them wrong leaves the seed *at* the reference determinant
    rather than obviously broken. One layer holds only the leading factorization
    term, so the recovered fraction is well short of UCCSD's (measured 0.36).
    """
    ensemble = h8_frontier_lassqd(dummy_expval_backend)
    state = ensemble.initial_state()

    for index, fragment in enumerate(state.fragments):
        spec = fragment.spec
        h_alpha, h_beta, g_frag = fragment_integrals(
            ensemble, state.mo_coeff, state.fragments, index
        )
        n_params = LUCJAnsatz.n_params_per_layer(
            2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
        )
        exact = fci.direct_spin1.kernel(
            h_alpha, g_frag, spec.n_orbitals, (spec.n_alpha, spec.n_beta)
        )[0]
        seed = _workflow._ccsd_seed_params(
            0.5 * (h_alpha + h_beta), g_frag, spec, n_params, LUCJAnsatz(), {}
        )
        assert seed is not None

        reference = ansatz_energy(
            np.zeros(n_params), h_alpha, g_frag, spec, LUCJAnsatz()
        )
        seeded = ansatz_energy(seed, h_alpha, g_frag, spec, LUCJAnsatz())

        assert seeded > exact - 1e-8
        assert (reference - seeded) / (reference - exact) > 0.3


def _seed_gain(ansatz, h_alpha, h_beta, g_frag, spec, seed_integrals=None, **kwargs):
    """``_seed_energy_gain`` for a seed on one fragment's integrals.

    ``seed_integrals`` builds the seed from a *different* ``(h, g)`` than the
    Hamiltonian it is scored against, which is what a basis mismatch is.
    """
    build_kwargs = {
        "n_electrons": spec.n_alpha + spec.n_beta,
        "n_alpha": spec.n_alpha,
        "n_beta": spec.n_beta,
        **kwargs,
    }
    n_params = type(ansatz).n_params_per_layer(2 * spec.n_orbitals, **build_kwargs)
    h_seed, g_seed = seed_integrals or (0.5 * (h_alpha + h_beta), g_frag)
    seed = _workflow._ccsd_seed_params(h_seed, g_seed, spec, n_params, ansatz, kwargs)
    assert seed is not None
    hamiltonian = _spo_from_integrals(
        h_alpha, g_frag, constant=0.0, one_body_beta=h_beta
    )
    return _workflow._seed_energy_gain(
        seed, hamiltonian, ansatz, 2 * spec.n_orbitals, 1, build_kwargs
    )


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_acceptance_rejects_amplitudes_on_the_wrong_excitations():
    """The failure class the check exists for, and the one a stationarity
    precondition provably cannot see.

    Running CCSD in a basis rotated within the occupied and virtual blocks --
    what an SCF canonicalization does -- leaves the reference determinant's
    energy identical while permuting which amplitude belongs to which orbital
    pair. ``F_ov`` transforms as ``U_o^T F_ov U_v``, so a stationarity check
    stays satisfied throughout. Permuting the seed vector is that same corruption
    applied directly, and the energy comparison catches it.
    """
    ensemble = h8_frontier_lassqd(None)
    state = ensemble.initial_state()
    spec = state.fragments[0].spec
    h_eff, _, g_frag = fragment_integrals(ensemble, state.mo_coeff, state.fragments, 0)

    ansatz = UCCSDAnsatz()
    build_kwargs = {
        "n_electrons": spec.n_alpha + spec.n_beta,
        "n_alpha": spec.n_alpha,
        "n_beta": spec.n_beta,
    }
    n_qubits = 2 * spec.n_orbitals
    n_params = UCCSDAnsatz.n_params_per_layer(n_qubits, **build_kwargs)
    seed = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, ansatz, {})
    assert seed is not None
    hamiltonian = _spo_from_integrals(h_eff, g_frag, constant=0.0)

    def gain(params):
        return _workflow._seed_energy_gain(
            params, hamiltonian, ansatz, n_qubits, 1, build_kwargs
        )

    scrambled = np.random.default_rng(0).permutation(seed)

    assert gain(seed) > _workflow._SEED_ACCEPTANCE_MARGIN
    assert gain(scrambled) < _workflow._SEED_ACCEPTANCE_MARGIN


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_acceptance_admits_a_polarized_non_stationary_fragment():
    """A spin-polarized fragment has no shared spatial basis making both spin
    channels Hartree-Fock stationary, so the old precondition refused it
    outright -- which is what left the diiron benchmark unseeded. The seed is
    worth keeping there."""
    ensemble = h8_frontier_lassqd(
        local_spins=[2, -2], fragment_atoms=([0, 1, 2, 3], [4, 5, 6, 7])
    )
    state = ensemble.initial_state()
    spec = state.fragments[0].spec
    assert spec.n_alpha != spec.n_beta
    h_alpha, h_beta, g_frag = fragment_integrals(
        ensemble, state.mo_coeff, state.fragments, 0
    )

    gain = _seed_gain(
        LUCJAnsatz(), h_alpha, h_beta, g_frag, spec, trailing_rotation=True
    )

    assert gain > _workflow._SEED_ACCEPTANCE_MARGIN


def test_seed_acceptance_skips_a_fragment_too_wide_to_check():
    """Above the exact-check width the seed is accepted unchecked rather than
    discarded: refusing a good seed is the failure this replaced."""
    spec = FragmentSpec(orbitals=tuple(range(11)), n_alpha=2, n_beta=2)
    n_qubits = 2 * spec.n_orbitals
    assert n_qubits > _workflow._SEED_CHECK_MAX_QUBITS

    gain = _workflow._seed_energy_gain(
        np.zeros(4), object(), LUCJAnsatz(), n_qubits, 1, {}
    )

    assert gain is None


def test_uccsd_seed_singles_match_the_ccsd_t1_amplitudes():
    """The energy assertion above cannot cover the singles, so pin their values.

    That fragment's orbitals are canonical MOs, so ``t1`` is
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
        mol, np.asarray(mean_field.mo_coeff), n_core=0, n_act=4
    )
    placeholder = FragmentState(
        spec=spec, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, _, g_frag = _workflow.fragment_effective_integrals(
        integrals, [placeholder], 0
    )
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


@pytest.mark.filterwarnings("ignore:.*only UCCSDAnsatz has parameters")
@pytest.mark.filterwarnings("ignore:CCSD seeding skipped")
@pytest.mark.parametrize(
    "ansatz_kwargs",
    [
        {"trailing_rotation": True},
        {"shared_spin_params": True},
        {"rotation_depth": 1},
        {"same_spin_pairs": [], "opposite_spin_pairs": [(0, 1)]},
    ],
    ids=lambda kwargs: "-".join(sorted(kwargs)),
)
def test_ansatz_kwargs_reach_every_fragment_vqe(dummy_expval_backend, ansatz_kwargs):
    """``ansatz_kwargs`` must reach the fragment VQE, the seed-length
    calculation, and the seed's own layout -- three places
    ``_build_fragment_program`` derives separately.

    A mismatch is silent: the VQE would reject a seed vector sized for the
    other circuit, or the optimizer would tune parameters no gate reads.
    """
    ensemble = _lassqd(
        dummy_expval_backend,
        ansatz=LUCJAnsatz(),
        ansatz_kwargs=ansatz_kwargs,
    )
    ensemble.create_programs(ensemble.initial_state())

    for program in ensemble.programs.values():
        expected = LUCJAnsatz.n_params_per_layer(program.n_qubits, **ansatz_kwargs)
        assert program.n_params_per_layer == expected
        assert len(program.cost_circuit.parameters) == program.n_params
        if program._seed_params is not None:
            assert program._seed_params.shape == (program.n_params,)


def test_lucj_seeding_skips_a_shared_spin_params_ansatz(dummy_expval_backend):
    """The factorization gives each spin sector its own rotation, so there is no
    seed to write into a single shared set. Warning and deferring beats writing
    the alpha sector's angles into both."""
    ensemble = _lassqd(
        dummy_expval_backend,
        ansatz=LUCJAnsatz(),
        ansatz_kwargs={"shared_spin_params": True},
    )

    with pytest.warns(UserWarning, match="shared_spin_params cannot hold"):
        ensemble.create_programs(ensemble.initial_state())

    for program in ensemble.programs.values():
        assert program._seed_params is None


def test_round_reports_record_each_round_as_it_completes(exact_sampler_lassqd):
    """The per-round record must land when the round does, not at the end of the
    run, so a capped or interrupted run keeps every finished round's numbers."""
    ensemble, state = exact_sampler_lassqd

    assert ensemble.round_reports == ()

    for expected_rounds in (1, 2):
        ensemble.create_programs(state)
        ensemble.run_one_round(blocking=True)
        state = ensemble.update_state(state)
        ensemble._clear_completed_round()

        assert len(ensemble.round_reports) == expected_rounds
        report = ensemble.round_reports[-1]
        assert report.number == expected_rounds
        assert report.energy == ensemble.energy_history[-1]
        assert report.rotation_pairs > 0
        assert report.orbital_evaluations > report.orbital_iterations >= 0
        assert len(report.subspace_sizes) == len(state.fragments)
        assert all(size >= 1 for size in report.subspace_sizes)
        assert report.orbital_seconds >= 0.0
        assert f"Round {expected_rounds} done" in report.summary()

    # The initial state's energy is ``inf``, so round 1 has no predecessor to
    # subtract and must not report an infinite change.
    assert ensemble.round_reports[0].energy_change is None
    assert "first round" in ensemble.round_reports[0].summary()
    assert ensemble.round_reports[1].energy_change == pytest.approx(
        ensemble.energy_history[1] - ensemble.energy_history[0]
    )


def test_round_reports_are_cleared_by_a_workflow_reset(exact_sampler_lassqd):
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    ensemble.update_state(state)
    assert ensemble.round_reports

    ensemble._reset_workflow_state()

    assert ensemble.round_reports == ()


def test_update_state_names_each_reduction_stage(exact_sampler_lassqd, mocker):
    """Each classical stage of the reduction reports itself, so the display
    distinguishes SQD recovery from the orbital solve rather than only showing
    that a round is open."""
    ensemble, state = exact_sampler_lassqd
    spy = mocker.spy(ensemble, "_emit_workflow_round_stage")

    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    ensemble.update_state(state)

    reported = [call.args[0] for call in spy.call_args_list]
    assert any("SQD" in message for message in reported)
    assert any("RDM" in message for message in reported)
    assert any("orbital" in message.lower() for message in reported)
    assert reported[-1] == ensemble.round_reports[-1].summary()
    # Only the outcome is marked final; the stages it passed through are not.
    assert [call.kwargs.get("final", False) for call in spy.call_args_list][-1] is True
    assert not any(call.kwargs.get("final", False) for call in spy.call_args_list[:-1])


def test_round_summary_is_logged_not_only_painted_on_the_progress_row(
    exact_sampler_lassqd, caplog
):
    """The round's outcome must survive a run whose output is redirected.

    The workflow-round row is transient -- later frames overwrite its text -- so
    a summary written only there leaves no record of a completed round. The
    stage names are progress and may stay transient; the outcome may not.
    """
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)

    with caplog.at_level(logging.INFO, logger="divi.qprog.ensemble"):
        ensemble.update_state(state)

    summary = ensemble.round_reports[-1].summary()
    assert summary in caplog.text
    # A stage label is progress only, and must not be logged alongside it.
    assert "Re-optimizing orbitals" not in caplog.text


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


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_converged_energy_is_pinned_and_above_fci(
    exact_sampler_lassqd, h4_chain_mean_field
):
    """Golden pin on the converged macro-cycle, plus the bound that makes it
    physical.

    The product-state RDM is N-representable, so the energy is a real
    expectation value and cannot dip below full-space FCI. What the pin covers
    is the deterministic pipeline: integral transform, orbital permutation,
    effective-integral construction, RDM assembly and orbital re-optimization.
    """
    ensemble, _ = exact_sampler_lassqd
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


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_macro_cycle_uses_separate_optimization_and_sampling_backends(
    default_test_simulator, sampling_test_simulator, mocker
):
    """One LASSQD round batches final fragment samples on its second backend."""
    optimization_submit = mocker.spy(default_test_simulator, "submit_circuits")
    sampling_submit = mocker.spy(sampling_test_simulator, "submit_circuits")
    ensemble = _lassqd(
        default_test_simulator,
        sampling_backend=sampling_test_simulator,
        max_iterations=1,
        max_orbital_iterations=1,
        n_batches=1,
        n_recovery_iterations=1,
    )

    with pytest.warns(UserWarning, match="Orbital optimisation stopped"):
        ensemble.run(max_rounds=1)

    assert optimization_submit.call_count > 0
    sampling_submit.assert_called_once()
    assert len(ensemble.round_history) == 1


def test_carryover_survives_a_full_macro_cycle(
    default_test_simulator, mocker, h4_chain_mean_field
):
    """Carryover through a whole macro-cycle rather than ``SQDSolver.solve``
    alone: RDM assembly and the orbital step must accept the enlarged subspace,
    and the energy must stay a valid bound.

    Only the integration is asserted, not a gain. The exact sampler's
    distribution is fixed and already symmetry-valid, so bit-flip recovery is a
    no-op and every iteration draws from the same peaked distribution -- leaving
    retention nothing that varies to compound. The size of the gain is measured
    in ``test_sqd.py``, over a flat distribution where draws do differ.
    """
    specs = [FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)]
    settings = dict(
        active_spaces=specs,
        n_batches=2,
        batch_size=6,
        n_recovery_iterations=3,
    )

    plain, _ = build_exact_sampler_lassqd(default_test_simulator, mocker, **settings)
    plain.run(max_rounds=1)

    carrying, _ = build_exact_sampler_lassqd(
        default_test_simulator, mocker, carryover_cutoff=1e-2, **settings
    )
    carrying.run(max_rounds=1)

    exact = fci.FCI(h4_chain_mean_field).kernel()[0]
    assert carrying.stop_reason in (
        WorkflowStatus.COMPLETE,
        WorkflowStatus.MAX_ROUNDS,
    )
    assert np.isfinite(carrying.best_energy)
    assert carrying.best_energy > exact - 1e-8
    assert carrying.best_energy <= plain.best_energy + 1e-9
    assert (
        carrying.round_reports[0].subspace_sizes[0]
        >= plain.round_reports[0].subspace_sizes[0]
    )


def test_best_energy_tracks_the_lowest_round_not_the_last(exact_sampler_lassqd, mocker):
    """The macro-cycle is not guaranteed monotone, and ``energy`` reports the
    last round. Since every round's energy is a variational upper bound, the
    lowest is the tightest bound the run established."""
    ensemble, _ = exact_sampler_lassqd
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
    ensemble.run(max_rounds=1)
    assert ensemble.stop_reason is WorkflowStatus.MAX_ROUNDS


def test_stop_reason_is_complete_when_converged(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd
    ensemble.run(max_rounds=12)
    assert ensemble.stop_reason is WorkflowStatus.COMPLETE


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_energy_is_monotonically_non_increasing(exact_sampler_lassqd):
    """Catches sign and orbital-indexing errors that still converge.

    Drives the macro-cycles directly rather than through ``run()``, so all 4
    rounds happen regardless of ``is_complete``, giving 3 comparisons that the
    ``energy_tol`` gate does not already force to agree.
    """
    ensemble, state = exact_sampler_lassqd
    energies = []
    for _ in range(4):
        ensemble.create_programs(state)
        ensemble.run_one_round(blocking=True)
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
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=dummy_expval_backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(
            n_active_orbitals=4,
            max_orbitals_per_fragment=2,
            n_batches=2,
            batch_size=8,
            n_recovery_iterations=2,
            seed=0,
        ),
    )
    mocker.patch.object(LASSQD, "_build_fragment_program", _build_exact_sampler_program)

    ensemble.run(max_rounds=2)
    first_energy = ensemble.energy
    first_mo_coeff = ensemble.workflow_state.mo_coeff.copy()
    first_rounds = len(ensemble.round_history)

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
    how many circuits the *process* has already simulated. The budget below is
    raised until the correlated determinant is captured regardless.

    ``best_energy`` rather than ``energy``, since the macro-cycle need not be
    monotone and every round is a valid upper bound.
    """
    mean_field = scf.RHF(h2_molecule()).run(verbose=0)
    exact = fci.FCI(mean_field).kernel()[0]

    ensemble = LASSQD(
        h2_molecule(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=default_test_simulator,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(
            active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
            max_iterations=200,
            n_batches=24,
            batch_size=256,
            n_recovery_iterations=3,
            seed=7,
        ),
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
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=default_test_simulator,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(
            active_spaces=[
                FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
                FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
            ],
            max_iterations=60,
            n_batches=6,
            batch_size=16,
            n_recovery_iterations=3,
            seed=7,
        ),
    )
    ensemble.run(max_rounds=5)

    mean_field = scf.RHF(h4_chain()).run(verbose=0)
    casci = mcscf.CASCI(mean_field, 4, 4).kernel()[0]

    assert ensemble.energy > casci, "a product state cannot beat CASCI"
    assert ensemble.energy <= mean_field.e_tot + 1e-5
