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

import copy
import dataclasses
import logging

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import cc, fci, gto, mcscf, scf
from pyscf.cc import addons as cc_addons
from pyscf.fci import cistring
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog import (
    LASSQD,
    FragmentationConfig,
    LASSQDPreparationMode,
    ReportingLevel,
    SQDConfig,
    WorkflowStatus,
)
from divi.qprog.algorithms import LUCJAnsatz, QCCAnsatz, UCCSDAnsatz
from divi.qprog.algorithms._ansatze import (
    _uccsd_excitations,
    lucj_jastrow_pairs,
    n_rotation_params,
)
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import HamiltonianProblem, MolecularProblem
from divi.qprog.workflows._lassqd import _workflow
from divi.qprog.workflows._lassqd._integrals import OrbitalSolve
from divi.qprog.workflows._lassqd._preparation import (
    LinearMethodFragmentProgram,
)
from divi.qprog.workflows._lassqd._sqd import SQDResult, SQDSolver
from divi.qprog.workflows._lassqd._state import (
    FragmentSpec,
    FragmentState,
    LASSQDState,
    validate_fragment_specs,
)
from tests._helpers import exact_match
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    PRODUCT_STATE_ENERGY,
    PRODUCT_STATE_MO_TRACE,
    _build_exact_sampler_program,
    ansatz_energy,
    build_exact_sampler_lassqd,
    dense_fci_energy,
    embedded_fragment_ccsd,
    exact_sampler_lassqd,
    fragment_integrals,
    fragment_problem,
    h2_mean_field,
    h2_molecule,
    h4_chain,
    h4_chain_mean_field,
    h8_chain,
    h8_frontier_lassqd,
    lassqd_kwargs,
    mo_integrals,
    uniform_full_space_probs,
)


def test_fragment_spec_normalizes_orbitals_to_a_tuple():
    spec = FragmentSpec(orbitals=[2, 3], n_alpha=1, n_beta=1)
    assert spec.orbitals == (2, 3)
    assert spec.n_orbitals == 2


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


@pytest.mark.parametrize(
    "specs",
    [
        pytest.param([((0, 1), 1, 1), ((2, 3), 1, 1)], id="disjoint_in_range"),
        pytest.param([((0, 3), 1, 1), ((1, 2), 1, 1)], id="consistent_electron_count"),
        pytest.param([((0, 1), 1, 0), ((2, 3), 1, 2)], id="spin_imbalanced"),
    ],
)
def test_validate_fragment_specs_accepts(specs):
    """Spin-imbalanced fragments are the antiferromagnetic case a localised
    active space exists to describe, so they must be accepted as long as the
    fragments' electrons still add up and each keeps an excitation available in
    at least one spin channel."""
    validate_fragment_specs(
        [FragmentSpec(orbitals=o, n_alpha=a, n_beta=b) for o, a, b in specs],
        n_orbitals_total=4,
        n_occupied=2,
    )


@pytest.mark.parametrize(
    "specs, match",
    [
        pytest.param([((0, 1), 1, 1), ((1, 2), 1, 1)], "overlap", id="overlap"),
        pytest.param([((0, 9), 1, 1)], "out of range", id="out_of_range"),
        pytest.param([((0, 4), 1, 1)], "out of range", id="one_past_the_end"),
        pytest.param([((0, 1, 2), 1, 1)], "electron", id="electron_count_mismatch"),
        pytest.param(
            [((0, 1), 2, 0), ((2, 3), 0, 2)],
            "no excitation available",
            id="spin_saturated_fragment",
        ),
        pytest.param([((0, 1), 2, 1), ((2, 3), 1, 0)], "Sz", id="nonzero_total_sz"),
        pytest.param([((0, 1), 1, 0)], "declare", id="inconsistent_electron_totals"),
        pytest.param(
            [((0, 1), 2, 2)], "no excitation available", id="fully_occupied_fragment"
        ),
        pytest.param(
            [((2, 0), 1, 1), ((1, 3), 1, 1)],
            "lists virtual orbital 2 before occupied orbital 0",
            id="virtual_listed_before_occupied",
        ),
    ],
)
def test_validate_fragment_specs_rejects(specs, match):
    """Fragments must cover exactly the active electrons (a shortfall silently
    yields an energy for the wrong electron count), sum to zero total Sz (an
    Sz=1 split of a closed-shell molecule ran to ``COMPLETE`` in the wrong spin
    sector), leave each fragment an excitation in some spin channel (a
    saturated or fully occupied fragment gives UCCSD zero parameters), and list
    occupied orbitals first (the reference determinant and the initial density
    fill a fragment's orbitals in order, so a virtual listed first is filled)."""
    with pytest.raises(ValueError, match=match):
        validate_fragment_specs(
            [FragmentSpec(orbitals=o, n_alpha=a, n_beta=b) for o, a, b in specs],
            n_orbitals_total=4,
            n_occupied=2,
        )


_H4_FRAGMENTS = (
    FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
    FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
)


def _lassqd(
    backend,
    *,
    problem=None,
    preparation_mode=LASSQDPreparationMode.VQE,
    **overrides,
):
    """Two-fragment H4 ensemble with a sampling budget sized for the suite.

    VQE mode supplies an optimizer and UCCSD ansatz unless overridden, while
    linear-method mode keeps its production defaults.
    """
    kwargs = dict(
        active_spaces=list(_H4_FRAGMENTS),
        n_batches=2,
        batch_size=8,
        n_recovery_iterations=2,
        seed=42,
    )
    kwargs.update(overrides)
    if preparation_mode is LASSQDPreparationMode.VQE:
        kwargs.setdefault("max_iterations", 3)
        kwargs.setdefault("optimizer", ScipyOptimizer(ScipyMethod.COBYLA))
        kwargs.setdefault("ansatz", UCCSDAnsatz())
    return LASSQD(
        problem or MolecularProblem.from_molecule(h4_chain()),
        preparation_mode=preparation_mode,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(**kwargs),
    )


def _raw_lassqd(backend, **kwargs):
    """Construct LASSQD directly, without :func:`_lassqd`'s mode-aware defaults."""
    return LASSQD(
        MolecularProblem.from_molecule(h4_chain()),
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **kwargs,
        **lassqd_kwargs(active_spaces=list(_H4_FRAGMENTS)),
    )


def test_defaults_to_paper_linear_method_and_lucj(dummy_expval_backend):
    ensemble = _raw_lassqd(dummy_expval_backend)

    assert ensemble.preparation_mode is LASSQDPreparationMode.LINEAR_METHOD
    assert isinstance(ensemble.ansatz, LUCJAnsatz)


def test_rejects_unknown_preparation_mode(dummy_expval_backend):
    with pytest.raises(ValueError, match="Unknown LASSQD preparation_mode"):
        _raw_lassqd(dummy_expval_backend, preparation_mode="not-a-mode")


class CustomLUCJAnsatz(LUCJAnsatz):
    """LUCJ variant whose behavior cannot be represented by the paper path."""


def test_linear_method_rejects_custom_lucj_subclass(dummy_expval_backend):
    message = (
        "The linear_method preparation mode requires LUCJAnsatz in its default "
        "form; select preparation_mode='vqe' to use another ansatz."
    )
    with pytest.raises(TypeError, match=exact_match(message)):
        _raw_lassqd(dummy_expval_backend, ansatz=CustomLUCJAnsatz())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            dict(preparation_mode=LASSQDPreparationMode.VQE),
            "optimizer is required",
            id="vqe-without-optimizer",
        ),
        pytest.param(
            dict(ansatz=UCCSDAnsatz()),
            "requires LUCJAnsatz",
            id="linear-method-with-foreign-ansatz",
        ),
        pytest.param(
            dict(optimizer=ScipyOptimizer(ScipyMethod.COBYLA)),
            "only used by",
            id="linear-method-with-optimizer",
        ),
    ],
)
def test_preparation_mode_rejects_mismatched_configuration(
    dummy_expval_backend, kwargs, match
):
    """Each message must be distinguishable: both optimizer errors mention
    'optimizer' and 'vqe', so a loose pattern would let one guard pass the
    other's test."""
    with pytest.raises(TypeError, match=match):
        _raw_lassqd(dummy_expval_backend, **kwargs)


@pytest.mark.parametrize(
    ("ansatz_kwargs", "expected_warnings"),
    [
        pytest.param({}, 1, id="implicit-uccsd-warns"),
        pytest.param({"ansatz": UCCSDAnsatz()}, 0, id="explicit-uccsd-silent"),
    ],
)
def test_vqe_mode_warns_only_when_it_implicitly_selects_uccsd(
    dummy_expval_backend, recwarn, ansatz_kwargs, expected_warnings
):
    ensemble = _raw_lassqd(
        dummy_expval_backend,
        preparation_mode="vqe",
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        **ansatz_kwargs,
    )

    ansatz_warnings = [
        recorded
        for recorded in recwarn
        if issubclass(recorded.category, UserWarning)
        and "UCCSDAnsatz" in str(recorded.message)
    ]
    assert len(ansatz_warnings) == expected_warnings
    assert ensemble.preparation_mode is LASSQDPreparationMode.VQE
    assert isinstance(ensemble.ansatz, UCCSDAnsatz)


@pytest.mark.parametrize(
    "override",
    [{"n_active_orbitals": 4}, {"active_spaces": None}],
    ids=["both", "neither"],
)
def test_requires_exactly_one_fragment_specification(dummy_expval_backend, override):
    message = (
        "Pass exactly one of active_spaces (explicit fragment layout), "
        "n_active_orbitals (frontier selection), or active_orbitals "
        "(explicit MO indices)."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        _lassqd(dummy_expval_backend, **override)


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
        ({"carryover_cutoff": 1.0}, "carryover_cutoff must be below 1"),
        ({"carryover_cutoff": None, "max_carryover": 4}, "needs carryover_cutoff"),
        ({"max_carryover": 0}, "max_carryover must be at least 1"),
        ({"max_dim": 0}, "max_dim entries must be at least 1"),
        ({"max_dim": (2, 0)}, "max_dim entries must be at least 1"),
        ({"max_dim": (1, 2, 3)}, "got 3 entries"),
        ({"recovery_energy_tol": -1.0}, "recovery_energy_tol"),
        ({"recovery_occupancies_tol": -1.0}, "recovery_occupancies_tol"),
        ({"max_orbitals_per_fragment": 0}, "max_orbitals_per_fragment"),
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
        ({"n_active_orbitals": 1}, "n_active_orbitals must be at least 2"),
        (
            {"n_active_orbitals": 4, "max_orbitals_per_fragment": 1},
            "max_orbitals_per_fragment must be at least 2",
        ),
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
    message = "ansatz must be an Ansatz instance; got str."
    with pytest.raises(TypeError, match=exact_match(message)):
        _lassqd(dummy_expval_backend, ansatz="UCCSD")


def _pennylane_problem():
    qp = pytest.importorskip("pennylane")
    molecule = qp.qchem.Molecule(["H", "H"], h2_molecule().atom_coords())
    return MolecularProblem.from_molecule(molecule)


def _integral_problem():
    return MolecularProblem(np.zeros((2, 2)), np.zeros((2,) * 4), n_alpha=1, n_beta=1)


def _needs_from_molecule(got):
    return (
        "LASSQD expects a MolecularProblem built with from_molecule; bare "
        f"integrals carry no atomic-orbital basis. Got {got}."
    )


@pytest.mark.parametrize(
    "make_problem, message",
    [
        (_integral_problem, _needs_from_molecule("MolecularProblem")),
        (
            _pennylane_problem,
            "LASSQD needs a problem built from a pyscf Mole or restricted "
            "mean-field object, got Molecule.",
        ),
        (h4_chain, _needs_from_molecule("Mole")),
    ],
    ids=["integrals", "pennylane", "bare-molecule"],
)
def test_rejects_problems_without_a_pyscf_molecule(
    dummy_expval_backend, make_problem, message
):
    with pytest.raises(TypeError, match=exact_match(message)):
        LASSQD(
            make_problem(),
            backend=dummy_expval_backend,
            **lassqd_kwargs(active_spaces=list(_H4_FRAGMENTS)),
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


@pytest.mark.parametrize(
    "n_alpha, n_beta", [(2, 2), (2, 1)], ids=["closed-shell", "polarized"]
)
def test_diagonal_rdm_guess_is_the_reference_determinant(n_alpha, n_beta):
    """PySCF's RDMs of the aufbau determinant are the oracle. Exchange is the
    part a product of occupations misses: a doubly occupied orbital carries
    ``Gamma[p, p, p, p] = 2``, not ``n_p ** 2 = 4``. The polarized case keeps
    the alpha and beta halves apart, which a spin-traced guess would not."""
    n_orb = 3
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=n_alpha, n_beta=n_beta)
    civec = np.zeros(
        (cistring.num_strings(n_orb, n_alpha), cistring.num_strings(n_orb, n_beta))
    )
    civec[0, 0] = 1.0
    expected = (
        *fci.direct_spin1.make_rdm12(civec, n_orb, (n_alpha, n_beta)),
        *fci.direct_spin1.make_rdm1s(civec, n_orb, (n_alpha, n_beta)),
    )

    for actual, reference in zip(
        _workflow._diagonal_rdm_guess(spec), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, atol=1e-12)


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

    state = ensemble.initial_state()

    assert spy.call_args.args[3] is ensemble._rng
    assert state.mo_coeff.shape[1] == 4


def test_initial_state_forwards_the_mean_field_core_hamiltonian_to_auto_fragment_specs(
    dummy_expval_backend, mocker
):
    mean_field = scf.RHF(h4_chain()).sfx2c1e().run(verbose=0)
    ensemble = _lassqd(
        dummy_expval_backend,
        problem=MolecularProblem.from_molecule(mean_field),
        active_spaces=None,
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )
    spy = mocker.spy(_workflow, "auto_fragment_specs")

    ensemble.initial_state()

    np.testing.assert_allclose(
        spy.call_args.kwargs["h_ao"], mean_field.get_hcore(), atol=1e-12
    )


def test_create_programs_makes_one_vqe_per_fragment(dummy_expval_backend):
    ensemble = _lassqd(dummy_expval_backend)
    ensemble.create_programs(ensemble.initial_state())

    assert len(ensemble.programs) == 2
    for program in ensemble.programs.values():
        # 2 spatial orbitals per fragment -> 4 qubits.
        assert program.n_qubits == 4


@pytest.mark.parametrize(
    ("preparation_mode", "expected_type", "ansatz"),
    [
        pytest.param(
            LASSQDPreparationMode.LINEAR_METHOD,
            LinearMethodFragmentProgram,
            None,
            id="linear-method",
        ),
        pytest.param(
            LASSQDPreparationMode.VQE,
            _workflow._FragmentVQE,
            LUCJAnsatz(),
            id="vqe",
            marks=pytest.mark.filterwarnings("ignore:.*only UCCSDAnsatz"),
        ),
    ],
)
def test_create_programs_builds_the_program_type_for_its_mode(
    dummy_expval_backend, preparation_mode, expected_type, ansatz
):
    """VQE mode hands the configured non-default ansatz to every program."""
    overrides = {} if ansatz is None else {"ansatz": ansatz}
    ensemble = _lassqd(
        dummy_expval_backend, preparation_mode=preparation_mode, **overrides
    )
    ensemble.create_programs(ensemble.initial_state())

    assert len(ensemble.programs) == 2
    assert all(
        isinstance(program, expected_type) for program in ensemble.programs.values()
    )
    if ansatz is not None:
        assert all(program.ansatz is ansatz for program in ensemble.programs.values())


def test_linear_method_defers_classical_preparation_to_dispatch(
    dummy_expval_backend, mocker
):
    prepare = mocker.patch(
        "divi.qprog.workflows._lassqd._preparation.prepare_lucj_fragment"
    )
    ensemble = _lassqd(
        dummy_expval_backend, preparation_mode=LASSQDPreparationMode.LINEAR_METHOD
    )

    ensemble.create_programs(ensemble.initial_state())

    prepare.assert_not_called()


def test_linear_method_samples_once_per_fragment_over_a_macro_cycle(
    default_test_simulator,
):
    ensemble = _lassqd(
        default_test_simulator,
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        seed=9,
    )

    ensemble.run(max_rounds=1)

    assert all(program.has_results() for program in ensemble.programs.values())
    assert all(
        program.total_circuit_count == 1 for program in ensemble.programs.values()
    )
    assert ensemble.total_circuit_count == 2
    assert np.isfinite(ensemble.energy)
    assert ensemble.stop_reason is WorkflowStatus.MAX_ROUNDS


@pytest.mark.parametrize("preparation_mode", list(LASSQDPreparationMode))
def test_fragment_programs_get_distinct_reproducible_seeds(
    dummy_expval_backend, preparation_mode
):
    seeds = []
    for _ in range(2):
        ensemble = _lassqd(dummy_expval_backend, preparation_mode=preparation_mode)
        ensemble.create_programs(ensemble.initial_state())
        seeds.append([program._seed for program in ensemble.programs.values()])

    assert seeds[0] == seeds[1]
    assert len(set(seeds[0])) == 2


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


@pytest.mark.parametrize("converged", [True, False], ids=["run", "not-run"])
def test_fragment_integrals_use_the_mean_field_core_hamiltonian(
    dummy_expval_backend, converged
):
    """A scalar-relativistic mean field changes only the one-electron
    Hamiltonian, so a workflow that rebuilds it from ``mol`` drops the
    relativistic correction while every other integral stays right. With one
    fragment and a single frozen core orbital the fragment's one-body integrals
    are CASCI's effective Hamiltonian, which PySCF builds from the mean field's
    ``get_hcore``. A mean field passed before it has run must be run as given,
    not replaced by a plain RHF."""
    reference = scf.RHF(h4_chain()).sfx2c1e().run(verbose=0)
    mean_field = reference if converged else scf.RHF(h4_chain()).sfx2c1e()
    ensemble = _lassqd(
        dummy_expval_backend,
        problem=MolecularProblem.from_molecule(mean_field),
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        active_spaces=[FragmentSpec(orbitals=(1, 2), n_alpha=1, n_beta=1)],
    )

    ensemble.create_programs(ensemble.initial_state())

    casci = mcscf.CASCI(reference, 2, 2)
    casci.mo_coeff = reference.mo_coeff
    expected, _ = casci.get_h1eff()

    np.testing.assert_allclose(
        ensemble.programs["fragment_0"].problem.one_body, expected, atol=1e-10
    )


def test_run_uses_programs_the_caller_materialised(exact_sampler_lassqd):
    """``run()`` takes a program map built by ``create_programs()`` as its
    first round, so that round must be reduced against the state those programs
    were built from."""
    ensemble, _ = exact_sampler_lassqd
    ensemble.create_programs()
    materialised = ensemble.programs

    ensemble.run(max_rounds=1)

    assert ensemble.stop_reason is WorkflowStatus.MAX_ROUNDS
    assert len(ensemble.energy_history) == 1
    assert all(
        ensemble.programs[program_id] is program
        for program_id, program in materialised.items()
    )


def test_missing_backend_raises_type_error():
    message = "LASSQD.__init__ missing required keyword-only argument: 'backend'."
    with pytest.raises(TypeError, match=exact_match(message)):
        LASSQD(
            MolecularProblem.from_molecule(h4_chain()),
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            preparation_mode=LASSQDPreparationMode.VQE,
            **lassqd_kwargs(
                active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)]
            ),
        )


@pytest.mark.parametrize("attribute, value", [("n_layers", 2), ("max_iterations", 5)])
def test_settings_are_forwarded_to_each_fragment_vqe(
    dummy_expval_backend, attribute, value
):
    ensemble = _lassqd(dummy_expval_backend, **{attribute: value})
    ensemble.create_programs(ensemble.initial_state())
    for program in ensemble.programs.values():
        assert getattr(program, attribute) == value


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


@pytest.mark.parametrize(
    "settings, names",
    [
        pytest.param({"n_layers": 2}, "n_layers", id="n_layers"),
        pytest.param(
            {"ansatz_kwargs": {"trailing_rotation": True}},
            "ansatz_kwargs",
            id="ansatz_kwargs",
        ),
        pytest.param({"max_iterations": 5}, "max_iterations", id="max_iterations"),
        pytest.param(
            {"max_iterations": 5, "n_layers": 2},
            "max_iterations, n_layers",
            id="several",
        ),
    ],
)
def test_linear_method_rejects_vqe_settings_at_construction(
    dummy_expval_backend, settings, names
):
    """These configure the VQE fragment programs. The linear-method programs
    either ignored them (``max_iterations``) or raised only inside
    ``create_programs``, after the mean field and localisation had run."""
    message = (
        f"preparation_mode='linear_method' does not take {names}; its fragment "
        "programs accept only precision, qem_protocol, "
        "suppress_performance_warnings. VQE options need preparation_mode='vqe'."
    )
    with pytest.raises(TypeError, match=exact_match(message)):
        _raw_lassqd(dummy_expval_backend, **settings)


def test_linear_method_forwards_the_shared_program_options(dummy_expval_backend):
    ensemble = _lassqd(
        dummy_expval_backend,
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        precision=6,
        suppress_performance_warnings=True,
        qem_protocol=None,
    )

    ensemble.create_programs(ensemble.initial_state())

    assert all(program._precision == 6 for program in ensemble.programs.values())


def test_fragment_vqe_rejects_mismatched_seed_params_length(dummy_expval_backend):
    problem = HamiltonianProblem(SparsePauliOp(["ZZ"], [1.0]), n_electrons=2)
    baseline = _workflow._FragmentVQE(
        problem,
        ansatz=LUCJAnsatz(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        backend=dummy_expval_backend,
    )
    bad_params = np.ones(baseline.n_params + 1)

    with pytest.raises(ValueError, match="seed_params"):
        _workflow._FragmentVQE(
            problem,
            ansatz=LUCJAnsatz(),
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
            backend=dummy_expval_backend,
            seed_params=bad_params,
        )


@pytest.mark.parametrize(
    "energies, expected",
    [
        pytest.param({}, False, id="before-any-round"),
        pytest.param(
            {"energy": -2.0, "previous_energy": -2.0 + 1e-9}, True, id="converged"
        ),
        pytest.param({"energy": -2.0, "previous_energy": -1.0}, False, id="moving"),
    ],
)
def test_is_complete_compares_the_energy_change_to_the_tolerance(
    dummy_expval_backend, energies, expected
):
    ensemble = _lassqd(dummy_expval_backend, energy_tol=1e-5)
    state = dataclasses.replace(ensemble.initial_state(), **energies)
    assert ensemble.is_complete(state) is expected


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


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param(
            {},
            {"lambda_penalty": 0.2, "carryover_cutoff": 1e-5, "max_carryover": None},
            id="defaults",
        ),
        pytest.param(
            {"carryover_cutoff": None},
            {"carryover_cutoff": None},
            id="carryover_turned_off",
        ),
    ],
)
def test_solver_settings_are_threaded_to_the_solver(
    dummy_expval_backend, overrides, expected
):
    """An option users cannot reach from ``LASSQD`` is not an option. Carryover
    is on by default because conventional SQD oscillates across macro-cycles
    where sampling covers only a fraction of the determinant space."""
    ensemble = _lassqd(dummy_expval_backend, **overrides)
    state = ensemble.initial_state()
    solver = ensemble._solver_for(0, state.fragments[0].spec)
    for attribute, value in expected.items():
        if value is None:
            assert getattr(solver, attribute) is None
        else:
            assert getattr(solver, attribute) == pytest.approx(value)


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


def test_aggregate_results_matches_energy_after_one_round(exact_sampler_lassqd):
    """``aggregate_results`` returns the state ``update_state`` produced, not
    the one that built the round's programs, so after a single round its energy
    is the round's finite energy rather than the initial state's ``inf``."""
    ensemble, _ = exact_sampler_lassqd
    assert ensemble.energy == float("inf")
    ensemble.run(max_rounds=1)

    result = ensemble.aggregate_results()

    assert result.energy == ensemble.energy
    assert np.isfinite(result.energy)
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


_H8_HALF_CHAINS = ([0, 1, 2, 3], [4, 5, 6, 7])


@pytest.mark.parametrize(
    "local_spins, expected",
    [
        pytest.param([2, -2], [(3, 1), (1, 3)], id="polarized"),
        pytest.param(None, [(2, 2), (2, 2)], id="closed-shell"),
    ],
)
def test_local_spins_set_the_automatically_built_fragment_spins(
    dummy_expval_backend, local_spins, expected
):
    """``local_spins`` must survive the remap from localized-column indices to
    register positions, which is where the automatic path rewrites every spec."""
    ensemble = h8_frontier_lassqd(
        dummy_expval_backend, fragment_atoms=_H8_HALF_CHAINS, local_spins=local_spins
    )

    state = ensemble.initial_state()

    assert [(f.spec.n_alpha, f.spec.n_beta) for f in state.fragments] == expected


def test_local_spins_must_sum_to_zero_sz(dummy_expval_backend):
    """A per-fragment override can break the global Sz = 0 invariant, which
    ``auto_fragment_specs`` cannot see on its own since it checks each
    fragment's electron count independently."""
    ensemble = h8_frontier_lassqd(
        dummy_expval_backend, fragment_atoms=_H8_HALF_CHAINS, local_spins=[2, 2]
    )

    with pytest.raises(ValueError, match="total Sz"):
        ensemble.initial_state()


POLARIZED_SPECS = [
    FragmentSpec(orbitals=(0, 1), n_alpha=2, n_beta=1),
    FragmentSpec(orbitals=(2, 3), n_alpha=0, n_beta=1),
]


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_default_linear_method_runs_a_beta_majority_fragment(
    default_test_simulator,
):
    ensemble = _lassqd(
        default_test_simulator,
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        active_spaces=POLARIZED_SPECS,
        seed=9,
    )

    ensemble.run(max_rounds=1)

    assert np.isfinite(ensemble.energy)
    assert ensemble.best_energy == ensemble.energy
    assert all(program.has_results() for program in ensemble.programs.values())


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
                fragment,
                fragment_problem(h_alpha, h_beta, g_frag, fragment.spec),
                seed=0,
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
            fragments[0],
            fragment_problem(h_alpha, h_beta, g_frag, fragments[0].spec),
            seed=0,
        )


@pytest.mark.parametrize(
    "n_beta, n_params, ansatz_kwargs, match",
    [
        # Neither the t1/t2 read-off nor the factorization means anything for an
        # ansatz whose parameters are unrelated to coupled-cluster amplitudes.
        pytest.param(
            1, 6, {"ansatz": QCCAnsatz()}, "no correspondence is defined", id="qcc"
        ),
        # Restricted CCSD cannot represent a polarized fragment, which is
        # reachable whenever the fragments together still sum to Sz = 0.
        pytest.param(0, 4, {}, "CCSD", id="spin-imbalanced"),
    ],
)
def test_ccsd_seed_params_warns_and_skips(n_beta, n_params, ansatz_kwargs, match):
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=n_beta)

    with pytest.warns(UserWarning, match=match):
        result = _workflow._ccsd_seed_params(
            np.eye(2), np.zeros((2,) * 4), spec, n_params=n_params, **ansatz_kwargs
        )

    assert result is None


_H4_WHOLE_SPACE = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)


def _h4_as_one_fragment(mean_field):
    """``(h_eff, g_frag)`` for H4's whole canonical-MO space as one fragment."""
    integrals = _workflow.transform_integrals(
        mean_field.mol, np.asarray(mean_field.mo_coeff), n_core=0, n_act=4
    )
    placeholder = FragmentState(
        spec=_H4_WHOLE_SPACE, rdm1=np.zeros((4, 4)), rdm2=np.zeros((4, 4, 4, 4))
    )
    h_eff, _, g_frag = _workflow.fragment_effective_integrals(
        integrals, [placeholder], 0
    )
    return h_eff, g_frag


def test_ccsd_seed_params_uses_amplitude_correspondence_for_uccsd(h4_chain_mean_field):
    """The seed's exact-statevector energy must land near the fragment's own
    CCSD energy, and clearly beats a permutation of the same values.

    A fragment with only one double excitation (e.g. a minimal H2 fragment)
    cannot exercise ordering, sign, or magnitude errors: with a single value
    there is nothing to permute, no crossed-spin case, and no scale to get
    wrong. This uses a single 4-orbital fragment (8 singles, 18 doubles of
    differing sign) so a scrambled index map, a wrong spin pairing, or an
    un-doubled amplitude all produce a detectably worse energy.
    """
    spec = _H4_WHOLE_SPACE
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = UCCSDAnsatz.n_params_per_layer(
        2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
    )

    seed = _workflow._ccsd_seed_params(h_eff, g_frag, spec, n_params, UCCSDAnsatz())

    assert seed is not None
    coupled_cluster = cc.CCSD(h4_chain_mean_field)
    coupled_cluster.kernel()
    ccsd_electronic_energy = (
        coupled_cluster.e_tot - h4_chain_mean_field.mol.energy_nuc()
    )

    seed_energy = ansatz_energy(seed, h_eff, g_frag, spec)
    permuted_energy = ansatz_energy(
        np.random.default_rng(0).permutation(seed), h_eff, g_frag, spec
    )

    assert seed_energy == pytest.approx(ccsd_electronic_energy, abs=1e-3)
    assert seed_energy < permuted_energy - 0.05


@pytest.mark.parametrize(
    "ansatz, min_recovered_fraction",
    [(UCCSDAnsatz(), 0.9), (LUCJAnsatz(), 0.3)],
    ids=["uccsd", "lucj"],
)
def test_seed_beats_the_reference_determinant_in_a_localized_basis(
    dummy_expval_backend, ansatz, min_recovered_fraction
):
    """The seed must recover correlation in the basis the fragments actually use.

    The canonical-MO test above cannot see this: there the fragment basis and
    the basis an SCF on the fragment's integrals converges to coincide. Real
    fragments are localized, and an SCF rotates within their occupied and
    virtual blocks, permuting which amplitude belongs to which orbital pair.
    Amplitudes read off that rotated solution seeded a state *above* the
    reference determinant.

    LUCJ's seed runs through the double factorization, where every link is a
    sign convention -- the Jastrow's factor of ``i`` absorbed into one sector's
    rotation, the conjugate transpose of the sandwiched block, the
    ``RZZ(-J / 2)`` pair term -- and getting one wrong leaves the seed *at* the
    reference. One layer holds only the leading factorization term, so its
    recovered fraction is well short of UCCSD's (measured 0.36).
    """
    ensemble = h8_frontier_lassqd(dummy_expval_backend)
    state = ensemble.initial_state()

    for index, fragment in enumerate(state.fragments):
        spec = fragment.spec
        h_alpha, h_beta, g_frag = fragment_integrals(
            ensemble, state.mo_coeff, state.fragments, index
        )
        n_params = type(ansatz).n_params_per_layer(
            2 * spec.n_orbitals, n_electrons=spec.n_alpha + spec.n_beta
        )
        exact = fci.direct_spin1.kernel(
            h_alpha, g_frag, spec.n_orbitals, (spec.n_alpha, spec.n_beta)
        )[0]
        seed = _workflow._ccsd_seed_params(
            0.5 * (h_alpha + h_beta), g_frag, spec, n_params, ansatz, {}
        )
        assert seed is not None

        reference = ansatz_energy(np.zeros(n_params), h_alpha, g_frag, spec, ansatz)
        seeded = ansatz_energy(seed, h_alpha, g_frag, spec, ansatz)

        assert seeded > exact - 1e-8
        assert (reference - seeded) / (reference - exact) > min_recovered_fraction


def _seed_gain(
    ansatz,
    h_alpha,
    h_beta,
    g_frag,
    spec,
    seed_integrals=None,
    permute=False,
    **kwargs,
):
    """``_seed_energy_gain`` for a seed on one fragment's integrals.

    ``seed_integrals`` builds the seed from a *different* ``(h, g)`` than the
    Hamiltonian it is scored against, which is what a basis mismatch is.
    ``permute`` scores a permutation of the seed instead.
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
    if permute:
        seed = np.random.default_rng(0).permutation(seed)
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

    def gain(permute):
        return _seed_gain(UCCSDAnsatz(), h_eff, h_eff, g_frag, spec, permute=permute)

    assert gain(permute=False) > _workflow._SEED_ACCEPTANCE_MARGIN
    assert gain(permute=True) < _workflow._SEED_ACCEPTANCE_MARGIN


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_seed_acceptance_admits_a_polarized_non_stationary_fragment():
    """A spin-polarized fragment has no shared spatial basis making both spin
    channels Hartree-Fock stationary, so the old precondition refused it
    outright -- which is what left the diiron benchmark unseeded. The seed is
    worth keeping there."""
    ensemble = h8_frontier_lassqd(local_spins=[2, -2], fragment_atoms=_H8_HALF_CHAINS)
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


def test_uccsd_seed_singles_match_the_ccsd_t1_amplitudes(h4_chain_mean_field):
    """The energy assertion above cannot cover the singles, so pin their values.

    That fragment's orbitals are canonical MOs, so ``t1`` is
    Brillouin-suppressed: measured ``max|t1|`` is 1.25e-03 against ``max|t2|``
    of 8.2e-02. Zeroing every single moves the seeded energy by 7.5e-06 and
    sign-flipping them by 2.4e-05 -- both far inside that test's 1e-3 tolerance,
    so the whole singles block could be scrambled or deleted unnoticed. The
    doubles are caught there with 40x margin.
    """
    spec = _H4_WHOLE_SPACE
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
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
    state = ensemble.initial_state()
    ensemble.create_programs(state)

    for fragment, program in zip(state.fragments, ensemble.programs.values()):
        expected_length = 2 * UCCSDAnsatz.n_params_per_layer(
            program.n_qubits, n_electrons=fragment.spec.n_alpha + fragment.spec.n_beta
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
    spy = mocker.spy(ensemble, "_emit_workflow_stage")

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
        "pyscf.cc.CCSD",
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

    with pytest.warns(UserWarning, match="Orbital optimisation ended without"):
        ensemble.run(max_rounds=1)

    assert optimization_submit.call_count > 0
    sampling_submit.assert_called_once()
    assert len(ensemble.round_history) == 1


def test_linear_method_samples_each_fragment_on_the_sampling_backend(
    default_test_simulator, sampling_test_simulator, mocker
):
    optimization_submit = mocker.spy(default_test_simulator, "submit_circuits")
    sampling_submit = mocker.spy(sampling_test_simulator, "submit_circuits")
    ensemble = _lassqd(
        default_test_simulator,
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        sampling_backend=sampling_test_simulator,
        seed=9,
    )

    ensemble.run(max_rounds=1)

    assert ensemble.sampling_backend is sampling_test_simulator
    assert sampling_submit.call_count == len(ensemble.programs) == 2
    optimization_submit.assert_not_called()
    assert np.isfinite(ensemble.energy)


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


@pytest.mark.parametrize(
    "max_rounds, expected",
    [(1, WorkflowStatus.MAX_ROUNDS), (12, WorkflowStatus.COMPLETE)],
    ids=["capped", "converged"],
)
def test_stop_reason(exact_sampler_lassqd, max_rounds, expected):
    ensemble, _ = exact_sampler_lassqd
    ensemble.run(max_rounds=max_rounds)
    assert ensemble.stop_reason is expected


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
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        active_spaces=None,
        n_active_orbitals=4,
        max_orbitals_per_fragment=2,
    )

    ensemble.run(max_rounds=2)
    first_energy = ensemble.energy
    first_mo_coeff = ensemble.workflow_state.mo_coeff.copy()
    first_rounds = len(ensemble.round_history)

    ensemble.run(max_rounds=2)

    assert len(ensemble.round_history) == first_rounds
    assert ensemble.energy == pytest.approx(first_energy, abs=1e-9)
    np.testing.assert_allclose(ensemble.workflow_state.mo_coeff, first_mo_coeff)


_H2_E2E_SETTINGS = dict(
    active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
    n_batches=24,
    batch_size=256,
    n_recovery_iterations=3,
    seed=7,
)


@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_default_linear_method_h2_reaches_chemical_accuracy_over_two_rounds(
    default_test_simulator, h2_mean_field
):
    exact = fci.FCI(h2_mean_field).kernel()[0]
    ensemble = _lassqd(
        default_test_simulator,
        problem=MolecularProblem.from_molecule(h2_molecule()),
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        **_H2_E2E_SETTINGS,
    )

    ensemble.run(max_rounds=2)

    assert len(ensemble.round_history) == 2
    assert ensemble.best_energy == pytest.approx(exact, abs=1e-6)
    assert ensemble.best_energy >= exact - 1e-8


@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_single_fragment_h2_reaches_chemical_accuracy(
    default_test_simulator, h2_mean_field
):
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
    exact = fci.FCI(h2_mean_field).kernel()[0]

    ensemble = _lassqd(
        default_test_simulator,
        problem=MolecularProblem.from_molecule(h2_molecule()),
        max_iterations=200,
        **_H2_E2E_SETTINGS,
    )
    ensemble.run(max_rounds=5)

    assert ensemble.best_energy == pytest.approx(exact, abs=1e-6)
    assert ensemble.best_energy >= exact - 1e-8


@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_two_fragment_h4_lands_on_the_product_state_energy(
    default_test_simulator, h4_chain_mean_field
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
    ensemble = _lassqd(
        default_test_simulator,
        max_iterations=60,
        n_batches=6,
        batch_size=16,
        n_recovery_iterations=3,
        seed=7,
    )
    ensemble.run(max_rounds=5)

    casci = mcscf.CASCI(h4_chain_mean_field, 4, 4).kernel()[0]

    assert ensemble.energy > casci, "a product state cannot beat CASCI"
    assert ensemble.energy <= h4_chain_mean_field.e_tot + 1e-5


def _round_report(number, energy_change):
    return _workflow.LASSQDRoundReport(
        number=number,
        energy=-1.25,
        energy_change=energy_change,
        subspace_sizes=(2, 3),
        orbital_iterations=3,
        orbital_evaluations=4,
        orbital_gradient_norm=0.01,
        orbital_converged=False,
        rotation_pairs=5,
        recovery_seconds=0.2,
        orbital_seconds=0.3,
    )


def _checkpoint_state(state, *, orbitals_converged=False):
    """A post-round state whose every array differs from ``state``'s."""
    fragments = tuple(
        FragmentState(
            spec=fragment.spec,
            rdm1=fragment.rdm1 + 0.1 * (index + 1),
            rdm2=fragment.rdm2 + 0.2 * (index + 1),
            params=np.arange(3, dtype=float) + index,
            rdm1_alpha=fragment.rdm1 / 3 + index,
            rdm1_beta=fragment.rdm1 * 2 / 3 - index,
        )
        for index, fragment in enumerate(state.fragments)
    )
    # A rotation of the first two orbitals, so the orbitals stay orthonormal.
    rotation = np.eye(state.mo_coeff.shape[1])
    rotation[:2, :2] = [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]
    return LASSQDState(
        mo_coeff=state.mo_coeff @ rotation,
        fragments=fragments,
        energy=-1.25,
        previous_energy=-1.0,
        orbitals_converged=orbitals_converged,
    )


def _save_checkpoint(ensemble, state, directory):
    """Save ``state`` after two recorded rounds and one cached solver."""
    ensemble._energy_history = [-1.0, -1.25]
    ensemble._round_reports = [_round_report(1, None), _round_report(2, -0.25)]
    ensemble._solver_for(0, state.fragments[0].spec)
    return ensemble._save_workflow_checkpoint_state(state, directory, "output_state")


def _rewrite_checkpoint(directory, payload, corrupt):
    """Apply ``corrupt(arrays, payload)`` to a saved checkpoint in place."""
    path = directory / payload["artifact"]
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    corrupt(arrays, payload)
    np.savez(path, **arrays)


def _set_array(name, value):
    def corrupt(arrays, payload):
        arrays[name] = value

    return corrupt


def _drop_array(name):
    def corrupt(arrays, payload):
        del arrays[name]

    return corrupt


def _set_payload(key, value):
    def corrupt(arrays, payload):
        payload[key] = value

    return corrupt


def _drop_payload(key):
    def corrupt(arrays, payload):
        del payload[key]

    return corrupt


def _duplicate_solver(arrays, payload):
    payload["solvers"] = payload["solvers"] * 2


def _solver_one_past_the_end(arrays, payload):
    payload["solvers"][0]["index"] = len(payload["fragments"])


def _stretch_orbitals(arrays, payload):
    arrays["mo_coeff"] = 1.01 * arrays["mo_coeff"]


class _PickleTripwire:
    """Records whether an instance was ever unpickled."""

    unpickled = False

    def __init__(self):
        self.marker = 1

    def __setstate__(self, state):
        type(self).unpickled = True
        self.__dict__.update(state)


def _resume_one_round(ensemble, payload, directory, *, restore_solvers):
    """Load a checkpoint into ``ensemble``, run one round, return its history."""
    state = ensemble._load_workflow_checkpoint_state(payload, directory, "output_state")
    if not restore_solvers:
        ensemble._solvers.clear()
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)
    ensemble.update_state(state)
    return ensemble.energy_history


@pytest.mark.parametrize("orbitals_converged", [True, False])
def test_workflow_checkpoint_state_round_trips_npz(
    exact_sampler_lassqd, tmp_path, orbitals_converged
):
    ensemble, initial = exact_sampler_lassqd
    state = _checkpoint_state(initial, orbitals_converged=orbitals_converged)
    payload = _save_checkpoint(ensemble, state, tmp_path)
    expected_reports = ensemble.round_reports
    expected_rng_state = ensemble._rng.bit_generator.state
    expected_solver_state = ensemble._solvers[0]._rng.bit_generator.state
    ensemble._rng.random()
    ensemble._solvers[0]._rng.random()
    ensemble._energy_history.clear()
    ensemble._round_reports.clear()
    ensemble._solvers.clear()

    restored = ensemble._load_workflow_checkpoint_state(
        payload, tmp_path, "output_state"
    )

    np.testing.assert_array_equal(restored.mo_coeff, state.mo_coeff)
    for loaded, saved in zip(restored.fragments, state.fragments, strict=True):
        assert loaded.spec == saved.spec
        for name in ("rdm1", "rdm2", "params", "rdm1_alpha", "rdm1_beta"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(saved, name))
    assert (restored.energy, restored.previous_energy) == (-1.25, -1.0)
    assert restored.orbitals_converged is orbitals_converged
    assert ensemble.energy_history == (-1.0, -1.25)
    assert ensemble.round_reports == expected_reports
    assert ensemble.round_reports[0].energy_change is None
    assert ensemble._rng.bit_generator.state == expected_rng_state
    assert set(ensemble._solvers) == {0}
    assert ensemble._solvers[0]._rng.bit_generator.state == expected_solver_state


def test_resuming_from_a_checkpoint_reproduces_an_uninterrupted_run(
    dummy_expval_backend, mocker, tmp_path
):
    """The batch is far smaller than the 36-determinant sector, so the second
    round's subspace depends on where each generator was left."""
    settings = dict(
        active_spaces=[_H4_WHOLE_SPACE], batch_size=3, n_recovery_iterations=1
    )
    straight, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    straight.run(max_rounds=2)

    interrupted, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, **settings
    )
    interrupted.run(max_rounds=1)
    payload = interrupted._save_workflow_checkpoint_state(
        interrupted.workflow_state, tmp_path, "output_state"
    )

    resumed, control = (
        build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)[0]
        for _ in range(2)
    )

    assert len(straight.energy_history) == 2
    assert (
        _resume_one_round(resumed, payload, tmp_path, restore_solvers=True)
        == straight.energy_history
    )
    assert (
        _resume_one_round(control, payload, tmp_path, restore_solvers=False)
        != straight.energy_history
    )


def test_restoring_a_run_checkpoint_reproduces_an_uninterrupted_run(
    dummy_expval_backend, mocker, tmp_path
):
    """The path a user takes: ``run()`` writes the checkpoint, ``restore_state``
    reads it and ``run()`` continues. Restoring a completed round also rebuilds
    that round's programs from its input state, which must not leave the
    workflow's generators, solvers or history at that earlier point."""
    settings = dict(
        active_spaces=[_H4_WHOLE_SPACE], batch_size=3, n_recovery_iterations=1
    )
    straight, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    straight.run(max_rounds=2)
    interrupted, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, **settings
    )
    interrupted.run(
        max_rounds=1, checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path)
    )

    resumed, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **settings)
    resumed.restore_state(tmp_path).run(max_rounds=2)

    assert resumed.energy_history == straight.energy_history
    assert [report.number for report in resumed.round_reports] == [1, 2]


@pytest.mark.parametrize(
    "corrupt, match",
    [
        pytest.param(
            _set_payload("fragments", []),
            "no fragment metadata",
            id="empty-fragment-list",
        ),
        pytest.param(
            _drop_array("fragment_0_rdm2"),
            "missing fragment_0_rdm2",
            id="missing-array",
        ),
        pytest.param(
            _set_array("fragment_0_rdm1", np.zeros((3, 3))),
            r"fragment_0_rdm1 has shape \(3, 3\)",
            id="rdm-shape",
        ),
        pytest.param(
            _set_array("fragment_1_rdm1_beta", np.zeros((3, 3))),
            r"fragment_1_rdm1_beta has shape \(3, 3\)",
            id="spin-rdm-shape",
        ),
        pytest.param(
            _duplicate_solver,
            "invalid solver fragment indices",
            id="duplicate-solver",
        ),
        pytest.param(
            _solver_one_past_the_end,
            "invalid solver fragment indices",
            id="out-of-range-solver",
        ),
        pytest.param(
            _set_array("fragment_0_params", np.zeros((3, 1))),
            "parameters are not 1-D",
            id="2d-params",
        ),
        pytest.param(
            _set_array("fragment_0_rdm2", np.full((2, 2, 2, 2), np.nan)),
            "fragment_0_rdm2 contains non-finite values",
            id="nan",
        ),
        pytest.param(
            _set_array("mo_coeff", np.full((4, 4), "x")),
            "mo_coeff has non-numeric dtype",
            id="string-dtype",
        ),
        pytest.param(
            _drop_payload("solvers"),
            "solver metadata must be a list",
            id="missing-solvers",
        ),
        pytest.param(
            _drop_payload("round_reports"),
            "round reports must be a list",
            id="missing-round-reports",
        ),
        pytest.param(
            _stretch_orbitals,
            "mo_coeff is not orthonormal",
            id="non-orthonormal-orbitals",
        ),
        pytest.param(
            _drop_payload("molecule"),
            "missing its molecule fingerprint",
            id="missing-molecule",
        ),
        pytest.param(
            _drop_payload("configuration"),
            "missing its configuration",
            id="missing-configuration",
        ),
    ],
)
def test_workflow_checkpoint_rejects_a_corrupt_snapshot(
    exact_sampler_lassqd, tmp_path, corrupt, match
):
    ensemble, initial = exact_sampler_lassqd
    payload = _save_checkpoint(ensemble, _checkpoint_state(initial), tmp_path)
    _rewrite_checkpoint(tmp_path, payload, corrupt)

    with pytest.raises(ValueError, match=match):
        ensemble._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def test_workflow_checkpoint_refuses_to_unpickle_an_object_array(
    exact_sampler_lassqd, tmp_path
):
    ensemble, initial = exact_sampler_lassqd
    payload = _save_checkpoint(ensemble, _checkpoint_state(initial), tmp_path)
    tripwire = np.empty(1, dtype=object)
    tripwire[0] = _PickleTripwire()
    _rewrite_checkpoint(tmp_path, payload, _set_array("fragment_0_rdm1", tripwire))

    with pytest.raises(ValueError, match="allow_pickle=False"):
        ensemble._load_workflow_checkpoint_state(payload, tmp_path, "output_state")
    assert not _PickleTripwire.unpickled


def test_a_failed_checkpoint_save_leaves_no_temporary_file(
    exact_sampler_lassqd, tmp_path, mocker
):
    ensemble, state = exact_sampler_lassqd
    mocker.patch.object(_workflow.np, "savez", side_effect=OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    assert list(tmp_path.iterdir()) == []


def test_rejects_a_frontier_selection_with_no_virtual_orbital(dummy_expval_backend):
    """Helium in STO-3G has one orbital, occupied, so no frontier selection can
    reach a virtual -- caught before any mean field is computed."""
    with pytest.raises(ValueError, match="at least one occupied and one virtual"):
        LASSQD(
            MolecularProblem.from_molecule(
                gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
            ),
            backend=dummy_expval_backend,
            reporting_level=ReportingLevel.OFF,
            **lassqd_kwargs(n_active_orbitals=2),
        )


@pytest.mark.parametrize("spin_half", ["rdm1_alpha", "rdm1_beta"])
def test_fragment_state_rejects_a_single_spin_rdm(spin_half):
    with pytest.raises(ValueError, match="given together"):
        FragmentState(
            spec=_H4_FRAGMENTS[0],
            rdm1=np.eye(2),
            rdm2=np.zeros((2,) * 4),
            **{spin_half: np.eye(2) / 2},
        )


def test_sqd_config_max_dim_accepts_any_pair_sequence():
    assert SQDConfig(max_dim=[2, 3]).max_dim == (2, 3)


def test_sqd_settings_reach_the_solver_constructor(dummy_expval_backend, mocker):
    constructor = mocker.patch.object(_workflow, "SQDSolver", wraps=SQDSolver)
    ensemble = _lassqd(
        dummy_expval_backend,
        n_batches=3,
        batch_size=5,
        n_recovery_iterations=4,
        lambda_penalty=0.7,
        carryover_cutoff=1e-3,
        max_carryover=7,
        max_dim=(2, 3),
        include_reference=False,
        symmetrize_spin=True,
        recovery_energy_tol=1e-4,
        recovery_occupancies_tol=1e-3,
    )
    spec = ensemble.initial_state().fragments[0].spec

    ensemble._solver_for(0, spec)

    constructor.assert_called_once()
    assert constructor.call_args.args == (spec.n_orbitals, spec.n_alpha, spec.n_beta)
    keywords = dict(constructor.call_args.kwargs)
    assert isinstance(keywords.pop("rng"), np.random.Generator)
    assert keywords == dict(
        n_batches=3,
        batch_size=5,
        n_iterations=4,
        lambda_penalty=0.7,
        recovery=True,
        carryover_cutoff=1e-3,
        max_carryover=7,
        max_dim=(2, 3),
        include_reference=False,
        symmetrize_spin=True,
        energy_tol=1e-4,
        occupancies_tol=1e-3,
    )


def test_sqd_defaults_to_the_default_config(dummy_expval_backend):
    ensemble = LASSQD(
        MolecularProblem.from_molecule(h4_chain()),
        backend=dummy_expval_backend,
        reporting_level=ReportingLevel.OFF,
        fragmentation=FragmentationConfig(active_spaces=_H4_FRAGMENTS),
    )

    assert ensemble._sqd == SQDConfig()


def test_accepts_boundary_sqd_and_fragmentation_values(dummy_expval_backend):
    _lassqd(
        dummy_expval_backend,
        batch_size=1,
        lambda_penalty=0.0,
        max_carryover=1,
        max_dim=1,
    )
    _lassqd(
        dummy_expval_backend,
        active_spaces=None,
        n_active_orbitals=2,
        max_orbitals_per_fragment=2,
        coupling_threshold=0.0,
    )


def test_same_seed_automatic_fragmentation_is_reproducible(dummy_expval_backend):
    states = [
        _lassqd(
            dummy_expval_backend,
            active_spaces=None,
            n_active_orbitals=4,
            max_orbitals_per_fragment=2,
            seed=5,
        ).initial_state()
        for _ in range(2)
    ]

    np.testing.assert_array_equal(states[0].mo_coeff, states[1].mo_coeff)


def _rotated_mean_field(mean_field, angle):
    """``mean_field`` with its two orbitals mixed by a real rotation."""
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated = copy.copy(mean_field)
    rotated.mo_coeff = np.asarray(mean_field.mo_coeff) @ np.array(
        [[cosine, -sine], [sine, cosine]]
    )
    return rotated


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_linear_method_rotates_fragment_rdms_back_to_the_workflow_basis(
    default_test_simulator, h2_mean_field
):
    """The fragment's ROHF undoes the rotation, so SQD's RDMs come back in a
    basis other than the one the energy is evaluated in. A single fragment over
    every orbital leaves the orbital step nothing to rotate, so a missing
    rotation back would show up directly in the energy."""
    exact = fci.FCI(h2_mean_field).kernel()[0]
    ensemble = _lassqd(
        default_test_simulator,
        problem=MolecularProblem.from_molecule(
            _rotated_mean_field(h2_mean_field, np.pi / 4)
        ),
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
        n_batches=4,
        batch_size=64,
        n_recovery_iterations=2,
        seed=7,
    )

    ensemble.run(max_rounds=1)

    assert ensemble.best_energy == pytest.approx(exact, abs=1e-6)


def _h6_chain():
    """Linear H6 in STO-3G: 6 orbitals, 3 of them occupied."""
    return gto.M(
        atom="; ".join(f"H 0 0 {index * 0.9:.1f}" for index in range(6)),
        basis="sto-3g",
        verbose=0,
    )


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_frozen_core_and_virtual_orbitals_bracket_the_active_space(
    dummy_expval_backend, mocker
):
    """Orbital 0 stays a frozen core and orbital 5 a frozen virtual around two
    interleaved fragments, so the register reads core | (1, 3) | (2, 4) |
    virtual."""
    mean_field = scf.RHF(_h6_chain()).run(verbose=0)
    ensemble, state = build_exact_sampler_lassqd(
        dummy_expval_backend,
        mocker,
        problem=MolecularProblem.from_molecule(mean_field),
        active_spaces=[
            FragmentSpec(orbitals=(1, 3), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 4), n_alpha=1, n_beta=1),
        ],
    )

    ensemble.run(max_rounds=1)

    np.testing.assert_array_equal(
        state.mo_coeff, np.asarray(mean_field.mo_coeff)[:, [0, 1, 3, 2, 4, 5]]
    )
    casci = mcscf.CASCI(mean_field, 4, 4).kernel()[0]
    assert casci - 1e-8 < ensemble.energy <= mean_field.e_tot + 1e-8


@pytest.mark.filterwarnings("ignore:.*recovered subspace contains only one")
def test_sqd_receives_each_linear_method_program_s_spin_integrals(
    default_test_simulator, mocker
):
    """A polarised neighbour splits the embedding by spin, so the beta channel
    is only exercised if SQD receives the program's own ``h_beta``."""
    solve = mocker.spy(SQDSolver, "solve")
    ensemble = _lassqd(
        default_test_simulator,
        preparation_mode=LASSQDPreparationMode.LINEAR_METHOD,
        active_spaces=POLARIZED_SPECS,
        seed=9,
    )

    ensemble.run(max_rounds=1)

    assert solve.call_count == 2
    for call, program in zip(
        solve.call_args_list, ensemble.programs.values(), strict=True
    ):
        h_alpha = call.args[2]
        h_beta = call.kwargs["one_body_beta"]
        assert h_alpha is program.h_alpha
        assert h_beta is program.h_beta
        assert not np.allclose(h_alpha, h_beta)


def test_update_state_reports_the_orbital_solve_it_received(
    exact_sampler_lassqd, mocker
):
    ensemble, state = exact_sampler_lassqd
    solves = [
        OrbitalSolve(
            mo_coeff=state.mo_coeff.copy(),
            energy=-2.0 - 0.1 * index,
            converged=False,
            n_iterations=7 + index,
            n_evaluations=11 + index,
            gradient_norm=0.5 + index,
            n_rotation_pairs=4,
        )
        for index in range(2)
    ]
    mocker.patch.object(_workflow, "optimize_orbitals", side_effect=solves)

    for solve in solves:
        ensemble.create_programs(state)
        ensemble.run_one_round(blocking=True)
        state = ensemble.update_state(state)
        ensemble._clear_completed_round()

        assert state.mo_coeff is solve.mo_coeff
        assert state.energy == solve.energy
        assert state.orbitals_converged is False
        report = ensemble.round_reports[-1]
        assert (
            report.energy,
            report.orbital_iterations,
            report.orbital_evaluations,
            report.orbital_gradient_norm,
            report.orbital_converged,
            report.rotation_pairs,
        ) == (
            solve.energy,
            solve.n_iterations,
            solve.n_evaluations,
            solve.gradient_norm,
            False,
            4,
        )

    assert ensemble.round_reports[1].energy_change == pytest.approx(-0.1)
    assert "change -1.000e-01" in ensemble.round_reports[1].summary()


def test_orbital_solves_converge_at_the_square_root_of_the_energy_tolerance(
    dummy_expval_backend, mocker
):
    """PySCF's CASSCF convention: near a minimum the energy error grows as the
    square of the gradient, so a gradient of ``sqrt(energy_tol)`` matches the
    energy resolution the macro-cycle asks for."""
    ensemble, _ = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, energy_tol=1e-4
    )
    spy = mocker.spy(_workflow, "optimize_orbitals")

    ensemble.run(max_rounds=1)

    assert spy.call_args.kwargs["gradient_tol"] == pytest.approx(1e-2)


def test_a_non_finite_round_energy_fails_the_round(exact_sampler_lassqd, mocker):
    """A NaN energy never satisfies ``is_complete``, so unguarded it would run
    every remaining round on a meaningless state."""
    ensemble, state = exact_sampler_lassqd
    mocker.patch.object(
        _workflow,
        "optimize_orbitals",
        return_value=OrbitalSolve(
            mo_coeff=state.mo_coeff.copy(),
            energy=float("nan"),
            converged=False,
            n_iterations=1,
            n_evaluations=2,
            gradient_norm=float("nan"),
            n_rotation_pairs=4,
        ),
    )

    with pytest.raises(ValueError, match="Round 1 produced a non-finite energy"):
        ensemble.run(max_rounds=2)

    assert ensemble.stop_reason is WorkflowStatus.FAILED
    assert ensemble.energy_history == ()


def test_aggregate_results_raises_before_a_round_is_reduced(exact_sampler_lassqd):
    """The pre-round state would read as a result while carrying none."""
    ensemble, state = exact_sampler_lassqd
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)

    with pytest.raises(RuntimeError, match="No LASSQD round"):
        ensemble.aggregate_results()


@pytest.mark.parametrize(
    "local_spins, match",
    [
        pytest.param([1, -1], "wrong parity", id="odd"),
        pytest.param([6, -6], "cannot supply", id="above-electron-count"),
        pytest.param([4, -4], "no excitation available", id="at-electron-count"),
    ],
)
def test_local_spins_are_bounded_by_parity_and_electron_count(
    dummy_expval_backend, local_spins, match
):
    """``2S`` equal to a fragment's electron count passes the spin checks; on
    H8's four-orbital halves it then fills the alpha channel, which fragment
    validation rejects."""
    with pytest.raises(ValueError, match=match):
        h8_frontier_lassqd(
            dummy_expval_backend,
            fragment_atoms=_H8_HALF_CHAINS,
            local_spins=local_spins,
        ).initial_state()


def test_a_supplied_mean_field_is_used_without_rerunning_scf(
    dummy_expval_backend, h4_chain_mean_field, mocker
):
    rhf = mocker.spy(_workflow.scf, "RHF")
    ensemble = LASSQD(
        MolecularProblem.from_molecule(h4_chain_mean_field),
        backend=dummy_expval_backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(active_spaces=list(_H4_FRAGMENTS)),
    )

    state = ensemble.initial_state()

    rhf.assert_not_called()
    np.testing.assert_array_equal(state.mo_coeff, h4_chain_mean_field.mo_coeff)


def test_warns_when_frontier_selection_starts_from_a_non_aufbau_reference(
    dummy_expval_backend, h4_chain_mean_field
):
    swapped = copy.copy(h4_chain_mean_field)
    energies = np.array(h4_chain_mean_field.mo_energy, copy=True)
    energies[[1, 2]] = energies[[2, 1]]
    swapped.mo_energy = energies
    ensemble = LASSQD(
        MolecularProblem.from_molecule(swapped),
        backend=dummy_expval_backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(n_active_orbitals=4, max_orbitals_per_fragment=2),
    )

    with pytest.warns(UserWarning, match="not aufbau"):
        ensemble.initial_state()


def test_linear_method_accepts_an_explicit_default_lucj_ansatz(dummy_expval_backend):
    ansatz = LUCJAnsatz()

    ensemble = _raw_lassqd(dummy_expval_backend, ansatz=ansatz)

    assert ensemble.ansatz is ansatz


def test_run_rejects_unknown_keywords(exact_sampler_lassqd):
    ensemble, _ = exact_sampler_lassqd

    with pytest.raises(TypeError, match="bogus"):
        ensemble.run(bogus=1)


def test_ao_integrals_are_computed_once_per_ensemble(exact_sampler_lassqd, mocker):
    ensemble, _ = exact_sampler_lassqd
    ao_eri = mocker.spy(_workflow, "cached_ao_eri")
    h_ao = mocker.spy(ensemble._mean_field, "get_hcore")

    ensemble.run(max_rounds=2)

    assert len(ensemble.energy_history) == 2
    ao_eri.assert_called_once()
    h_ao.assert_called_once()


def test_a_seed_with_no_energy_gain_is_rejected(dummy_expval_backend, mocker):
    mocker.patch.object(_workflow, "_seed_energy_gain", return_value=0.0)
    ensemble = _lassqd(dummy_expval_backend)

    with pytest.warns(UserWarning, match="seeding rejected"):
        ensemble.create_programs(ensemble.initial_state())

    assert all(program._seed_params is None for program in ensemble.programs.values())


@pytest.mark.parametrize(
    "ansatz, match",
    [
        pytest.param(UCCSDAnsatz(), "^CCSD did not converge", id="uccsd"),
        pytest.param(LUCJAnsatz(), "^UCCSD did not converge", id="lucj"),
    ],
)
def test_an_unconverged_seed_calculation_warns_but_still_seeds(
    h4_chain_mean_field, mocker, ansatz, match
):
    mocker.patch.object(_workflow, "_SEED_CC_MAX_CYCLE", 1)
    h_eff, g_frag = _h4_as_one_fragment(h4_chain_mean_field)
    n_params = type(ansatz).n_params_per_layer(8, n_electrons=4, n_alpha=2, n_beta=2)

    with pytest.warns(UserWarning, match=match):
        seed = _workflow._ccsd_seed_params(
            h_eff, g_frag, _H4_WHOLE_SPACE, n_params, ansatz, {}
        )

    assert seed is not None
    assert seed.shape == (n_params,)


def test_a_failed_lucj_seed_calculation_falls_back_with_a_warning(mocker):
    mocker.patch.object(_workflow.cc, "UCCSD", side_effect=RuntimeError("boom"))
    spec = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)

    with pytest.warns(UserWarning, match="CCSD seeding failed.*boom"):
        seed = _workflow._lucj_seed_params(
            np.eye(2), np.zeros((2,) * 4), spec, n_params=6, ansatz_kwargs={}
        )

    assert seed is None


_ONE_PAIR_SPEC = FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)


def _one_pair_coupled_cluster(mocker, double, single=0.0):
    """UCCSD amplitudes for one occupied and one virtual orbital per spin."""
    return mocker.Mock(
        t1=(np.full((1, 1), single), np.full((1, 1), -single)),
        t2=(
            np.zeros((1, 1, 1, 1)),
            np.full((1, 1, 1, 1), double),
            np.zeros((1, 1, 1, 1)),
        ),
    )


def _lucj_n_params(**ansatz_kwargs):
    return LUCJAnsatz.n_params_per_layer(
        4, n_electrons=2, n_alpha=1, n_beta=1, **ansatz_kwargs
    )


@pytest.mark.parametrize("double", [0.3, -0.3])
@pytest.mark.parametrize(
    "ansatz_kwargs",
    [{}, {"rotation_depth": 1}, {"same_spin_pairs": [(0, 1)]}],
    ids=["default", "rotation_depth", "same_spin_pairs"],
)
def test_lucj_seed_places_the_factorised_jastrow_in_its_layout(
    mocker, ansatz_kwargs, double
):
    """With one pair per spin the doubles tensor is a single number ``t``: its
    singular value is ``|t|`` and each sector's one-body operator has
    eigenvalues ``(-1, 1)``, so every opposite-spin angle is
    ``-|t| d_p d_q / 2`` and every same-spin angle stays zero."""
    n_params = _lucj_n_params(**ansatz_kwargs)
    same_pairs, opposite_pairs = lucj_jastrow_pairs(
        2, ansatz_kwargs.get("same_spin_pairs"), None
    )
    jastrow_start = 2 * n_rotation_params(
        2, orbital_phases=False, depth=ansatz_kwargs.get("rotation_depth")
    )
    diagonal = np.array([-1.0, 1.0])

    seed = _workflow._lucj_amplitude_seed(
        _one_pair_coupled_cluster(mocker, double),
        _ONE_PAIR_SPEC,
        n_params,
        ansatz_kwargs,
    )

    assert seed is not None
    assert seed.shape == (n_params,)
    jastrow_end = jastrow_start + len(opposite_pairs)
    np.testing.assert_allclose(
        seed[jastrow_start:jastrow_end],
        [-0.5 * abs(double) * diagonal[p] * diagonal[q] for p, q in opposite_pairs],
        atol=1e-12,
    )
    assert jastrow_end + 2 * len(same_pairs) == n_params
    assert not seed[jastrow_end:].any()


def _fit_only_the_first(count):
    """``rotation_angles`` that gives up after its first ``count`` fits."""
    fit = _workflow.rotation_angles
    calls = iter(range(count))

    def first_only(target, *, depth=None):
        return fit(target, depth=depth) if next(calls, None) is not None else None

    return first_only


def test_lucj_seed_keeps_the_rest_when_the_trailing_rotation_cannot_be_fit(mocker):
    mocker.patch.object(
        _workflow, "rotation_angles", side_effect=_fit_only_the_first(2)
    )
    n_params = _lucj_n_params(trailing_rotation=True)
    n_trailing = 2 * n_rotation_params(2, orbital_phases=True)

    with pytest.warns(UserWarning, match="could not realize the trailing rotation"):
        seed = _workflow._lucj_amplitude_seed(
            _one_pair_coupled_cluster(mocker, 0.3, single=0.05),
            _ONE_PAIR_SPEC,
            n_params,
            {"trailing_rotation": True},
        )

    assert seed is not None
    assert seed.shape == (n_params,)
    assert not seed[-n_trailing:].any()
    assert seed[:-n_trailing].any()


_STRETCHED_H4 = "H 0 0 0; H 0 0 0.80; H 0 0 2.0; H 0 0 2.80"


@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param(
            {
                "problem": MolecularProblem.from_molecule(
                    gto.M(atom=_STRETCHED_H4, basis="sto-3g", verbose=0)
                )
            },
            "written for a different molecule",
            id="geometry",
        ),
        pytest.param(
            {"active_spaces": [_H4_WHOLE_SPACE]},
            "written with a different LASSQD configuration",
            id="fragment-layout",
        ),
        pytest.param(
            {"n_batches": 3},
            "written with a different LASSQD configuration",
            id="sqd-budget",
        ),
    ],
)
def test_workflow_checkpoint_rejects_a_different_problem(
    exact_sampler_lassqd, dummy_expval_backend, mocker, tmp_path, overrides, match
):
    """A snapshot from another molecule with the same orbital count, or another
    fragmentation or SQD setup, would otherwise resume silently against the
    wrong orbitals or budget."""
    source, state = exact_sampler_lassqd
    payload = source._save_workflow_checkpoint_state(state, tmp_path, "output_state")
    target, _ = build_exact_sampler_lassqd(dummy_expval_backend, mocker, **overrides)

    with pytest.raises(ValueError, match=match):
        target._load_workflow_checkpoint_state(payload, tmp_path, "output_state")


def _linearly_dependent_h2():
    """H2 whose basis repeats a function, so the mean field keeps 4 of 6."""
    shells = gto.basis.parse("H S\n 1.0 1.0\nH S\n 1.0 1.0\nH S\n 0.3 1.0")
    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis={"H": shells}, verbose=0)
    mean_field = scf.addons.remove_linear_dep_(scf.RHF(molecule))
    mean_field.init_guess = "1e"
    return mean_field.run(verbose=0)


def test_orbital_indices_are_validated_against_the_mean_field_register(
    dummy_expval_backend,
):
    mean_field = _linearly_dependent_h2()
    assert mean_field.mo_coeff.shape == (6, 4)

    with pytest.raises(ValueError, match="out of range for a molecule with 4"):
        _lassqd(
            dummy_expval_backend,
            problem=MolecularProblem.from_molecule(mean_field),
            active_spaces=[FragmentSpec(orbitals=(0, 5), n_alpha=1, n_beta=1)],
        )


def test_workflow_checkpoint_round_trips_a_linearly_dependent_basis(
    dummy_expval_backend, tmp_path
):
    """Dropping linearly dependent functions leaves fewer orbitals than basis
    functions, so ``mo_coeff`` is not square."""
    ensemble = _lassqd(
        dummy_expval_backend,
        problem=MolecularProblem.from_molecule(_linearly_dependent_h2()),
        active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
    )
    state = ensemble.initial_state()
    payload = ensemble._save_workflow_checkpoint_state(state, tmp_path, "output_state")

    restored = ensemble._load_workflow_checkpoint_state(
        payload, tmp_path, "output_state"
    )

    np.testing.assert_array_equal(restored.mo_coeff, state.mo_coeff)
