# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for QSCISolver, the PySCF active-space solver backed by QSCI."""

from itertools import combinations

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, RYGate

from divi.backends import MaestroConfig
from divi.qprog.algorithms import VQE, GenericLayerAnsatz, UCCSDAnsatz
from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.qsci import QSCI, QSCISolver, QSCIState, _fcisolver
from divi.qprog.ensembles.qsci._subspace import ConfigurationSubspace
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from tests._helpers import (
    H3_CHAIN_ATOM,
    WATER_ATOM,
    build_degenerate_orbital_integrals,
    exact_match,
    run_mcscf,
    run_mean_field,
)
from tests.qprog._active_space_solver_contracts import (
    ELECTRON_COUNT_SPLITS,
    verify_kernel_rejects_several_roots,
    verify_kernel_rejects_unrestricted_integrals,
    verify_total_electron_count_splits_like_pyscf,
    verify_unconverged_vqe_leaves_casci_unconverged,
)
from tests.qprog.algorithms._helpers import needs_qiskit_nature

mcscf = pytest.importorskip("pyscf.mcscf")
ao2mo = pytest.importorskip("pyscf.ao2mo")

pytestmark = needs_qiskit_nature

#: ``(mean field, active orbitals, active electrons)`` checked against PySCF.
_ACTIVE_SPACES = {
    "water-singlet": (lambda: run_mean_field(WATER_ATOM), 4, 4),
    "h3-doublet": (lambda: run_mean_field(H3_CHAIN_ATOM, spin=1), 3, (2, 1)),
    "water-triplet": (lambda: run_mean_field(WATER_ATOM, spin=2), 3, (2, 0)),
}


def _build_solver(
    backend, max_iterations=50, ansatz=None, qsci_options=None, **vqe_kwargs
):
    return QSCISolver(
        backend=backend,
        optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ansatz=ansatz or UCCSDAnsatz(),
        max_iterations=max_iterations,
        qsci_options={"reporting_level": ReportingLevel.OFF, **(qsci_options or {})},
        **vqe_kwargs,
    )


def _collect_rdms_and_spin(solver, ci, n_orbitals, n_electrons):
    """Every RDM and the spin of ``ci`` through each solver method, by name."""
    args = (n_orbitals, n_electrons)
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = solver.make_rdm12s(ci, *args)
    rdm1s_alpha, rdm1s_beta = solver.make_rdm1s(ci, *args)
    rdm12_one, rdm12_two = solver.make_rdm12(ci, *args)
    spin_square, multiplicity = solver.spin_square(ci, *args)
    return {
        "make_rdm1": solver.make_rdm1(ci, *args),
        "make_rdm1s alpha": rdm1s_alpha,
        "make_rdm1s beta": rdm1s_beta,
        "make_rdm12 one-body": rdm12_one,
        "make_rdm12 two-body": rdm12_two,
        "make_rdm12s alpha": dm1a,
        "make_rdm12s beta": dm1b,
        "make_rdm12s alpha-alpha": dm2aa,
        "make_rdm12s alpha-beta": dm2ab,
        "make_rdm12s beta-beta": dm2bb,
        "spin_square": spin_square,
        "multiplicity": multiplicity,
    }


def _is_in_sector(config, n_alpha, n_beta):
    return config[0::2].count("1") == n_alpha and config[1::2].count("1") == n_beta


@pytest.mark.parametrize(
    ("build_mean_field", "n_orbitals", "n_electrons"),
    _ACTIVE_SPACES.values(),
    ids=_ACTIVE_SPACES.keys(),
)
def test_casci_matches_pyscf(
    build_mean_field, n_orbitals, n_electrons, default_test_simulator
):
    mean_field = build_mean_field()
    reference = run_mcscf(mcscf.CASCI, mean_field, n_orbitals, n_electrons)
    solver = _build_solver(default_test_simulator)

    casci = run_mcscf(mcscf.CASCI, mean_field, n_orbitals, n_electrons, solver)

    # Expansion reaches the whole sector of these active spaces, so QSCI is exact.
    assert casci.e_tot == pytest.approx(reference.e_tot, abs=1e-8)
    assert casci.converged
    assert isinstance(casci.ci, QSCIState)
    measured = _collect_rdms_and_spin(solver, casci.ci, n_orbitals, casci.nelecas)
    expected = _collect_rdms_and_spin(
        reference.fcisolver, reference.ci, n_orbitals, casci.nelecas
    )
    for name, oracle in expected.items():
        np.testing.assert_allclose(measured[name], oracle, atol=1e-6, err_msg=name)


def test_rdms_carry_the_fermionic_sign_of_every_configuration(dummy_simulator):
    mean_field = run_mean_field("N 0 0 0; N 0 0 1.6")
    reference = run_mcscf(mcscf.CASCI, mean_field, 6, 6)
    strings = [
        "".join("1" if p in occupied else "0" for p in range(6))
        for occupied in combinations(range(6), 3)
    ]
    subspace = ConfigurationSubspace(
        MolecularProblem.from_casci(mean_field, 6, 6).hamiltonian
    )
    subspace.add_configurations(
        "".join(a + b for a, b in zip(alpha, beta, strict=True))
        for alpha in strings
        for beta in strings
    )
    subspace.extend_hamiltonian_matrix()
    subspace.solve_eigensystem()
    state = QSCIState(subspace.energy, tuple(subspace.configs), subspace.eigenvector)

    dm1, dm2 = _build_solver(dummy_simulator).make_rdm12(state, 6, (3, 3))

    expected_dm1, expected_dm2 = reference.fcisolver.make_rdm12(reference.ci, 6, (3, 3))
    np.testing.assert_allclose(dm1, expected_dm1, atol=1e-9)
    np.testing.assert_allclose(dm2, expected_dm2, atol=1e-9)


def test_rdms_of_a_truncated_subspace_reproduce_its_energy(make_maestro_simulator):
    casci = mcscf.CASCI(run_mean_field(WATER_ATOM), 4, 4)
    h1, ecore = casci.get_h1eff()
    eri = ao2mo.restore(1, casci.get_h2eff(), 4)
    solver = _build_solver(
        make_maestro_simulator(maestro_config=MaestroConfig(seed=7)),
        qsci_options={"n_expansions": 0, "max_seed_configs": 3},
    )

    energy, state = solver.kernel(h1, eri, 4, (2, 2), ecore=ecore)

    dm1, dm2 = solver.make_rdm12(state, 4, (2, 2))
    assert len(state.configs) == 3
    assert energy == pytest.approx(
        ecore + np.einsum("pq,pq", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2),
        abs=1e-10,
    )


def test_casscf_matches_pyscf(default_test_simulator):
    mean_field = run_mean_field(WATER_ATOM)
    reference = run_mcscf(mcscf.CASSCF, mean_field, 4, 4)

    casscf = run_mcscf(
        mcscf.CASSCF, mean_field, 4, 4, _build_solver(default_test_simulator)
    )

    assert casscf.e_tot == pytest.approx(reference.e_tot, abs=1e-6)
    assert casscf.converged


def test_natural_orbitals_rerun_qsci_in_the_rotated_space(
    mocker, h2_mean_field, default_test_simulator
):
    reference = run_mcscf(mcscf.CASCI, h2_mean_field, 2, 2, natorb=True)
    run = mocker.spy(QSCI, "run")

    casci = run_mcscf(
        mcscf.CASCI,
        h2_mean_field,
        2,
        2,
        _build_solver(default_test_simulator),
        natorb=True,
    )

    assert run.call_count == 2
    assert casci.e_tot == pytest.approx(reference.e_tot, abs=1e-8)
    np.testing.assert_allclose(casci.mo_occ, reference.mo_occ, atol=1e-6)


def test_seed_is_the_most_probable_vqe_sample(
    h2_mean_field, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space

    _, state = _build_solver(
        default_test_simulator,
        qsci_options={"n_expansions": 0, "max_seed_configs": 1},
    ).kernel(h1, eri, 2, (1, 1), ecore=ecore)

    assert state.configs == ("1100",)
    assert len(state.records) == 1
    assert state.energy == pytest.approx(h2_mean_field.e_tot, abs=1e-8)


def test_seed_keeps_only_the_active_space_sector(
    mocker, h2_active_space, make_maestro_simulator
):
    h1, eri, ecore = h2_active_space
    run = mocker.spy(VQE, "run")
    solver = _build_solver(
        make_maestro_simulator(maestro_config=MaestroConfig(seed=3)),
        max_iterations=1,
        ansatz=GenericLayerAnsatz([RYGate], CXGate),
        qsci_options={"n_expansions": 0},
        seed=3,
    )

    _, state = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)

    (sampled,) = run.spy_return.best_probs.values()
    in_sector = [_is_in_sector(config, 1, 1) for config in sampled]
    assert any(in_sector) and not all(in_sector)
    assert set(state.configs) == {
        config for config, kept in zip(sampled, in_sector, strict=True) if kept
    }


def test_approx_kernel_runs_qsci_only_when_there_is_no_result(
    mocker, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator)
    run = mocker.spy(QSCI, "run")

    energy, state = solver.approx_kernel(h1, eri, 2, (1, 1), ecore=ecore)

    assert isinstance(state, QSCIState)
    again_energy, again_state = solver.approx_kernel(h1, eri, 2, (1, 1), ecore=ecore)
    assert again_state is state
    assert again_energy == energy
    assert run.call_count == 1


def test_vqe_keyword_arguments_reach_the_vqe(
    mocker, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    run = mocker.spy(VQE, "run")

    _build_solver(default_test_simulator, n_layers=2).kernel(
        h1, eri, 2, (1, 1), ecore=ecore
    )

    assert run.spy_return.n_layers == 2


@pytest.mark.parametrize(
    ("lambda_penalty", "energy", "spin_square"),
    [(0.0, -0.3, 2.0), (0.2, 0.3, 0.0)],
    ids=["unpenalised-triplet", "penalised-singlet"],
)
def test_spin_penalty_lifts_a_lower_triplet_above_the_singlet(
    mocker, default_test_simulator, lambda_penalty, energy, spin_square
):
    h1, eri, _ = build_degenerate_orbital_integrals(exchange=0.3)
    sector = dict.fromkeys(("1100", "1001", "0110", "0011"), 0.25)
    mocker.patch.object(
        VQE, "best_probs", new_callable=mocker.PropertyMock, return_value={0: sector}
    )
    solver = _build_solver(
        default_test_simulator,
        max_iterations=1,
        qsci_options={"n_expansions": 0, "lambda_penalty": lambda_penalty},
    )

    found, state = solver.kernel(h1, eri, 2, (1, 1))

    assert found == pytest.approx(energy, abs=1e-10)
    assert solver.spin_square(state, 2, (1, 1))[0] == pytest.approx(
        spin_square, abs=1e-10
    )


@pytest.mark.parametrize(
    "options",
    [["config_filter"], ["penalty"], ["config_filter", "penalty"]],
    ids=["config-filter", "penalty", "both"],
)
def test_qsci_options_cannot_set_what_the_solver_derives(dummy_simulator, options):
    with pytest.raises(
        ValueError,
        match=exact_match(
            "QSCISolver derives config_filter and penalty from the active space's "
            f"electron counts; qsci_options sets {', '.join(options)}."
        ),
    ):
        _build_solver(dummy_simulator, qsci_options=dict.fromkeys(options))


def test_rdms_reject_active_spaces_wider_than_pyscf_addresses(dummy_simulator):
    state = QSCIState(-1.0, ("10" * 64,), np.ones(1))

    with pytest.raises(
        ValueError,
        match=exact_match(
            "PySCF's selected CI takes at most 63 active orbitals; got 64."
        ),
    ):
        _build_solver(dummy_simulator).make_rdm1s(state, 64, (64, 0))


def test_rdms_reject_a_selected_ci_vector_past_the_size_limit(mocker, dummy_simulator):
    mocker.patch.object(_fcisolver, "_MAX_SCI_VECTOR_ENTRIES", 3)
    state = QSCIState(-1.0, ("1001", "0110"), np.array([0.6, 0.8]))

    with pytest.raises(
        ValueError,
        match=exact_match(
            "The state's 2 alpha and 2 beta strings span 4 pairs, more than the 3 "
            "PySCF's dense selected-CI vector is built for here; lower "
            "max_subspace_size."
        ),
    ):
        _build_solver(dummy_simulator).make_rdm1s(state, 2, (1, 1))


def test_unconverged_vqe_leaves_casci_unconverged(
    h2_mean_field, default_test_simulator
):
    verify_unconverged_vqe_leaves_casci_unconverged(
        _build_solver(default_test_simulator, max_iterations=1), h2_mean_field
    )


@ELECTRON_COUNT_SPLITS
def test_total_electron_count_splits_like_pyscf(
    total, pair, h2_active_space, default_test_simulator
):
    verify_total_electron_count_splits_like_pyscf(
        lambda: _build_solver(default_test_simulator), h2_active_space, total, pair
    )


def test_kernel_rejects_several_roots(h2_active_space, dummy_simulator):
    verify_kernel_rejects_several_roots(_build_solver(dummy_simulator), h2_active_space)


def test_kernel_rejects_unrestricted_integrals(h2_active_space, dummy_simulator):
    verify_kernel_rejects_unrestricted_integrals(
        _build_solver(dummy_simulator), h2_active_space
    )
