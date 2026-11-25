# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for VQEFCISolver, the PySCF active-space solver backed by VQE."""

import numpy as np
import pytest

from divi.qprog.algorithms import VQE, UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tests._helpers import (
    H2_ATOM,
    H3_CHAIN_ATOM,
    WATER_ATOM,
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

pytestmark = needs_qiskit_nature

#: ``(mean field, active orbitals, active electrons)`` checked against PySCF.
_ACTIVE_SPACES = {
    "h2-singlet": (lambda: run_mean_field(H2_ATOM, basis="6-31g"), 2, 2),
    "h3-doublet": (lambda: run_mean_field(H3_CHAIN_ATOM, spin=1), 3, (2, 1)),
}


def _build_solver(backend, optimizer=None, max_iterations=50, **vqe_kwargs):
    return VQEFCISolver(
        backend=backend,
        optimizer=optimizer or ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=max_iterations,
        **vqe_kwargs,
    )


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

    assert casci.e_tot == pytest.approx(reference.e_tot, abs=1e-6)
    assert casci.converged
    assert isinstance(casci.ci, VQE)
    args = (n_orbitals, casci.nelecas)
    measured = (
        solver.make_rdm1(casci.ci, *args),
        *solver.make_rdm12(casci.ci, *args),
        solver.spin_square(casci.ci, *args),
    )
    expected = (
        reference.fcisolver.make_rdm1(reference.ci, *args),
        *reference.fcisolver.make_rdm12(reference.ci, *args),
        reference.fcisolver.spin_square(reference.ci, *args),
    )
    # RDM errors scale as the square root of energy errors.
    for value, oracle in zip(measured, expected, strict=True):
        np.testing.assert_allclose(value, oracle, atol=1e-4)


def test_casscf_matches_pyscf_for_two_electrons(h2_mean_field, default_test_simulator):
    reference = run_mcscf(mcscf.CASSCF, h2_mean_field, 2, 2)

    casscf = run_mcscf(
        mcscf.CASSCF, h2_mean_field, 2, 2, _build_solver(default_test_simulator)
    )

    assert casscf.e_tot == pytest.approx(reference.e_tot, abs=1e-6)
    assert casscf.converged


def test_natural_orbitals_rerun_vqe_in_the_rotated_space(
    mocker, h2_mean_field, default_test_simulator
):
    reference = run_mcscf(mcscf.CASCI, h2_mean_field, 2, 2, natorb=True)
    run = mocker.spy(VQE, "run")

    casci = run_mcscf(
        mcscf.CASCI,
        h2_mean_field,
        2,
        2,
        _build_solver(default_test_simulator),
        natorb=True,
    )

    assert run.call_count == 2
    assert casci.e_tot == pytest.approx(reference.e_tot, abs=1e-6)
    np.testing.assert_allclose(casci.mo_occ, reference.mo_occ, atol=1e-6)


def test_open_shell_active_space_keeps_its_spin(default_test_simulator):
    mean_field = run_mean_field(WATER_ATOM, spin=2)
    reference = run_mcscf(mcscf.CASCI, mean_field, 3, (2, 0))
    solver = _build_solver(default_test_simulator)

    casci = run_mcscf(mcscf.CASCI, mean_field, 3, (2, 0), solver)

    assert casci.e_tot == pytest.approx(reference.e_tot, abs=1e-6)
    dm1a, dm1b = solver.make_rdm1s(casci.ci, 3, (2, 0))
    assert np.trace(dm1a) == pytest.approx(2.0, abs=1e-9)
    assert np.trace(dm1b) == pytest.approx(0.0, abs=1e-9)
    spin_square, multiplicity = solver.spin_square(casci.ci, 3, (2, 0))
    assert spin_square == pytest.approx(2.0, abs=1e-8)
    assert multiplicity == pytest.approx(3.0, abs=1e-8)


def test_kernel_warm_starts_from_the_previous_parameters(
    mocker, h2_active_space, default_test_simulator, default_optimizer
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator, default_optimizer, max_iterations=2)
    run = mocker.spy(VQE, "run")

    _, first = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)
    solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)

    first_call, second_call = run.call_args_list
    assert first_call.kwargs["initial_params"] is None
    np.testing.assert_array_equal(
        second_call.kwargs["initial_params"], first.best_params
    )


def test_kernel_starts_afresh_when_the_parameter_count_changes(
    mocker, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator)
    run = mocker.spy(VQE, "run")

    _, first = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)
    _, second = solver.kernel(h1, eri, 2, (2, 1), ecore=ecore)

    assert first.n_params != second.n_params
    assert run.call_args_list[1].kwargs["initial_params"] is None


def test_approx_kernel_trains_only_when_there_is_no_result(
    mocker, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator)
    run = mocker.spy(VQE, "run")

    result = solver.approx_kernel(h1, eri, 2, (1, 1), ecore=ecore)

    assert isinstance(result[1], VQE)
    assert solver.approx_kernel(h1, eri, 2, (1, 1), ecore=ecore) == result
    assert run.call_count == 1


def test_last_rdms_are_reused_as_copies(
    mocker, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator)
    measure = mocker.spy(VQE, "measure_rdms")
    _, vqe = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)

    (dm1a, _), _ = solver.make_rdm12s(vqe, 2, (1, 1))
    dm1a[0, 0] = 99.0

    (dm1a_again, _), _ = solver.make_rdm12s(vqe, 2, (1, 1))
    assert measure.call_count == 1
    assert dm1a_again[0, 0] != 99.0


def test_rdms_of_an_earlier_program_are_measured_again(
    h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(default_test_simulator)
    _, first = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)
    _, last = solver.kernel(h1 + np.diag([0.0, -1.0]), eri, 2, (1, 1), ecore=ecore)

    dm1s = solver.make_rdm1s(first, 2, (1, 1))

    first_dm1s, _ = first.measure_rdms()
    last_dm1s, _ = solver.make_rdm12s(last, 2, (1, 1))
    np.testing.assert_allclose(dm1s, first_dm1s, atol=1e-12)
    assert np.abs(np.subtract(dm1s, last_dm1s)).max() > 1e-3


def test_vqe_keyword_arguments_reach_every_vqe(h2_active_space, default_test_simulator):
    h1, eri, ecore = h2_active_space

    _, vqe = _build_solver(default_test_simulator, n_layers=2).kernel(
        h1, eri, 2, (1, 1), ecore=ecore
    )

    assert vqe.n_layers == 2


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
