# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behaviour shared by divi's PySCF active-space solvers, VQEFCISolver and
QSCISolver."""

import numpy as np
import pytest

from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.ensembles import ReportingLevel
from divi.qprog.ensembles.qsci import QSCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tests._helpers import exact_match, run_mcscf
from tests.qprog.algorithms._helpers import needs_qiskit_nature

mcscf = pytest.importorskip("pyscf.mcscf")

pytestmark = needs_qiskit_nature

#: Each solver's options beyond the ones they share.
_EXTRA_OPTIONS = {
    VQEFCISolver: {},
    QSCISolver: {"qsci_options": {"reporting_level": ReportingLevel.OFF}},
}


@pytest.fixture(params=[VQEFCISolver, QSCISolver], ids=["vqe", "qsci"])
def solver_class(request):
    return request.param


def _build_solver(solver_class, backend, max_iterations=50):
    return solver_class(
        backend=backend,
        optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=max_iterations,
        **_EXTRA_OPTIONS[solver_class],
    )


def test_unconverged_vqe_leaves_casci_unconverged(
    solver_class, h2_mean_field, default_test_simulator
):
    casci = run_mcscf(
        mcscf.CASCI,
        h2_mean_field,
        2,
        2,
        _build_solver(solver_class, default_test_simulator, max_iterations=1),
    )

    assert not casci.converged


@pytest.mark.parametrize(
    ("total", "pair"),
    [(2, (1, 1)), (3, (2, 1)), (np.int64(2), (1, 1))],
    ids=["even", "odd", "numpy-integer"],
)
def test_total_electron_count_splits_like_pyscf(
    solver_class, total, pair, h2_active_space, default_test_simulator
):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(solver_class, default_test_simulator)

    energy, ci = solver.kernel(h1, eri, 2, total, ecore=ecore)

    expected, _ = _build_solver(solver_class, default_test_simulator).kernel(
        h1, eri, 2, pair, ecore=ecore
    )
    assert energy == pytest.approx(expected, abs=1e-9)
    dm1a, dm1b = solver.make_rdm1s(ci, 2, total)
    assert (np.trace(dm1a), np.trace(dm1b)) == pytest.approx(pair, abs=1e-6)


def test_kernel_rejects_several_roots(solver_class, h2_active_space, dummy_simulator):
    h1, eri, ecore = h2_active_space
    solver = _build_solver(solver_class, dummy_simulator)
    solver.nroots = 2

    with pytest.raises(
        NotImplementedError,
        match=exact_match(f"{solver_class.__name__} finds one root; got nroots=2."),
    ):
        solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)


def test_kernel_rejects_unrestricted_integrals(
    solver_class, h2_active_space, dummy_simulator
):
    h1, eri, ecore = h2_active_space

    with pytest.raises(
        NotImplementedError,
        match=exact_match(
            f"{solver_class.__name__} needs restricted integrals; got separate "
            "alpha and beta ones."
        ),
    ):
        _build_solver(solver_class, dummy_simulator).kernel(
            (h1, h1), (eri, eri, eri), 2, (1, 1), ecore=ecore
        )
