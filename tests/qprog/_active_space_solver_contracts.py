# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural contracts shared by divi's PySCF active-space solvers,
``VQEFCISolver`` and ``QSCISolver``."""

import numpy as np
import pytest

from tests._helpers import exact_match, run_mcscf

#: ``(total, pair)``: total electron counts and the pair PySCF splits each into.
ELECTRON_COUNT_SPLITS = pytest.mark.parametrize(
    ("total", "pair"),
    [(2, (1, 1)), (3, (2, 1)), (np.int64(2), (1, 1))],
    ids=["even", "odd", "numpy-integer"],
)


def verify_unconverged_vqe_leaves_casci_unconverged(solver, mean_field) -> None:
    """A CASCI over H2's CAS(2,2) reports the solver's unconverged VQE."""
    mcscf = pytest.importorskip("pyscf.mcscf")

    casci = run_mcscf(mcscf.CASCI, mean_field, 2, 2, solver)

    assert not casci.converged


def verify_total_electron_count_splits_like_pyscf(
    build_solver, active_space, total, pair
) -> None:
    """A total electron count gives the energy and per-spin occupations of the
    pair PySCF splits it into."""
    h1, eri, ecore = active_space
    solver = build_solver()

    energy, ci = solver.kernel(h1, eri, 2, total, ecore=ecore)

    expected, _ = build_solver().kernel(h1, eri, 2, pair, ecore=ecore)
    assert energy == pytest.approx(expected, abs=1e-9)
    dm1a, dm1b = solver.make_rdm1s(ci, 2, total)
    assert (np.trace(dm1a), np.trace(dm1b)) == pytest.approx(pair, abs=1e-6)


def verify_kernel_rejects_several_roots(solver, active_space) -> None:
    h1, eri, ecore = active_space
    solver.nroots = 2

    with pytest.raises(
        NotImplementedError,
        match=exact_match(f"{type(solver).__name__} finds one root; got nroots=2."),
    ):
        solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)


def verify_kernel_rejects_unrestricted_integrals(solver, active_space) -> None:
    h1, eri, ecore = active_space

    with pytest.raises(
        NotImplementedError,
        match=exact_match(
            f"{type(solver).__name__} needs restricted integrals; got separate "
            "alpha and beta ones."
        ),
    ):
        solver.kernel((h1, h1), (eri, eri, eri), 2, (1, 1), ecore=ecore)
