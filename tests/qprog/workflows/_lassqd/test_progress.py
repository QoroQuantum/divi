# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's progress reporting during long classical stages."""

import numpy as np
import pytest

pytest.importorskip("pyscf")

from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows._lassqd._integrals import optimize_orbitals
from divi.qprog.workflows._lassqd._preparation import (
    LinearMethodFragmentProgram,
    prepare_lucj_fragment,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec
from divi.reporting._events import EventKind
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    exact_sampler_lassqd,
    orbital_rotation_case,
)


def test_the_orbital_solve_reports_every_iteration(orbital_rotation_case):
    """Minutes pass in this solve at production size with nothing else to show."""
    messages = []

    solve = optimize_orbitals(
        *orbital_rotation_case, gradient_tol=1e-3, report=messages.append
    )

    assert len(messages) == solve.n_iterations
    assert messages[0].startswith("Orbital solve: iteration 1, energy ")
    assert all("|g|" in message and " s" in message for message in messages)


def _three_orbital_fragment():
    spec = FragmentSpec(orbitals=(0, 1, 2), n_alpha=2, n_beta=1)
    rng = np.random.default_rng(3)
    one_body = rng.normal(size=(3, 3))
    one_body = one_body + one_body.T
    return spec, one_body, np.zeros((3,) * 4)


def test_lucj_preparation_reports_each_stage_and_iteration_energy():
    spec, one_body, two_body = _three_orbital_fragment()
    stages, energies = [], []

    prepare_lucj_fragment(
        one_body,
        one_body,
        two_body,
        spec,
        report=stages.append,
        on_iteration=energies.append,
    )

    assert stages == ["Fragment ROHF", "CCSD seed", "Linear method"]
    assert energies and all(np.isfinite(energy) for energy in energies)


def test_a_linear_method_program_reports_through_its_progress_row(dummy_simulator):
    """Fragment preparation runs in a worker thread, so it reports through the
    program's own progress channel: stages as row messages, and each
    linear-method iteration as an advance carrying its energy, as a
    variational program reports its iterations."""
    spec, one_body, two_body = _three_orbital_fragment()
    problem = MolecularProblem(one_body, two_body, n_alpha=2, n_beta=1)
    program = LinearMethodFragmentProgram(problem, spec, backend=dummy_simulator)
    events = []
    program._progress_emitter = events.append

    program.run()

    shown = [
        event.message
        for event in events
        if event.kind is EventKind.SHOW and event.message
    ]
    advances = [event for event in events if event.kind is EventKind.ADVANCE]
    assert shown[:3] == ["Fragment ROHF", "CCSD seed", "Linear method"]
    assert "Sampling the prepared circuit" in shown
    assert advances and all(event.loss is not None for event in advances)
    assert all(event.progress_key == program._progress_key for event in events)


def test_update_state_reports_each_fragment_and_the_orbital_solve(
    exact_sampler_lassqd, mocker
):
    ensemble, state = exact_sampler_lassqd
    stages = mocker.spy(ensemble, "_emit_workflow_stage")
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)

    ensemble.update_state(state)

    messages = [call.args[0] for call in stages.call_args_list]
    assert any(message.startswith("Recovered fragment_0: ") for message in messages)
    assert any(message.startswith("Recovered fragment_1: ") for message in messages)
    assert any(message.startswith("Orbital solve: iteration ") for message in messages)
