# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LASSQD's progress reporting during long classical stages."""

import numpy as np
import pytest

from divi.qprog.problems import MolecularProblem
from divi.qprog.workflows._lassqd._config import (
    FullOrbitalSolve,
    SecondOrderOrbitalSolve,
)
from divi.qprog.workflows._lassqd._preparation import (
    LUCJFragmentProgram,
    prepare_lucj_fragment,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec
from divi.reporting._events import EventKind
from tests.qprog.workflows._lassqd._helpers import (  # noqa: F401
    build_exact_sampler_lassqd,
    orbital_rotation_case,
    orbital_solver,
)


def test_an_orbital_solve_reports_every_iteration(
    orbital_rotation_case, orbital_solver
):
    messages = []

    solve = orbital_solver(
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


def test_a_lucj_fragment_program_reports_through_its_progress_row(dummy_simulator):
    """Preparation stages arrive as row messages and each optimiser iteration
    as an advance carrying its energy, on the program's own progress key."""
    spec, one_body, two_body = _three_orbital_fragment()
    problem = MolecularProblem(one_body, two_body, n_alpha=2, n_beta=1)
    program = LUCJFragmentProgram(problem, spec, backend=dummy_simulator)
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


@pytest.mark.parametrize(
    "orbital_update",
    [FullOrbitalSolve(), SecondOrderOrbitalSolve()],
    ids=["full", "second-order"],
)
def test_update_state_reports_each_fragment_and_the_orbital_solve(
    dummy_expval_backend, mocker, orbital_update
):
    ensemble, state = build_exact_sampler_lassqd(
        dummy_expval_backend, mocker, orbital_update=orbital_update
    )
    stages = mocker.spy(ensemble, "_emit_workflow_stage")
    ensemble.create_programs(state)
    ensemble.run_one_round(blocking=True)

    ensemble.update_state(state)

    messages = [call.args[0] for call in stages.call_args_list]
    assert any(message.startswith("Recovered fragment_0: ") for message in messages)
    assert any(message.startswith("Recovered fragment_1: ") for message in messages)
    assert any(message.startswith("Orbital solve: iteration ") for message in messages)
