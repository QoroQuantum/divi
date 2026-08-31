# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for pure progress-state transitions."""

import pytest

from divi.backends._job_status import JobStatus
from divi.reporting._events import ProgressEvent, ProgressScope, TerminalStatus
from divi.reporting._state import ProgressState


def registered_program_state(
    *, hide_successful_programs: bool = False
) -> ProgressState:
    """Create state containing one visible program target."""
    state = ProgressState(hide_successful_programs=hide_successful_programs)
    state.apply(ProgressEvent.register("p", ProgressScope.PROGRAM, "Program", 3))
    return state


def registered_batch_with_programs() -> ProgressState:
    """Create state with two programs assigned to one active batch."""
    state = ProgressState()
    for target in ("p1", "p2"):
        state.apply(ProgressEvent.register(target, ProgressScope.PROGRAM, target, 2))
    state.apply(
        ProgressEvent.register(
            "batch",
            ProgressScope.BATCH,
            "Batch",
            None,
            batch_color="cyan",
            program_keys=("p1", "p2"),
        )
    )
    return state


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            ProgressEvent.advance("p", amount=2, loss=-0.5),
            {"completed": 2, "loss": -0.5},
        ),
        (
            ProgressEvent.show("p", "submitting"),
            {"message": "submitting", "service_job_id": None},
        ),
        (
            ProgressEvent.polling(
                "p",
                job_id="abc-def",
                status=JobStatus.RUNNING,
                attempt=1,
                limit=3,
            ),
            {
                "message": None,
                "service_job_id": "abc-def",
                "job_status": JobStatus.RUNNING,
                "poll_attempt": 1,
                "max_retries": 3,
            },
        ),
    ],
)
def test_active_program_events_update_their_rendered_fields(event, expected):
    """Each active event updates only its corresponding rendered state."""
    state = registered_program_state()

    assert state.apply(event) == {"p"}

    target = state.get("p")
    for name, value in expected.items():
        assert getattr(target, name) == value


def test_active_registration_is_idempotent():
    state = registered_program_state()

    affected = state.apply(
        ProgressEvent.register("p", ProgressScope.PROGRAM, "Replacement", 99)
    )

    target = state.get("p")
    assert affected == set()
    assert target.label == "Program"
    assert target.total == 3


def test_active_registration_rejects_a_different_scope():
    state = registered_program_state()

    with pytest.raises(ValueError, match="scope"):
        state.apply(ProgressEvent.register("p", ProgressScope.BATCH, "Batch", None))


def test_terminal_registration_starts_a_clean_lifecycle():
    state = registered_program_state()
    state.apply(ProgressEvent.advance("p", loss=-0.5))
    state.apply(ProgressEvent.finish("p", TerminalStatus.SUCCESS, detail="complete"))

    affected = state.apply(
        ProgressEvent.register("p", ProgressScope.PROGRAM, "Retry", 4, visible=False)
    )

    target = state.get("p")
    assert affected == {"p"}
    assert target.label == "Retry"
    assert target.total == 4
    assert target.completed == 0
    assert target.loss is None
    assert target.terminal_status is None
    assert target.detail is None
    assert target.visible is False


@pytest.mark.parametrize(
    "event",
    [
        ProgressEvent.advance("missing"),
        ProgressEvent.show("missing", "phase"),
        ProgressEvent.polling(
            "missing",
            job_id="abc-def",
            status=JobStatus.RUNNING,
            attempt=0,
            limit=None,
        ),
        ProgressEvent.finish("missing", TerminalStatus.FAILED),
    ],
)
def test_events_for_unknown_targets_raise_key_error(event):
    state = ProgressState()

    with pytest.raises(KeyError, match="missing"):
        state.apply(event)


def test_late_advance_after_terminal_completion_is_ignored():
    state = registered_program_state()
    state.apply(ProgressEvent.advance("p", amount=2, loss=-0.5))
    state.apply(ProgressEvent.finish("p", TerminalStatus.SUCCESS))

    affected = state.apply(ProgressEvent.advance("p", amount=1, loss=-1.0))

    target = state.get("p")
    assert affected == set()
    assert target.completed == 3
    assert target.loss == -0.5


def test_show_after_completed_polling_clears_polling_state():
    state = registered_program_state()
    state.apply(
        ProgressEvent.polling(
            "p", job_id="abc-def", status=JobStatus.COMPLETED, attempt=2, limit=None
        )
    )

    state.apply(ProgressEvent.show("p", "next phase"))

    target = state.get("p")
    assert target.message == "next phase"
    assert target.service_job_id is None
    assert target.job_status is None
    assert target.poll_attempt is None
    assert target.max_retries is None


def test_repeated_show_with_the_same_message_reports_no_change():
    state = registered_program_state()
    event = ProgressEvent.show("p", "next phase")
    state.apply(event)

    assert state.apply(event) == set()


def test_repeated_polling_with_the_same_fields_reports_no_change():
    state = registered_program_state()
    event = ProgressEvent.polling(
        "p", job_id="abc-def", status=JobStatus.RUNNING, attempt=2, limit=3
    )
    state.apply(event)

    assert state.apply(event) == set()


def test_finish_preserves_loss_and_detail_but_clears_polling():
    state = registered_program_state()
    state.apply(ProgressEvent.advance("p", loss=-1.25))
    state.apply(
        ProgressEvent.polling(
            "p", job_id="abc-def", status=JobStatus.RUNNING, attempt=2, limit=None
        )
    )

    state.apply(
        ProgressEvent.finish(
            "p",
            TerminalStatus.FAILED,
            job_status=JobStatus.TIMED_OUT,
            detail="deadline exceeded",
        )
    )

    target = state.get("p")
    assert target.loss == -1.25
    assert target.terminal_status is TerminalStatus.FAILED
    assert target.job_status is JobStatus.TIMED_OUT
    assert target.detail == "deadline exceeded"
    assert target.service_job_id is None
    assert target.poll_attempt is None
    assert target.max_retries is None


def test_successful_finish_fills_a_finite_total():
    state = registered_program_state()
    state.apply(ProgressEvent.advance("p", amount=1))

    state.apply(ProgressEvent.finish("p", TerminalStatus.SUCCESS))

    target = state.get("p")
    assert target.completed == target.total == 3


@pytest.mark.parametrize(
    ("hide_successful_programs", "status", "visible"),
    [
        (False, TerminalStatus.SUCCESS, True),
        (True, TerminalStatus.SUCCESS, False),
        (True, TerminalStatus.FAILED, True),
        (True, TerminalStatus.CANCELLED, True),
        (True, TerminalStatus.ABORTED, True),
    ],
)
def test_program_terminal_visibility_follows_reporting_level(
    hide_successful_programs, status, visible
):
    state = registered_program_state(hide_successful_programs=hide_successful_programs)

    state.apply(ProgressEvent.finish("p", status))

    assert state.get("p").visible is visible


@pytest.mark.parametrize(
    "scope",
    [ProgressScope.PREPARATION, ProgressScope.WORKFLOW],
)
def test_preparation_and_workflow_scopes_use_standard_transitions(scope):
    state = ProgressState()
    state.apply(ProgressEvent.register(scope, scope, scope.value, 2))

    affected = state.apply(ProgressEvent.advance(scope))
    state.apply(ProgressEvent.show(scope, "running"))

    target = state.get(scope)
    assert affected == {scope}
    assert target.completed == 1
    assert target.message == "running"
    assert target.scope is scope


def test_batch_registration_assigns_its_colour_to_member_programs():
    state = registered_batch_with_programs()

    assert state.get("batch").batch_color == "cyan"
    assert state.get("p1").batch_color == "cyan"
    assert state.get("p2").batch_color == "cyan"


def test_finishing_batch_clears_program_colours():
    state = registered_batch_with_programs()

    affected = state.apply(ProgressEvent.finish("batch", TerminalStatus.SUCCESS))

    assert affected == {"batch", "p1", "p2"}
    assert state.get("p1").batch_color == ""
    assert state.get("p2").batch_color == ""


def test_targets_exposes_registered_target_states():
    state = registered_program_state()

    assert set(state.targets) == {"p"}
    assert state.targets["p"] == state.get("p")


def test_externally_obtained_target_state_cannot_mutate_reducer_state():
    state = registered_program_state()
    from_get = state.get("p")
    from_targets = state.targets["p"]

    from_get.completed = 99
    from_targets.loss = -1.25

    stored = state.get("p")
    assert stored.completed == 0
    assert stored.loss is None
