# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from divi.backends._job_status import JobStatus
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
    discard_progress_event,
)


def test_register_keeps_progress_metadata():
    event = ProgressEvent.register(
        progress_key="batch-1",
        scope=ProgressScope.BATCH,
        label="Batch 1",
        total=12,
        visible=False,
        batch_color="cyan",
        program_keys=("program-1", "program-2"),
    )

    assert event.kind is EventKind.REGISTER
    assert event.progress_key == "batch-1"
    assert event.scope is ProgressScope.BATCH
    assert event.label == "Batch 1"
    assert event.total == 12
    assert event.visible is False
    assert event.batch_color == "cyan"
    assert event.program_keys == ("program-1", "program-2")


def test_register_normalizes_batch_members_to_a_tuple():
    event = ProgressEvent.register(
        progress_key="batch-1",
        scope=ProgressScope.BATCH,
        label="Batch 1",
        total=None,
        program_keys=(key for key in ("program-1", "program-2")),
    )

    assert event.program_keys == ("program-1", "program-2")
    assert isinstance(event.program_keys, tuple)


def test_show_keeps_an_explicit_empty_message_to_clear_a_phase():
    event = ProgressEvent.show("program-1", "")

    assert event.kind is EventKind.SHOW
    assert event.message == ""


def test_polling_preserves_job_status_enum():
    event = ProgressEvent.polling(
        progress_key="program-1",
        job_id="abc-def",
        status=JobStatus.RUNNING,
        attempt=4,
        limit=None,
    )
    assert event.kind is EventKind.POLLING
    assert event.job_status is JobStatus.RUNNING
    assert event.poll_attempt == 4
    assert event.max_retries is None


@pytest.mark.parametrize(
    ("status", "kind", "job_status", "message"),
    [
        ("RUNNING", EventKind.POLLING, JobStatus.RUNNING, None),
        (
            "BACKEND_SPECIFIC",
            EventKind.SHOW,
            None,
            "Backend status BACKEND_SPECIFIC for job abc-def (attempt 4)",
        ),
    ],
)
def test_polling_normalizes_known_and_preserves_unknown_statuses(
    status, kind, job_status, message
):
    event = ProgressEvent.polling(
        progress_key="program-1",
        job_id="abc-def",
        status=status,
        attempt=4,
        limit=None,
    )

    assert event.kind is kind
    assert event.job_status is job_status
    assert event.message == message


def test_finish_keeps_terminal_status_separate_from_job_status():
    event = ProgressEvent.finish(
        progress_key="program-1",
        status=TerminalStatus.FAILED,
        job_status=JobStatus.TIMED_OUT,
        detail="deadline exceeded",
    )
    assert event.terminal_status is TerminalStatus.FAILED
    assert event.job_status is JobStatus.TIMED_OUT
    assert event.detail == "deadline exceeded"


def test_progress_events_are_frozen():
    event = ProgressEvent.advance("program-1")

    with pytest.raises(FrozenInstanceError):
        event.amount = 2


def test_progress_event_fields_are_keyword_only():
    with pytest.raises(TypeError):
        ProgressEvent(EventKind.ADVANCE, "program-1", amount=1)


def test_discard_progress_event_is_a_no_op():
    discard_progress_event(ProgressEvent.advance("program-1"))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: ProgressEvent.register("p", ProgressScope.PROGRAM, "P", total=-1),
            "total",
        ),
        (
            lambda: ProgressEvent.register(
                "p",
                ProgressScope.PROGRAM,
                "P",
                total=1,
                program_keys=("other",),
            ),
            "batch metadata",
        ),
        (lambda: ProgressEvent.advance("p", amount=0), "amount"),
        (
            lambda: ProgressEvent.polling(
                "p", job_id="", status=JobStatus.RUNNING, attempt=1, limit=3
            ),
            "job_id",
        ),
        (
            lambda: ProgressEvent.polling(
                "p", job_id="j", status=JobStatus.RUNNING, attempt=-1, limit=3
            ),
            "attempt",
        ),
        (
            lambda: ProgressEvent.polling(
                "p", job_id="j", status=JobStatus.RUNNING, attempt=1, limit=-1
            ),
            "limit",
        ),
        (
            lambda: ProgressEvent.polling(
                "p", job_id="j", status=JobStatus.RUNNING, attempt=4, limit=3
            ),
            "attempt",
        ),
    ],
)
def test_invalid_event_arguments_fail_at_construction(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
