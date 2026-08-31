# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for Rich views rendered from progress state."""

import logging
from io import StringIO
from threading import Event, Thread

import pytest
from rich.console import Console
from rich.progress import Progress, TextColumn

import divi.reporting._rich as rich_module
from divi.backends._job_status import JobStatus
from divi.reporting._events import ProgressEvent, ProgressScope, TerminalStatus
from divi.reporting._rich import (
    BatchIndicatorColumn,
    ConditionalSpinnerColumn,
    _ProgressBearingColumn,
    make_ensemble_view,
    make_standalone_view,
    render_status_text,
)
from divi.reporting._state import ProgressState


def registered_program_state() -> ProgressState:
    """Create state containing one visible program row."""
    state = ProgressState()
    state.apply(ProgressEvent.register("p", ProgressScope.PROGRAM, "Program", 3))
    return state


def completed_poll_then_next_phase_state() -> ProgressState:
    """Create state after a completed poll transitions to another phase."""
    state = registered_program_state()
    state.apply(
        ProgressEvent.polling(
            "p", job_id="abc-def", status=JobStatus.COMPLETED, attempt=2, limit=None
        )
    )
    state.apply(ProgressEvent.show("p", "next phase"))
    return state


def task_for_target(column, target):
    """Create a real Rich task from a state-owned target snapshot."""
    progress = Progress(column, auto_refresh=False)
    task_id = progress.add_task(
        "",
        total=target.total,
        completed=target.completed,
        scope=target.scope,
        terminal_status=target.terminal_status,
        batch_color=target.batch_color,
        label=target.label,
    )
    return next(task for task in progress.tasks if task.id == task_id)


@pytest.mark.parametrize(
    ("status", "icon", "punctuation"),
    [
        (TerminalStatus.SUCCESS, "✅", "!"),
        (TerminalStatus.FAILED, "❌", "!"),
        (TerminalStatus.CANCELLED, "⏹️", ""),
        (TerminalStatus.ABORTED, "⚠️", ""),
    ],
)
def test_terminal_status_renders_icon_and_final_loss(status, icon, punctuation):
    state = registered_program_state()
    state.apply(ProgressEvent.advance("p", loss=-0.123456789))
    state.apply(ProgressEvent.finish("p", status, detail="details"))

    text = render_status_text(state.get("p"))

    assert str(text) == (
        f"• {status.value}{punctuation} {icon}  (details) [loss: -0.123457]"
    )


def test_unlimited_polling_renders_short_job_id_and_highlights_it():
    state = registered_program_state()
    state.apply(
        ProgressEvent.polling(
            "p", job_id="abc-def", status=JobStatus.RUNNING, attempt=3, limit=None
        )
    )

    text = render_status_text(state.get("p"))

    assert str(text) == " [Job abc is RUNNING. Polling attempt 3 / ∞]"
    assert any(str(span.style) == "blue" for span in text.spans)


def test_next_phase_does_not_render_stale_completed_job():
    state = completed_poll_then_next_phase_state()

    text = render_status_text(state.get("p"))

    assert str(text) == "[next phase]"
    assert "complete" not in str(text)


def test_active_target_shows_spinner_and_terminal_target_hides_it():
    state = registered_program_state()
    column = ConditionalSpinnerColumn()

    assert str(column.render(task_for_target(column, state.get("p"))))

    state.apply(ProgressEvent.finish("p", TerminalStatus.SUCCESS))
    assert str(column.render(task_for_target(column, state.get("p")))) == ""


def test_progress_bearing_column_renders_program_and_preparation_progress():
    column = _ProgressBearingColumn(TextColumn("{task.fields[label]}"))
    state = registered_program_state()

    program_text = column.render(task_for_target(column, state.get("p")))

    state.apply(
        ProgressEvent.register(
            "preparation",
            ProgressScope.PREPARATION,
            "Submitting circuits",
            3,
        )
    )
    preparation_text = column.render(task_for_target(column, state.get("preparation")))

    state.apply(ProgressEvent.register("batch", ProgressScope.BATCH, "Batch", None))
    batch_text = column.render(task_for_target(column, state.get("batch")))

    assert str(program_text) == "Program"
    assert str(preparation_text) == "Submitting circuits"
    assert str(batch_text) == ""


def test_batch_indicator_uses_state_assigned_colour():
    state = registered_program_state()
    state.apply(
        ProgressEvent.register(
            "batch",
            ProgressScope.BATCH,
            "Batch",
            None,
            batch_color="cyan",
            program_keys=("p",),
        )
    )
    column = BatchIndicatorColumn()

    indicator = column.render(task_for_target(column, state.get("p")))

    assert str(indicator) == "■ "
    assert str(indicator.style) == "cyan"


def test_render_failure_uses_the_supplied_console():
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)

    rich_module.render_failure(
        RuntimeError("backend exploded"),
        label=" (Program alpha)",
        console=console,
    )

    rendered = output.getvalue()
    assert "Program Failure (Program alpha)" in rendered
    assert "RuntimeError: backend exploded" in rendered
    assert "Traceback follows" in rendered


def test_ensemble_view_honours_state_row_visibility():
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=120)
    state = ProgressState(hide_successful_programs=True)
    affected = state.apply(
        ProgressEvent.register(
            "p", ProgressScope.PROGRAM, "Hidden program", 1, visible=False
        )
    )
    render, close = make_ensemble_view(console, is_jupyter=True)

    try:
        render(state, affected)
        assert "Hidden program" not in output.getvalue()

        affected = state.apply(ProgressEvent.finish("p", TerminalStatus.FAILED))
        render(state, affected)
    finally:
        close()

    assert "Hidden program" in output.getvalue()
    assert "Failed" in output.getvalue()


def test_view_closures_are_idempotent():
    console = Console(file=StringIO(), force_terminal=True)

    _, close_standalone = make_standalone_view(console)
    _, close_ensemble = make_ensemble_view(console, is_jupyter=True)

    close_standalone()
    close_standalone()
    close_ensemble()
    close_ensemble()


def test_ensemble_view_close_serializes_with_inflight_render(monkeypatch):
    console = Console(file=StringIO(), force_terminal=True)
    state = registered_program_state()
    render_started = Event()
    release_render = Event()
    close_finished = Event()
    original_add_target = rich_module._add_target

    def blocking_add_target(progress, target):
        render_started.set()
        if not release_render.wait(timeout=2):
            raise RuntimeError("test did not release render")
        return original_add_target(progress, target)

    monkeypatch.setattr(rich_module, "_add_target", blocking_add_target)
    render, close = make_ensemble_view(console, is_jupyter=True)
    renderer = Thread(target=render, args=(state, {"p"}))
    renderer.start()
    assert render_started.wait(timeout=1)

    def close_view():
        close()
        close_finished.set()

    closer = Thread(target=close_view)
    closer.start()
    closed_while_rendering = close_finished.wait(timeout=0.5)
    release_render.set()
    renderer.join(timeout=2)
    closer.join(timeout=2)

    live_after_close = tuple(console._live_stack)
    for live in reversed(live_after_close):
        live.stop()
    assert closed_while_rendering
    assert not renderer.is_alive()
    assert not closer.is_alive()
    assert live_after_close == ()


def test_standalone_view_logs_terminal_state_with_service_status(caplog):
    console = Console(file=StringIO(), force_terminal=True, color_system=None)
    state = registered_program_state()
    render, close = make_standalone_view(console)

    with caplog.at_level(logging.INFO, logger="divi"):
        render(state, {"p"})
        affected = state.apply(
            ProgressEvent.finish(
                "p",
                TerminalStatus.FAILED,
                job_status=JobStatus.TIMED_OUT,
                detail="deadline exceeded",
            )
        )
        render(state, affected)
    close()

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.startswith("Program: • Failed! ❌")
    assert "deadline exceeded" in message
    assert "TIMED_OUT" in message
