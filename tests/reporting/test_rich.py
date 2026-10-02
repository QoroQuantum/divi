# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for Rich views rendered from progress state."""

import logging
import re
from io import StringIO
from threading import Event, Thread

import pytest
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, TextColumn
from rich.spinner import Spinner
from rich.status import Status

import divi.reporting._rich as rich_module
from divi.backends._job_status import JobStatus
from divi.reporting._events import ProgressEvent, ProgressScope, TerminalStatus
from divi.reporting._rich import (
    BatchIndicatorColumn,
    ConditionalSpinnerColumn,
    _ProgressBearingColumn,
    _short_job_id,
    make_ensemble_view,
    make_standalone_view,
    render_status_text,
)
from divi.reporting._state import ProgressState


def captured_console(*, force_terminal: bool) -> tuple[Console, StringIO]:
    """Return a colourless console writing to an inspectable buffer."""
    output = StringIO()
    console = Console(
        file=output, force_terminal=force_terminal, color_system=None, width=120
    )
    return console, output


def block_calls(monkeypatch, name: str) -> tuple[Event, Event]:
    """Make ``rich_module.<name>`` wait for release; return (started, release)."""
    started = Event()
    release = Event()
    original = getattr(rich_module, name)

    def blocking(*args):
        started.set()
        if not release.wait(timeout=2):
            raise RuntimeError("test did not release render")
        return original(*args)

    monkeypatch.setattr(rich_module, name, blocking)
    return started, release


def failure_from_backend() -> RuntimeError:
    """Return a raised exception that carries its traceback."""
    try:
        raise RuntimeError("backend exploded")
    except RuntimeError as exc:
        return exc


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


@pytest.mark.parametrize(
    ("attempt", "limit", "expected"),
    [
        (3, None, " [Job abc is RUNNING. Polling attempt 3 / ∞]"),
        (1, None, " [Job abc is RUNNING. Polling attempt 1 / ∞]"),
        (2, 3, " [Job abc is RUNNING. Polling attempt 2 / 3]"),
    ],
)
def test_polling_renders_short_job_id_and_highlights_it(attempt, limit, expected):
    state = registered_program_state()
    state.apply(
        ProgressEvent.polling(
            "p",
            job_id="abc-def",
            status=JobStatus.RUNNING,
            attempt=attempt,
            limit=limit,
        )
    )

    text = render_status_text(state.get("p"))

    assert str(text) == expected
    assert any(str(span.style) == "blue" for span in text.spans)


def test_short_job_id_keeps_the_text_before_the_first_hyphen():
    assert _short_job_id("abc-def-ghi") == "abc"


def test_next_phase_does_not_render_stale_completed_job():
    state = completed_poll_then_next_phase_state()

    text = render_status_text(state.get("p"))

    assert str(text) == "[next phase]"
    assert "complete" not in str(text)


def test_active_target_shows_spinner_and_terminal_target_hides_it():
    state = registered_program_state()
    column = ConditionalSpinnerColumn()

    assert isinstance(column.render(task_for_target(column, state.get("p"))), Spinner)

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


def test_batch_indicator_leaves_rows_without_a_batch_blank():
    column = BatchIndicatorColumn()

    indicator = column.render(
        task_for_target(column, registered_program_state().get("p"))
    )

    assert str(indicator) == "  "


def test_render_failure_prints_panel_and_traceback_to_the_supplied_console():
    console, output = captured_console(force_terminal=False)

    rich_module.render_failure(
        failure_from_backend(),
        label=" (Program alpha)",
        console=console,
    )

    rendered = output.getvalue()
    assert "Program Failure (Program alpha)" in rendered
    assert rendered.count("RuntimeError: backend exploded") == 2
    assert "Traceback follows" in rendered
    assert "failure_from_backend" in rendered


def test_render_failure_defaults_to_standard_error(capsys):
    rich_module.render_failure(failure_from_backend())

    assert "Program Failure" in capsys.readouterr().err


def test_ensemble_view_honours_state_row_visibility():
    console, output = captured_console(force_terminal=True)
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


@pytest.mark.parametrize(
    ("make_view", "view_cls"),
    [
        (make_standalone_view, Status),
        (lambda console: make_ensemble_view(console, is_jupyter=True), Live),
    ],
    ids=["standalone", "ensemble"],
)
def test_view_closures_are_idempotent(mocker, make_view, view_cls):
    stop = mocker.spy(view_cls, "stop")
    render, close = make_view(Console(file=StringIO(), force_terminal=True))
    render(registered_program_state(), {"p"})

    close()
    close()

    assert stop.call_count == 1


def test_ensemble_view_rows_carry_target_state(mocker):
    console, output = captured_console(force_terminal=True)
    make_progress = mocker.spy(rich_module, "_make_progress")
    state = registered_program_state()
    render, close = make_ensemble_view(console, is_jupyter=True)

    try:
        render(state, {"p"})
        render(
            state,
            state.apply(
                ProgressEvent.register(
                    "batch",
                    ProgressScope.BATCH,
                    "Batch",
                    None,
                    batch_color="cyan",
                    program_keys=("p",),
                )
            ),
        )
        render(state, state.apply(ProgressEvent.advance("p")))
        render(state, state.apply(ProgressEvent.finish("p", TerminalStatus.FAILED)))
    finally:
        close()

    task = next(
        task
        for task in make_progress.spy_return.tasks
        if task.fields["label"] == "Program"
    )
    assert task.completed == 1
    assert task.fields["scope"] is ProgressScope.PROGRAM
    assert task.fields["batch_color"] == "cyan"
    assert task.fields["terminal_status"] is TerminalStatus.FAILED
    rendered = output.getvalue()
    assert "Program" in rendered
    assert "■" in rendered
    assert "1/3" in rendered
    assert re.search(r"\d+:\d{2}:\d{2}", rendered)


@pytest.mark.parametrize("is_jupyter", [False, True])
def test_ensemble_view_auto_refreshes_only_outside_jupyter(mocker, is_jupyter):
    live = mocker.patch.object(rich_module, "Live", wraps=Live)

    make_ensemble_view(Console(file=StringIO()), is_jupyter=is_jupyter)

    assert live.call_args.kwargs["auto_refresh"] is (not is_jupyter)


def test_ensemble_view_adds_each_target_once(mocker):
    add_target = mocker.spy(rich_module, "_add_target")
    make_progress = mocker.spy(rich_module, "_make_progress")
    state = registered_program_state()
    render, close = make_ensemble_view(
        Console(file=StringIO(), force_terminal=True), is_jupyter=True
    )

    try:
        render(state, {"p"})
        render(state, {"p"})
    finally:
        close()

    assert add_target.call_count == 1
    assert len(make_progress.spy_return.tasks) == 1


def test_ensemble_view_stays_live_until_closed():
    console = Console(file=StringIO(), force_terminal=True)
    render, close = make_ensemble_view(console, is_jupyter=True)

    try:
        render(registered_program_state(), {"p"})
        assert console._live_stack
    finally:
        close()


def test_ensemble_view_close_during_render_stops_live_when_render_finishes(
    mocker, monkeypatch
):
    stop = mocker.spy(Live, "stop")
    state = registered_program_state()
    render, close = make_ensemble_view(
        Console(file=StringIO(), force_terminal=True), is_jupyter=True
    )
    render(state, {"p"})
    render_started, release_render = block_calls(monkeypatch, "_update_target")
    renderer = Thread(target=render, args=(state, {"p"}))
    renderer.start()
    assert render_started.wait(timeout=1)

    close()
    stops_before_release = stop.call_count
    release_render.set()
    renderer.join(timeout=2)

    assert stops_before_release == 0
    assert stop.call_count == 1


def test_ensemble_view_close_serializes_with_inflight_render(monkeypatch):
    console = Console(file=StringIO(), force_terminal=True)
    state = registered_program_state()
    close_finished = Event()
    render_started, release_render = block_calls(monkeypatch, "_add_target")
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


def test_standalone_view_renders_active_target_to_its_console():
    console, output = captured_console(force_terminal=True)
    render, close = make_standalone_view(console)

    try:
        render(registered_program_state(), {"p"})
    finally:
        close()

    assert "Program:" in output.getvalue()


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
