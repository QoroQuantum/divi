# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Rich views that render already-reduced progress state."""

import logging
from collections.abc import Callable, Hashable
from threading import Lock
from typing import TypeAlias

from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.status import Status
from rich.text import Text
from rich.traceback import Traceback

from divi.backends._job_status import JobStatus

from ._events import ProgressScope, TerminalStatus
from ._state import ProgressState, _TargetState

RenderFn: TypeAlias = Callable[[ProgressState, set[Hashable]], None]
CloseViewFn: TypeAlias = Callable[[], None]

logger = logging.getLogger("divi")


def console_supports_live_rendering(
    console: Console, *, is_jupyter: bool = False
) -> bool:
    """Return whether a console can safely host a transient Rich view."""
    return console.is_terminal or is_jupyter or console.is_jupyter


def render_failure(
    exc: BaseException,
    *,
    label: str = "",
    console: Console | None = None,
) -> None:
    """Render one execution failure and its traceback."""
    resolved_console = console if console is not None else Console(stderr=True)
    resolved_console.print(
        Panel(
            f"[bold]{type(exc).__name__}[/bold]: {exc}",
            title=f"[bold red]Program Failure{label}[/bold red]",
            subtitle="[dim]Traceback follows[/dim]",
            border_style="red",
        )
    )
    resolved_console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))


class BatchIndicatorColumn(ProgressColumn):
    """Render a coloured square for programs associated with a batch."""

    def render(self, task: Task) -> RenderableType:
        color = task.fields.get("batch_color", "")
        if color:
            return Text("■ ", style=color)
        return Text("  ")


class _ProgressBearingColumn(ProgressColumn):
    """Render a stock progress column for rows that track completed work."""

    def __init__(self, inner: ProgressColumn) -> None:
        super().__init__()
        self._inner = inner

    def render(self, task: Task) -> RenderableType:
        scope = task.fields.get("scope")
        if scope is ProgressScope.PROGRAM or scope is ProgressScope.PREPARATION:
            return self._inner.render(task)
        return Text("")


class ConditionalSpinnerColumn(ProgressColumn):
    """Render the spinner only while the state target is non-terminal."""

    def __init__(self) -> None:
        super().__init__()
        self._spinner = Spinner("point")

    def render(self, task: Task) -> RenderableType:
        if task.fields.get("terminal_status") is not None:
            return Text("")
        return self._spinner


class PhaseStatusColumn(ProgressColumn):
    """Render status text prepared directly from the target state."""

    def render(self, task: Task) -> RenderableType:
        return task.fields["status_text"]


def render_status_text(target: _TargetState) -> Text:
    """Return Rich text for one target's current rendered status."""
    loss_text = _render_loss_text(target.loss)
    terminal = target.terminal_status
    if terminal is not None:
        message, style = _terminal_message(terminal)
        detail = target.detail
        suffix = f" ({detail})" if detail else ""
        service_status = (
            f" [Job status: {target.job_status}]"
            if target.job_status not in {None, JobStatus.COMPLETED}
            else ""
        )
        return Text(f"{message}{suffix}{loss_text}{service_status}", style=style)

    message = f"[{target.message}]" if target.message else ""
    polling = _render_polling_text(target)
    text = Text(f"{message}{loss_text}{polling}")
    if target.service_job_id is not None:
        text.highlight_words([_short_job_id(target.service_job_id)], "blue")
    return text


def _terminal_message(status: TerminalStatus) -> tuple[str, str]:
    messages = {
        TerminalStatus.SUCCESS: ("• Success! ✅ ", "bold green"),
        TerminalStatus.FAILED: ("• Failed! ❌ ", "bold red"),
        TerminalStatus.CANCELLED: ("• Cancelled ⏹️ ", "bold yellow"),
        TerminalStatus.ABORTED: ("• Aborted ⚠️ ", "dim magenta"),
    }
    return messages[status]


def _render_loss_text(loss: float | None) -> str:
    if loss is None:
        return ""
    return f" [loss: {float(loss):.6f}]"


def _render_polling_text(target: _TargetState) -> str:
    if target.service_job_id is None or target.job_status is None:
        return ""
    job_id = _short_job_id(target.service_job_id)
    if target.job_status is JobStatus.COMPLETED:
        return f" [Job {job_id} is complete.]"
    if target.poll_attempt is None or target.poll_attempt == 0:
        return ""
    limit = "∞" if target.max_retries is None else target.max_retries
    return (
        f" [Job {job_id} is {target.job_status}. Polling attempt "
        f"{target.poll_attempt} / {limit}]"
    )


def _short_job_id(service_job_id: str) -> str:
    return service_job_id.split("-", maxsplit=1)[0]


def _add_target(progress: Progress, target: _TargetState) -> TaskID:
    task_id = progress.add_task("", total=target.total)
    _update_target(progress, task_id, target)
    return task_id


def _update_target(progress: Progress, task_id: TaskID, target: _TargetState) -> None:
    progress.update(
        task_id,
        total=target.total,
        completed=target.completed,
        visible=target.visible,
        scope=target.scope,
        label=target.label,
        batch_color=target.batch_color,
        terminal_status=target.terminal_status,
        status_text=render_status_text(target),
    )


def _make_progress(console: Console) -> Progress:
    return Progress(
        BatchIndicatorColumn(),
        TextColumn("[bold blue]{task.fields[label]}"),
        _ProgressBearingColumn(BarColumn()),
        _ProgressBearingColumn(MofNCompleteColumn()),
        _ProgressBearingColumn(TimeElapsedColumn()),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
        console=console,
        auto_refresh=False,
    )


def make_standalone_view(console: Console) -> tuple[RenderFn, CloseViewFn]:
    """Create a transient status view for directly rendered progress state."""
    status = Status("", console=console, spinner="point")
    started = False

    def render(state: ProgressState, affected: set[Hashable]) -> None:
        nonlocal started
        for target_id in affected:
            target = state.get(target_id)
            text = Text.assemble(f"{target.label}: ", render_status_text(target))
            if not started:
                status.start()
                started = True
            status.update(text)
            if target.terminal_status is not None:
                status.stop()
                started = False
                logger.info("%s", str(text).strip())

    def close() -> None:
        nonlocal started
        if started:
            status.stop()
            started = False

    return render, close


def make_ensemble_view(
    console: Console, is_jupyter: bool
) -> tuple[RenderFn, CloseViewFn]:
    """Create a single-writer live view for queued ensemble state changes."""
    progress = _make_progress(console)
    live = Live(
        progress,
        console=console,
        auto_refresh=not is_jupyter,
        refresh_per_second=10,
    )
    task_ids: dict[Hashable, TaskID] = {}
    started = False
    closed = False
    active_renders = 0
    view_lock = Lock()

    def render(state: ProgressState, affected: set[Hashable]) -> None:
        nonlocal active_renders, started
        with view_lock:
            if closed:
                return
            active_renders += 1
        try:
            for target_id in affected:
                target = state.get(target_id)
                task_id = task_ids.get(target_id)
                if task_id is None:
                    task_ids[target_id] = _add_target(progress, target)
                else:
                    _update_target(progress, task_id, target)
            with view_lock:
                if closed:
                    return
                if not started:
                    live.start()
                    started = True
            if is_jupyter:
                live.refresh()
        finally:
            with view_lock:
                active_renders -= 1
                if closed and active_renders == 0 and started:
                    live.stop()
                    started = False

    def close() -> None:
        nonlocal closed, started
        with view_lock:
            if closed:
                return
            closed = True
            if active_renders == 0 and started:
                live.stop()
                started = False

    return render, close


__all__ = [
    "CloseViewFn",
    "RenderFn",
    "console_supports_live_rendering",
    "make_ensemble_view",
    "make_standalone_view",
    "render_failure",
]
