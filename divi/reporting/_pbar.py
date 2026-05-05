# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, cast

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from divi.reporting._qlogger import _ensure_unbuffered_stdout

#: Color cycle assigned to flush groups in ensemble progress displays.
#:
#: Each running flush group is tinted with the next color in this tuple so
#: that progress rows can be visually associated with their participating
#: programs.
BATCH_COLORS = ("green", "cyan", "magenta", "yellow", "red", "blue")

_PROGRESS_DISABLE_ENV = "DIVI_DISABLE_PROGRESS"
_PROGRESS_DISABLE_TRUTHY = frozenset({"1", "true", "yes", "on"})


def progress_disabled() -> bool:
    """Return True if ``DIVI_DISABLE_PROGRESS`` is set to a truthy value
    (``1``, ``true``, ``yes``, ``on``; case-insensitive)."""
    return (
        os.getenv(_PROGRESS_DISABLE_ENV, "").strip().lower() in _PROGRESS_DISABLE_TRUTHY
    )


class BatchIndicatorColumn(ProgressColumn):
    """Renders a colored square prefix to associate programs with their batch."""

    def render(self, task):
        color = task.fields.get("batch_color", "")
        if color:
            return Text("■ ", style=color)
        return Text("  ")


class _UnfinishedTaskWrapper:
    """Wrapper that forces a task to appear unfinished for spinner animation."""

    def __init__(self, task):
        self._task = task

    def __getattr__(self, name):
        if name == "finished":
            return False
        return getattr(self._task, name)


class ConditionalSpinnerColumn(ProgressColumn):
    _FINAL_STATUSES = ("Success", "Failed", "Cancelled", "Aborted")

    def __init__(self):
        super().__init__()
        self.spinner = SpinnerColumn("point")

    def render(self, task):
        status = task.fields.get("final_status")

        if status in self._FINAL_STATUSES:
            return Text("")

        # Force the task to appear unfinished for spinner animation
        return self.spinner.render(cast(Task, _UnfinishedTaskWrapper(task)))


class PhaseStatusColumn(ProgressColumn):
    _STATUS_MESSAGES = {
        "Success": ("• Success! ✅ ", "bold green"),
        "Failed": ("• Failed! ❌ ", "bold red"),
        "Cancelled": ("• Cancelled ⏹️ ", "bold yellow"),
        "Aborted": ("• Aborted ⚠️ ", "dim magenta"),
    }

    def __init__(self, table_column=None):
        super().__init__(table_column)

    def _build_polling_string(
        self, split_job_id: str, job_status: str, poll_attempt: int, max_retries: int
    ) -> str:
        """Build the polling status string for service job tracking."""
        if job_status == "COMPLETED":
            return f" [Job {split_job_id} is complete.]"
        elif poll_attempt > 0:
            return f" [Job {split_job_id} is {job_status}. Polling attempt {poll_attempt} / {max_retries}]"

        return ""

    @staticmethod
    def _build_loss_string(loss: float | None) -> str:
        """Build a compact loss display when a numeric loss is present."""
        if loss is None:
            return ""
        return f" [loss: {float(loss):.6f}]"

    def render(self, task):
        final_status = task.fields.get("final_status")
        loss = task.fields.get("loss")
        loss_str = self._build_loss_string(loss)

        # Early return for final statuses
        if final_status in self._STATUS_MESSAGES:
            status_text, style = self._STATUS_MESSAGES[final_status]
            detail = task.fields.get("message", "")
            suffix = f" ({detail})" if detail else ""
            return Text(f"{status_text}{suffix}{loss_str}", style=style)

        # Build message with polling information
        message = task.fields.get("message")
        service_job_id = task.fields.get("service_job_id")
        job_status = task.fields.get("job_status")
        poll_attempt = task.fields.get("poll_attempt", 0)
        max_retries = task.fields.get("max_retries")

        polling_str = ""
        split_job_id = None
        if service_job_id is not None:
            split_job_id = service_job_id.split("-")[0]
            polling_str = self._build_polling_string(
                split_job_id, job_status, poll_attempt, max_retries
            )
        msg_str = f"[{message}]" if message else ""
        final_text = Text(f"{msg_str}{loss_str}{polling_str}")

        # Highlight job ID if present
        if split_job_id is not None:
            final_text.highlight_words([split_job_id], "blue")

        return final_text


def make_progress_bar() -> Progress:
    """Create a Rich Progress bar for per-program tracking.

    Auto-refresh is disabled; the enclosing ``Live`` display handles refresh.
    """
    return Progress(
        BatchIndicatorColumn(),
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
        auto_refresh=False,
    )


def _make_batch_status_bar() -> Progress:
    """Create a lightweight Progress bar for batch status lines (no bar/M-of-N/time)."""
    return Progress(
        BatchIndicatorColumn(),
        TextColumn("[bold blue]{task.fields[job_name]}"),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
        auto_refresh=False,
    )


def make_progress_display(
    is_jupyter: bool = False,
) -> tuple[Progress | None, Live | None]:
    """Create a ``Live``-wrapped progress bar for per-program tracking.

    Returns ``(None, None)`` when :func:`progress_disabled` is true.
    """
    if progress_disabled():
        return None, None

    _ensure_unbuffered_stdout()
    program_progress = make_progress_bar()
    live = Live(
        program_progress,
        auto_refresh=not is_jupyter,
        refresh_per_second=10,
    )
    return program_progress, live


def make_batch_display(
    is_jupyter: bool = False,
) -> tuple[Progress | None, Progress | None, Live | None]:
    """Create a composed Live display with separate batch status and program progress bars.

    In Jupyter environments auto-refresh is disabled to avoid double-rendering
    issues (rich#1737). The caller is responsible for calling ``live.refresh()``
    after each update.

    Returns ``(None, None, None)`` when :func:`progress_disabled` is true.
    """
    if progress_disabled():
        return None, None, None

    _ensure_unbuffered_stdout()
    batch_progress = _make_batch_status_bar()
    program_progress = make_progress_bar()
    group = Group(program_progress, batch_progress)
    live = Live(
        group,
        auto_refresh=not is_jupyter,
        refresh_per_second=10,
    )
    return batch_progress, program_progress, live


# ---------------------------------------------------------------------------
# Queue listener & batch message handler
# ---------------------------------------------------------------------------


def _safe_log(console, msg: str) -> None:
    """Best-effort console log — swallow any errors from Rich (especially
    during interpreter / live-display teardown) so the listener thread
    never dies on its own diagnostics."""
    try:
        console.log(msg)
    except Exception:
        pass


def _safe_call(fn, /, *args, **kwargs) -> None:
    """Run *fn(*args, **kwargs)* and swallow any exception; same intent
    as :func:`_safe_log` for non-logging Rich calls (e.g. ``live.refresh``)."""
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


def queue_listener(
    queue: Queue,
    progress_bar: Progress,
    pb_task_map: dict[Any, TaskID],
    done_event: Event,
    lock: Lock,
    *,
    batch_progress: Progress | None = None,
    batch_task_ids: dict[int, TaskID] | None = None,
    program_key_to_task_ids: dict[str, list[TaskID]] | None = None,
    live_display: Live | None = None,
    is_jupyter: bool = False,
    hide_on_success: bool = False,
):
    """Drain a message queue and update progress bars accordingly.

    Runs in a daemon thread until *done_event* is set.  Messages with
    ``batch=True`` are routed to :func:`handle_batch_message`; all others
    update the per-program *progress_bar*.

    Each per-message body is wrapped so a malformed message (e.g. unknown
    ``job_id``, Rich raising during teardown) cannot starve ``queue.join()``
    or kill the listener thread.  Exceptions are logged and the queue
    advances to the next message.
    """
    console = progress_bar.console

    while not done_event.is_set():
        try:
            msg: dict[str, Any] = queue.get(timeout=0.1)
        except Empty:
            continue
        except Exception as e:
            _safe_log(console, f"[queue_listener] queue.get failed: {e}")
            continue

        try:
            # --- Batch-level messages from the coordinator ---
            if msg.get("batch"):
                if batch_progress is not None and batch_task_ids is not None:
                    handle_batch_message(
                        msg,
                        batch_progress,
                        batch_task_ids,
                        progress_bar,
                        program_key_to_task_ids or {},
                        lock,
                    )
                if is_jupyter and live_display is not None:
                    _safe_call(live_display.refresh)
                continue

            # --- Regular per-program messages ---
            with lock:
                task_id = pb_task_map.get(msg["job_id"])
            if task_id is None:
                # Stale or unknown job_id (e.g. a late progress message
                # arriving after the program's task was torn down).  Drop
                # it rather than letting a KeyError kill the listener.
                _safe_log(
                    console,
                    f"[queue_listener] dropped message for unknown job_id "
                    f"{msg.get('job_id')!r}",
                )
                continue

            update_args: dict[str, Any] = {"advance": msg["progress"]}
            for key in (
                "poll_attempt",
                "max_retries",
                "service_job_id",
                "job_status",
                "loss",
            ):
                if key in msg:
                    update_args[key] = msg[key]
            if msg.get("message"):
                update_args["message"] = msg["message"]
            if "final_status" in msg:
                update_args["final_status"] = msg.get("final_status", "")
                # Fill the bar so a successful program isn't displayed
                # as 0/N when it didn't tick the counter incrementally.
                if msg["final_status"] == "Success":
                    task = progress_bar._tasks.get(task_id)
                    if task is not None and task.total is not None:
                        update_args["completed"] = task.total
                        update_args.pop("advance", None)
                    # In large ensembles, hide successful rows so the
                    # batch row remains the focal signal.  Failed,
                    # cancelled, and aborted rows stay visible so users
                    # can diagnose what went wrong.
                    if hide_on_success:
                        update_args["visible"] = False

            try:
                progress_bar.update(task_id, **update_args)
            except Exception as e:
                _safe_log(console, f"[queue_listener] progress_bar.update failed: {e}")

            if is_jupyter and live_display is not None:
                _safe_call(live_display.refresh)

        except Exception as e:
            # Final safety net: any unexpected exception in the per-message
            # body is logged and swallowed.  Without this, queue.join() in
            # ProgramEnsemble would block forever waiting for the
            # task_done() that the dead listener thread never makes.
            _safe_log(
                console,
                f"[queue_listener] unexpected exception while handling message: {e}",
            )
        finally:
            queue.task_done()


def handle_batch_message(
    msg: dict[str, Any],
    batch_progress: Progress,
    batch_task_ids: dict[int, TaskID],
    program_progress: Progress,
    program_key_to_task_ids: dict[str, list[TaskID]],
    lock: Lock,
):
    """Process a batch-level progress message using the separate batch Progress bar.

    Batch rows are created dynamically per ``batch_id`` so that split
    sub-batches (e.g. expval vs shots) each get their own status line.
    """
    batch_id = msg.get("batch_id")
    if not isinstance(batch_id, int):
        return
    color = msg.get("batch_color", "")
    label = msg.get("batch_label", "")
    n_circuits = msg.get("n_circuits", 0)
    n_programs = msg.get("n_programs", 0)
    final_status = msg.get("final_status")

    # Lazily create a batch row for this batch_id
    if batch_id not in batch_task_ids:
        batch_task_ids[batch_id] = batch_progress.add_task(
            "",
            job_name="",
            total=0,
            visible=False,
            batch_color="",
            message="",
            final_status="",
        )
    task_id = batch_task_ids[batch_id]

    # Build update args for the batch row
    update_args: dict[str, Any] = {}

    if not final_status:
        # Show the batch row and update its label/color
        update_args["visible"] = True
        prefix = f"Batch ({label})" if label else "Batch"
        update_args["job_name"] = (
            f"{prefix}: {n_circuits} circuits, {n_programs} programs"
        )
        update_args["batch_color"] = color

        # Color-code participating programs
        with lock:
            for prog_key in msg.get("program_keys", []):
                for tid in program_key_to_task_ids.get(prog_key, []):
                    program_progress.update(tid, batch_color=color)

    if "poll_attempt" in msg:
        update_args["poll_attempt"] = msg["poll_attempt"]
    if "max_retries" in msg:
        update_args["max_retries"] = msg["max_retries"]
    if "service_job_id" in msg:
        update_args["service_job_id"] = msg["service_job_id"]
    if "job_status" in msg:
        update_args["job_status"] = msg["job_status"]
    if msg.get("message"):
        update_args["message"] = msg["message"]

    if final_status:
        # Hide the batch row and clear program colors
        update_args["visible"] = False
        with lock:
            for prog_key in msg.get("program_keys", []):
                for tid in program_key_to_task_ids.get(prog_key, []):
                    program_progress.update(tid, batch_color="")
        # Clean up the mapping
        del batch_task_ids[batch_id]

    batch_progress.update(task_id, **update_args)
