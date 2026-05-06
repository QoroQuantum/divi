# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, cast

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


class TerminalStatus(str, Enum):
    """Terminal status for a progress row; members equal their string value."""

    SUCCESS = "Success"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    ABORTED = "Aborted"


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


class _ProgramOnlyColumn(ProgressColumn):
    """Wrap a stock column so it renders only on rows tagged
    ``row_kind="program"``; batch-row cells fall back to empty text."""

    def __init__(self, inner: ProgressColumn):
        super().__init__()
        self._inner = inner

    def render(self, task):
        if task.fields.get("row_kind") == "program":
            return self._inner.render(task)
        return Text("")


class ConditionalSpinnerColumn(ProgressColumn):
    _FINAL_STATUSES = frozenset(TerminalStatus)

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
        TerminalStatus.SUCCESS: ("• Success! ✅ ", "bold green"),
        TerminalStatus.FAILED: ("• Failed! ❌ ", "bold red"),
        TerminalStatus.CANCELLED: ("• Cancelled ⏹️ ", "bold yellow"),
        TerminalStatus.ABORTED: ("• Aborted ⚠️ ", "dim magenta"),
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
    """Create the unified Rich Progress bar.

    Per-program rows render the full bar/M-of-N/elapsed columns; batch
    rows render only the indicator/text/spinner/status columns thanks
    to the :class:`_ProgramOnlyColumn` wrappers.  Tasks distinguish
    themselves via the ``row_kind`` field.
    """
    return Progress(
        BatchIndicatorColumn(),
        TextColumn("[bold blue]{task.fields[job_name]}"),
        _ProgramOnlyColumn(BarColumn()),
        _ProgramOnlyColumn(MofNCompleteColumn()),
        _ProgramOnlyColumn(TimeElapsedColumn()),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
    )


def make_progress_display(
    is_jupyter: bool = False,
) -> tuple[Progress | None, Live | None]:
    """Create a ``Live``-wrapped progress bar covering both per-program
    and batch rows.

    In Jupyter, ``auto_refresh`` is disabled to avoid double-rendering
    (rich#1737); the caller is responsible for ``live.refresh()`` after
    each update in that mode.

    Returns ``(None, None)`` when :func:`progress_disabled` is true.
    """
    if progress_disabled():
        return None, None

    _ensure_unbuffered_stdout()
    progress_bar = make_progress_bar()
    live = Live(
        progress_bar,
        auto_refresh=not is_jupyter,
        refresh_per_second=10,
    )
    return progress_bar, live


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


def _drain_queue_quietly(queue: Queue) -> None:
    """Drain remaining messages and ``task_done()`` each so callers
    blocked on ``queue.join()`` aren't held hostage by a dead listener."""
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break
        except Exception:
            break
        try:
            queue.task_done()
        except Exception:
            pass


def queue_listener(
    queue: Queue,
    progress_bar: Progress,
    pb_task_map: dict[Any, TaskID],
    done_event: Event,
    lock: Lock,
    *,
    hide_program_rows: bool = False,
    prep_task_id: TaskID | None = None,
):
    """Drain a message queue and update the unified progress bar.

    Runs in a daemon thread until *done_event* is set.  Messages with
    ``batch=True`` are routed to :func:`handle_batch_message`; messages
    with ``prep_advance=True`` advance the prep row; all others are
    program-level updates resolved through ``pb_task_map``.

    When ``hide_program_rows`` is set, per-program rows were created
    invisible by the ensemble — the listener reveals them only on a
    non-Success terminal status so failures stay diagnosable.

    The body is fully guarded: per-message exceptions are caught by the
    inner try; anything that escapes (including a thread-construction
    error before this body even runs) is caught by the outer
    ``BaseException`` handler in the spawned thread, which drains the
    queue so ``queue.join()`` callers never hang on a dead listener.

    Each per-message body is wrapped so a malformed message (e.g. unknown
    ``job_id``, Rich raising during teardown) cannot starve ``queue.join()``
    or kill the listener thread.  Exceptions are logged and the queue
    advances to the next message.
    """
    console = progress_bar.console
    # Batch rows live in the same Progress as program rows now; their
    # TaskIDs are tracked locally so split flush groups (expval vs
    # shots) each get their own row.
    batch_task_ids: dict[int, TaskID] = {}

    while not done_event.is_set():
        # Outer-loop guard: this listener is the sole writer of progress
        # updates after the queue-routing refactor.  A dead listener
        # silently freezes the display and hangs ``ensemble.join()``'s
        # drain wait, so any escaping ``Exception`` is logged and the
        # loop continues — the inner per-message try is the primary
        # defence; this is belt-and-suspenders.
        try:
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
                    handle_batch_message(
                        msg,
                        progress_bar,
                        batch_task_ids,
                        lock,
                    )
                    continue

                # --- Prep-progress signals from the coordinator ---
                if msg.get("prep_advance"):
                    if prep_task_id is None:
                        continue
                    prep_update: dict[str, Any] = {"advance": 1}
                    prep_task = progress_bar._tasks.get(prep_task_id)
                    if (
                        prep_task is not None
                        and prep_task.total is not None
                        and prep_task.completed + 1 >= prep_task.total
                    ):
                        # Last program reached submit — barrier is about
                        # to fire.  Mark the prep row as final so the
                        # display reads "Success ✅" instead of leaving
                        # an active spinner.
                        prep_update["final_status"] = TerminalStatus.SUCCESS
                        prep_update.pop("advance", None)
                        prep_update["completed"] = prep_task.total
                    progress_bar.update(prep_task_id, **prep_update)
                    continue

                # --- Regular per-program messages ---
                with lock:
                    task_id = pb_task_map.get(msg["job_id"])
                if task_id is None:
                    # Stale or unknown job_id (e.g. a late progress message
                    # arriving after the program's task was torn down).
                    # Drop it rather than letting a KeyError kill the
                    # listener.
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
                    final_status = msg["final_status"]
                    update_args["final_status"] = final_status
                    if final_status == TerminalStatus.SUCCESS:
                        # Fill the bar so a successful program isn't
                        # displayed as 0/N when it didn't tick the
                        # counter incrementally.
                        task = progress_bar._tasks.get(task_id)
                        if task is not None and task.total is not None:
                            update_args["completed"] = task.total
                            update_args.pop("advance", None)
                    elif hide_program_rows and final_status in (
                        TerminalStatus.FAILED,
                        TerminalStatus.CANCELLED,
                        TerminalStatus.ABORTED,
                    ):
                        # Per-program rows were created hidden by the
                        # ensemble; reveal this one so the user can see
                        # what went wrong.
                        update_args["visible"] = True

                try:
                    progress_bar.update(task_id, **update_args)
                except Exception as e:
                    _safe_log(
                        console,
                        f"[queue_listener] progress_bar.update failed: {e}",
                    )

            except Exception as e:
                # Per-message safety net: any unexpected exception in the
                # processing body is logged and swallowed.  Without this,
                # queue.join() in ProgramEnsemble would block forever
                # waiting for the task_done() that the dead listener
                # thread never makes.
                _safe_log(
                    console,
                    f"[queue_listener] unexpected exception while handling message: {e}",
                )
            finally:
                queue.task_done()
        except Exception as e:
            _safe_log(
                console,
                f"[queue_listener] outer-loop exception (continuing): {e}",
            )


def handle_batch_message(
    msg: dict[str, Any],
    progress_bar: Progress,
    batch_task_ids: dict[int, TaskID],
    lock: Lock,
):
    """Process a batch-level progress message in the unified progress bar.

    Batch rows are created dynamically per ``batch_id`` so that split
    sub-batches (e.g. expval vs shots) each get their own status line.
    The conditional column wrappers keep the bar/M-of-N/elapsed cells
    empty for batch rows; the indicator/text/spinner/status columns
    render normally.

    Program-row coloring works by reading each task's ``program_key``
    field — no parallel ``program_key → TaskID`` index needed.  Reading
    ``progress_bar._tasks`` mirrors the same pattern used elsewhere in
    the listener.
    """
    batch_id = msg.get("batch_id")
    if not isinstance(batch_id, int):
        return
    color = msg.get("batch_color", "")
    label = msg.get("batch_label", "")
    n_circuits = msg.get("n_circuits", 0)
    n_programs = msg.get("n_programs", 0)
    final_status = msg.get("final_status")

    # Lazily create a batch row for this batch_id.
    if batch_id not in batch_task_ids:
        batch_task_ids[batch_id] = progress_bar.add_task(
            "",
            job_name="",
            total=0,
            visible=False,
            row_kind="batch",
            program_key=None,
            batch_color="",
            message="",
            final_status=None,
        )
    task_id = batch_task_ids[batch_id]

    # Build update args for the batch row
    update_args: dict[str, Any] = {}

    if not final_status:
        update_args["visible"] = True
        prefix = f"Batch ({label})" if label else "Batch"
        update_args["job_name"] = (
            f"{prefix}: {n_circuits} circuits, {n_programs} programs"
        )
        update_args["batch_color"] = color
        _apply_color_to_program_rows(
            progress_bar, msg.get("program_keys", ()), color, lock
        )

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
        update_args["visible"] = False
        _apply_color_to_program_rows(
            progress_bar, msg.get("program_keys", ()), "", lock
        )
        del batch_task_ids[batch_id]

    progress_bar.update(task_id, **update_args)


def _apply_color_to_program_rows(
    progress_bar: Progress,
    program_keys,
    color: str,
    lock: Lock,
) -> None:
    """Set ``batch_color`` on every program row whose ``program_key``
    field appears in *program_keys*.  Pass ``color=""`` to clear."""
    if not program_keys:
        return
    keys_set = set(program_keys)
    # Snapshot under the lock so a concurrent ``add_task`` can't resize
    # ``_tasks`` mid-iteration.  ``progress_bar.update`` does not need
    # the lock — it only mutates per-task fields, not the dict.
    with lock:
        snapshot = list(progress_bar._tasks.items())
    for tid, task in snapshot:
        if task.fields.get("program_key") in keys_set:
            progress_bar.update(tid, batch_color=color)
