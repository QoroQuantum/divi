# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from queue import Empty, Queue
from threading import Event, Lock
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
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
        return self.spinner.render(_UnfinishedTaskWrapper(task))


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
) -> tuple[Progress, Live]:
    """Create a ``Live``-wrapped progress bar for per-program tracking.

    Returns:
        Tuple of (program_progress, live_display).
    """
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
) -> tuple[Progress, Progress, Live]:
    """Create a composed Live display with separate batch status and program progress bars.

    In Jupyter environments auto-refresh is disabled to avoid double-rendering
    issues (rich#1737). The caller is responsible for calling ``live.refresh()``
    after each update.

    Returns:
        Tuple of (batch_progress, program_progress, live_display).
    """
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
):
    """Drain a message queue and update progress bars accordingly.

    Runs in a daemon thread until *done_event* is set.  Messages with
    ``batch=True`` are routed to :func:`handle_batch_message`; all others
    update the per-program *progress_bar*.
    """
    while not done_event.is_set():
        try:
            msg: dict[str, Any] = queue.get(timeout=0.1)
        except Empty:
            continue
        except Exception as e:
            progress_bar.console.log(f"[queue_listener] Unexpected exception: {e}")
            continue

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
                live_display.refresh()
            queue.task_done()
            continue

        # --- Regular per-program messages ---
        with lock:
            task_id = pb_task_map[msg["job_id"]]

        # Prepare update arguments, starting with progress.
        update_args = {"advance": msg["progress"]}

        if "poll_attempt" in msg:
            update_args["poll_attempt"] = msg.get("poll_attempt", 0)
        if "max_retries" in msg:
            update_args["max_retries"] = msg.get("max_retries")
        if "service_job_id" in msg:
            update_args["service_job_id"] = msg.get("service_job_id")
        if "job_status" in msg:
            update_args["job_status"] = msg.get("job_status")
        if msg.get("message"):
            update_args["message"] = msg.get("message")
        if "final_status" in msg:
            update_args["final_status"] = msg.get("final_status", "")
        if "loss" in msg:
            update_args["loss"] = msg.get("loss")

        progress_bar.update(task_id, **update_args)

        if is_jupyter and live_display is not None:
            live_display.refresh()

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
