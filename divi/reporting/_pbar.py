# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


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
    def __init__(self, table_column=None):
        super().__init__(table_column)

    def render(self, task):
        final_status = task.fields.get("final_status")

        if final_status == "Success":
            return Text("• Success! ✅", style="bold green")
        elif final_status == "Failed":
            return Text("• Failed! ❌", style="bold red")
        elif final_status == "Cancelled":
            return Text("• Cancelled ⏹️", style="bold yellow")
        elif final_status == "Aborted":
            return Text("• Aborted ⚠️", style="dim magenta")

        message = task.fields.get("message")

        poll_attempt = task.fields.get("poll_attempt", 0)
        polling_str = ""
        service_job_id = task.fields.get("service_job_id")

        if service_job_id:
            split_job_id = service_job_id.split("-")[0]
            job_status = task.fields.get("job_status")

            if job_status == "COMPLETED":
                polling_str = f" [Job {split_job_id} is complete.]"
            elif poll_attempt > 0:
                max_retries = task.fields.get("max_retries")
                polling_str = f" [Job {split_job_id} is {job_status}. Polling attempt {poll_attempt} / {max_retries}]"

        final_text = Text(f"[{message}]{polling_str}")
        if service_job_id:
            final_text.highlight_words([split_job_id], "blue")

        return final_text


def make_progress_bar(is_jupyter: bool = False) -> Progress:
    """
    Create a customized Rich progress bar for tracking quantum program execution.

    Builds a progress bar with custom columns including job name, completion status,
    elapsed time, spinner, and phase status indicators. Automatically adapts refresh
    behavior for Jupyter notebook environments.

    Args:
        is_jupyter (bool, optional): Whether the progress bar is being displayed in
            a Jupyter notebook environment. Affects refresh behavior. Defaults to False.

    Returns:
        Progress: A configured Rich Progress instance with custom columns for
            quantum program tracking.
    """
    return Progress(
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
        # For jupyter notebooks, refresh manually instead
        auto_refresh=not is_jupyter,
        # Give a dummy positive value if is_jupyter
        refresh_per_second=10 if not is_jupyter else 999,
    )
