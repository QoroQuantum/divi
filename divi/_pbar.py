from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
)
from rich.text import Text


class ConditionalSpinnerColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.spinner = SpinnerColumn("point")

    def render(self, task):
        status = task.fields.get("final_status")

        if status in ("Success", "Failed"):
            return Text("")

        return self.spinner.render(task)


class PhaseStatusColumn(ProgressColumn):
    def __init__(self, max_retries: int, table_column=None):
        super().__init__(table_column)

        self._max_retries = max_retries
        self._curr_message = ""

    def render(self, task):
        final_status = task.fields.get("final_status")

        if final_status == "Success":
            return Text("• Success! ✅", style="bold green")
        elif final_status == "Failed":
            return Text("• Failed! ❌", style="bold red")

        phase = task.fields.get("message")
        if phase != "":
            self._curr_message = phase

        mode = task.fields.get("mode")
        if mode == "simulation":
            return Text(f"[{self._curr_message}]")

        poll_attempt = task.fields.get("poll_attempt")
        if poll_attempt > 0:
            return Text(f"[{phase}] Polling {poll_attempt}/{self._max_retries}")

        return Text(f"[{phase}]")


def make_progress_bar(max_retries: Optional[int] = None):
    return Progress(
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(max_retries=max_retries),
    )
