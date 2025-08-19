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
        self._last_message = ""

    def render(self, task):
        final_status = task.fields.get("final_status")

        if final_status == "Success":
            return Text("• Success! ✅", style="bold green")
        elif final_status == "Failed":
            return Text("• Failed! ❌", style="bold red")

        message = task.fields.get("message")
        if message != "":
            self._last_message = message

        poll_attempt = task.fields.get("poll_attempt")
        if poll_attempt > 0:
            return Text(
                f"[{self._last_message}] Polling {poll_attempt}/{self._max_retries}"
            )

        return Text(f"[{self._last_message}]")


def make_progress_bar(
    max_retries: int | None = None, is_jupyter: bool = False
) -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(max_retries=max_retries),
        # For jupyter notebooks, refresh manually instead
        auto_refresh=not is_jupyter,
        # Give a dummy positive value if is_jupyter
        refresh_per_second=10 if not is_jupyter else 999,
    )
