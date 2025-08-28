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
    def __init__(self, table_column=None):
        super().__init__(table_column)

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
            max_retries = task.fields.get("max_retries")
            service_job_id = task.fields.get("service_job_id").split("-")[0]
            output_str = f"[{self._last_message}] [Polling Job {service_job_id}: {poll_attempt} / {max_retries} Retries]"

            txt = Text(output_str)
            txt.stylize("blue", output_str.index("Job") + 4, output_str.index(":"))

            return txt

        return Text(f"[{self._last_message}]")


def make_progress_bar(is_jupyter: bool = False) -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.fields[job_name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        ConditionalSpinnerColumn(),
        PhaseStatusColumn(),
        # For jupyter notebooks, refresh manually instead
        auto_refresh=not is_jupyter,
        # Give a dummy positive value if is_jupyter
        refresh_per_second=10 if not is_jupyter else 999,
    )
