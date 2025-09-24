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

    def render(self, task):
        final_status = task.fields.get("final_status")

        if final_status == "Success":
            return Text("• Success! ✅", style="bold green")
        elif final_status == "Failed":
            return Text("• Failed! ❌", style="bold red")

        message = task.fields.get("message")

        poll_attempt = task.fields.get("poll_attempt")
        polling_str = ""
        service_job_id = ""
        if poll_attempt > 0:
            max_retries = task.fields.get("max_retries")
            service_job_id = task.fields.get("service_job_id").split("-")[0]
            job_status = task.fields.get("job_status")
            polling_str = f" [Job {service_job_id} is {job_status}. Polling attempt {poll_attempt} / {max_retries}]"

        final_text = Text(f"[{message}]{polling_str}")
        final_text.highlight_words([service_job_id], "blue")

        return final_text


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
