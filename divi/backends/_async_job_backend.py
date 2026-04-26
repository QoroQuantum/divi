# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for backends that execute circuits as async jobs.

Implemented by backends like :class:`~divi.backends.QoroService` that submit
work to a remote scheduler and return a job handle, then expose polling,
result-fetching, and cancellation against that handle.
"""

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import requests

from divi.backends._execution_result import ExecutionResult


@runtime_checkable
class AsyncJobBackend(Protocol):
    """A backend whose ``submit_circuits`` returns a job handle to poll."""

    @property
    def shots(self) -> int: ...

    def submit_circuits(
        self, circuits: Mapping[str, str], **kwargs
    ) -> ExecutionResult: ...

    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        on_complete: Callable[[requests.Response], None] | None = None,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
    ): ...

    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult: ...

    def cancel_job(self, execution_result: ExecutionResult) -> requests.Response: ...
