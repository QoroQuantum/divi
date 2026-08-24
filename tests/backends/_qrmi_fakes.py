# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for the ``qrmi`` package, so the backend's tests need no wheel.

The real API is duck-typed rather than subclassed: ``qrmi`` ships a compiled
pyo3 extension whose classes cannot be constructed from Python.
"""

import json
from enum import Enum
from types import SimpleNamespace


class FakeTaskStatus(Enum):
    Queued = "Queued"
    Running = "Running"
    Completed = "Completed"
    Failed = "Failed"
    Cancelled = "Cancelled"


class FakePayload:
    """Mirrors ``qrmi.Payload``; the variant is a constructor, not a class."""

    @staticmethod
    def MaestroLocal(**fields):
        return SimpleNamespace(**fields)


FAKE_QRMI = SimpleNamespace(Payload=FakePayload, TaskStatus=FakeTaskStatus)


class FakeQuantumResource:
    """A ``qrmi.QuantumResource`` for a ``maestro-local`` resource.

    Records every payload it is handed and replays canned results, so tests
    assert on what went over the wire.
    """

    def __init__(
        self,
        *,
        resource_id: str = "MAESTRO_LOCAL",
        counts: dict[str, int] | None = None,
        expectation_values: list[float] | None = None,
        statuses: list[FakeTaskStatus] | None = None,
    ):
        self._resource_id = resource_id
        self._counts = counts if counts is not None else {"00": 4, "11": 6}
        self._expectation_values = expectation_values
        self._statuses = list(statuses) if statuses else None

        self.payloads: list[SimpleNamespace] = []
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.stopped: list[str] = []
        self._next_session = 0

    # --- qrmi.QuantumResource surface ---

    def resource_id(self) -> str:
        return self._resource_id

    def acquire(self) -> str:
        self._next_session += 1
        session = str(self._next_session)
        self.acquired.append(session)
        return session

    def release(self, session: str) -> None:
        self.released.append(session)

    def task_start(self, payload) -> str:
        self.payloads.append(payload)
        return f"task-{len(self.payloads) - 1}"

    def task_status(self, task_id: str):
        if self._statuses:
            return self._statuses.pop(0)
        return FakeTaskStatus.Completed

    def task_result(self, task_id: str):
        index = int(task_id.rsplit("-", 1)[1])
        request = json.loads(self.payloads[index].input)
        if request["operation"] == "estimate":
            values = self._expectation_values
            if values is None:
                values = [1.0] * len(request["observables"])
            body = {"expectation_values": values}
        else:
            body = {"counts": dict(self._counts)}
        body |= {"ok": True, "time_taken": 0.001, "seed": 7}
        return SimpleNamespace(value=json.dumps(body))

    def task_logs(self, task_id: str) -> str:
        return f"logs for {task_id}"

    def task_stop(self, task_id: str) -> None:
        self.stopped.append(task_id)

    # --- helpers for assertions ---

    @property
    def requests(self) -> list[dict]:
        return [json.loads(payload.input) for payload in self.payloads]

    @property
    def last_request(self) -> dict:
        return self.requests[-1]
