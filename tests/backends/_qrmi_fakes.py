# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for the ``qrmi`` package, so the backend's tests need no wheel.

The real API is duck-typed rather than subclassed: ``qrmi`` ships a compiled
pyo3 extension whose classes cannot be constructed from Python.
"""

import json
from enum import Enum


class FakeTaskStatus(Enum):
    Queued = "Queued"
    Running = "Running"
    Completed = "Completed"
    Failed = "Failed"
    Cancelled = "Cancelled"


class FakeConfigError(RuntimeError):
    """``qrmi.ConfigError``: no task exists after it."""


class FakeInvalidInputError(RuntimeError):
    """``qrmi.InvalidInputError``."""


class FakeUnsupportedFunctionError(RuntimeError):
    """``qrmi.UnsupportedFunctionError``."""


def fake_qrmi_module(mocker):
    """The ``qrmi`` module, its payload helper checking a native request as
    ``qrmi.maestro.request_payload`` does."""

    def request_payload(request):
        if request.get("schema_version") != 2:
            raise ValueError("Maestro request must be an object with schema_version=2")
        return mocker.Mock(
            input=json.dumps(request, allow_nan=False), job_type="request"
        )

    qrmi = mocker.MagicMock(
        TaskStatus=FakeTaskStatus,
        ConfigError=FakeConfigError,
        InvalidInputError=FakeInvalidInputError,
        UnsupportedFunctionError=FakeUnsupportedFunctionError,
    )
    qrmi.maestro.request_payload.side_effect = request_payload
    return qrmi


SESSION_NOT_FOUND = "invalid configuration: session_not_found: Session does not exist"


class FakeQuantumResource:
    """A ``qrmi.QuantumResource`` for a ``maestro-local`` resource.

    Records every payload it is handed and replays canned results, so tests
    assert on what went over the wire.
    """

    def __init__(
        self,
        mocker,
        *,
        resource_id: str = "MAESTRO_LOCAL",
        counts: dict[str, int] | None = None,
        expectation_values: list[float] | None = None,
        statuses: list[FakeTaskStatus] | None = None,
    ):
        self._mocker = mocker
        self._resource_id = resource_id
        self._counts = counts if counts is not None else {"00": 4, "11": 6}
        self._expectation_values = expectation_values
        self._statuses = list(statuses) if statuses else None
        self._failure = {
            "error": {"code": "backend_unavailable", "message": "no such backend"},
            "logs": "",
        }

        self.payloads: list = []
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
        elif request["operation"] == "statevector":
            # Amplitude k is k itself, so tests can see which entries come back.
            dimension = 2 ** request["circuit"]["num_qubits"]
            body = {"amplitudes": [[float(k), 0.0] for k in range(dimension)]}
        elif request["operation"] == "adjoint_vjp":
            rows = request["parameter_rows"]
            body = {
                "expectation_values": [[0.5] * len(request["observables"])] * len(rows),
                "vjp": [[1.0] * len(row) for row in rows],
            }
        else:
            body = {"counts": dict(self._counts)}
        body |= {"ok": True, "time_taken": 0.001, "seed": 7}
        return self._mocker.Mock(value=json.dumps(body))

    def task_logs(self, task_id: str) -> str:
        return json.dumps(self._failure)

    def task_stop(self, task_id: str) -> None:
        self.stopped.append(task_id)

    # --- helpers for assertions ---

    @property
    def requests(self) -> list[dict]:
        return [json.loads(payload.input) for payload in self.payloads]

    @property
    def last_request(self) -> dict:
        return self.requests[-1]
