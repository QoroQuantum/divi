# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for the ``qrmi`` package, so the backend's tests need no wheel.

The real API is duck-typed rather than subclassed: ``qrmi`` ships a compiled
pyo3 extension whose classes cannot be constructed from Python.
"""

import json
from collections.abc import Callable
from enum import Enum

from qiskit.primitives import BitArray, DataBin, PrimitiveResult, SamplerPubResult


class FakeTaskStatus(Enum):
    Queued = "Queued"
    Running = "Running"
    Completed = "Completed"
    Failed = "Failed"
    Cancelled = "Cancelled"


class FakeResourceType(Enum):
    IBMQuantumSystem = "ibm-quantum-system"
    IBMQiskitRuntimeService = "qiskit-runtime-service"
    IBMQuantumComputeService = "ibm-quantum-compute-service"
    PasqalLocal = "pasqal-local"
    MaestroLocal = "maestro-local"


class FakeUpstreamResourceType(Enum):
    """Upstream qrmi's ``ResourceType``, which has no ``MaestroLocal``."""

    IBMQuantumSystem = "ibm-quantum-system"
    IBMQiskitRuntimeService = "qiskit-runtime-service"
    IBMQuantumComputeService = "ibm-quantum-compute-service"
    PasqalLocal = "pasqal-local"


class FakeConfigError(RuntimeError):
    """``qrmi.ConfigError``: no task exists after it."""


class FakeInvalidInputError(RuntimeError):
    """``qrmi.InvalidInputError``."""


class FakeUnsupportedFunctionError(RuntimeError):
    """``qrmi.UnsupportedFunctionError``."""


class FakeTaskNotReadyError(RuntimeError):
    """``qrmi.TaskNotReadyError``: IBM's result for a failed or cancelled job."""


def fake_qrmi_module(mocker):
    """The ``qrmi`` module, its payload helper checking a native request as
    ``qrmi.maestro.request_payload`` does. Only names ``qrmi`` defines exist,
    so a mistyped call fails."""

    def request_payload(request):
        if request.get("schema_version") != 2:
            raise ValueError("Maestro request must be an object with schema_version=2")
        return mocker.Mock(
            input=json.dumps(request, allow_nan=False), job_type="request"
        )

    def qiskit_primitive(input, program_id):
        return mocker.Mock(input=input, program_id=program_id)

    return mocker.NonCallableMock(
        spec_set=[
            "ResourceType",
            "TaskStatus",
            "ConfigError",
            "InvalidInputError",
            "UnsupportedFunctionError",
            "TaskNotReadyError",
            "maestro",
            "Payload",
        ],
        ResourceType=FakeResourceType,
        TaskStatus=FakeTaskStatus,
        ConfigError=FakeConfigError,
        InvalidInputError=FakeInvalidInputError,
        UnsupportedFunctionError=FakeUnsupportedFunctionError,
        TaskNotReadyError=FakeTaskNotReadyError,
        maestro=mocker.NonCallableMock(
            spec_set=["request_payload"],
            request_payload=mocker.Mock(side_effect=request_payload),
        ),
        Payload=mocker.NonCallableMock(
            spec_set=["QiskitPrimitive"],
            QiskitPrimitive=mocker.Mock(side_effect=qiskit_primitive),
        ),
    )


SESSION_NOT_FOUND = "invalid configuration: session_not_found: Session does not exist"


class FakeQuantumResource:
    """A ``qrmi.QuantumResource``, ``maestro-local`` unless told otherwise.

    Records every payload it is handed and replays canned results, so tests
    assert on what went over the wire. The public attributes set what it
    returns and which calls fail.
    """

    resource_id_value = "MAESTRO_LOCAL"

    def __init__(
        self,
        mocker,
        *,
        resource_type: FakeResourceType = FakeResourceType.MaestroLocal,
        counts: dict[str, int] | None = None,
    ):
        self._mocker = mocker
        self._resource_type = resource_type
        self.counts = counts if counts is not None else {"00": 4, "11": 6}
        self.expectation_values: list[float] | None = None
        #: Statuses ``task_status`` reports in turn, then Completed.
        self.statuses: list[FakeTaskStatus] = []
        #: What ``task_logs`` reports for a failed task, as a Maestro server does.
        self.failure: dict = {
            "error": {"code": "backend_unavailable", "message": "no such backend"},
            "logs": "",
        }
        #: Raw ``task_logs`` text, in place of :attr:`failure` when set.
        self.logs: str | None = None
        #: Errors ``task_start`` raises in turn before accepting payloads.
        self.start_errors: list[Exception] = []
        #: Called with each payload ``task_start`` accepts, and each task id
        #: ``task_result`` is asked for.
        self.on_task_start: Callable[[object], None] | None = None
        self.on_task_result: Callable[[str], None] | None = None
        self.status_error: Exception | None = None
        self.result_error: Exception | None = None
        self.logs_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.release_error: Exception | None = None

        self.payloads: list = []
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.stopped: list[str] = []
        self._next_session = 0

    # --- qrmi.QuantumResource surface ---

    def resource_id(self) -> str:
        return self.resource_id_value

    def resource_type(self) -> FakeResourceType:
        return self._resource_type

    def acquire(self) -> str:
        self._next_session += 1
        session = str(self._next_session)
        self.acquired.append(session)
        return session

    def release(self, session: str) -> None:
        if self.release_error is not None:
            raise self.release_error
        self.released.append(session)

    def task_start(self, payload) -> str:
        if self.start_errors:
            raise self.start_errors.pop(0)
        self.payloads.append(payload)
        if self.on_task_start is not None:
            self.on_task_start(payload)
        return f"task-{len(self.payloads) - 1}"

    def task_status(self, task_id: str):
        if self.status_error is not None:
            raise self.status_error
        if self.statuses:
            return self.statuses.pop(0)
        return FakeTaskStatus.Completed

    def task_result(self, task_id: str):
        if self.on_task_result is not None:
            self.on_task_result(task_id)
        if self.result_error is not None:
            raise self.result_error
        return self._mocker.Mock(value=self._result(task_id))

    def task_logs(self, task_id: str) -> str:
        if self.logs_error is not None:
            raise self.logs_error
        return self.logs if self.logs is not None else json.dumps(self.failure)

    def task_stop(self, task_id: str) -> None:
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped.append(task_id)

    # --- helpers for assertions ---

    @property
    def requests(self) -> list[dict]:
        return [json.loads(payload.input) for payload in self.payloads]

    @property
    def last_request(self) -> dict:
        return self.requests[-1]

    def _request_for(self, task_id: str) -> dict:
        return self.requests[int(task_id.rsplit("-", 1)[1])]

    def _result(self, task_id: str) -> str:
        request = self._request_for(task_id)
        body = {
            "estimate": self._estimate_body,
            "statevector": self._statevector_body,
            "adjoint_vjp": self._vjp_body,
        }.get(request["operation"], self._counts_body)(request)
        return json.dumps(body | {"ok": True, "time_taken": 0.001, "seed": 7})

    def _counts_body(self, _request: dict) -> dict:
        return {"counts": dict(self.counts)}

    def _estimate_body(self, request: dict) -> dict:
        values = self.expectation_values
        return {"expectation_values": values or [1.0] * len(request["observables"])}

    def _statevector_body(self, request: dict) -> dict:
        # Amplitude k is k itself, so tests can see which entries come back.
        dimension = 2 ** request["circuit"]["num_qubits"]
        return {"amplitudes": [[float(k), 0.0] for k in range(dimension)]}

    def _vjp_body(self, request: dict) -> dict:
        rows = request["parameter_rows"]
        return {
            "expectation_values": [[0.5] * len(request["observables"])] * len(rows),
            "vjp": [[1.0] * len(row) for row in rows],
        }


class FakeIBMResource(FakeQuantumResource):
    """An IBM ``qrmi.QuantumResource`` that returns :attr:`registers`' counts
    for every pub of a sampler job, encoded as Qiskit Runtime encodes results.

    ``task_stop`` deletes a job on IBM, so :attr:`stopped` lists deleted jobs.
    """

    resource_id_value = "fake_lagos"

    def __init__(
        self,
        mocker,
        result_encoder,
        *,
        resource_type: FakeResourceType = FakeResourceType.IBMQuantumSystem,
    ):
        super().__init__(mocker, resource_type=resource_type)
        self._result_encoder = result_encoder
        self.registers = {"c": self.counts}
        self.logs = "INFO Executing\nERROR the sampler rejected the circuit\n"
        #: IBM's reason for a failed job, which ``task_result`` raises with.
        self.failure_reason = (
            "task failed. code: 1517, message: circuit too wide, solution: retry"
        )
        self._ended_badly: set[str] = set()

    def task_status(self, task_id: str):
        status = super().task_status(task_id)
        if status in (FakeTaskStatus.Failed, FakeTaskStatus.Cancelled):
            self._ended_badly.add(task_id)
        return status

    def task_result(self, task_id: str):
        if task_id in self._ended_badly:
            raise FakeTaskNotReadyError(self.failure_reason)
        return super().task_result(task_id)

    def _result(self, task_id: str) -> str:
        data = DataBin(
            **{
                name: BitArray.from_counts(counts, num_bits=len(next(iter(counts))))
                for name, counts in self.registers.items()
            },
            shape=(),
        )
        pubs = [
            SamplerPubResult(data, metadata={"shots": shots})
            for _source, _parameters, shots in self._request_for(task_id)["pubs"]
        ]
        result = PrimitiveResult(pubs, metadata={"version": 2})
        return json.dumps(result, cls=self._result_encoder)
