# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for backend tests."""

from collections.abc import Callable
from http import HTTPStatus

import pytest
from pydantic import ValidationError
from qiskit.circuit import Parameter

from divi.backends import ExecutionResult, JobStatus
from divi.backends.runners._qoro import API_URL
from divi.circuits._payloads import CircuitPayload

SHOT_GROUPS_WITH_HAM_OPS_MESSAGE = (
    "shot_groups is incompatible with ham_ops: expectation-value mode is "
    "analytical and ignores shot counts. Pass exactly one."
)


def padding_warning(short: str, padded: str) -> str:
    """The warning raised when ``short`` is padded to ``padded`` (first example)."""
    return (
        "Observables shorter than their circuit are padded onto its first "
        f"qubits, e.g. {short!r} -> {padded!r}."
    )


def reset_unknown_message(name: str, suggestion: str) -> str:
    return f"Cannot reset unknown fields {name!r} (did you mean {suggestion!r}?)."


def uncovered_circuits_message(missing: list[int]) -> str:
    return f"Shot ranges do not cover every circuit; missing indices {missing}."


def validation_error_message(build: Callable[[], object]) -> str:
    """The message of the one error a validator raised while ``build`` ran."""
    with pytest.raises(ValidationError) as exc_info:
        build()
    (error,) = exc_info.value.errors()
    return str(error["ctx"]["error"])


def make_execution_result(job_id: str = "test_job") -> ExecutionResult:
    """Helper to create ExecutionResult instances."""
    return ExecutionResult(job_id=job_id)


def make_mock_init_response(mocker, job_id: str = "mock_job_id"):
    """Helper to create mock init response."""
    mock = mocker.MagicMock()
    mock.status_code = HTTPStatus.CREATED
    mock.json.return_value = {"job_id": job_id}
    return mock


def make_mock_add_response(mocker, status_code: int = HTTPStatus.OK):
    """Helper to create mock add_circuits response."""
    mock = mocker.MagicMock()
    mock.status_code = status_code
    return mock


def make_mock_status_response(mocker, status: JobStatus):
    """Helper to create mock status response."""
    return mocker.MagicMock(json=lambda: {"status": status.value})


def http_response(
    mocker, status_code=HTTPStatus.OK, *, body=None, text="", reason="OK"
):
    """A ``requests`` response; ``body=None`` makes ``json()`` fail as on an HTML page."""
    response = mocker.MagicMock(
        status_code=status_code, reason=reason, url=f"{API_URL}/endpoint", text=text
    )
    if body is None:
        response.json.side_effect = ValueError("not JSON")
    else:
        response.json.return_value = body
    return response


def patch_transport(mocker, *, session=None, plain=None):
    """Patch the retrying session and the plain ``requests.request`` path."""
    return (
        mocker.patch("requests.Session.request", side_effect=session),
        mocker.patch("requests.request", side_effect=plain),
    )


def create_failed_job(service):
    """Create a job pre-marked as FAILED via the create_failed endpoint.

    This is a test-only helper; the endpoint is not part of the public SDK.
    """
    response = service._make_request(
        "post", "job/create_failed/", json={"tag": "test"}, timeout=10
    )
    job_id = response.json()["job_id"]
    return ExecutionResult(job_id=job_id)


def make_qasm_payload(
    n_param_sets: int = 2, n_params: int = 2, label_prefix: str = "iter"
) -> CircuitPayload:
    """Build a QASM-encoded CircuitPayload whose parameter values are derived from the
    set/param indices, making per-set assertions deterministic."""
    parameters = tuple(Parameter(f"theta_{i}") for i in range(n_params))
    sets = tuple(
        (f"{label_prefix}_{i}", tuple(float(i + j) for j in range(n_params)))
        for i in range(n_param_sets)
    )
    return CircuitPayload(
        circuit=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
            "ry(theta_0) q[0];\nrz(theta_1) q[0];\nmeasure q[0] -> c[0];\n"
        ),
        parameters=parameters,
        parameter_sets=sets,
    )


def encode_uleb128(value: int) -> bytes:
    """Encode a non-negative integer as ULEB128."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def build_qh_histogram(
    n_bits: int, entries: list[tuple[int, int]], magic: bytes = b"QH1"
) -> bytes:
    """Build a QH1/QH2 histogram payload from ``(index, count)`` entries."""
    entries = sorted(entries, key=lambda entry: entry[0])
    indices = [index for index, _ in entries]
    counts = [count for _, count in entries]

    gaps = []
    previous = 0
    for index in indices:
        gaps.append(index - previous)
        previous = index

    is_one = [count == 1 for count in counts]
    if not is_one:
        rle_body = b""
    else:
        runs = []
        current = is_one[0]
        run_length = 1
        for value in is_one[1:]:
            if value == current:
                run_length += 1
            else:
                runs.append(run_length)
                current = not current
                run_length = 1
        runs.append(run_length)
        first_value = 1 if is_one[0] else 0
        rle_body = (
            encode_uleb128(len(runs))
            + bytes([first_value])
            + b"".join(encode_uleb128(run) for run in runs)
        )

    extras = [count - 2 for count in counts if count != 1]
    width = encode_uleb128(n_bits) if magic == b"QH2" else bytes([n_bits])
    return b"".join(
        [
            magic,
            width,
            encode_uleb128(len(entries)),
            encode_uleb128(sum(counts)),
            encode_uleb128(len(gaps)),
            *(encode_uleb128(gap) for gap in gaps),
            encode_uleb128(len(rle_body)),
            rle_body,
            encode_uleb128(len(extras)),
            *(encode_uleb128(extra) for extra in extras),
        ]
    )
