# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for backend tests."""

from http import HTTPStatus

from divi.backends import ExecutionResult, JobStatus
from divi.circuits import TemplateEntry


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


def assert_delete_successful(service, result):
    """Helper to assert successful job deletion."""
    res = service.delete_job(result)
    assert res.status_code == 204, "Deletion should be successful"


def create_failed_job(service):
    """Create a job pre-marked as FAILED via the create_failed endpoint.

    This is a test-only helper; the endpoint is not part of the public SDK.
    """
    response = service._make_request(
        "post", "job/create_failed/", json={"tag": "test"}, timeout=10
    )
    job_id = response.json()["job_id"]
    return ExecutionResult(job_id=job_id)


def make_template_entry(
    n_param_sets: int = 2, n_params: int = 2, label_prefix: str = "iter"
) -> TemplateEntry:
    """Build a TemplateEntry whose parameter values are derived from the
    set/param indices, making per-set assertions deterministic."""
    param_names = tuple(f"theta_{i}" for i in range(n_params))
    sets = tuple(
        (f"{label_prefix}_{i}", tuple(float(i + j) for j in range(n_params)))
        for i in range(n_param_sets)
    )
    return TemplateEntry(
        template_qasm=(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
            "ry(theta_0) q[0];\nrz(theta_1) q[0];\nmeasure q[0] -> c[0];\n"
        ),
        parameter_names=param_names,
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
