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
