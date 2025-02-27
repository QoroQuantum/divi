import pytest
import requests

from divi.services.qoro_service import JobStatus, MaxRetriesReachedError, QoroService


@pytest.fixture
def qoro_service(api_token):
    qoro_service = QoroService(api_token)

    return qoro_service


@pytest.fixture
def circuits():
    test_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = test_qasm

    return circuits


@pytest.mark.requires_api_token
def test_service_initialization(qoro_service):
    # Test if QoroService is initialized correctly
    response = qoro_service.test_connection()

    assert response.status_code == 200, "Connection should be successful"


@pytest.mark.requires_api_token
def test_send_circuits(qoro_service, circuits):
    # Test if QoroService can send circuits

    job_id = qoro_service.send_circuits(circuits)
    assert job_id is not None, "Job ID should not be None"

    res = qoro_service.delete_job(job_id)
    res.status_code == 204, "Deletion should be successful"


@pytest.mark.requires_api_token
def test_get_job_status(qoro_service, circuits):
    # Test if QoroService can get the status of a job
    job_id = qoro_service.send_circuits(circuits)
    status = qoro_service.job_status(job_id)

    assert status is not None, "Status should not be None"
    assert status != "", "Status should not be empty"
    assert status == JobStatus.PENDING.value, "Status should be PENDING"

    res = qoro_service.delete_job(job_id)

    assert res.status_code == 204, "Deletion should be successful"


@pytest.mark.requires_api_token
def test_retry_get_job_status(qoro_service, circuits):
    # Test getting the job status with retries

    job_id = qoro_service.send_circuits(circuits)

    with pytest.raises(MaxRetriesReachedError):
        qoro_service.job_status(
            job_id,
            loop_until_complete=True,
            max_retries=5,
            timeout=0.05,
        )

    res = qoro_service.delete_job(job_id)
    assert res.status_code == 204, "Deletion should be successful"


def test_fail_api_connection():
    # Test if QoroService fails to connect to the API
    api_token = "invalid_token"

    service = QoroService(api_token)

    response = service.test_connection()

    assert response.status_code != 200, "Connection should fail with invalid token"


def test_fail_send_circuits(circuits):
    # Test if QoroService fails to connect send circuits
    api_token = "invalid_token"

    service = QoroService(api_token)

    with pytest.raises(requests.exceptions.HTTPError):
        service.send_circuits(circuits)
