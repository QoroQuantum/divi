import pytest
import requests

from divi.services.qoro_service import JobStatus, MaxRetriesReachedError, QoroService


@pytest.fixture
def qoro_service(api_token):
    qoro_service = QoroService(api_token)

    return qoro_service


@pytest.fixture
def qoro_service_mock():
    qoro_service = QoroService("mock_token")

    return qoro_service


@pytest.fixture
def circuits():
    test_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = test_qasm

    return circuits


def test_service_connection_test_mock(mocker, qoro_service_mock):
    with pytest.raises(
        requests.exceptions.HTTPError,
        match="Connection failed with error: 401: Unauthorized",
    ):
        qoro_service_mock.test_connection()

    mock_response = mocker.Mock()
    mock_response.status_code = 200

    mocker.patch("requests.Session.get", return_value=mock_response)

    response = qoro_service_mock.test_connection()
    assert response.status_code == 200


def test_fail_send_circuits(circuits):
    # Test if QoroService fails to connect send circuits
    api_token = "invalid_token"

    service = QoroService(api_token)

    with pytest.raises(requests.exceptions.HTTPError):
        service.send_circuits(circuits)


def test_send_circuits_single_chunk_mock(mocker, qoro_service_mock):
    mock_response = mocker.Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"job_id": "mock_job_id"}

    mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

    job_id = qoro_service_mock.send_circuits({"circuit_1": "mock_qasm"})
    assert job_id == "mock_job_id"

    assert mock_post.call_count == 1


def test_send_circuits_multiple_chunks_mock(mocker, qoro_service_mock):
    mocker.patch(
        "divi.services.qoro_service.MAX_PAYLOAD_SIZE_MB", new=60.0 / 1024 / 1024
    )

    mock_response_1 = mocker.Mock(
        status_code=201, json=lambda: {"job_id": "mock_job_id_1"}
    )

    mock_response_2 = mocker.Mock(
        status_code=201, json=lambda: {"job_id": "mock_job_id_2"}
    )

    mock_post = mocker.patch(
        "requests.Session.post", side_effect=[mock_response_1, mock_response_2]
    )

    job_ids = qoro_service_mock.send_circuits(
        {"circuit_1": "mock_qasm", "circuit_2": "mock_qasm"}
    )

    assert mock_post.call_count == 2

    assert job_ids == ["mock_job_id_1", "mock_job_id_2"]


def test_poll_job_status_success_mock(mocker, qoro_service_mock):
    mock_response_pending = mocker.Mock()
    mock_response_pending.status_code = 200
    mock_response_pending.json.return_value = {"status": JobStatus.PENDING.value}

    mock_response_completed = mocker.Mock()
    mock_response_completed.status_code = 200
    mock_response_completed.json.return_value = {"status": JobStatus.COMPLETED.value}

    mock_get = mocker.patch(
        "requests.Session.get",
        side_effect=[
            mock_response_pending,
            mock_response_pending,
            mock_response_completed,
        ],
    )

    job_id = "mock_job_id"

    status = qoro_service_mock.poll_job_status(
        job_ids=job_id,
        loop_until_complete=True,
        timeout=0.01,
        max_retries=5,
        verbose=False,
    )

    # Original attempt + 2 retries before completion
    assert mock_get.call_count == 3
    assert status == JobStatus.COMPLETED


def test_poll_job_status_failure_mock(mocker, qoro_service_mock):
    mock_response_pending = mocker.Mock()
    mock_response_pending.status_code = 200
    mock_response_pending.json.return_value = {"status": JobStatus.PENDING.value}

    mock_response_failed = mocker.Mock()
    mock_response_failed.status_code = 200
    mock_response_failed.json.return_value = {"status": JobStatus.FAILED.value}

    mock_get = mocker.patch(
        "requests.Session.get",
        side_effect=[
            mock_response_pending,
            mock_response_pending,
            mock_response_failed,
        ],
    )

    job_id = "mock_job_id"

    with pytest.raises(
        MaxRetriesReachedError, match="Maximum retries reached: 2 retries attempted"
    ):
        qoro_service_mock.poll_job_status(
            job_ids=job_id,
            loop_until_complete=True,
            timeout=0.01,
            max_retries=2,
            verbose=False,
        )

    # Original attempt + 2 retries
    assert mock_get.call_count == 3


@pytest.mark.requires_api_token
def test_service_connection_test(qoro_service):
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
