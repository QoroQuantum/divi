# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from divi.qoro_service import JobStatus, MaxRetriesReachedError, QoroService


@pytest.fixture
def qoro_service(api_token):
    return QoroService(api_token)


@pytest.fixture
def qoro_service_mock():
    return QoroService("mock_token", max_retries=2, polling_interval=0.01)


@pytest.fixture
def circuits():
    test_qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n'
        "x q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];"
        "cx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];"
        "\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n"
    )
    return {f"circuit_{i}": test_qasm for i in range(10)}


class TestQoroServiceMock:
    def test_fail_submit_circuits(self, circuits):
        service = QoroService("invalid_token")
        with pytest.raises(requests.exceptions.HTTPError):
            service.submit_circuits(circuits)

    def test_service_connection_test_mock(self, mocker, qoro_service_mock):
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

    def test_submit_circuits_single_chunk_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"job_id": "mock_job_id"}

        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        job_id = qoro_service_mock.submit_circuits({"circuit_1": "mock_qasm"})
        assert job_id == "mock_job_id"
        assert mock_post.call_count == 1

    def test_submit_circuits_multiple_chunks_mock(self, mocker, qoro_service_mock):
        mocker.patch("divi.qoro_service.MAX_PAYLOAD_SIZE_MB", new=60.0 / 1024 / 1024)

        mock_response_1 = mocker.Mock(
            status_code=201, json=lambda: {"job_id": "mock_job_id_1"}
        )
        mock_response_2 = mocker.Mock(
            status_code=201, json=lambda: {"job_id": "mock_job_id_2"}
        )

        mock_post = mocker.patch(
            "requests.Session.post", side_effect=[mock_response_1, mock_response_2]
        )

        job_ids = qoro_service_mock.submit_circuits(
            {"circuit_1": "mock_qasm", "circuit_2": "mock_qasm"}
        )

        assert mock_post.call_count == 2
        assert job_ids == ["mock_job_id_1", "mock_job_id_2"]

    def test_poll_job_status_success_mock(self, mocker, qoro_service_mock):
        mock_response_pending = mocker.Mock(status_code=200)
        mock_response_pending.json.return_value = {"status": JobStatus.PENDING.value}

        mock_response_completed = mocker.Mock(status_code=200)
        mock_response_completed.json.return_value = {
            "status": JobStatus.COMPLETED.value
        }

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
            verbose=False,
        )

        assert mock_get.call_count == 3
        assert status == JobStatus.COMPLETED

    def test_poll_job_status_failure_mock(self, mocker, qoro_service_mock):
        mock_response_pending = mocker.Mock(status_code=200)
        mock_response_pending.json.return_value = {"status": JobStatus.PENDING.value}

        mock_response_failed = mocker.Mock(status_code=200)
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
                verbose=False,
            )

        assert mock_get.call_count == 3

    @pytest.mark.parametrize("original_flag", [True, False])
    def test_submit_circuits_override_packing_mock(
        self, mocker, qoro_service_mock, original_flag
    ):
        qoro_service_mock.use_circuit_packing = original_flag

        mock_response = mocker.Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"job_id": "mock_job_id_packed"}

        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        circuits = {"circuit_1": "mock_qasm"}

        job_id = qoro_service_mock.submit_circuits(
            circuits, override_circuit_packing=True
        )
        assert job_id == "mock_job_id_packed"
        assert mock_post.call_count == 1

        # Extract actual payload from call args
        _, called_kwargs = mock_post.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("use_packing") is True
        assert "circuit_1" in payload.get("circuits")

        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)
        job_id = qoro_service_mock.submit_circuits(
            circuits, override_circuit_packing=False
        )
        assert job_id == "mock_job_id_packed"
        assert mock_post.call_count == 1

        # Extract actual payload from call args
        _, called_kwargs = mock_post.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("use_packing") is False


@pytest.mark.requires_api_token
class TestQoroServiceWithApiToken:

    def test_service_connection_test(self, qoro_service):
        response = qoro_service.test_connection()
        assert response.status_code == 200, "Connection should be successful"

    def test_submit_circuits(self, qoro_service, circuits):
        job_id = qoro_service.submit_circuits(circuits)
        assert job_id is not None, "Job ID should not be None"

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"

    def test_get_job_status(self, qoro_service, circuits):
        job_id = qoro_service.submit_circuits(circuits)
        status = qoro_service.job_status(job_id)

        assert status is not None, "Status should not be None"
        assert status != "", "Status should not be empty"
        assert status == JobStatus.PENDING.value, "Status should be PENDING"

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"

    def test_retry_get_job_status(self, qoro_service, circuits):
        job_id = qoro_service.submit_circuits(circuits)

        with pytest.raises(MaxRetriesReachedError):
            qoro_service.job_status(
                job_id,
                loop_until_complete=True,
                max_retries=5,
                timeout=0.05,
            )

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"
