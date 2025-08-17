# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from divi.qoro_service import JobStatus, JobType, MaxRetriesReachedError, QoroService

# --- Test Fixtures ---


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
    # --- Tests for test_connection ---

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

    # --- Tests for submit_circuits ---

    def test_submit_circuits_single_chunk_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"job_id": "mock_job_id"}

        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)

        job_id = qoro_service_mock.submit_circuits({"circuit_1": "mock_qasm"})
        assert job_id == "mock_job_id"
        assert mock_post.call_count == 1

    def test_submit_circuits_multiple_chunks_mock(self, mocker, qoro_service_mock):
        mocker.patch("divi.qoro_service.MAX_PAYLOAD_SIZE_MB", new=60.0 / 1024 / 1024)
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)

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

    def test_submit_circuits_invalid_qasm_mock(self, mocker, qoro_service_mock):
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=False)
        with pytest.raises(
            ValueError, match="Circuit circuit_1 is not a valid QASM string."
        ):
            qoro_service_mock.submit_circuits({"circuit_1": "invalid_qasm"})

    def test_submit_circuits_circuit_cut_constraint_mock(self, qoro_service_mock):
        with pytest.raises(
            ValueError, match="Only one circuit allowed for circuit-cutting jobs."
        ):
            qoro_service_mock.submit_circuits(
                {"c1": "qasm1", "c2": "qasm2"}, job_type=JobType.CIRCUIT_CUT
            )

    def test_submit_circuits_with_tag_and_job_type_mock(
        self, mocker, qoro_service_mock
    ):
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)
        mock_response = mocker.Mock(
            status_code=201, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        qoro_service_mock.submit_circuits(
            {"c1": "qasm"}, tag="my_custom_tag", job_type=JobType.EXECUTE
        )

        _, called_kwargs = mock_post.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("tag") == "my_custom_tag"
        assert payload.get("job_type") == JobType.EXECUTE.value

    def test_submit_circuits_api_error_mock(self, mocker, qoro_service_mock):
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)
        mock_response = mocker.Mock(status_code=500, reason="Internal Server Error")
        mocker.patch("requests.Session.post", return_value=mock_response)

        with pytest.raises(
            requests.exceptions.HTTPError, match="500: Internal Server Error"
        ):
            qoro_service_mock.submit_circuits({"c1": "qasm"})

    @pytest.mark.parametrize("original_flag", [True, False])
    def test_submit_circuits_override_packing_mock(
        self, mocker, qoro_service_mock, original_flag
    ):
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)

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

    # --- Tests for delete_job ---

    def test_delete_job_single_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=204)
        mock_delete = mocker.patch(
            "requests.Session.delete", return_value=mock_response
        )
        response = qoro_service_mock.delete_job("job_1")
        mock_delete.assert_called_once_with(
            "https://app.qoroquantum.net/api/job/job_1",
            headers={"Authorization": "Bearer mock_token"},
            timeout=50,
        )
        assert response.status_code == 204

    def test_delete_job_multiple_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=204)
        mock_delete = mocker.patch(
            "requests.Session.delete", return_value=mock_response
        )
        responses = qoro_service_mock.delete_job(["job_1", "job_2"])
        assert mock_delete.call_count == 2
        assert all(res.status_code == 204 for res in responses)

    def test_delete_job_api_error_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=404, reason="Not Found")
        mocker.patch("requests.Session.delete", return_value=mock_response)
        # The current implementation doesn't raise on delete failure, it just returns the response
        response = qoro_service_mock.delete_job("job_1")
        assert response.status_code == 404

    # --- Tests for get_job_results ---

    def test_get_job_results_success_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=200, json=lambda: [{"result": "data"}])
        mock_get = mocker.patch("requests.Session.get", return_value=mock_response)
        results = qoro_service_mock.get_job_results("job_1")
        mock_get.assert_called_once()
        assert results == [{"result": "data"}]

    def test_get_job_results_multiple_success_mock(self, mocker, qoro_service_mock):
        mock_response_1 = mocker.Mock(
            status_code=200, json=lambda: [{"result": "data1"}]
        )
        mock_response_2 = mocker.Mock(
            status_code=200, json=lambda: [{"result": "data2"}]
        )
        mock_get = mocker.patch(
            "requests.Session.get", side_effect=[mock_response_1, mock_response_2]
        )
        results = qoro_service_mock.get_job_results(["job_1", "job_2"])
        assert mock_get.call_count == 2
        assert results == [{"result": "data1"}, {"result": "data2"}]

    def test_get_job_results_still_running_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=400)
        mocker.patch("requests.Session.get", return_value=mock_response)
        with pytest.raises(requests.exceptions.HTTPError, match="400 Bad Request"):
            qoro_service_mock.get_job_results("job_1")

    def test_get_job_results_api_error_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(status_code=404, reason="Not Found")
        mocker.patch("requests.Session.get", return_value=mock_response)
        with pytest.raises(requests.exceptions.HTTPError, match="404: Not Found"):
            qoro_service_mock.get_job_results("job_1")

    # --- Tests for poll_job_status ---

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

    def test_poll_job_status_no_loop_mock(self, mocker, qoro_service_mock):
        mock_response = mocker.Mock(
            status_code=200, json=lambda: {"status": JobStatus.RUNNING.value}
        )
        mock_get = mocker.patch("requests.Session.get", return_value=mock_response)
        status = qoro_service_mock.poll_job_status("job_1", loop_until_complete=False)
        mock_get.assert_called_once()
        assert status == JobStatus.RUNNING.value

    def test_poll_job_status_multiple_jobs_mock(self, mocker, qoro_service_mock):
        # Job 1: PENDING -> COMPLETED
        # Job 2: PENDING -> PENDING -> COMPLETED
        mock_response_pending = mocker.Mock(
            status_code=200, json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_response_completed = mocker.Mock(
            status_code=200, json=lambda: {"status": JobStatus.COMPLETED.value}
        )

        # Polling loop 1: job1=PENDING, job2=PENDING
        # Polling loop 2: job1=COMPLETED, job2=PENDING
        # Polling loop 3: job1=COMPLETED, job2=COMPLETED
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                # Loop 1
                mock_response_pending,
                mock_response_pending,
                # Loop 2
                mock_response_completed,
                mock_response_pending,
                # Loop 3
                mock_response_completed,
                mock_response_completed,
            ],
        )

        status = qoro_service_mock.poll_job_status(
            ["job_1", "job_2"], loop_until_complete=True, verbose=False
        )
        assert mock_get.call_count == 6
        assert status == JobStatus.COMPLETED

    def test_poll_job_status_on_complete_callback_mock(self, mocker, qoro_service_mock):
        mock_response_completed = mocker.Mock(
            status_code=200,
            json=lambda: {"status": JobStatus.COMPLETED.value, "data": "results"},
        )
        mocker.patch("requests.Session.get", return_value=mock_response_completed)

        callback_mock = mocker.Mock()
        status = qoro_service_mock.poll_job_status(
            "job_1", loop_until_complete=True, on_complete=callback_mock
        )

        assert status == JobStatus.COMPLETED
        callback_mock.assert_called_once()
        # The callback receives the list of response.json() results
        callback_mock.assert_called_with([{"status": "COMPLETED", "data": "results"}])

    def test_poll_job_status_pbar_update_fn_mock(self, mocker, qoro_service_mock):
        mock_response_pending = mocker.Mock(
            status_code=200, json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_response_completed = mocker.Mock(
            status_code=200, json=lambda: {"status": JobStatus.COMPLETED.value}
        )
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                mock_response_pending,
                mock_response_pending,
                mock_response_completed,
            ],
        )

        pbar_mock = mocker.Mock()
        qoro_service_mock.poll_job_status(
            "job_1", loop_until_complete=True, pbar_update_fn=pbar_mock, verbose=True
        )

        # Called on the first and second retry
        assert pbar_mock.call_count == 2
        pbar_mock.assert_any_call(1)  # First retry
        pbar_mock.assert_any_call(2)  # Second retry


# --- Integration Tests (require API token) ---


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
        status = qoro_service.poll_job_status(job_id)

        assert status is not None, "Status should not be None"
        assert status != "", "Status should not be empty"
        assert status == JobStatus.PENDING.value, "Status should be PENDING"

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"

    def test_retry_get_job_status(self, qoro_service, circuits):
        job_id = qoro_service.submit_circuits(circuits)

        qoro_service_temp = QoroService(
            qoro_service.auth_token.split(" ")[1], max_retries=5, polling_interval=0.05
        )

        with pytest.raises(MaxRetriesReachedError):
            qoro_service_temp.poll_job_status(
                job_id,
                loop_until_complete=True,
            )

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"
