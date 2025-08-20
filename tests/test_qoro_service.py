# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from divi.qoro_service import JobStatus, JobType, MaxRetriesReachedError, QoroService

# --- Test Fixtures ---


@pytest.fixture
def qoro_service(api_token):
    """Provides a QoroService instance with a real API token for integration tests."""
    return QoroService(api_token)


@pytest.fixture
def qoro_service_mock():
    """Provides a mocked QoroService instance for unit tests."""
    return QoroService("mock_token", max_retries=3, polling_interval=0.01)


@pytest.fixture
def circuits():
    """Provides a dictionary of test circuits."""
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
        """Tests that submitting circuits with an invalid token raises an HTTPError."""
        service = QoroService("invalid_token")
        with pytest.raises(requests.exceptions.HTTPError):
            service.submit_circuits(circuits)

    def test_service_connection_test_mock(self, mocker, qoro_service_mock):
        """Tests the connection test functionality with a mock."""
        # Test for failure
        mock_response_fail = mocker.MagicMock()
        mock_response_fail.status_code = 401
        mock_response_fail.reason = "Unauthorized"
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=requests.exceptions.HTTPError("401: Unauthorized"),
        )

        with pytest.raises(requests.exceptions.HTTPError, match="401: Unauthorized"):
            qoro_service_mock.test_connection()

        # Test for success
        mock_response_success = mocker.MagicMock()
        mock_response_success.status_code = 200
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_success
        )

        response = qoro_service_mock.test_connection()
        assert response.status_code == 200

    # --- Tests for submit_circuits ---

    def test_submit_circuits_single_chunk_mock(self, mocker, qoro_service_mock):
        """Tests submitting a single chunk of circuits."""
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"job_id": "mock_job_id"}
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)

        job_id = qoro_service_mock.submit_circuits({"circuit_1": "mock_qasm"})

        assert job_id == "mock_job_id"
        mock_make_request.assert_called_once()

    def test_submit_circuits_multiple_chunks_mock(self, mocker, qoro_service_mock):
        """Tests submitting multiple chunks of circuits."""
        mocker.patch("divi.qoro_service.MAX_PAYLOAD_SIZE_MB", new=60.0 / 1024 / 1024)
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)

        mock_response_1 = mocker.MagicMock(json=lambda: {"job_id": "mock_job_id_1"})
        mock_response_2 = mocker.MagicMock(json=lambda: {"job_id": "mock_job_id_2"})

        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_response_1, mock_response_2],
        )

        job_ids = qoro_service_mock.submit_circuits(
            {"circuit_1": "mock_qasm", "circuit_2": "mock_qasm"}
        )

        assert mock_make_request.call_count == 2
        assert job_ids == ["mock_job_id_1", "mock_job_id_2"]

    def test_submit_circuits_invalid_qasm_mock(self, mocker, qoro_service_mock):
        """Tests that submitting an invalid QASM string raises a ValueError."""
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=False)
        with pytest.raises(
            ValueError, match="Circuit circuit_1 is not a valid QASM string."
        ):
            qoro_service_mock.submit_circuits({"circuit_1": "invalid_qasm"})

    def test_submit_circuits_circuit_cut_constraint_mock(self, qoro_service_mock):
        """Tests the constraint for circuit cutting jobs."""
        with pytest.raises(
            ValueError, match="Only one circuit allowed for circuit-cutting jobs."
        ):
            qoro_service_mock.submit_circuits(
                {"c1": "qasm1", "c2": "qasm2"}, job_type=JobType.CIRCUIT_CUT
            )

    def test_submit_circuits_with_tag_and_job_type_mock(
        self, mocker, qoro_service_mock
    ):
        """Tests submitting circuits with a custom tag and job type."""
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)
        mock_response = mocker.MagicMock(json=lambda: {"job_id": "mock_job_id"})
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        qoro_service_mock.submit_circuits(
            {"c1": "qasm"}, tag="my_custom_tag", job_type=JobType.EXECUTE
        )

        _, called_kwargs = mock_make_request.call_args
        payload = called_kwargs.get("json", {})
        assert payload.get("tag") == "my_custom_tag"
        assert payload.get("job_type") == JobType.EXECUTE.value

    def test_submit_circuits_api_error_mock(self, mocker, qoro_service_mock):
        """Tests API error handling during circuit submission."""
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=requests.exceptions.HTTPError(
                "API Error: 500 Internal Server Error for URL http://mock.url"
            ),
        )

        with pytest.raises(requests.exceptions.HTTPError):
            qoro_service_mock.submit_circuits({"c1": "qasm"})

    @pytest.mark.parametrize("original_flag", [True, False])
    def test_submit_circuits_override_packing_mock(
        self, mocker, qoro_service_mock, original_flag
    ):
        """Tests overriding the circuit packing setting."""
        mocker.patch("divi.qoro_service.is_valid_qasm", return_value=True)
        qoro_service_mock.use_circuit_packing = original_flag

        mock_response = mocker.MagicMock(json=lambda: {"job_id": "mock_job_id_packed"})
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        circuits = {"circuit_1": "mock_qasm"}

        # Test overriding to True
        job_id = qoro_service_mock.submit_circuits(
            circuits, override_circuit_packing=True
        )
        assert job_id == "mock_job_id_packed"
        _, called_kwargs = mock_make_request.call_args
        assert called_kwargs.get("json", {}).get("use_packing") is True

        # Test overriding to False
        job_id = qoro_service_mock.submit_circuits(
            circuits, override_circuit_packing=False
        )
        assert job_id == "mock_job_id_packed"
        _, called_kwargs = mock_make_request.call_args
        assert called_kwargs.get("json", {}).get("use_packing") is False

    # --- Tests for delete_job ---

    def test_delete_job_single_mock(self, mocker, qoro_service_mock):
        """Tests deleting a single job."""
        mock_response = mocker.MagicMock(status_code=204)
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        response = qoro_service_mock.delete_job("job_1")

        mock_make_request.assert_called_once_with("delete", "job/job_1", timeout=50)
        assert response.status_code == 204

    def test_delete_job_multiple_mock(self, mocker, qoro_service_mock):
        """Tests deleting multiple jobs."""
        mock_response = mocker.MagicMock(status_code=204)
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        responses = qoro_service_mock.delete_job(["job_1", "job_2"])

        assert mock_make_request.call_count == 2
        assert all(res.status_code == 204 for res in responses)

    def test_delete_job_api_error_mock(self, mocker, qoro_service_mock):
        """Tests API error handling during job deletion."""
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=requests.exceptions.HTTPError(
                "API Error: 404 Not Found for URL http://mock.url"
            ),
        )

        with pytest.raises(requests.exceptions.HTTPError):
            qoro_service_mock.delete_job("job_1")

    # --- Tests for get_job_results ---

    def test_get_job_results_success_mock(self, mocker, qoro_service_mock):
        """Tests successfully fetching job results."""
        mock_response = mocker.MagicMock(
            status_code=200, json=lambda: [{"result": "data"}]
        )
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        results = qoro_service_mock.get_job_results("job_1")

        assert results == [{"result": "data"}]

    def test_get_job_results_multiple_success_mock(self, mocker, qoro_service_mock):
        """Tests fetching results for multiple jobs."""
        mock_response_1 = mocker.MagicMock(
            status_code=200, json=lambda: [{"result": "data1"}]
        )
        mock_response_2 = mocker.MagicMock(
            status_code=200, json=lambda: [{"result": "data2"}]
        )
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_response_1, mock_response_2],
        )

        results = qoro_service_mock.get_job_results(["job_1", "job_2"])

        assert results == [{"result": "data1"}, {"result": "data2"}]

    def test_get_job_results_still_running_mock(self, mocker, qoro_service_mock):
        """Tests handling of a 'still running' job."""
        mock_response = mocker.MagicMock(status_code=400)
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        with pytest.raises(requests.exceptions.HTTPError, match="400 Bad Request"):
            qoro_service_mock.get_job_results("job_1")

    def test_get_job_results_api_error_mock(self, mocker, qoro_service_mock):
        """Tests API error handling when fetching job results."""
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=requests.exceptions.HTTPError(
                "API Error: 404 Not Found for URL http://mock.url"
            ),
        )

        with pytest.raises(requests.exceptions.HTTPError):
            qoro_service_mock.get_job_results("job_1")

    # --- Tests for poll_job_status ---

    def test_poll_job_status_success_mock(self, mocker, qoro_service_mock):
        """Tests successful polling of job status until completion."""
        mock_response_pending = mocker.MagicMock(
            json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_response_completed = mocker.MagicMock(
            json=lambda: {"status": JobStatus.COMPLETED.value}
        )
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_response_pending, mock_response_completed],
        )

        status = qoro_service_mock.poll_job_status(
            "mock_job_id", loop_until_complete=True, verbose=False
        )

        assert mock_make_request.call_count == 2
        assert status == JobStatus.COMPLETED

    def test_poll_job_status_failure_mock(self, mocker, qoro_service_mock):
        """Tests polling that results in a FAILED status."""
        mock_response_pending = mocker.MagicMock(
            json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_pending
        )

        with pytest.raises(
            MaxRetriesReachedError, match="Maximum retries reached: 3 retries attempted"
        ):
            qoro_service_mock.poll_job_status(
                "mock_job_id", loop_until_complete=True, verbose=False
            )

        assert mock_make_request.call_count == 3

    def test_poll_job_status_no_loop_mock(self, mocker, qoro_service_mock):
        """Tests polling without looping."""
        mock_response_running = mocker.MagicMock(
            json=lambda: {"status": JobStatus.RUNNING.value}
        )
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_running
        )

        status = qoro_service_mock.poll_job_status("job_1", loop_until_complete=False)

        mock_make_request.assert_called_once()
        assert status == JobStatus.RUNNING.value

    def test_poll_job_status_multiple_jobs_mock(self, mocker, qoro_service_mock):
        """Tests polling for multiple jobs."""
        mock_response_pending = mocker.MagicMock(
            json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_response_completed = mocker.MagicMock(
            json=lambda: {"status": JobStatus.COMPLETED.value}
        )

        # Simulation:
        # Loop 1: job1=PENDING, job2=PENDING
        # Loop 2: job1=COMPLETED, job2=PENDING
        # Loop 3: job2=COMPLETED
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[
                mock_response_pending,
                mock_response_pending,  # Loop 1
                mock_response_completed,
                mock_response_pending,  # Loop 2
                mock_response_completed,  # Loop 3
            ],
        )

        status = qoro_service_mock.poll_job_status(
            ["job_1", "job_2"], loop_until_complete=True, verbose=False
        )

        assert mock_make_request.call_count == 5
        assert status == JobStatus.COMPLETED

    def test_poll_job_status_on_complete_callback_mock(self, mocker, qoro_service_mock):
        """Tests the on_complete callback functionality."""
        mock_response_completed = mocker.MagicMock(
            json=lambda: {"status": JobStatus.COMPLETED.value, "data": "results"}
        )
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_completed
        )

        callback_mock = mocker.MagicMock()
        status = qoro_service_mock.poll_job_status(
            "job_1", loop_until_complete=True, on_complete=callback_mock
        )

        assert status == JobStatus.COMPLETED
        callback_mock.assert_called_once_with([mock_response_completed])

    def test_poll_job_status_pbar_update_fn_mock(self, mocker, qoro_service_mock):
        """Tests the progress bar update function."""
        mock_response_pending = mocker.MagicMock(
            json=lambda: {"status": JobStatus.PENDING.value}
        )
        mock_response_completed = mocker.MagicMock(
            json=lambda: {"status": JobStatus.COMPLETED.value}
        )
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[
                mock_response_pending,
                mock_response_pending,
                mock_response_completed,
            ],
        )

        pbar_mock = mocker.MagicMock()
        qoro_service_mock.poll_job_status(
            "job_1", loop_until_complete=True, pbar_update_fn=pbar_mock, verbose=True
        )

        assert pbar_mock.call_count == 2
        pbar_mock.assert_has_calls(
            [
                mocker.call.__bool__(),
                mocker.call(1),
                mocker.call.__bool__(),
                mocker.call(2),
            ]
        )


# --- Integration Tests (require API token) ---


@pytest.mark.requires_api_token
class TestQoroServiceWithApiToken:
    """Integration tests for the QoroService, requiring a valid API token."""

    def test_service_connection_test(self, qoro_service):
        """Tests the connection to the live service."""
        response = qoro_service.test_connection()
        assert response.status_code == 200, "Connection should be successful"

    def test_submit_and_delete_circuits(self, qoro_service, circuits):
        """Tests submitting and then deleting circuits."""
        job_id = qoro_service.submit_circuits(circuits)
        assert job_id is not None, "Job ID should not be None"

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"

    def test_get_job_status(self, qoro_service, circuits):
        """Tests retrieving the status of a submitted job."""
        job_id = qoro_service.submit_circuits(circuits)
        status = qoro_service.poll_job_status(job_id)

        assert status is not None, "Status should not be None"
        assert status in [
            s.value for s in JobStatus
        ], "Status should be a valid JobStatus"

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"

    def test_retry_get_job_status(self, qoro_service, circuits):
        """Tests the retry mechanism for polling job status."""
        job_id = qoro_service.submit_circuits(circuits)

        qoro_service_temp = QoroService(
            qoro_service.auth_token.split(" ")[1], max_retries=5, polling_interval=0.05
        )

        with pytest.raises(MaxRetriesReachedError):
            qoro_service_temp.poll_job_status(job_id, loop_until_complete=True)

        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"
