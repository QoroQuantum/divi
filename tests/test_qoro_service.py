# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import pytest
import requests

from divi.backends import _qoro_service
from divi.backends._qoro_service import (
    JobStatus,
    JobType,
    MaxRetriesReachedError,
    QoroService,
    _raise_with_details,
    is_valid_qasm,
)
from divi.backends._qpu_system import QPUSystem

# --- Test Fixtures ---


@pytest.fixture
def qoro_service(api_key):
    """Provides a QoroService instance with a real API token for integration tests."""
    return QoroService(api_key)


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
        """Tests submitting a single chunk of circuits using the new init/add flow."""
        mock_init_response = mocker.MagicMock()
        mock_init_response.status_code = HTTPStatus.CREATED
        mock_init_response.json.return_value = {"job_id": "mock_job_id"}

        mock_add_response = mocker.MagicMock()
        mock_add_response.status_code = HTTPStatus.OK

        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )

        job_id = qoro_service_mock.submit_circuits({"circuit_1": "mock_qasm"})

        assert job_id == "mock_job_id"
        assert mock_make_request.call_count == 2
        # Check init call
        mock_make_request.call_args_list[0].assert_called_with(
            "post", "job/init/", json=mocker.ANY, timeout=100
        )
        # Check add_circuits call
        add_circuits_call = mock_make_request.call_args_list[1]
        assert add_circuits_call.args == ("post", "job/mock_job_id/add_circuits/")
        assert add_circuits_call.kwargs["json"]["finalized"] == "true"

    def test_submit_circuits_multiple_chunks_mock(self, mocker, qoro_service_mock):
        """Tests submitting multiple chunks to a single job."""
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )
        # Set a small payload size to force chunking
        mocker.patch.object(
            _qoro_service, "_MAX_PAYLOAD_SIZE_MB", new=60.0 / 1024 / 1024
        )

        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "single_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)

        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response, mock_add_response],
        )

        job_id = qoro_service_mock.submit_circuits(
            {"circuit_1": "mock_qasm", "circuit_2": "mock_qasm"}
        )

        assert job_id == "single_job_id"
        assert mock_make_request.call_count == 3  # 1 for init, 2 for add_circuits

        # Check that the first add_circuits call is not finalized
        first_add_payload = mock_make_request.call_args_list[1].kwargs["json"]
        assert first_add_payload["finalized"] == "false"

        # Check that the second (last) add_circuits call is finalized
        second_add_payload = mock_make_request.call_args_list[2].kwargs["json"]
        assert second_add_payload["finalized"] == "true"

    def test_submit_circuits_invalid_qasm_mock(self, mocker, qoro_service_mock):
        """Tests that submitting an invalid QASM string raises a ValueError."""
        mocker.patch(
            f"{_qoro_service.__name__}.{is_valid_qasm.__name__}", return_value=False
        )
        with pytest.raises(ValueError, match="Circuit 'circuit_1' is not a valid QASM"):
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
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        qoro_service_mock.submit_circuits(
            {"c1": "qasm"}, tag="my_custom_tag", job_type=JobType.EXECUTE
        )

        # The parameters should be in the first (init) call
        _, called_kwargs = mock_make_request.call_args_list[0]
        payload = called_kwargs.get("json", {})
        assert payload.get("tag") == "my_custom_tag"
        assert payload.get("job_type") == JobType.EXECUTE.value

    def test_submit_circuits_api_error_mock(self, mocker, qoro_service_mock):
        """Tests API error handling during circuit submission."""
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )
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
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )
        qoro_service_mock.use_circuit_packing = original_flag

        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response]
            * 2,  # For two calls below
        )

        circuits = {"circuit_1": "mock_qasm"}

        # Test overriding to True
        qoro_service_mock.submit_circuits(circuits, override_circuit_packing=True)
        _, called_kwargs = mock_make_request.call_args_list[0]
        assert called_kwargs.get("json", {}).get("use_packing") is True

        # Test overriding to False
        qoro_service_mock.submit_circuits(circuits, override_circuit_packing=False)
        _, called_kwargs = mock_make_request.call_args_list[2]  # 3rd call overall
        assert called_kwargs.get("json", {}).get("use_packing") is False

    def test_submit_circuits_add_circuits_fails_mock(self, mocker, qoro_service_mock):
        """Tests that an error during the 'add_circuits' step is handled."""
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=True
        )
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )

        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[
                mock_init_response,
                requests.exceptions.HTTPError("API Error: 500"),
            ],
        )

        with pytest.raises(requests.exceptions.HTTPError, match="API Error: 500"):
            qoro_service_mock.submit_circuits({"c1": "qasm"})

    def test_raise_with_details_json_body(self, mocker):
        """
        Tests that _raise_with_details formats the error message correctly
        when the response body is valid JSON.
        """
        mock_response = mocker.MagicMock()
        mock_response.status_code = 400
        mock_response.reason = "Bad Request"
        mock_response.json.return_value = {"error": "Invalid input", "code": 123}

        expected_msg = '400 Bad Request: {"error": "Invalid input", "code": 123}'
        with pytest.raises(requests.HTTPError, match=expected_msg):
            _raise_with_details(mock_response)

    def test_raise_with_details_text_body(self, mocker):
        """
        Tests that _raise_with_details falls back to using the raw text body
        when the response body is not valid JSON.
        """
        mock_response = mocker.MagicMock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.text = "A fatal server error occurred."
        mock_response.json.side_effect = ValueError

        expected_msg = "500 Internal Server Error: A fatal server error occurred."
        with pytest.raises(requests.HTTPError, match=expected_msg):
            _raise_with_details(mock_response)

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
        mocker.patch(
            "divi.backends._qoro_service._decode_qh1_b64",
            return_value={"decoded": "data"},
        )
        mock_json = {
            "results": [
                {"label": "circuit_0", "results": {"encoding": "qh1", "payload": "..."}}
            ]
        }
        mock_response = mocker.MagicMock(status_code=200, json=lambda: mock_json)
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        results = qoro_service_mock.get_job_results("job_1")

        expected = [{"label": "circuit_0", "results": {"decoded": "data"}}]
        assert results == expected

    def test_get_job_results_empty_results_mock(self, mocker, qoro_service_mock):
        """Tests fetching a job that completed with an empty results list."""
        mock_json = {"results": []}
        mock_response = mocker.MagicMock(status_code=200, json=lambda: mock_json)
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )
        mock_decode = mocker.patch("divi.backends._qoro_service._decode_qh1_b64")

        results = qoro_service_mock.get_job_results("job_1")

        assert results == []
        mock_decode.assert_not_called()

    def test_get_job_results_decoding_error_mock(self, mocker, qoro_service_mock):
        """Tests that an error during decoding propagates."""
        mocker.patch(
            "divi.backends._qoro_service._decode_qh1_b64",
            side_effect=ValueError("corrupt stream"),
        )
        mock_json = {
            "results": [
                {"label": "circuit_0", "results": {"encoding": "qh1", "payload": "..."}}
            ]
        }
        mock_response = mocker.MagicMock(status_code=200, json=lambda: mock_json)
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        with pytest.raises(ValueError, match="corrupt stream"):
            qoro_service_mock.get_job_results("job_1")

    def test_get_job_results_still_running_mock(self, mocker, qoro_service_mock):
        """Tests handling of a 'still running' job."""
        # Create a mock response object to attach to the error
        mock_response = mocker.MagicMock()
        mock_response.status_code = 400

        # Create an HTTPError instance with the response attached
        http_error = requests.exceptions.HTTPError(response=mock_response)

        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=http_error,  # Use side_effect to raise the error
        )

        with pytest.raises(requests.exceptions.HTTPError, match="400 Bad Request"):
            qoro_service_mock.get_job_results("job_1")

    def test_get_job_results_api_error_mock(self, mocker, qoro_service_mock):
        """Tests API error handling when fetching job results."""
        # Create a mock response with a different error code (e.g., 404)
        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://mock.url"

        # Create an HTTPError that includes the response
        http_error = requests.exceptions.HTTPError(
            "API Error: 404 Not Found for URL http://mock.url",
            response=mock_response,
        )

        mocker.patch.object(qoro_service_mock, "_make_request", side_effect=http_error)

        with pytest.raises(requests.exceptions.HTTPError, match="API Error: 404"):
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
        callback_mock.assert_called_once_with(mock_response_completed)

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
            "job_1", loop_until_complete=True, poll_callback=pbar_mock, verbose=True
        )

        assert pbar_mock.call_count == 2
        pbar_mock.assert_has_calls(
            [
                mocker.call.__bool__(),
                mocker.call(1, "PENDING"),
                mocker.call(2, "PENDING"),
            ]
        )


# --- Integration Tests (require API key) ---


class TestQoroServiceWithApiKey:
    """Integration tests for the QoroService, requiring a valid API key."""

    def test_service_connection_test(self, qoro_service):
        """Tests the connection to the live service."""
        response = qoro_service.test_connection()
        assert response.status_code == 200, "Connection should be successful"

    def test_submit_and_delete_circuits(self, qoro_service, circuits):
        """Tests submitting and then deleting circuits."""
        job_id = qoro_service.submit_circuits(circuits)
        assert isinstance(job_id, str), "Job ID should be a string"

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

    def test_fetch_qpu_systems(self, qoro_service):
        """Tests fetching the list of QPU systems."""
        systems = qoro_service.fetch_qpu_systems()
        assert isinstance(systems, list)
        if systems:
            assert isinstance(systems[0], QPUSystem)

    def test_get_job_results(self, qoro_service, circuits):
        """Tests submitting a job, polling until complete, and fetching results."""
        # Use only one circuit for a quicker test
        single_circuit = {"circuit_1": circuits["circuit_0"]}
        job_id = qoro_service.submit_circuits(single_circuit)

        # Poll for completion
        status = qoro_service.poll_job_status(job_id, loop_until_complete=True)
        assert status == JobStatus.COMPLETED

        # Fetch results
        results = qoro_service.get_job_results(job_id)
        assert isinstance(results, list)
        assert len(results) == 1
        assert "label" in results[0]
        assert "results" in results[0]
        assert isinstance(results[0]["results"], dict)

        # Cleanup
        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"
