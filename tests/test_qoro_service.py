# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

import pytest
import requests

from divi.backends import _qoro_service
from divi.backends._qoro_service import (
    JobConfig,
    JobStatus,
    JobType,
    MaxRetriesReachedError,
    QoroService,
    _decode_qh1_b64,
    _decompress_histogram,
    _int_to_bitstr,
    _raise_with_details,
    _rle_bool_decode,
    _uleb128_decode,
    is_valid_qasm,
)
from divi.backends._qpu_system import (
    QPUSystem,
    get_available_qpu_systems,
    get_qpu_system,
    parse_qpu_systems,
    update_qpu_systems_cache,
)

# --- Test Fixtures ---


@pytest.fixture
def qoro_service(api_key):
    """Provides a QoroService instance with a real API token for integration tests."""
    return QoroService(auth_token=api_key)


@pytest.fixture
def qoro_service_mock():
    """Provides a mocked QoroService instance for unit tests."""
    return QoroService(auth_token="mock_token", max_retries=3, polling_interval=0.01)


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


class TestQoroServiceUtilities:
    """
    Test suite for QoroService utility functions for histogram decompression.
    This suite has been corrected to remove misleading tests and add proper
    validation for both success and failure paths.
    """

    # --- Top-level Wrapper Function Tests ---

    def test_decode_qh1_b64_empty_or_no_payload(self):
        """Tests that _decode_qh1_b64 handles empty inputs correctly."""
        assert _decode_qh1_b64(None) == None
        assert _decode_qh1_b64({}) == {}
        assert _decode_qh1_b64({"encoding": "qh1", "payload": ""}) == {
            "encoding": "qh1",
            "payload": "",
        }

    def test_decode_qh1_b64_unsupported_encoding(self):
        """Tests that _decode_qh1_b64 raises an error for unsupported encodings."""
        with pytest.raises(ValueError, match="Unsupported encoding: invalid"):
            _decode_qh1_b64({"encoding": "invalid", "payload": "dGVzdA=="})

    def test_decode_qh1_b64_delegates_correctly(self, mocker):
        """Tests that _decode_qh1_b64 correctly decodes and calls the decompressor."""
        mock_decompress = mocker.patch(
            "divi.backends._qoro_service._decompress_histogram"
        )
        mock_decompress.return_value = {"01": 100}

        # "test" -> base64 -> "dGVzdA=="
        encoded_data = {"encoding": "qh1", "payload": "dGVzdA=="}
        result = _decode_qh1_b64(encoded_data)

        # Assert it passed the correctly decoded bytes to the decompressor
        mock_decompress.assert_called_once_with(b"test")
        assert result == {"01": 100}

    # --- Core Decompression Logic Tests ---

    def test_decompress_histogram_empty_buffer(self):
        """Tests that an empty byte buffer returns an empty histogram."""
        assert _decompress_histogram(b"") == {}

    def test_decompress_histogram_bad_magic(self):
        """Tests that a payload with an invalid magic header raises a ValueError."""
        with pytest.raises(ValueError, match="bad magic"):
            _decompress_histogram(b"INVALID_MAGIC")

    def test_decompress_histogram_successful(self):
        """
        ⭐ Tests the entire decompression 'happy path' with a valid, non-trivial
        QH1 payload. This is the most critical test for the decoder.
        """
        # This payload encodes the histogram: {"001": 1, "101": 3, "111": 1}
        # Details: n_bits=3, unique=3, total_shots=5
        # Gaps between sorted indices [1, 5, 7] are [1, 4, 2]
        # Counts [1, 3, 1] are RLE-encoded.
        valid_payload = (
            b"QH1"  # Magic header
            b"\x03"  # n_bits = 3
            b"\x03"  # unique = 3
            b"\x05"  # total_shots = 5
            b"\x03"  # num_gaps = 3
            b"\x01\x04\x02"  # gaps data
            b"\x05"  # RLE data length
            b"\x03\x01\x01\x01"  # RLE data for [True, False, True]
            b"\x01"  # extras_len = 1
            b"\x01" + b"\x01"  # extras data for count=3
        )

        expected_histogram = {"001": 1, "101": 3, "111": 1}
        result = _decompress_histogram(valid_payload)
        assert result == expected_histogram

    def test_decompress_histogram_shot_sum_mismatch_error(self):
        """Tests that a corrupt stream with a shot sum mismatch raises a ValueError."""
        # This payload is now correctly formed but has an invalid `total_shots` value.
        # The RLE data is b"\x03\x01\x01\x01\x01" (num_runs=3, first_val=T, len1=1, len2=1, len3=1)
        # The length of this RLE data is 5 bytes, so rb_len is b"\x05".
        invalid_payload = (
            b"QH1"  # Magic header
            b"\x03"  # n_bits = 3
            b"\x03"  # unique = 3
            b"\x0a"  # total_shots = 10 (INCORRECT)
            b"\x03"  # num_gaps = 3
            b"\x01\x04\x02"  # gaps data
            b"\x05"  # rb_len = 5
            b"\x03\x01\x01\x01\x01"  # Correct RLE data
            b"\x01"  # extras_len = 1
            b"\x01"  # extras data
        )
        with pytest.raises(ValueError, match="corrupt stream: shot sum mismatch"):
            _decompress_histogram(invalid_payload)

    def test_decompress_histogram_unique_mismatch_error(self):
        """Tests that a corrupt stream with a unique count mismatch raises a ValueError."""
        # This payload is correctly formed but has an invalid `unique` value.
        # The RLE data is b"\x03\x01\x01\x01\x01" (num_runs=3, first_val=T, len1=1, len2=1, len3=1)
        # The length of this RLE data is 5 bytes, so rb_len is b"\x05".
        invalid_payload = (
            b"QH1"  # Magic header
            b"\x03"  # n_bits = 3
            b"\x02"  # unique = 2 (INCORRECT)
            b"\x05"  # total_shots = 5
            b"\x03"  # num_gaps = 3
            b"\x01\x04\x02"  # gaps data
            b"\x05"  # rb_len = 5
            b"\x03\x01\x01\x01\x01"  # Correct RLE data
            b"\x01"  # extras_len = 1
            b"\x01"  # extras data
        )

        with pytest.raises(ValueError, match="corrupt stream: unique mismatch"):
            _decompress_histogram(invalid_payload)

    # --- Low-level Utility Function Tests ---

    def test_uleb128_decode(self):
        """Tests ULEB128 decoding for single-byte, multi-byte, and truncated inputs."""
        # Single-byte value
        val, pos = _uleb128_decode(b"\x05", 0)
        assert val == 5 and pos == 1

        # Multi-byte value (128)
        val, pos = _uleb128_decode(b"\x80\x01", 0)
        assert val == 128 and pos == 2

        # Decoding with an offset
        val, pos = _uleb128_decode(b"\x00\x05", 1)
        assert val == 5 and pos == 2

        # Truncated varint raises an error
        with pytest.raises(ValueError, match="truncated varint"):
            _uleb128_decode(b"\x80")

    def test_rle_bool_decode(self):
        """Tests RLE boolean decoding for zero, single, and multiple runs."""
        # Zero runs
        result, pos = _rle_bool_decode(b"\x00")
        assert result == [] and pos == 1

        # Multiple runs: decodes to [True, False, False]
        # num_runs=2, first_val=True, len1=1, len2=2
        data = b"\x02\x01\x01\x02"
        result, pos = _rle_bool_decode(data)
        assert result == [True, False, False]
        assert pos == 4

    def test_int_to_bitstr(self):
        """Tests integer to bitstring conversion with zero-padding."""
        assert _int_to_bitstr(5, 4) == "0101"
        assert _int_to_bitstr(1, 2) == "01"
        assert _int_to_bitstr(7, 3) == "111"


class TestQoroServiceMock:
    """Test suite for QoroService with mocked dependencies."""

    # --- Tests for initialization ---

    def test_initialization_without_api_key_and_no_env_file(self, mocker):
        """Test initialization without API key and no .env file."""
        # Mock dotenv_values to raise KeyError (simulating missing QORO_API_KEY)
        mock_dotenv = mocker.patch("divi.backends._qoro_service.dotenv_values")
        mock_dotenv.return_value = {}  # Empty env file

        with pytest.raises(
            ValueError, match="Qoro API key not provided nor found in a .env file"
        ):
            QoroService(auth_token=None)

    def test_initialization_with_env_api_key(self, mocker):
        """Test initialization with API key from .env file."""
        # Mock dotenv_values to return API key
        mock_dotenv = mocker.patch("divi.backends._qoro_service.dotenv_values")
        mock_dotenv.return_value = {"QORO_API_KEY": "env_api_key"}

        service = QoroService(auth_token=None)
        assert service.auth_token == "Bearer env_api_key"

    @pytest.mark.parametrize(
        "input_value, expected_stored_value",
        [
            ("my_qpu_system", QPUSystem(name="my_qpu_system")),
            (
                QPUSystem(name="qpu_from_object"),
                QPUSystem(name="qpu_from_object"),
            ),
            (None, None),
        ],
        ids=["string_input", "QPUSystem_object_input", "None_input"],
    )
    def test_job_config_qpu_system_success(
        self, input_value, expected_stored_value, mocker
    ):
        """
        Tests that JobConfig correctly handles valid qpu_system types.
        """
        mocker.patch(
            "divi.backends._qoro_service.get_qpu_system",
            return_value=QPUSystem(name="my_qpu_system"),
        )
        config = JobConfig(qpu_system=input_value)
        assert config.qpu_system == expected_stored_value

    @pytest.mark.parametrize(
        "invalid_input",
        [123, ["a", "list"], {"a": "dict"}],
        ids=["integer_input", "list_input", "dict_input"],
    )
    def test_job_config_qpu_system_failure(self, invalid_input):
        """
        Tests that JobConfig raises a TypeError for invalid qpu_system types.
        """
        with pytest.raises(TypeError):
            JobConfig(qpu_system=invalid_input)

    def test_job_config_shots_validation(self):
        """Tests that JobConfig validates shots values."""
        # Valid shots
        config = JobConfig(shots=100)
        assert config.shots == 100

        # Invalid: shots <= 0
        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            JobConfig(shots=0)

        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            JobConfig(shots=-1)

    def test_job_config_use_circuit_packing_type_validation(self):
        """Tests that JobConfig validates use_circuit_packing type."""
        # Valid boolean
        config = JobConfig(use_circuit_packing=True)
        assert config.use_circuit_packing is True

        # Invalid: not a bool
        with pytest.raises(TypeError, match="Expected a bool"):
            JobConfig(use_circuit_packing="true")

        with pytest.raises(TypeError, match="Expected a bool"):
            JobConfig(use_circuit_packing=1)

    def test_job_config_override_basic(self):
        """Tests that JobConfig.override() method works correctly."""
        base = JobConfig(shots=1000, tag="base", use_circuit_packing=False)
        override = JobConfig(shots=500, tag="override")

        result = base.override(override)
        assert result.shots == 500
        assert result.tag == "override"
        assert result.use_circuit_packing is False  # Not overridden

    def test_job_config_override_none_values_ignored(self):
        """Tests that None values in override config are ignored."""
        base = JobConfig(
            shots=1000, tag="base", qpu_system=QPUSystem(name="qoro_maestro")
        )
        override = JobConfig(shots=None, tag="override", qpu_system=None)

        result = base.override(override)
        assert result.shots == 1000  # Not overridden
        assert result.tag == "override"  # Overridden
        assert result.qpu_system == QPUSystem(name="qoro_maestro")  # Not overridden

    def test_job_config_override_immutability(self):
        """Tests that JobConfig.override() returns a new instance."""
        base = JobConfig(shots=1000)
        override = JobConfig(shots=500)

        result = base.override(override)

        # Original should be unchanged
        assert base.shots == 1000
        # Result should have new value
        assert result.shots == 500
        # Should be different objects
        assert result is not base
        assert result is not override

    def test_job_config_override_all_fields(self, mocker):
        """Tests overriding all fields."""
        base = JobConfig(
            shots=1000,
            tag="base",
            qpu_system=QPUSystem(name="system1"),
            use_circuit_packing=False,
        )

        # Populate the cache before creating the JobConfig that uses a string
        update_qpu_systems_cache([QPUSystem(name="system2")])

        override = JobConfig(
            shots=2000,
            tag="override",
            qpu_system="system2",
            use_circuit_packing=True,
        )

        result = base.override(override)
        assert result.shots == 2000
        assert result.tag == "override"
        assert result.qpu_system == QPUSystem(name="system2")
        assert result.use_circuit_packing is True

    # --- Tests for core functionality ---

    def test_make_request_comprehensive(self, mocker):
        """Test _make_request functionality."""

        service = QoroService(auth_token="test_token")

        # Test 1: Successful GET request
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://app.qoroquantum.net/api/test"

        mock_request = mocker.patch(
            "requests.Session.request", return_value=mock_response
        )

        response = service._make_request("get", "test")

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "get"  # First positional argument is method
        assert (
            call_args[0][1] == "https://app.qoroquantum.net/api/test"
        )  # Second is URL
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_token"
        assert "Content-Type" not in call_args[1]["headers"]

        # Test 2: POST request with Content-Type header
        mock_request.reset_mock()
        service._make_request("post", "test", json={"data": "test"})

        call_args = mock_request.call_args
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

        # Test 3: Custom headers override
        mock_request.reset_mock()
        service._make_request("get", "test", headers={"Custom": "Header"})

        call_args = mock_request.call_args
        assert call_args[1]["headers"]["Custom"] == "Header"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_token"

        # Test 4: HTTP error handling
        mock_response_error = mocker.MagicMock()
        mock_response_error.status_code = 400
        mock_response_error.reason = "Bad Request"
        mock_response_error.url = "https://app.qoroquantum.net/api/test"

        mock_request.return_value = mock_response_error

        with pytest.raises(
            requests.exceptions.HTTPError, match="API Error: 400 Bad Request"
        ):
            service._make_request("get", "test")

    def test_compress_data_and_split_circuits(self, mocker):
        """Test _compress_data and _split_circuits functionality."""

        service = QoroService(auth_token="test_token")

        # Test _compress_data
        compressed = service._compress_data("test circuit")
        assert isinstance(compressed, str)
        assert len(compressed) > 0

        # Test _split_circuits with small payload
        circuits = {
            "circuit1": 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;',
            "circuit2": 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; h q[0]; measure q -> c;',
        }

        chunks = service._split_circuits(circuits)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, dict) for chunk in chunks)

        # Test _split_circuits with large payload (force chunking)
        mocker.patch(
            "divi.backends._qoro_service._MAX_PAYLOAD_SIZE_MB", new=0.0001
        )  # Very small limit

        large_circuits = {
            f"circuit{i}": 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
            * 50
            for i in range(3)
        }

        chunks = service._split_circuits(large_circuits)
        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_fetch_qpu_systems(self, mocker):
        """Test fetch_qpu_systems and parse_qpu_systems."""

        service = QoroService(auth_token="test_token")

        # Test parse_qpu_systems
        mock_json_data = [
            {
                "name": "test_qpu",
                "qpus": [
                    {
                        "nickname": "qpu1",
                        "q_bits": 5,
                        "status": "active",
                        "system_kind": "superconducting",
                    },
                    {
                        "nickname": "qpu2",
                        "q_bits": 7,
                        "status": "active",
                        "system_kind": "trapped_ion",
                    },
                ],
                "access_level": "basic",
            }
        ]

        qpu_systems = parse_qpu_systems(mock_json_data)
        assert len(qpu_systems) == 1
        assert qpu_systems[0].name == "test_qpu"
        assert len(qpu_systems[0].qpus) == 2

        # Test fetch_qpu_systems
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = mock_json_data
        mocker.patch.object(service, "_make_request", return_value=mock_response)
        mock_update_cache = mocker.patch.object(
            _qoro_service, "update_qpu_systems_cache"
        )

        result = service.fetch_qpu_systems()
        assert len(result) == 1
        assert result[0].name == "test_qpu"
        mock_update_cache.assert_called_once_with(result)

    def test_update_qpu_systems_cache(self):
        """Test that the cache is correctly updated, including special handling."""
        systems = [
            QPUSystem(name="system1"),
            QPUSystem(name="qoro_maestro", supports_expval=False),
        ]
        update_qpu_systems_cache(systems)

        cached_systems = get_available_qpu_systems()
        assert len(cached_systems) == 2

        # Verify that `supports_expval` was correctly updated for `qoro_maestro`
        maestro = get_qpu_system("qoro_maestro")
        assert maestro.supports_expval is True

        # Verify that other systems were not affected
        system1 = get_qpu_system("system1")
        assert system1.supports_expval is False

    def test_get_qpu_system(self):
        """Test get_qpu_system functionality with caching."""
        # Test 1: Cache is empty
        update_qpu_systems_cache([])
        with pytest.raises(ValueError, match="QPU systems cache is empty"):
            get_qpu_system("system1")

        # Test 2: Cache is populated, system found
        mock_systems = [
            QPUSystem(name="system1"),
            QPUSystem(name="system2"),
        ]
        update_qpu_systems_cache(mock_systems)
        system = get_qpu_system("system1")
        assert system.name == "system1"

        # Test 3: Cache is populated, system not found
        with pytest.raises(ValueError, match="QPUSystem with name 'system3' not found"):
            get_qpu_system("system3")

    def test_poll_job_status_comprehensive(self, mocker):
        """Test poll_job_status functionality."""

        service = QoroService("test_token", max_retries=3, polling_interval=0.01)

        # Test 1: Single status check
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"status": "RUNNING"}

        mocker.patch.object(service, "_make_request", return_value=mock_response)

        status = service.poll_job_status("test_job")
        assert status == "RUNNING"

        # Test 2: Loop until completed
        mock_responses = [
            mocker.MagicMock(json=lambda: {"status": "RUNNING"}),
            mocker.MagicMock(json=lambda: {"status": "RUNNING"}),
            mocker.MagicMock(json=lambda: {"status": "COMPLETED"}),
        ]

        mocker.patch.object(service, "_make_request", side_effect=mock_responses)

        on_complete_callback = mocker.MagicMock()
        status = service.poll_job_status(
            "test_job", loop_until_complete=True, on_complete=on_complete_callback
        )

        assert status == JobStatus.COMPLETED
        on_complete_callback.assert_called_once()

        # Test 3: Loop until failed
        mock_responses = [
            mocker.MagicMock(json=lambda: {"status": "RUNNING"}),
            mocker.MagicMock(json=lambda: {"status": "FAILED"}),
        ]

        mocker.patch.object(service, "_make_request", side_effect=mock_responses)

        on_complete_callback = mocker.MagicMock()
        status = service.poll_job_status(
            "test_job", loop_until_complete=True, on_complete=on_complete_callback
        )

        assert status == JobStatus.FAILED
        on_complete_callback.assert_called_once()

        # Test 4: Max retries reached
        mock_responses = [mocker.MagicMock(json=lambda: {"status": "RUNNING"})] * 4

        mocker.patch.object(service, "_make_request", side_effect=mock_responses)

        with pytest.raises(MaxRetriesReachedError):
            service.poll_job_status("test_job", loop_until_complete=True)

        # Test 5: Poll callback functionality
        mock_responses = [
            mocker.MagicMock(json=lambda: {"status": "RUNNING"}),
            mocker.MagicMock(json=lambda: {"status": "COMPLETED"}),
        ]

        mocker.patch.object(service, "_make_request", side_effect=mock_responses)

        poll_callback = mocker.MagicMock()
        status = service.poll_job_status(
            "test_job", loop_until_complete=True, poll_callback=poll_callback
        )

        assert status == JobStatus.COMPLETED
        poll_callback.assert_called()

    def test_get_job_results_error_handling(self, mocker):
        """Test get_job_results error handling."""
        from divi.backends._qoro_service import QoroService

        service = QoroService(auth_token="test_token")

        # Test 400 Bad Request handling
        mock_response = mocker.MagicMock()
        mock_response.status_code = 400

        mock_error = requests.exceptions.HTTPError("400 Bad Request")
        mock_error.response = mock_response

        mocker.patch.object(service, "_make_request", side_effect=mock_error)

        with pytest.raises(
            requests.exceptions.HTTPError,
            match="Job results not available, likely job is still running",
        ):
            service.get_job_results("test_job")

    # --- Tests for error handling ---

    def test_raise_with_details_json_response(self, mocker):
        """Test _raise_with_details with JSON response."""
        # Create a mock response with JSON data
        mock_response = mocker.MagicMock()
        mock_response.status_code = 400
        mock_response.reason = "Bad Request"
        mock_response.json.return_value = {"error": "Invalid circuit", "code": 400}

        with pytest.raises(
            requests.exceptions.HTTPError, match="400 Bad Request: .*Invalid circuit.*"
        ):
            _raise_with_details(mock_response)

    def test_raise_with_details_text_response(self, mocker):
        """Test _raise_with_details with text response."""
        # Create a mock response that fails JSON parsing
        mock_response = mocker.MagicMock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Internal server error occurred"

        with pytest.raises(
            requests.exceptions.HTTPError,
            match="500 Internal Server Error: Internal server error occurred",
        ):
            _raise_with_details(mock_response)

    # --- Tests for test_connection ---

    def test_fail_submit_circuits(self, circuits):
        """Tests that submitting circuits with an invalid token raises an HTTPError."""
        service = QoroService(auth_token="invalid_token")
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

    def test_submit_circuits_single_chunk(self, mocker, qoro_service_mock):
        """Test submitting circuits in a single chunk."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
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

    def test_submit_circuits_multiple_chunks(self, mocker, qoro_service_mock):
        """Test submitting circuits that are split into multiple chunks."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
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

    def test_submit_circuits_with_custom_options(self, mocker, qoro_service_mock):
        """Test submitting circuits with custom tag and job type."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
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
            {"c1": "qasm"},
            job_type=JobType.EXECUTE,
            override_config=JobConfig(tag="my_custom_tag"),
        )

        # The parameters should be in the first (init) call
        _, called_kwargs = mock_make_request.call_args_list[0]
        payload = called_kwargs.get("json", {})
        assert payload.get("tag") == "my_custom_tag"
        assert payload.get("job_type") == JobType.EXECUTE.value

    def test_submit_circuits_with_packing_override(self, mocker, qoro_service_mock):
        """Test submitting circuits with circuit packing override."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        circuits = {"circuit_1": "mock_qasm"}
        qoro_service_mock.submit_circuits(
            circuits, override_config=JobConfig(use_circuit_packing=True)
        )
        _, called_kwargs = mock_make_request.call_args_list[0]
        assert called_kwargs.get("json", {}).get("use_packing") is True

    def test_submit_circuits_with_config_override(self, mocker):
        """Verify that override_config merges with service config and is used consistently."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)

        # Create a service with default config
        service_with_default = QoroService(
            auth_token="test_token", max_retries=3, polling_interval=0.01
        )

        mock_make_request = mocker.patch.object(
            service_with_default,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        # Override shots in submit_circuits
        service_with_default.submit_circuits(
            {"circuit_1": "mock_qasm"}, override_config=JobConfig(shots=2000)
        )

        # Verify init payload is minimal (no shots/ham_ops)
        _, init_kwargs = mock_make_request.call_args_list[0]
        assert "shots" not in init_kwargs.get("json", {})
        assert "observables" not in init_kwargs.get("json", {})

        # Verify add_circuits payload includes shots
        _, add_kwargs = mock_make_request.call_args_list[1]
        assert add_kwargs.get("json", {}).get("shots") == 2000

    def test_submit_circuits_with_expectation_value(self, mocker, qoro_service_mock):
        """Test submitting circuits with expectation value job type and ham_ops."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        # Submit with expectation value
        circuits = {"circuit_1": "mock_qasm"}
        ham_ops = "XII;ZII"
        qoro_service_mock.submit_circuits(
            circuits, ham_ops=ham_ops, job_type=JobType.EXPECTATION
        )

        # Verify init payload is minimal (no shots/observables)
        _, init_kwargs = mock_make_request.call_args_list[0]
        assert "shots" not in init_kwargs.get("json", {})
        assert "observables" not in init_kwargs.get("json", {})

        # Verify add_circuits payload includes observables
        _, add_kwargs = mock_make_request.call_args_list[1]
        assert add_kwargs.get("json", {}).get("observables") == ham_ops
        assert "shots" not in add_kwargs.get("json", {})

    @pytest.mark.parametrize(
        "ham_ops, error_msg",
        [
            ("XX;YYY", "All Hamiltonian operators must have the same length"),
            (
                "XYZ;ABC",
                "Hamiltonian operators must contain only I, X, Y, Z characters",
            ),
        ],
    )
    def test_submit_circuits_ham_ops_validation_errors(
        self, mocker, qoro_service_mock, ham_ops, error_msg
    ):
        """Test ham_ops validation for various invalid inputs."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        with pytest.raises(ValueError, match=error_msg):
            qoro_service_mock.submit_circuits(
                {"c1": "qasm"}, ham_ops=ham_ops, job_type=JobType.EXPECTATION
            )

    def test_submit_circuits_ham_ops_with_non_expectation_error(
        self, mocker, qoro_service_mock
    ):
        """Test that ham_ops with non-EXPECTATION job_type issues an error."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        # Should error when ham_ops is used with SIMULATE job
        with pytest.raises(
            ValueError,
            match="Hamiltonian operators are only supported for EXPECTATION job type.",
        ):
            qoro_service_mock.submit_circuits(
                {"c1": "qasm"}, ham_ops="XII", job_type=JobType.SIMULATE
            )

    def test_submit_circuits_ham_ops_auto_infers_expectation_job_type(
        self, mocker, qoro_service_mock
    ):
        """Test that job_type is automatically set to EXPECTATION when ham_ops is provided without job_type."""
        mocker.patch(f"{_qoro_service.__name__}.is_valid_qasm", return_value=2)
        mock_init_response = mocker.MagicMock(
            status_code=HTTPStatus.CREATED, json=lambda: {"job_id": "mock_job_id"}
        )
        mock_add_response = mocker.MagicMock(status_code=HTTPStatus.OK)
        mock_make_request = mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=[mock_init_response, mock_add_response],
        )

        # Submit with ham_ops but without specifying job_type (should auto-infer EXPECTATION)
        circuits = {"circuit_1": "mock_qasm"}
        ham_ops = "XII;ZII"
        qoro_service_mock.submit_circuits(circuits, ham_ops=ham_ops, job_type=None)

        # Verify init payload has job_type set to EXPECTATION
        _, init_kwargs = mock_make_request.call_args_list[0]
        assert init_kwargs.get("json", {}).get("job_type") == JobType.EXPECTATION.value

        # Verify add_circuits payload includes observables
        _, add_kwargs = mock_make_request.call_args_list[1]
        assert add_kwargs.get("json", {}).get("observables") == ham_ops

    def test_submit_circuits_invalid_qasm_constraints_and_api_errors(
        self, mocker, qoro_service_mock
    ):
        """Test submit_circuits with invalid QASM, constraints, and API errors."""

        # Test 1: Invalid QASM
        mocker.patch(
            f"{_qoro_service.__name__}.{is_valid_qasm.__name__}",
            return_value="Invalid QASM syntax",
        )
        with pytest.raises(ValueError, match="Circuit 'circuit_1' is not a valid QASM"):
            qoro_service_mock.submit_circuits({"circuit_1": "invalid_qasm"})

        # Test 2: Circuit cut constraint
        mocker.patch(
            f"{_qoro_service.__name__}.{is_valid_qasm.__name__}", return_value=2
        )
        with pytest.raises(
            ValueError, match="Only one circuit allowed for circuit-cutting jobs."
        ):
            qoro_service_mock.submit_circuits(
                {"c1": "qasm1", "c2": "qasm2"}, job_type=JobType.CIRCUIT_CUT
            )

        # Test 3: API error during init
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=2
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

        # Test 4: API error during add_circuits
        mocker.patch(
            f"{is_valid_qasm.__module__}.{is_valid_qasm.__name__}", return_value=2
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

    def test_raise_with_details_json_and_text_response_formatting(self, mocker):
        """Test _raise_with_details JSON and text response formatting."""

        # Test 1: JSON response body
        mock_response_json = mocker.MagicMock()
        mock_response_json.status_code = 400
        mock_response_json.reason = "Bad Request"
        mock_response_json.json.return_value = {"error": "Invalid input", "code": 123}

        expected_msg = '400 Bad Request: {"error": "Invalid input", "code": 123}'
        with pytest.raises(requests.HTTPError, match=expected_msg):
            _raise_with_details(mock_response_json)

        # Test 2: Text response body (JSON parsing fails)
        mock_response_text = mocker.MagicMock()
        mock_response_text.status_code = 500
        mock_response_text.reason = "Internal Server Error"
        mock_response_text.text = "A fatal server error occurred."
        mock_response_text.json.side_effect = ValueError

        expected_msg = "500 Internal Server Error: A fatal server error occurred."
        with pytest.raises(requests.HTTPError, match=expected_msg):
            _raise_with_details(mock_response_text)

    # --- Tests for job management ---

    def test_delete_job_and_get_results_with_decoding(self, mocker, qoro_service_mock):
        """Test delete job and get results with decoding."""

        # Test 1: Delete job success
        mock_response = mocker.MagicMock(status_code=204)
        mock_make_request = mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response
        )

        response = qoro_service_mock.delete_job("job_1")

        mock_make_request.assert_called_once_with("delete", "job/job_1", timeout=50)
        assert response.status_code == 204

        # Test 2: Delete job API error
        mocker.patch.object(
            qoro_service_mock,
            "_make_request",
            side_effect=requests.exceptions.HTTPError(
                "API Error: 404 Not Found for URL http://mock.url"
            ),
        )

        with pytest.raises(requests.exceptions.HTTPError):
            qoro_service_mock.delete_job("job_1")

        # Test 3: Get job results success
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

        # Test 4: Get job results empty
        mock_json_empty = {"results": []}
        mock_response_empty = mocker.MagicMock(
            status_code=200, json=lambda: mock_json_empty
        )
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_empty
        )
        mock_decode = mocker.patch("divi.backends._qoro_service._decode_qh1_b64")

        results_empty = qoro_service_mock.get_job_results("job_1")

        assert results_empty == []
        mock_decode.assert_not_called()

        # Test 5: Get job results decoding error
        mocker.patch(
            "divi.backends._qoro_service._decode_qh1_b64",
            side_effect=ValueError("corrupt stream"),
        )
        mock_json_error = {
            "results": [
                {"label": "circuit_0", "results": {"encoding": "qh1", "payload": "..."}}
            ]
        }
        mock_response_error = mocker.MagicMock(
            status_code=200, json=lambda: mock_json_error
        )
        mocker.patch.object(
            qoro_service_mock, "_make_request", return_value=mock_response_error
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


@pytest.mark.requires_api_key
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

    def test_get_job_results_expectation_value(self, qoro_service, circuits):
        """Tests submitting an expectation value job and fetching results."""
        # Use only one circuit for a quicker test
        single_circuit = {"circuit_1": circuits["circuit_0"]}
        ham_ops = "ZIII;IZII;IIZI;IIIZ"
        job_id = qoro_service.submit_circuits(
            single_circuit, ham_ops=ham_ops, job_type=JobType.EXPECTATION
        )

        # Poll for completion
        status = qoro_service.poll_job_status(job_id, loop_until_complete=True)
        assert status == JobStatus.COMPLETED

        # Fetch results
        results = qoro_service.get_job_results(job_id)
        assert isinstance(results, list)
        assert len(results) == 1
        assert "label" in results[0]
        assert "results" in results[0]

        # For expectation value jobs, results should be a dict of {str: float}
        exp_values = results[0]["results"]
        assert isinstance(exp_values, dict)
        ham_terms = ham_ops.split(";")
        assert len(exp_values) == len(ham_terms)
        assert set(exp_values.keys()) == set(ham_terms)
        assert all(isinstance(val, float) for val in exp_values.values())

        # Cleanup
        res = qoro_service.delete_job(job_id)
        assert res.status_code == 204, "Deletion should be successful"
