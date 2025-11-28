# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends._results_processing import (
    _decode_qh1_b64,
    _decompress_histogram,
    _int_to_bitstr,
    _rle_bool_decode,
    _uleb128_decode,
)


class TestQoroServiceUtilities:
    """
    Test suite for QoroService utility functions for histogram decompression.
    This suite has been corrected to remove misleading tests and add proper
    validation for both success and failure paths.
    """

    # --- Top-level Wrapper Function Tests ---

    def test_decode_qh1_b64_empty_or_no_payload(self):
        """Tests that _decode_qh1_b64 handles empty inputs correctly."""
        assert _decode_qh1_b64(None) is None
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
            "divi.backends._results_processing._decompress_histogram"
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
