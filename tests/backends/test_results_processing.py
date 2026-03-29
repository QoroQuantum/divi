# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from divi.backends._numba_kernels import _decompress_histogram_jit, _uleb128_decode_jit
from divi.backends._results_processing import (
    _decode_qh1_b64,
    _decompress_histogram,
    convert_counts_to_probs,
    reverse_dict_endianness,
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

    # --- JIT Kernel Tests ---

    def test_uleb128_decode_jit(self):
        """Tests JIT ULEB128 decoding for single-byte, multi-byte, and offset."""
        # Single-byte value
        val, pos = _uleb128_decode_jit(np.array([0x05], dtype=np.uint8), 0)
        assert val == 5 and pos == 1

        # Multi-byte value (128)
        val, pos = _uleb128_decode_jit(np.array([0x80, 0x01], dtype=np.uint8), 0)
        assert val == 128 and pos == 2

        # Decoding with an offset
        val, pos = _uleb128_decode_jit(np.array([0x00, 0x05], dtype=np.uint8), 1)
        assert val == 5 and pos == 2

    def test_decompress_histogram_jit_matches_wrapper(self):
        """Tests that the JIT kernel output matches the Python wrapper."""
        valid_payload = (
            b"QH1"
            b"\x03"  # n_bits = 3
            b"\x03"  # unique = 3
            b"\x05"  # total_shots = 5
            b"\x03"  # num_gaps = 3
            b"\x01\x04\x02"  # gaps
            b"\x05"  # RLE data length
            b"\x03\x01\x01\x01"  # RLE data
            b"\x01"  # extras_len = 1
            b"\x01" + b"\x01"  # extras
        )
        data = np.frombuffer(valid_payload, dtype=np.uint8)
        indices, counts, n_bits, unique, total_shots, n_decoded = (
            _decompress_histogram_jit(data)
        )

        assert n_bits == 3
        assert unique == 3
        assert total_shots == 5
        assert n_decoded == 3
        np.testing.assert_array_equal(indices, [1, 5, 7])
        np.testing.assert_array_equal(counts, [1, 3, 1])


class TestReverseDictEndianness:
    """Tests for reverse_dict_endianness."""

    def test_single_tag_single_bitstring(self):
        """Bitstrings are reversed within each tag."""
        result = reverse_dict_endianness({"tag0": {"100": 0.5, "011": 0.5}})
        assert result == {"tag0": {"001": 0.5, "110": 0.5}}

    def test_multiple_tags(self):
        """Each tag's bitstrings are reversed independently."""
        inp = {
            "a": {"10": 0.7, "01": 0.3},
            "b": {"11": 1.0},
        }
        result = reverse_dict_endianness(inp)
        assert result == {
            "a": {"01": 0.7, "10": 0.3},
            "b": {"11": 1.0},
        }

    def test_empty_outer_dict(self):
        """Empty input returns empty output."""
        assert reverse_dict_endianness({}) == {}

    def test_palindromic_bitstrings_unchanged(self):
        """Palindromic bitstrings are unaffected by reversal."""
        result = reverse_dict_endianness({"t": {"010": 0.4, "111": 0.6}})
        assert result == {"t": {"010": 0.4, "111": 0.6}}


class TestConvertCountsToProbs:
    """Tests for convert_counts_to_probs."""

    def test_basic_conversion(self):
        """Counts are divided by shots to produce probabilities."""
        counts = {"tag0": {"00": 30, "11": 70}}
        result = convert_counts_to_probs(counts, shots=100)
        assert result == {"tag0": {"00": 0.3, "11": 0.7}}

    def test_multiple_tags(self):
        """Each tag is converted independently."""
        counts = {
            "a": {"0": 5, "1": 5},
            "b": {"0": 2, "1": 8},
        }
        result = convert_counts_to_probs(counts, shots=10)
        assert result == {
            "a": {"0": 0.5, "1": 0.5},
            "b": {"0": 0.2, "1": 0.8},
        }

    def test_empty_dict(self):
        """Empty input returns empty output."""
        assert convert_counts_to_probs({}, shots=100) == {}

    def test_single_shot(self):
        """With shots=1, counts equal probabilities."""
        counts = {"t": {"101": 1}}
        result = convert_counts_to_probs(counts, shots=1)
        assert result == {"t": {"101": 1.0}}
