# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64

import numpy as np
import pytest

from divi.backends import _results_processing as results_processing
from divi.backends import convert_counts_to_probs, reverse_dict_endianness
from divi.backends._numba_kernels import (
    _decompress_histogram_jit,
    _uleb128_decode_limbs_jit,
)
from divi.backends._results_processing import (
    _MAX_HISTOGRAM_WIDTH,
    _decode_histogram_b64,
    _decompress_histogram,
    _limbs_to_bitstrings,
)
from tests.backends._helpers import build_qh_histogram, encode_uleb128


class TestQoroServiceUtilities:
    """
    Test suite for QoroService utility functions for histogram decompression.
    This suite has been corrected to remove misleading tests and add proper
    validation for both success and failure paths.
    """

    # --- Top-level Wrapper Function Tests ---

    def test_decode_histogram_b64_empty_or_no_payload(self):
        """Tests that _decode_histogram_b64 handles empty inputs correctly."""
        assert _decode_histogram_b64(None) is None
        assert _decode_histogram_b64({}) == {}
        assert _decode_histogram_b64({"encoding": "qh1", "payload": ""}) == {
            "encoding": "qh1",
            "payload": "",
        }

    def test_decode_histogram_b64_unsupported_encoding(self):
        """Tests that _decode_histogram_b64 raises an error for unsupported encodings."""
        with pytest.raises(ValueError, match="Unsupported encoding: invalid"):
            _decode_histogram_b64({"encoding": "invalid", "payload": "dGVzdA=="})

    @pytest.mark.parametrize(
        ("encoding", "magic", "n_bits"),
        [("qh1", b"QH1", 3), ("qh2", b"QH2", 256)],
    )
    def test_decode_histogram_b64_decodes_supported_envelopes(
        self, encoding, magic, n_bits
    ):
        """Base64 decoding and format dispatch produce the encoded histogram."""
        key = "1" + "0" * (n_bits - 1)
        blob = build_qh_histogram(n_bits, [(1 << (n_bits - 1), 2)], magic=magic)
        encoded = {
            "encoding": encoding,
            "n_bits": n_bits,
            "payload": base64.b64encode(blob).decode("ascii"),
        }

        assert _decode_histogram_b64(encoded) == {key: 2}

    @pytest.mark.parametrize(("encoding", "magic"), [("qh1", b"QH2"), ("qh2", b"QH1")])
    def test_decode_histogram_b64_rejects_magic_disagreement(self, encoding, magic):
        """The envelope encoding and payload magic must describe one format."""
        n_bits = 3
        blob = build_qh_histogram(n_bits, [(0, 1)], magic=magic)
        encoded = {
            "encoding": encoding,
            "n_bits": n_bits,
            "payload": base64.b64encode(blob).decode("ascii"),
        }

        with pytest.raises(ValueError, match="does not match"):
            _decode_histogram_b64(encoded)

    # --- Core Decompression Logic Tests ---

    def test_decompress_histogram_empty_buffer(self):
        """Tests that an empty byte buffer returns an empty histogram."""
        assert _decompress_histogram(b"") == {}

    def test_decompress_histogram_bad_magic(self):
        """Tests that a payload with an invalid magic header raises a ValueError."""
        with pytest.raises(ValueError, match="bad magic"):
            _decompress_histogram(b"INVALID_MAGIC")

    @pytest.mark.parametrize("payload", [b"QH1", b"QH2", b"QH2\x80"])
    def test_decompress_histogram_truncated_header_raises(self, payload):
        """Both header formats report truncation as a decoder ValueError."""
        with pytest.raises(ValueError, match="truncated"):
            _decompress_histogram(payload)

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

    def test_uleb128_decode_limbs_jit(self):
        """Tests JIT ULEB128 decoding into a limb buffer."""
        out = np.zeros(1, dtype=np.uint64)

        # Single-byte value
        pos = _uleb128_decode_limbs_jit(np.array([0x05], dtype=np.uint8), 0, out)
        assert out[0] == 5 and pos == 1

        # Multi-byte value (128)
        pos = _uleb128_decode_limbs_jit(np.array([0x80, 0x01], dtype=np.uint8), 0, out)
        assert out[0] == 128 and pos == 2

        # Decoding with an offset
        pos = _uleb128_decode_limbs_jit(np.array([0x00, 0x05], dtype=np.uint8), 1, out)
        assert out[0] == 5 and pos == 2

    def test_uleb128_decode_limbs_jit_buffer_overflow_raises(self):
        """A varint wider than the destination limb buffer must raise."""
        # 10 bytes encoding 2**64 — bit 64 must spill past a 1-limb buffer,
        # so the decoder must raise rather than silently truncate.
        data = np.frombuffer(encode_uleb128(1 << 64), dtype=np.uint8)
        out = np.zeros(1, dtype=np.uint64)
        with pytest.raises(ValueError, match="exceeds destination width"):
            _uleb128_decode_limbs_jit(data, 0, out)

        # Same varint decodes cleanly into a 2-limb buffer.
        out2 = np.zeros(2, dtype=np.uint64)
        pos = _uleb128_decode_limbs_jit(data, 0, out2)
        assert pos == len(data)
        assert (int(out2[1]) << 64) | int(out2[0]) == 1 << 64

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
        indices, counts, n_bits, L, unique, total_shots, n_decoded = (
            _decompress_histogram_jit(data, 3, 4)
        )

        assert n_bits == 3
        assert L == 1
        assert unique == 3
        assert total_shots == 5
        assert n_decoded == 3
        np.testing.assert_array_equal(indices.ravel(), [1, 5, 7])
        np.testing.assert_array_equal(counts, [1, 3, 1])

    @pytest.mark.parametrize(("num_gaps", "rle_count"), [(2, 1), (1, 2)])
    def test_decompress_rejects_gap_count_cardinality_mismatch(
        self, num_gaps, rle_count
    ):
        """Every decoded count must have exactly one decoded index."""
        rle_body = b"\x01\x01" + encode_uleb128(rle_count)
        blob = b"".join(
            [
                b"QH1\x03",
                encode_uleb128(rle_count),
                encode_uleb128(rle_count),
                encode_uleb128(num_gaps),
                b"\x00" * num_gaps,
                encode_uleb128(len(rle_body)),
                rle_body,
                b"\x00",
            ]
        )

        with pytest.raises(ValueError, match="gap/count length mismatch"):
            _decompress_histogram(blob)

    @pytest.mark.parametrize(("extras_len", "extras"), [(0, b""), (2, b"\x00\x00")])
    def test_decompress_rejects_extra_count_cardinality_mismatch(
        self, extras_len, extras
    ):
        """Every non-unit count must consume exactly one extras entry."""
        rle_body = b"\x01\x00\x01"
        blob = b"".join(
            [
                b"QH1\x03\x01\x02\x01\x00",
                encode_uleb128(len(rle_body)),
                rle_body,
                encode_uleb128(extras_len),
                extras,
            ]
        )

        with pytest.raises(ValueError, match="extras/count length mismatch"):
            _decompress_histogram(blob)

    # --- Wide-register (limb array) tests ---

    @pytest.mark.parametrize("n_bits", [63, 64, 65, 100, 128, 129, 200])
    def test_decompress_top_bit_set_index(self, n_bits):
        """Index with the top bit set must decode to the correct bitstring.

        This exercises the uint64 limb path: the pre-refactor kernel used a
        signed int64 accumulator, so any index >= 2**63 decoded to a negative
        number and produced garbage bitstrings (with a ``-`` prefix, wrong
        length, or silent truncation).
        """
        index = 1 << (n_bits - 1)
        blob = build_qh_histogram(n_bits, [(index, 1)])
        result = _decompress_histogram(blob)
        expected_key = "1" + "0" * (n_bits - 1)
        assert result == {expected_key: 1}
        assert len(expected_key) == n_bits

    @pytest.mark.parametrize("n_bits", [64, 100, 200])
    def test_decompress_multiple_entries_wide(self, n_bits):
        """Several entries mixing small and wide indices decode correctly."""
        top = 1 << (n_bits - 1)
        entries = [(0, 2), (1, 1), (top - 1, 4), (top, 3), (top | 1, 1)]
        blob = build_qh_histogram(n_bits, entries)
        result = _decompress_histogram(blob)

        def to_key(i: int) -> str:
            return format(i, f"0{n_bits}b")

        expected = {to_key(i): c for i, c in entries}
        assert result == expected
        assert all(len(k) == n_bits for k in result)

    def test_decompress_index_exceeding_n_bits_raises(self):
        """A blob whose encoded index doesn't fit in its declared n_bits must fail loudly."""
        # n_bits=4 claims indices live in [0, 15], but we encode index=16.
        # The top limb holds bits that must all be zero above bit 4; they
        # aren't, so the validator must raise rather than silently return a
        # wrong-width bitstring.
        blob = build_qh_histogram(4, [(16, 1)])
        with pytest.raises(ValueError, match="exceeds n_bits"):
            _decompress_histogram(blob)

    # --- QH2 (>255 bits, ULEB128 width) tests ---

    @pytest.mark.parametrize(
        ("n_bits", "magic"), [(255, b"QH1"), (256, b"QH2"), (300, b"QH2")]
    )
    def test_decompress_histogram_across_qh2_boundary(self, n_bits, magic):
        """Both width encodings decode exact histograms across the boundary."""
        top = 1 << (n_bits - 1)
        entries = [(0, 2), (1, 1), (top, 3), (top | 1, 1)]
        blob = build_qh_histogram(n_bits, entries, magic=magic)
        result = _decompress_histogram(blob)

        expected = {format(i, f"0{n_bits}b"): c for i, c in entries}
        assert result == expected

    @pytest.mark.parametrize("n_bits", [_MAX_HISTOGRAM_WIDTH + 1, 2**63 + 5])
    def test_decompress_qh2_width_above_the_maximum_raises(self, n_bits):
        """A few header bytes must not be able to demand a huge allocation."""
        blob = build_qh_histogram(n_bits, [(0, 1)], magic=b"QH2")
        with pytest.raises(ValueError, match="exceeds the maximum supported"):
            _decompress_histogram(blob)

    def test_decompress_envelope_width_disagreement_raises(self):
        """The envelope's n_bits and the payload's must agree."""
        blob = build_qh_histogram(256, [(1, 1)], magic=b"QH2")
        encoded = {
            "encoding": "qh2",
            "n_bits": 255,
            "payload": base64.b64encode(blob).decode("ascii"),
        }
        with pytest.raises(ValueError, match="envelope declares"):
            _decode_histogram_b64(encoded)

    @pytest.mark.parametrize("n_bits", [65, -3])
    def test_limbs_to_bitstrings_width_wider_than_limbs_raises(self, n_bits):
        """A width past the limb buffer would silently render a short slice."""
        with pytest.raises(ValueError, match="not renderable"):
            _limbs_to_bitstrings(np.zeros((1, 1), dtype=np.uint64), n_bits, 1)

    def test_limbs_to_bitstrings_spans_render_blocks(self, monkeypatch):
        """Rendering is chunked, so rows must not be dropped or reordered."""
        monkeypatch.setattr(results_processing, "_RENDER_BLOCK_CELLS", 40)
        n_rows = 7
        limbs = np.arange(n_rows, dtype=np.uint64).reshape(n_rows, 1)
        rendered = _limbs_to_bitstrings(limbs, 20, 1)
        assert rendered == [format(i, "020b") for i in range(n_rows)]

    def test_decompress_qh2_index_exceeding_n_bits_raises(self):
        """The width validator must still fire on the QH2 path."""
        # n_bits=300 -> 5 limbs, top limb carrying 44 valid bits; bit 300 is
        # inside the buffer but outside the declared width.
        blob = build_qh_histogram(300, [(1 << 300, 1)], magic=b"QH2")
        with pytest.raises(ValueError, match="exceeds n_bits"):
            _decompress_histogram(blob)

    # --- Reviewer defensive-decoding cases ---

    def test_limbs_to_bitstring_zero_width(self):
        """Zero-width render must return an empty string, not the full limb.

        Python's ``s[-0:]`` returns ``s`` in full, so a naive implementation
        would emit 64 chars for a ``n_bits == 0`` input.
        """
        limbs = np.zeros((1, 1), dtype=np.uint64)
        assert _limbs_to_bitstrings(limbs, 0, 1) == [""]

        # Non-zero limb content must also be ignored at width 0.
        limbs[0, 0] = np.uint64(0xDEADBEEF)
        assert _limbs_to_bitstrings(limbs, 0, 1) == [""]

    def test_limbs_to_bitstring_multiple_of_64(self):
        """Width exactly a multiple of 64 must return the full concatenation."""
        limbs = np.array([[np.uint64(0x5), np.uint64(0x1)]], dtype=np.uint64)
        (result,) = _limbs_to_bitstrings(limbs, 128, 2)
        assert len(result) == 128
        # Top limb (0x1) -> bit 0 of limb[1] set -> bit 64 of the 128-bit value
        # -> position (128 - 1 - 64) = 63 from the left.
        assert result[63] == "1"
        # Low limb (0x5) -> bits 0 and 2 set -> positions 127 and 125 from left.
        assert result[127] == "1" and result[125] == "1"
        # All other bits zero.
        ones = {63, 125, 127}
        assert {i for i, c in enumerate(result) if c == "1"} == ones

    @pytest.mark.parametrize("n_bits", [7, 65, 130])
    def test_limbs_to_bitstrings_batch_matches_scalar_format(self, n_bits):
        """A batch render equals per-row big-endian formatting at any width."""
        rng = np.random.default_rng(0)
        n_limbs = (n_bits + 63) // 64
        limbs = rng.integers(
            0, 1 << 63, size=(5, n_limbs), dtype=np.uint64, endpoint=True
        )
        # Keep every value inside the declared width.
        top_bits = n_bits - (n_limbs - 1) * 64
        if top_bits < 64:
            limbs[:, -1] &= np.uint64((1 << top_bits) - 1)

        expected = [
            format(
                sum(int(limb) << (64 * j) for j, limb in enumerate(row)), f"0{n_bits}b"
            )
            for row in limbs
        ]
        assert _limbs_to_bitstrings(limbs, n_bits, n_limbs) == expected

    def test_uleb128_decode_truncated_varint_raises(self):
        """A varint whose final byte still has the continuation bit set must raise.

        This covers the case where the stream ends in the middle of a
        varint — the pre-fix decoder silently returned a partial value.
        """
        # Single byte with continuation bit set and no follow-up.
        data = np.array([0x80], dtype=np.uint8)
        out = np.zeros(1, dtype=np.uint64)
        with pytest.raises(ValueError, match="truncated"):
            _uleb128_decode_limbs_jit(data, 0, out)

        # Multi-byte varint truncated mid-stream (all continuation bytes).
        data = np.array([0x80, 0x80, 0x80], dtype=np.uint8)
        with pytest.raises(ValueError, match="truncated"):
            _uleb128_decode_limbs_jit(data, 0, out)

    def test_decompress_histogram_gap_sum_overflow_raises(self):
        """Gap values that individually fit in L limbs but whose sum wraps past
        2**(64*L) must raise — the ripple-carry check catches what the
        top-bit validator cannot when ``n_bits`` is a multiple of 64.
        """
        # Two entries whose gaps are both 2**63. Each gap fits in a 1-limb
        # (uint64) buffer, but their sum is 2**64 which wraps to 0 inside
        # uint64 arithmetic. For n_bits=64 the top-limb mask check is
        # skipped (top_bits_used == 64), so only the final-carry guard can
        # detect this corruption.
        blob = build_qh_histogram(64, [(1 << 63, 1), (1 << 64, 1)])
        with pytest.raises(ValueError, match="overflows limb buffer"):
            _decompress_histogram(blob)


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
