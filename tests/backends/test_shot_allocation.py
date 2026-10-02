# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends._shot_allocation."""

import pytest

from divi.backends._shot_allocation import (
    ShotRange,
    bucket_by_shots,
    from_wire,
    per_circuit,
    restrict_to_chunk,
    to_wire,
    validate,
)
from tests._helpers import exact_match
from tests.backends._helpers import uncovered_circuits_message


class TestShotRange:
    def test_shift_offsets_endpoints_and_preserves_shots(self):
        r = ShotRange(2, 5, 100)
        shifted = r.shift(10)
        assert shifted == ShotRange(12, 15, 100)

    def test_shift_by_zero_is_identity(self):
        r = ShotRange(0, 3, 50)
        assert r.shift(0) == r


class TestValidate:
    @pytest.mark.parametrize(
        "ranges",
        [
            [ShotRange(0, 2, 100), ShotRange(2, 5, 200)],
            [ShotRange(2, 5, 200), ShotRange(0, 2, 100)],
        ],
        ids=["contiguous", "non_contiguous"],
    )
    def test_full_coverage_passes(self, ranges):
        validate(ranges, n_circuits=5)

    def test_partial_coverage_raises(self):
        ranges = [ShotRange(0, 3, 100)]
        with pytest.raises(
            ValueError, match=exact_match(uncovered_circuits_message([3, 4]))
        ):
            validate(ranges, n_circuits=5)

    def test_overlap_raises(self):
        ranges = [ShotRange(0, 3, 100), ShotRange(2, 5, 200)]
        message = (
            f"ShotRange {ShotRange(2, 5, 200)} overlaps an earlier range at index 2."
        )
        with pytest.raises(ValueError, match=exact_match(message)):
            validate(ranges, n_circuits=5)

    def test_non_positive_shots_raises(self):
        ranges = [ShotRange(0, 5, 0)]
        message = f"ShotRange {ShotRange(0, 5, 0)} has non-positive shot count."
        with pytest.raises(ValueError, match=exact_match(message)):
            validate(ranges, n_circuits=5)

    @pytest.mark.parametrize(
        "ranges",
        [
            [ShotRange(5, 2, 100)],
            [ShotRange(0, 10, 100)],
            [ShotRange(0, 5, 100), ShotRange(2, 2, 100)],
        ],
        ids=["reversed", "past_end", "empty"],
    )
    def test_out_of_bounds_raises(self, ranges):
        message = f"ShotRange {ranges[-1]} is out of bounds for n_circuits=5."
        with pytest.raises(ValueError, match=exact_match(message)):
            validate(ranges, n_circuits=5)


@pytest.mark.parametrize(
    "ranges, n_circuits, expected",
    [
        ([ShotRange(0, 3, 100)], 3, [100, 100, 100]),
        (
            [ShotRange(0, 2, 100), ShotRange(2, 5, 200)],
            5,
            [100, 100, 200, 200, 200],
        ),
        ([ShotRange(1, 2, 100)], 3, [0, 100, 0]),
    ],
    ids=["uniform", "heterogeneous", "uncovered_get_zero"],
)
def test_per_circuit_expands_ranges(ranges, n_circuits, expected):
    assert per_circuit(ranges, n_circuits) == expected


class TestBucketByShots:
    def test_groups_contiguous_indices_by_shots(self):
        ranges = [ShotRange(0, 2, 100), ShotRange(2, 5, 200)]
        assert bucket_by_shots(ranges) == {100: [0, 1], 200: [2, 3, 4]}

    def test_merges_non_contiguous_same_shot_ranges(self):
        ranges = [
            ShotRange(0, 2, 100),
            ShotRange(2, 4, 200),
            ShotRange(4, 6, 100),
        ]
        out = bucket_by_shots(ranges)
        assert out[100] == [0, 1, 4, 5]
        assert out[200] == [2, 3]


class TestRestrictToChunk:
    def test_restricts_and_rebases(self):
        ranges = [ShotRange(0, 10, 100)]
        out = restrict_to_chunk(ranges, offset=3, size=4)
        # Global [3, 7) intersected → local [0, 4).
        assert out == [ShotRange(0, 4, 100)]

    def test_drops_non_overlapping_ranges(self):
        ranges = [ShotRange(0, 3, 100), ShotRange(5, 8, 200)]
        out = restrict_to_chunk(ranges, offset=3, size=2)
        # [3, 5) chunk → only ShotRange(0, 3, 100) partially overlaps at
        # global [3, 3) which is empty; ShotRange(5, 8) doesn't overlap.
        assert out == []

    def test_splits_range_spanning_chunk_boundary(self):
        ranges = [ShotRange(0, 10, 100)]
        first = restrict_to_chunk(ranges, offset=0, size=5)
        second = restrict_to_chunk(ranges, offset=5, size=5)
        assert first == [ShotRange(0, 5, 100)]
        assert second == [ShotRange(0, 5, 100)]


class TestWireFormat:
    def test_to_wire_produces_triples(self):
        ranges = [ShotRange(0, 2, 100), ShotRange(2, 5, 200)]
        assert to_wire(ranges) == [[0, 2, 100], [2, 5, 200]]

    def test_from_wire_normalises_triples_and_shot_ranges(self):
        out = from_wire([ShotRange(0, 2, 100), [2, 5, 200]])
        assert out == [ShotRange(0, 2, 100), ShotRange(2, 5, 200)]

    def test_roundtrip(self):
        ranges = [ShotRange(0, 2, 100), ShotRange(2, 5, 200)]
        assert from_wire(to_wire(ranges)) == ranges
