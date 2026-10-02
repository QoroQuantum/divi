# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

from divi.pipeline._shot_distribution import (
    _compute_group_l1_norms,
    _compute_shot_distribution,
)
from tests._helpers import exact_match


@pytest.mark.parametrize(
    "coeffs, partition, expected",
    [
        pytest.param([0.5], [[0]], [0.5], id="single_group_single_term"),
        pytest.param(
            [-2.0, 3.0], [[0, 1]], [5.0], id="negative_coefficients_use_absolute_value"
        ),
        pytest.param(
            [1.0, -0.5, 2.0, 0.25],
            [[0, 1], [2], [3]],
            [1.5, 2.0, 0.25],
            id="multiple_groups",
        ),
        pytest.param([1.0, 2.0], [], [], id="empty_partition"),
    ],
)
def test_compute_group_l1_norms(coeffs, partition, expected):
    assert _compute_group_l1_norms(coeffs, partition) == expected


class TestComputeShotDistributionUniform:
    @pytest.mark.parametrize(
        "norms, total_shots, expected",
        [
            pytest.param([1.0, 1.0, 1.0], 300, [100, 100, 100], id="evenly_divisible"),
            pytest.param(
                [1.0, 1.0, 1.0], 10, [4, 3, 3], id="remainder_to_first_groups"
            ),
            pytest.param([1.0, 1.0], 0, [0, 0], id="zero_total_shots"),
        ],
    )
    def test_splits_evenly(self, norms, total_shots, expected):
        assert _compute_shot_distribution(norms, total_shots, "uniform") == expected

    @pytest.mark.parametrize(
        "strategy_kwargs",
        [{"strategy": "uniform"}, {}],
        ids=["explicit", "default"],
    )
    def test_uniform_ignores_norms(self, strategy_kwargs):
        assert _compute_shot_distribution([100.0, 0.01], 10, **strategy_kwargs) == [
            5,
            5,
        ]


class TestComputeShotDistributionWeighted:
    @pytest.mark.parametrize(
        ("norms", "total_shots", "expected"),
        [
            ([3.0, 1.0], 100, [75, 25]),
            ([1.0, 1.0, 1.0], 10, [4, 3, 3]),
            ([2.0, 1.0], 0, [0, 0]),
            ([0.0, 0.0, 0.0], 9, [3, 3, 3]),
            ([0.3, 0.7], 10, [3, 7]),
            ([7.0, 1.0, 2.0], 3, [2, 0, 1]),
        ],
        ids=[
            "proportional",
            "largest-remainder-tie-goes-first",
            "zero-total-shots",
            "all-zero-norms-fall-back-to-uniform",
            "norms-below-one",
            "leftover-to-largest-fractional-remainder",
        ],
    )
    def test_exact_allocation(self, norms, total_shots, expected):
        assert _compute_shot_distribution(norms, total_shots, "weighted") == expected

    def test_total_preserved_with_irrational_weights(self):
        # Weights chosen to produce non-trivial fractional parts.
        result = _compute_shot_distribution([0.7, 1.3, 2.0], 1000, "weighted")
        assert sum(result) == 1000
        # First group gets ~175, second ~325, third ~500
        assert result[2] >= result[1] >= result[0]

    def test_dominant_group_gets_almost_all_shots(self):
        result = _compute_shot_distribution([100.0, 0.01], 1000, "weighted")
        assert sum(result) == 1000
        assert result[0] > result[1]


class TestComputeShotDistributionWeightedRandom:
    def test_total_preserved(self):
        rng = np.random.default_rng(42)
        result = _compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=rng
        )
        assert sum(result) == 1000
        assert all(s >= 0 for s in result)

    def test_seeded_reproducibility(self):
        a = _compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=np.random.default_rng(7)
        )
        b = _compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=np.random.default_rng(7)
        )
        assert a == b

    def test_high_weight_group_concentrates_shots(self):
        rng = np.random.default_rng(0)
        # Run a few times because the result is stochastic.
        for _ in range(5):
            result = _compute_shot_distribution(
                [100.0, 1.0], 10000, "weighted_random", rng=rng
            )
            assert result[0] > result[1]

    @pytest.mark.parametrize(
        ("norms", "total_shots", "expected"),
        [
            ([2.0, 1.0], 0, [0, 0]),
            ([0.0, 0.0, 0.0], 9, [3, 3, 3]),
            ([0.0, 0.5], 10, [0, 10]),
            ([1.0, 0.0], 1, [1, 0]),
        ],
        ids=[
            "zero-total-shots",
            "all-zero-norms-fall-back-to-uniform",
            "norms-below-one",
            "one-shot",
        ],
    )
    def test_deterministic_outcomes(self, norms, total_shots, expected):
        assert (
            _compute_shot_distribution(
                norms, total_shots, "weighted_random", rng=np.random.default_rng(0)
            )
            == expected
        )


class TestComputeShotDistributionCallable:
    def test_callable_passthrough(self):
        def custom(norms, total):
            return [total, *([0] * (len(norms) - 1))]

        assert _compute_shot_distribution([1.0, 2.0, 3.0], 100, custom) == [100, 0, 0]

    def test_callable_wrong_length_raises(self):
        def bad(norms, total):
            return [total]

        with pytest.raises(
            ValueError,
            match=exact_match(
                "Custom shot distribution returned 1 entries, expected 3."
            ),
        ):
            _compute_shot_distribution([1.0, 1.0, 1.0], 10, bad)

    def test_callable_negative_shots_raises(self):
        def bad(norms, total):
            return [-1, total + 1]

        with pytest.raises(
            ValueError,
            match=exact_match(
                "Custom shot distribution returned negative shot counts."
            ),
        ):
            _compute_shot_distribution([1.0, 1.0], 10, bad)

    def test_callable_float_truncation_warns(self):
        """Float results that truncate to less than total_shots must warn."""

        def fractional(norms, total):
            # 100/3 fractions truncate to 33+33+33 = 99, dropping 1 shot.
            return [total / 3] * 3

        with pytest.warns(UserWarning, match="budget drift"):
            result = _compute_shot_distribution([1.0, 1.0, 1.0], 100, fractional)
        assert result == [33, 33, 33]

    def test_callable_integer_result_does_not_warn(self):
        """Integer-valued callables that sum to total_shots must not warn."""

        def exact(norms, total):
            return [total, 0, 0]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _compute_shot_distribution([1.0, 1.0, 1.0], 30, exact) == [30, 0, 0]


class TestComputeShotDistributionErrors:
    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="at least one entry"):
            _compute_shot_distribution([], 100, "uniform")

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _compute_shot_distribution([1.0, 1.0], -5, "uniform")

    def test_unknown_strategy_raises(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Unknown shot distribution strategy: 'magic'. "
                "Expected 'uniform', 'weighted', 'weighted_random', or a callable."
            ),
        ):
            _compute_shot_distribution([1.0], 10, "magic")
