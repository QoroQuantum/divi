# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

from divi.pipeline._shot_distribution import (
    compute_group_l1_norms,
    compute_shot_distribution,
)


class TestComputeGroupL1Norms:
    def test_single_group_single_term(self):
        assert compute_group_l1_norms([0.5], [[0]]) == [0.5]

    def test_negative_coefficients_use_absolute_value(self):
        assert compute_group_l1_norms([-2.0, 3.0], [[0, 1]]) == [5.0]

    def test_multiple_groups(self):
        coeffs = [1.0, -0.5, 2.0, 0.25]
        partition = [[0, 1], [2], [3]]
        assert compute_group_l1_norms(coeffs, partition) == [1.5, 2.0, 0.25]

    def test_empty_partition(self):
        # No groups -> empty norms list (degenerate but legal).
        assert compute_group_l1_norms([1.0, 2.0], []) == []


class TestComputeShotDistributionUniform:
    def test_evenly_divisible(self):
        assert compute_shot_distribution([1.0, 1.0, 1.0], 300, "uniform") == [
            100,
            100,
            100,
        ]

    def test_remainder_distributed_to_first_groups(self):
        # 10 shots / 3 groups = 3 base + 1 remainder distributed to first group
        assert compute_shot_distribution([1.0, 1.0, 1.0], 10, "uniform") == [4, 3, 3]

    def test_uniform_ignores_norms(self):
        # Even with skewed norms, uniform splits evenly.
        assert compute_shot_distribution([100.0, 0.01], 10, "uniform") == [5, 5]

    def test_zero_total_shots(self):
        assert compute_shot_distribution([1.0, 1.0], 0, "uniform") == [0, 0]


class TestComputeShotDistributionWeighted:
    def test_proportional_allocation(self):
        # Weights 3:1, 100 shots -> 75:25
        assert compute_shot_distribution([3.0, 1.0], 100, "weighted") == [75, 25]

    def test_largest_remainder_preserves_total(self):
        # Three groups with weights 1:1:1, 10 shots -> 4:3:3 (or some permutation summing to 10)
        result = compute_shot_distribution([1.0, 1.0, 1.0], 10, "weighted")
        assert sum(result) == 10
        assert all(s in (3, 4) for s in result)

    def test_total_preserved_with_irrational_weights(self):
        # Weights chosen to produce non-trivial fractional parts.
        result = compute_shot_distribution([0.7, 1.3, 2.0], 1000, "weighted")
        assert sum(result) == 1000
        # First group gets ~175, second ~325, third ~500
        assert result[2] >= result[1] >= result[0]

    def test_zero_total_shots(self):
        assert compute_shot_distribution([2.0, 1.0], 0, "weighted") == [0, 0]

    def test_all_zero_norms_falls_back_to_uniform(self):
        assert compute_shot_distribution([0.0, 0.0, 0.0], 9, "weighted") == [3, 3, 3]

    def test_dominant_group_gets_almost_all_shots(self):
        result = compute_shot_distribution([100.0, 0.01], 1000, "weighted")
        assert sum(result) == 1000
        assert result[0] > result[1]


class TestComputeShotDistributionWeightedRandom:
    def test_total_preserved(self):
        rng = np.random.default_rng(42)
        result = compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=rng
        )
        assert sum(result) == 1000
        assert all(s >= 0 for s in result)

    def test_seeded_reproducibility(self):
        a = compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=np.random.default_rng(7)
        )
        b = compute_shot_distribution(
            [1.0, 2.0, 3.0], 1000, "weighted_random", rng=np.random.default_rng(7)
        )
        assert a == b

    def test_high_weight_group_concentrates_shots(self):
        rng = np.random.default_rng(0)
        # Run a few times because the result is stochastic.
        for _ in range(5):
            result = compute_shot_distribution(
                [100.0, 1.0], 10000, "weighted_random", rng=rng
            )
            assert result[0] > result[1]

    def test_zero_total_shots(self):
        assert compute_shot_distribution(
            [2.0, 1.0], 0, "weighted_random", rng=np.random.default_rng(0)
        ) == [0, 0]

    def test_all_zero_norms_falls_back_to_uniform(self):
        assert compute_shot_distribution(
            [0.0, 0.0, 0.0], 9, "weighted_random", rng=np.random.default_rng(0)
        ) == [3, 3, 3]


class TestComputeShotDistributionCallable:
    def test_callable_passthrough(self):
        def custom(norms, total):
            return [total, *([0] * (len(norms) - 1))]

        assert compute_shot_distribution([1.0, 2.0, 3.0], 100, custom) == [100, 0, 0]

    def test_callable_wrong_length_raises(self):
        def bad(norms, total):
            return [total]

        with pytest.raises(ValueError, match="returned 1 entries, expected 3"):
            compute_shot_distribution([1.0, 1.0, 1.0], 10, bad)

    def test_callable_negative_shots_raises(self):
        def bad(norms, total):
            return [-1, total + 1]

        with pytest.raises(ValueError, match="negative shot counts"):
            compute_shot_distribution([1.0, 1.0], 10, bad)

    def test_callable_float_truncation_warns(self):
        """Float results that truncate to less than total_shots must warn."""

        def fractional(norms, total):
            # 100/3 fractions truncate to 33+33+33 = 99, dropping 1 shot.
            return [total / 3] * 3

        with pytest.warns(UserWarning, match="budget drift"):
            result = compute_shot_distribution([1.0, 1.0, 1.0], 100, fractional)
        assert result == [33, 33, 33]

    def test_callable_integer_result_does_not_warn(self):
        """Integer-valued callables that sum to total_shots must not warn."""

        def exact(norms, total):
            return [total, 0, 0]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert compute_shot_distribution([1.0, 1.0, 1.0], 30, exact) == [30, 0, 0]


class TestComputeShotDistributionErrors:
    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="at least one entry"):
            compute_shot_distribution([], 100, "uniform")

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_shot_distribution([1.0, 1.0], -5, "uniform")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown shot distribution strategy"):
            compute_shot_distribution([1.0], 10, "magic")
