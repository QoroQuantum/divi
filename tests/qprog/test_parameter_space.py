# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ParameterSpace, the harmonic structure of a program's parameters."""

import numpy as np
import pytest

from divi.qprog import ParameterSpace
from tests._helpers import exact_match

_TWO_PI = 2 * np.pi


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (((0.0, 1),), "omega must be positive; got 0.0."),
        (((np.nan, 1),), "omega must be positive; got nan."),
        (((1.0, 0),), "order must be a positive integer; got 0."),
        (((1.0, 1.5),), "order must be a positive integer; got 1.5."),
        (((1.0, True),), "order must be a positive integer; got True."),
    ],
    ids=["zero-omega", "nan-omega", "zero-order", "fractional-order", "boolean-order"],
)
def test_rejects_invalid_frequencies(frequencies, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        ParameterSpace(frequencies)


def test_periods_follow_the_base_frequency():
    space = ParameterSpace(((1.0, 1), (0.5, 2), (2.0, 3), None))

    np.testing.assert_allclose(space.periods, [_TWO_PI, 4 * np.pi, np.pi, np.nan])


def test_shift_rule_needs_every_frequency():
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Parameters [1, 3] have no known frequencies, so no parameter-shift "
            "rule exists for them."
        ),
    ):
        ParameterSpace(((1.0, 1), None, (1.0, 1), None)).compute_shift_rule()


def test_unit_frequency_reproduces_the_two_term_rule():
    shifts, weights = ParameterSpace(((1.0, 1),) * 2).compute_shift_rule()
    half = np.pi / 2
    np.testing.assert_allclose(shifts, [[half, 0], [-half, 0], [0, half], [0, -half]])
    np.testing.assert_allclose(weights, [[0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 0.5, -0.5]])


def test_frequencies_may_differ_between_parameters():
    """A ragged rule is one flat batch, not padded rows per parameter."""
    shifts, weights = ParameterSpace(((1.0, 1), (0.5, 4))).compute_shift_rule()
    assert shifts.shape == (2 + 8, 2)
    assert np.count_nonzero(weights[0]) == 2
    assert np.count_nonzero(weights[1]) == 8
    np.testing.assert_array_equal(np.nonzero(weights[0])[0], [0, 1])
    assert np.count_nonzero(shifts[:2, 1]) == 0
    assert np.count_nonzero(shifts[2:, 0]) == 0


def test_shifts_fold_into_the_principal_interval():
    shifts, _ = ParameterSpace(((0.5, 2),)).compute_shift_rule()
    # Unfolded these are (2k-1) * pi/2 for k = 1..4; the period is 4 * pi.
    np.testing.assert_allclose(
        shifts[:, 0], [np.pi / 2, 3 * np.pi / 2, -3 * np.pi / 2, -np.pi / 2]
    )


@pytest.mark.parametrize("omega, order", [(1.0, 1), (2.0, 1), (1.0, 2), (0.5, 4)])
def test_shift_rule_differentiates_its_own_frequency_family_exactly(omega, order):
    """The rule is exact on any trigonometric polynomial it declares."""
    rng = np.random.default_rng(0)
    cosines = rng.normal(size=order + 1)
    sines = rng.normal(size=order + 1)

    def energy(theta):
        return sum(
            cosines[r] * np.cos(r * omega * theta)
            + sines[r] * np.sin(r * omega * theta)
            for r in range(order + 1)
        )

    def derivative(theta):
        return sum(
            r
            * omega
            * (
                -cosines[r] * np.sin(r * omega * theta)
                + sines[r] * np.cos(r * omega * theta)
            )
            for r in range(order + 1)
        )

    shifts, weights = ParameterSpace(((omega, order),)).compute_shift_rule()
    for theta in rng.uniform(-3.0, 3.0, 5):
        values = np.array([energy(theta + row[0]) for row in shifts])
        assert (weights @ values)[0] == pytest.approx(derivative(theta), abs=1e-9)
