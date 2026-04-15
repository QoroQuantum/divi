# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from divi.viz import periodic_trajectory_wrap, periodic_wrap


class TestPeriodicWrap:
    def test_close_points_unchanged(self):
        ref = np.array([0.0, 0.0])
        point = np.array([0.1, -0.1])
        result = periodic_wrap(point, ref)
        np.testing.assert_allclose(result, point)

    def test_wraps_to_closest_copy(self):
        ref = np.array([0.1])
        point = np.array([0.1 + 2 * np.pi - 0.01])
        result = periodic_wrap(point, ref, period=2 * np.pi)
        # Should wrap to point - 2*pi = 0.09.
        np.testing.assert_allclose(result[0], point[0] - 2 * np.pi, atol=1e-12)

    def test_modular_equivalence(self):
        ref = np.array([1.0, 2.0])
        point = np.array([1.0 + 4 * np.pi, 2.0 - 6 * np.pi])
        result = periodic_wrap(point, ref, period=2 * np.pi)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_custom_period(self):
        ref = np.array([0.0])
        point = np.array([3.5])
        result = periodic_wrap(point, ref, period=4.0)
        assert abs(result[0]) <= 2.0 + 1e-12

    def test_exactly_half_period_away(self):
        ref = np.array([0.0])
        point = np.array([np.pi])
        result = periodic_wrap(point, ref, period=2 * np.pi)
        # Either -pi or pi is valid; both are half-period away.
        np.testing.assert_allclose(abs(result[0]), np.pi, atol=1e-12)


class TestPeriodicTrajectoryWrap:
    def test_crossing_boundary_becomes_monotonic(self):
        """A trajectory that jumps across 2*pi should be unwrapped."""
        traj = np.array([[3.0], [3.1], [3.2], [-3.0]])
        # -3.0 is equivalent to -3.0 + 2*pi = 3.283... which is close to 3.2
        result = periodic_trajectory_wrap(traj, period=2 * np.pi)

        diffs = np.diff(result[:, 0])
        # After unwrapping, all steps should be small and positive-ish.
        assert np.all(np.abs(diffs) < 0.5)

    def test_already_continuous_unchanged(self):
        traj = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        result = periodic_trajectory_wrap(traj)
        np.testing.assert_allclose(result, traj, atol=1e-12)

    def test_first_row_preserved(self):
        traj = np.array([[5.0], [5.1], [5.2]])
        result = periodic_trajectory_wrap(traj)
        np.testing.assert_allclose(result[0], traj[0])

    def test_multidimensional_wrapping(self):
        traj = np.array(
            [
                [0.0, 6.0],
                [0.1, 6.1],
                [0.2, 0.1],  # second param jumps from ~6.1 to 0.1 (crosses 2*pi)
            ]
        )
        result = periodic_trajectory_wrap(traj)
        # First column should be unchanged (no boundary crossing).
        np.testing.assert_allclose(result[:, 0], traj[:, 0], atol=1e-12)
        # Second column should be continuous.
        diffs = np.abs(np.diff(result[:, 1]))
        assert np.all(diffs < 1.0)

    def test_rejects_1d_array(self):
        with pytest.raises(ValueError, match="2-D array"):
            periodic_trajectory_wrap(np.array([1.0, 2.0, 3.0]))

    def test_rejects_single_row(self):
        with pytest.raises(ValueError, match="at least two rows"):
            periodic_trajectory_wrap(np.array([[1.0, 2.0]]))

    def test_custom_period(self):
        traj = np.array([[0.0], [0.9], [0.1]])  # jump in period=1.0
        result = periodic_trajectory_wrap(traj, period=1.0)
        diffs = np.abs(np.diff(result[:, 0]))
        assert np.all(diffs < 0.5)
