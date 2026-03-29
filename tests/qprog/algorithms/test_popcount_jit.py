# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Numba JIT kernel: _popcount_parity_jit."""

import numpy as np

from divi.qprog.algorithms._numba_kernels import _popcount_parity_jit


class TestPopcountParityJit:
    def test_known_values(self):
        """XOR-fold gives correct parity for known inputs."""
        # popcount(0) = 0 → parity 0
        # popcount(1) = 1 → parity 1
        # popcount(3) = 2 → parity 0
        # popcount(7) = 3 → parity 1
        # popcount(255) = 8 → parity 0
        arr = np.array([0, 1, 3, 7, 255], dtype=np.uint64)
        result = _popcount_parity_jit(arr)
        np.testing.assert_array_equal(result, [0, 1, 0, 1, 0])

    def test_powers_of_two(self):
        """All powers of two have popcount 1 → parity 1."""
        arr = np.array([2**i for i in range(64)], dtype=np.uint64)
        result = _popcount_parity_jit(arr)
        np.testing.assert_array_equal(result, np.ones(64, dtype=np.uint8))

    def test_all_ones(self):
        """0xFFFFFFFFFFFFFFFF has popcount 64 → parity 0."""
        arr = np.array([np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
        result = _popcount_parity_jit(arr)
        assert result[0] == 0

    def test_preserves_shape(self):
        """Output shape matches input shape for 2D arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
        result = _popcount_parity_jit(arr)
        assert result.shape == (2, 3)

    def test_large_random_array(self):
        """JIT matches naive popcount for a large random array."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 2**63, size=10_000, dtype=np.uint64)

        expected = np.array([bin(int(x)).count("1") % 2 for x in arr], dtype=np.uint8)
        result = _popcount_parity_jit(arr)
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        """Empty input returns empty output."""
        arr = np.array([], dtype=np.uint64)
        result = _popcount_parity_jit(arr)
        assert result.shape == (0,)
