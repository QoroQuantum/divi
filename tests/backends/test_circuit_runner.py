# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends import CircuitRunner


class ConcreteCircuitRunner(CircuitRunner):
    """Concrete implementation of CircuitRunner for testing."""

    @property
    def supports_expval(self) -> bool:
        return False

    @property
    def is_async(self) -> bool:
        return False

    def submit_circuits(self, circuits: dict[str, str], **kwargs):
        return []


class TestCircuitRunner:
    """Tests for CircuitRunner abstract base class."""

    def test_init_with_valid_shots(self):
        """Test initialization with valid shots."""
        runner = ConcreteCircuitRunner(shots=1000)
        assert runner.shots == 1000

    def test_init_with_zero_shots_raises(self):
        """Test that ValueError is raised when shots is 0 (line 15)."""
        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            ConcreteCircuitRunner(shots=0)

    def test_init_with_negative_shots_raises(self):
        """Test that ValueError is raised when shots is negative (line 15)."""
        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            ConcreteCircuitRunner(shots=-1)

    def test_shots_property(self):
        """Test shots property getter."""
        runner = ConcreteCircuitRunner(shots=5000)
        assert runner.shots == 5000

    def test_concrete_implementation(self):
        """Test that a concrete implementation works correctly."""
        runner = ConcreteCircuitRunner(shots=100)
        assert runner.shots == 100
        assert runner.supports_expval is False
        assert runner.is_async is False
        assert runner.submit_circuits({}) == []
