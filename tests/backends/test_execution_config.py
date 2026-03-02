# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends._config import (
    ExecutionConfig,
    SimulationMethod,
    Simulator,
)


class TestExecutionConfigOverride:
    """Tests for the ExecutionConfig.override() method."""

    def test_override_non_none_fields(self):
        """Non-None fields from `other` should win."""
        base = ExecutionConfig(bond_dimension=32, simulator=Simulator.QiskitAer)
        other = ExecutionConfig(bond_dimension=64)
        result = base.override(other)
        assert result.bond_dimension == 64
        assert result.simulator == Simulator.QiskitAer

    def test_override_none_fields_preserved(self):
        """None fields in `other` should not clobber existing values."""
        base = ExecutionConfig(
            bond_dimension=32,
            truncation_threshold=1e-8,
            simulator=Simulator.QCSim,
            simulation_method=SimulationMethod.MatrixProductState,
        )
        other = ExecutionConfig(bond_dimension=128)
        result = base.override(other)

        assert result.bond_dimension == 128
        assert result.truncation_threshold == 1e-8
        assert result.simulator == Simulator.QCSim
        assert result.simulation_method == SimulationMethod.MatrixProductState

    def test_override_returns_new_instance(self):
        """Override should return a new instance, not mutate the original."""
        base = ExecutionConfig(bond_dimension=32)
        other = ExecutionConfig(bond_dimension=64)
        result = base.override(other)

        assert result is not base
        assert result is not other
        assert base.bond_dimension == 32

    def test_override_both_empty(self):
        """Two empty configs should produce an empty config."""
        result = ExecutionConfig().override(ExecutionConfig())
        assert result.bond_dimension is None
        assert result.truncation_threshold is None
        assert result.simulator is None
        assert result.simulation_method is None
        assert result.api_meta is None

    def test_override_api_meta(self):
        """api_meta from `other` should replace (not merge) the base value."""
        base = ExecutionConfig(api_meta={"optimization_level": 1})
        other = ExecutionConfig(api_meta={"resilience_level": 2})
        result = base.override(other)
        assert result.api_meta == {"resilience_level": 2}


class TestExecutionConfigPayload:
    """Tests for to_payload() and from_response() serialization."""

    def test_to_payload_full(self):
        """All fields should serialize correctly, with enums converted to ints."""
        config = ExecutionConfig(
            bond_dimension=256,
            truncation_threshold=1e-8,
            simulator=Simulator.QCSim,
            simulation_method=SimulationMethod.MatrixProductState,
            api_meta={"optimization_level": 2},
        )
        payload = config.to_payload()
        assert payload == {
            "bond_dimension": 256,
            "truncation_threshold": 1e-8,
            "simulator_type": int(Simulator.QCSim),
            "simulation_type": int(SimulationMethod.MatrixProductState),
            "api_meta": {"optimization_level": 2},
        }

    def test_to_payload_partial(self):
        """Only non-None fields should appear in the payload."""
        config = ExecutionConfig(bond_dimension=64)
        payload = config.to_payload()
        assert payload == {"bond_dimension": 64}
        assert "simulator_type" not in payload
        assert "simulation_type" not in payload

    def test_to_payload_empty(self):
        """An empty config should produce an empty dict."""
        assert ExecutionConfig().to_payload() == {}

    def test_from_response_round_trip(self):
        """from_response(to_payload()) should reproduce the original config."""
        original = ExecutionConfig(
            bond_dimension=128,
            truncation_threshold=1e-6,
            simulator=Simulator.GpuSim,
            simulation_method=SimulationMethod.TensorNetwork,
            api_meta={"max_execution_time": 300},
        )
        reconstructed = ExecutionConfig.from_response(original.to_payload())
        assert reconstructed == original

    def test_from_response_partial(self):
        """from_response should handle missing fields gracefully."""
        data = {"bond_dimension": 64}
        config = ExecutionConfig.from_response(data)
        assert config.bond_dimension == 64
        assert config.simulator is None
        assert config.simulation_method is None


class TestExecutionConfigFrozen:
    """Tests for immutability of ExecutionConfig."""

    def test_frozen(self):
        """Mutating a frozen dataclass should raise an error."""
        config = ExecutionConfig(bond_dimension=32)
        with pytest.raises(AttributeError):
            config.bond_dimension = 64
