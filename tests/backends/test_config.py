# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`divi.backends._config`."""

import dataclasses

import pytest

from divi.backends import (
    ExecutionConfig,
    JobConfig,
    QPUSystem,
    SimulationMethod,
    Simulator,
    SimulatorCluster,
)
from divi.backends._systems import update_qpu_systems_cache


class TestJobConfig:
    """JobConfig field validation and ``override()`` behaviour."""

    @pytest.mark.parametrize(
        "input_value, expected_stored_value",
        [
            ("my_qpu_system", "my_qpu_system"),
            (
                QPUSystem(name="qpu_from_object"),
                QPUSystem(name="qpu_from_object"),
            ),
            (None, None),
        ],
        ids=["string_input", "QPUSystem_object_input", "None_input"],
    )
    def test_qpu_system_success(self, input_value, expected_stored_value):
        """Valid ``qpu_system`` types are stored as-is (resolution happens in QoroService)."""
        config = JobConfig(qpu_system=input_value)
        assert config.qpu_system == expected_stored_value

    @pytest.mark.parametrize(
        "invalid_input",
        [123, ["a", "list"], {"a": "dict"}],
        ids=["integer_input", "list_input", "dict_input"],
    )
    def test_qpu_system_failure(self, invalid_input):
        with pytest.raises(TypeError):
            JobConfig(qpu_system=invalid_input)

    def test_simulator_cluster_accepts_valid_types(self):
        assert (
            JobConfig(simulator_cluster="my_cluster").simulator_cluster == "my_cluster"
        )
        cluster = SimulatorCluster(name="c")
        assert JobConfig(simulator_cluster=cluster).simulator_cluster == cluster
        assert JobConfig(simulator_cluster=None).simulator_cluster is None

    def test_simulator_cluster_rejects_invalid_types(self):
        for invalid in (123, ["a"], {"a": "b"}):
            with pytest.raises(TypeError):
                JobConfig(simulator_cluster=invalid)

    def test_rejects_both_targets(self):
        with pytest.raises(ValueError, match="not both"):
            JobConfig(
                simulator_cluster=SimulatorCluster(name="cluster"),
                qpu_system=QPUSystem(name="qpu"),
            )

    def test_shots_validation(self):
        config = JobConfig(shots=100)
        assert config.shots == 100

        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            JobConfig(shots=0)

        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            JobConfig(shots=-1)

    def test_use_circuit_packing_type_validation(self):
        config = JobConfig(use_circuit_packing=True)
        assert config.use_circuit_packing is True

        with pytest.raises(TypeError, match="Expected a bool"):
            JobConfig(use_circuit_packing="true")

        with pytest.raises(TypeError, match="Expected a bool"):
            JobConfig(use_circuit_packing=1)

    def test_override_basic(self):
        base = JobConfig(shots=1000, tag="base", use_circuit_packing=False)
        override = JobConfig(shots=500, tag="override")

        result = base.override(override)
        assert result.shots == 500
        assert result.tag == "override"
        assert result.use_circuit_packing is False

    def test_override_none_values_ignored(self):
        base = JobConfig(
            shots=1000, tag="base", qpu_system=QPUSystem(name="qoro_maestro")
        )
        override = JobConfig(shots=None, tag="override", qpu_system=None)

        result = base.override(override)
        assert result.shots == 1000
        assert result.tag == "override"
        assert result.qpu_system == QPUSystem(name="qoro_maestro")

    def test_override_immutability(self):
        base = JobConfig(shots=1000)
        override = JobConfig(shots=500)

        result = base.override(override)

        assert base.shots == 1000
        assert result.shots == 500
        assert result is not base
        assert result is not override

    def test_override_all_fields(self):
        base = JobConfig(
            shots=1000,
            tag="base",
            qpu_system=QPUSystem(name="system1"),
            use_circuit_packing=False,
        )

        update_qpu_systems_cache([QPUSystem(name="system2")])

        override = JobConfig(
            shots=2000,
            tag="override",
            qpu_system="system2",
            use_circuit_packing=True,
        )

        result = base.override(override)
        assert result.shots == 2000
        assert result.tag == "override"
        assert result.qpu_system == "system2"
        assert result.use_circuit_packing is True

    def test_override_with_qpu_system_object(self):
        base = JobConfig(
            shots=1000,
            tag="base",
            qpu_system=QPUSystem(name="system1", supports_expval=True),
        )

        override_qpu = QPUSystem(name="system2", supports_expval=False)
        override = JobConfig(qpu_system=override_qpu, tag="base")

        result = base.override(override)
        assert result.shots == 1000
        assert result.tag == "base"
        assert result.qpu_system == override_qpu
        assert result.qpu_system.name == "system2"

    def test_override_with_empty_config(self):
        base = JobConfig(
            shots=1000,
            tag="base",
            qpu_system=QPUSystem(name="qoro_maestro"),
            use_circuit_packing=True,
        )

        empty_override = JobConfig(
            shots=None,
            tag=None,
            qpu_system=None,
            use_circuit_packing=None,
        )

        result = base.override(empty_override)

        assert result.shots == 1000
        assert result.tag == "base"
        assert result.qpu_system == QPUSystem(name="qoro_maestro")
        assert result.use_circuit_packing is True

    def test_override_boolean_false(self):
        base = JobConfig(
            shots=1000,
            use_circuit_packing=True,
        )

        override = JobConfig(use_circuit_packing=False)

        result = base.override(override)
        assert result.shots == 1000
        assert result.use_circuit_packing is False

    def test_override_chained(self):
        base = JobConfig(
            shots=1000,
            tag="base",
            use_circuit_packing=False,
        )

        override1 = JobConfig(shots=500, tag="override1")
        override2 = JobConfig(shots=250, tag=None, use_circuit_packing=True)

        result = base.override(override1).override(override2)

        assert result.shots == 250
        assert result.tag == "override1"
        assert result.use_circuit_packing is True

    def test_override_preserves_base_when_override_has_none(self):
        base = JobConfig(tag="custom_tag", shots=1000, use_circuit_packing=True)

        override = JobConfig(shots=500, tag=None, use_circuit_packing=None)

        result = base.override(override)
        assert result.shots == 500
        assert result.tag == "custom_tag"
        assert result.use_circuit_packing is True

        override_with_values = JobConfig(
            shots=300, tag="new_tag", use_circuit_packing=False
        )
        result_overridden = base.override(override_with_values)
        assert result_overridden.shots == 300
        assert result_overridden.tag == "new_tag"
        assert result_overridden.use_circuit_packing is False

    def test_override_validation_after_override(self):
        base = JobConfig(shots=1000)

        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            base.override(JobConfig(shots=-1))

        with pytest.raises(ValueError, match="Shots must be a positive integer"):
            base.override(JobConfig(shots=0))

        result = base.override(JobConfig(shots=500))
        assert result.shots == 500


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
        assert result.noisy_device is None
        assert result.noise_realizations is None
        assert result.noise_scaling_factor is None
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
            noisy_device="ibm_fake_fez",
            noise_realizations=10,
            noise_scaling_factor=0.5,
            api_meta={"optimization_level": 2},
        )
        payload = config.to_payload()
        assert payload == {
            "bond_dimension": 256,
            "truncation_threshold": 1e-8,
            "simulator_type": int(Simulator.QCSim),
            "simulation_type": int(SimulationMethod.MatrixProductState),
            "noisy_device": "ibm_fake_fez",
            "noise_realizations": 10,
            "noise_scaling_factor": 0.5,
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
            simulation_method=SimulationMethod.MatrixProductState,
            noisy_device="ibm_fake_fez",
            noise_realizations=5,
            noise_scaling_factor=0.25,
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


@pytest.mark.parametrize("invalid_value", [0, -1, -10])
def test_bond_dimension_rejects_non_positive(invalid_value):
    """bond_dimension must be a positive integer."""
    with pytest.raises(ValueError, match="bond_dimension must be a positive integer"):
        ExecutionConfig(bond_dimension=invalid_value)


@pytest.mark.parametrize("invalid_value", [-1e-8, -0.5, -10])
def test_truncation_threshold_rejects_negative(invalid_value):
    """truncation_threshold must be non-negative."""
    with pytest.raises(ValueError, match="truncation_threshold must be non-negative"):
        ExecutionConfig(truncation_threshold=invalid_value)


@pytest.mark.parametrize("invalid_value", [0, -1, -10])
def test_noise_realizations_rejects_non_positive(invalid_value):
    """noise_realizations must be a positive integer."""
    with pytest.raises(
        ValueError, match="noise_realizations must be a positive integer"
    ):
        ExecutionConfig(noise_realizations=invalid_value)


@pytest.mark.parametrize("field", ["bond_dimension", "truncation_threshold"])
def test_mps_only_fields_warn_with_non_mps_method(field):
    """MPS-only fields warn when paired with an explicit non-MPS method."""
    value = 64 if field == "bond_dimension" else 1e-8
    with pytest.warns(UserWarning, match="only apply to MatrixProductState"):
        ExecutionConfig(
            simulation_method=SimulationMethod.Statevector, **{field: value}
        )


def test_mps_only_fields_silent_with_mps_method(recwarn):
    """No warning when MPS-only fields pair with MatrixProductState."""
    ExecutionConfig(
        bond_dimension=64,
        truncation_threshold=1e-8,
        simulation_method=SimulationMethod.MatrixProductState,
    )
    assert len(recwarn) == 0


def test_mps_only_fields_silent_with_no_method(recwarn):
    """No warning when simulation_method is unset (server auto-picks)."""
    ExecutionConfig(bond_dimension=64, truncation_threshold=1e-8)
    assert len(recwarn) == 0


@pytest.mark.parametrize("invalid_value", [-0.1, 1.1, 2.0, -5])
def test_noise_scaling_factor_rejects_out_of_range(invalid_value):
    """noise_scaling_factor must lie between 0 and 1."""
    with pytest.raises(
        ValueError, match="noise_scaling_factor must be between 0 and 1"
    ):
        ExecutionConfig(noise_scaling_factor=invalid_value)


@pytest.mark.parametrize("valid_value", [0, 0.5, 1])
def test_noise_scaling_factor_accepts_unit_interval(valid_value):
    """Boundary and interior values in [0, 1] are accepted."""
    assert (
        ExecutionConfig(noise_scaling_factor=valid_value).noise_scaling_factor
        == valid_value
    )


def test_noise_scaling_factor_zero_with_device_warns():
    """Scaling a named device's noise to zero warns about the noiseless run."""
    with pytest.warns(UserWarning, match="cancels all noise from noisy_device"):
        ExecutionConfig(noisy_device="ibm_fake_fez", noise_scaling_factor=0)


def test_noise_scaling_factor_zero_without_device_is_silent(recwarn):
    """Scaling to zero with no device set is a valid baseline, no warning."""
    ExecutionConfig(noise_scaling_factor=0)
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "data",
    [
        {"bond_dimension": 0},
        {"noise_realizations": 0},
        {"truncation_threshold": -1e-8},
        {"noise_scaling_factor": 2.0},
    ],
)
def test_from_response_skips_input_validation(data):
    """Reading server state must round-trip even if it predates a guard."""
    config = ExecutionConfig.from_response(data)
    field, value = next(iter(data.items()))
    assert getattr(config, field) == value


def test_override_skips_validation_and_warnings(recwarn):
    """Merging two individually-valid configs neither raises nor warns."""
    base = ExecutionConfig(bond_dimension=64)
    other = ExecutionConfig(simulation_method=SimulationMethod.Statevector)
    result = base.override(other)
    assert result.bond_dimension == 64
    assert result.simulation_method == SimulationMethod.Statevector
    assert len(recwarn) == 0


def test_validate_input_flag_excluded_from_fields_and_payload():
    """The private _validate_input InitVar must not leak into fields or payload."""
    field_names = {f.name for f in dataclasses.fields(ExecutionConfig)}
    assert "_validate_input" not in field_names
    assert "_validate_input" not in ExecutionConfig(bond_dimension=64).to_payload()


def test_frozen():
    """Mutating a frozen dataclass should raise an error."""
    config = ExecutionConfig(bond_dimension=32)
    with pytest.raises(AttributeError):
        config.bond_dimension = 64
