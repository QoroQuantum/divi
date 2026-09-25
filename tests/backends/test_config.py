# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`divi.backends._config`."""

import pytest
from pydantic import ValidationError

from divi.backends import DeviceConfig, JobConfig, QPUSystem, SimulatorCluster
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
        with pytest.raises(ValidationError):
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
            with pytest.raises(ValidationError):
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

        with pytest.raises(ValidationError, match="greater than 0"):
            JobConfig(shots=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            JobConfig(shots=-1)

    def test_use_circuit_packing_type_validation(self):
        config = JobConfig(use_circuit_packing=True)
        assert config.use_circuit_packing is True

        with pytest.raises(ValidationError, match="valid boolean"):
            JobConfig(use_circuit_packing="true")

        with pytest.raises(ValidationError, match="valid boolean"):
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

        with pytest.raises(ValidationError, match="greater than 0"):
            base.override(JobConfig(shots=-1))

        with pytest.raises(ValidationError, match="greater than 0"):
            base.override(JobConfig(shots=0))

        result = base.override(JobConfig(shots=500))
        assert result.shots == 500


class TestDeviceConfig:
    """Per-job hardware options and their wire format."""

    def test_unset_options_are_omitted(self):
        assert DeviceConfig().to_payload() == {}

    def test_every_option_reaches_the_payload_together(self):
        """Guards against a serialisation regression that only drops some fields."""
        options = {
            "optimization_level": 1,
            "resilience_level": 2,
            "max_execution_time": 300,
            "transpilation_seed": 7,
            "layout_method": "dense",
            "routing_method": "sabre",
            "approximation_degree": 0.9,
        }
        assert DeviceConfig(**options).to_payload() == options

    def test_a_misspelled_option_is_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DeviceConfig(optimisation_level=1)

    @pytest.mark.parametrize(
        "field",
        [
            "optimization_level",
            "resilience_level",
            "max_execution_time",
            "transpilation_seed",
        ],
    )
    @pytest.mark.parametrize("value", [True, "3", 1.0])
    def test_integer_options_reject_coercible_values(self, field, value):
        """The service rejects a JSON ``true`` or ``"3"`` as an integer, so fail here."""
        with pytest.raises(ValidationError):
            DeviceConfig(**{field: value})

    @pytest.mark.parametrize("value", [True, "0.9"])
    def test_approximation_degree_rejects_coercible_values(self, value):
        with pytest.raises(ValidationError):
            DeviceConfig(approximation_degree=value)

    @pytest.mark.parametrize("value", [1, 0.9])
    def test_approximation_degree_takes_an_int_or_a_float(self, value):
        assert DeviceConfig(approximation_degree=value).approximation_degree == value

    def test_method_names_are_capped_at_64_characters(self):
        with pytest.raises(ValidationError, match="at most 64 characters"):
            DeviceConfig(layout_method="x" * 65)

    def test_frozen(self):
        """Mutating a constructed config should raise an error."""
        config = DeviceConfig(optimization_level=1)
        with pytest.raises(ValidationError):
            config.optimization_level = 2
