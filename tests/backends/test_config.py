# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`divi.backends._config`."""

import pytest
from pydantic import ValidationError

from divi.backends import DeviceConfig, JobConfig, QPUSystem, SimulatorCluster


class TestJobConfig:
    """JobConfig field validation, ``override()`` and ``reset()``."""

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

    def test_shots_default_to_1000(self):
        assert JobConfig(qpu_system="qpu").shots == 1000

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

    def test_override_sets_only_the_given_fields(self):
        base = JobConfig(shots=1000, tag="base", force_sampling=True)
        result = base.override(shots=500)
        assert result == JobConfig(shots=500, tag="base", force_sampling=True)

    def test_override_returns_a_new_config(self):
        base = JobConfig(shots=1000)
        result = base.override(shots=500)
        assert base.shots == 1000
        assert result is not base

    @pytest.mark.parametrize(
        "base, fields, cleared",
        [
            (
                JobConfig(simulator_cluster="sim"),
                {"qpu_system": "qpu"},
                "simulator_cluster",
            ),
            (JobConfig(qpu_system="qpu"), {"simulator_cluster": "sim"}, "qpu_system"),
        ],
        ids=["to_qpu", "to_simulator"],
    )
    def test_override_setting_one_target_clears_the_other(self, base, fields, cleared):
        assert getattr(base.override(**fields), cleared) is None

    def test_override_validates_like_the_constructor(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            JobConfig().override(shots=0)
        with pytest.raises(ValidationError, match="did you mean 'shots'"):
            JobConfig().override(shot=10)

    def test_model_copy_validates_like_override(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            JobConfig().model_copy(update={"shots": 0})

    @pytest.mark.parametrize(
        "field, kept",
        [("simulator_cluster", "qpu_system"), ("qpu_system", "simulator_cluster")],
    )
    def test_override_with_a_none_target_keeps_the_other(self, field, kept):
        base = JobConfig(**{kept: "target"})
        assert getattr(base.override(**{field: None}), kept) == "target"

    def test_reset_restores_the_defaults(self):
        base = JobConfig(shots=1000, tag="base", force_sampling=True)
        assert base.reset("tag", "force_sampling") == JobConfig(shots=1000)

    def test_reset_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="did you mean 'shots'"):
            JobConfig().reset("shot")

    def test_an_empty_reset_is_rejected(self):
        with pytest.raises(ValueError, match="at least one field"):
            JobConfig().reset()


class TestDeviceConfig:
    """Per-job hardware options and their wire format."""

    def test_options_travel_under_the_service_keys(self):
        config = DeviceConfig(transpile_level=3, use_mitigation=True)
        assert config.to_payload() == {"TRANSPILE_LEVEL": 3, "USE_MITIGATION": True}

    def test_unset_options_are_omitted(self):
        assert DeviceConfig(use_twirling=None).to_payload() == {}

    def test_key_case_does_not_matter(self):
        assert DeviceConfig(TRANSPILE_LEVEL=3) == DeviceConfig(transpile_level=3)

    def test_the_stored_form_rebuilds_the_config(self):
        """The service stores toggles as the strings its workers compare against."""
        stored = {
            "TRANSPILE_LEVEL": 1,
            "USE_TWIRLING": "true",
            "USE_MITIGATION": "false",
        }
        assert DeviceConfig.from_payload(stored) == DeviceConfig(
            transpile_level=1, use_twirling=True, use_mitigation=False
        )

    @pytest.mark.parametrize("key", ["ibm_device", "IQM_DEVICE_URL"])
    def test_device_selection_is_rejected(self, key):
        with pytest.raises(ValidationError, match="would choose the device"):
            DeviceConfig(**{key: "some-device"})

    def test_override_sets_options_in_any_case(self):
        config = DeviceConfig(transpile_level=1, use_mitigation=True)
        assert config.override(TRANSPILE_LEVEL=3) == DeviceConfig(
            transpile_level=3, use_mitigation=True
        )

    def test_reset_drops_options(self):
        config = DeviceConfig(transpile_level=1, use_mitigation=True)
        assert config.reset("USE_MITIGATION") == DeviceConfig(transpile_level=1)

    def test_reset_rejects_options_that_are_not_set(self):
        with pytest.raises(ValueError, match="did you mean 'transpile_level'"):
            DeviceConfig(transpile_level=1).reset("transpile_levl")

    def test_frozen(self):
        """Mutating a constructed config should raise an error."""
        config = DeviceConfig(transpile_level=1)
        with pytest.raises(ValidationError):
            config.transpile_level = 2
