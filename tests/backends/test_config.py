# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`divi.backends._config`."""

import pytest
from pydantic import ValidationError

from divi.backends import DeviceConfig, JobConfig, QPUSystem, SimulatorCluster
from tests._helpers import exact_match
from tests.backends._helpers import reset_unknown_message, validation_error_message


def _device_selection_message(key: str) -> str:
    return (
        f"['{key}'] would choose the device, which the job's QPU system "
        "already does."
    )


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

    @pytest.mark.parametrize("field", ["qpu_system", "simulator_cluster"])
    @pytest.mark.parametrize(
        "invalid_input",
        [123, ["a", "list"], {"a": "dict"}],
        ids=["integer_input", "list_input", "dict_input"],
    )
    def test_target_rejects_invalid_types(self, field, invalid_input):
        with pytest.raises(ValidationError):
            JobConfig(**{field: invalid_input})

    def test_simulator_cluster_accepts_valid_types(self):
        assert (
            JobConfig(simulator_cluster="my_cluster").simulator_cluster == "my_cluster"
        )
        cluster = SimulatorCluster(name="c")
        assert JobConfig(simulator_cluster=cluster).simulator_cluster == cluster
        assert JobConfig(simulator_cluster=None).simulator_cluster is None

    def test_rejects_both_targets(self):
        assert (
            validation_error_message(
                lambda: JobConfig(
                    simulator_cluster=SimulatorCluster(name="cluster"),
                    qpu_system=QPUSystem(name="qpu"),
                )
            )
            == "Provide either 'simulator_cluster' or 'qpu_system', not both."
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
        assert validation_error_message(lambda: JobConfig().override(shot=10)) == (
            "JobConfig got unknown fields 'shot' (did you mean 'shots'?)."
        )

    def test_model_copy_validates_like_override(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            JobConfig().model_copy(update={"shots": 0})

    def test_unknown_fields_are_all_listed_with_their_suggestions(self):
        assert validation_error_message(lambda: JobConfig(shot=1, zzz=2)) == (
            "JobConfig got unknown fields 'shot' (did you mean 'shots'?), 'zzz'."
        )

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
        with pytest.raises(
            ValueError, match=exact_match(reset_unknown_message("shot", "shots"))
        ):
            JobConfig().reset("shot")

    def test_an_empty_reset_is_rejected(self):
        with pytest.raises(
            ValueError, match=exact_match("reset() needs at least one field name.")
        ):
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
        """The service stores toggles as the strings its workers compare against;
        other strings are kept as they are."""
        stored = {
            "TRANSPILE_LEVEL": 1,
            "USE_TWIRLING": "true",
            "USE_MITIGATION": "false",
            "IBM_MODE": "fast",
        }
        assert DeviceConfig.from_payload(stored) == DeviceConfig(
            transpile_level=1,
            use_twirling=True,
            use_mitigation=False,
            ibm_mode="fast",
        )

    @pytest.mark.parametrize(
        "build, key",
        [
            (lambda: DeviceConfig(ibm_device="some-device"), "IBM_DEVICE"),
            (lambda: DeviceConfig(IQM_DEVICE_URL="some-device"), "IQM_DEVICE_URL"),
            (
                lambda: DeviceConfig().model_copy(update={"ibm_device": "some-device"}),
                "IBM_DEVICE",
            ),
        ],
        ids=["constructor_lowercase", "constructor_uppercase", "model_copy"],
    )
    def test_device_selection_is_rejected(self, build, key):
        assert validation_error_message(build) == _device_selection_message(key)

    @pytest.mark.parametrize(
        "change",
        [
            lambda config: config.override(TRANSPILE_LEVEL=3),
            lambda config: config.model_copy(update={"TRANSPILE_LEVEL": 3}),
        ],
        ids=["override", "model_copy"],
    )
    def test_changes_set_options_in_any_case(self, change):
        config = DeviceConfig(transpile_level=1, use_mitigation=True)
        assert change(config) == DeviceConfig(transpile_level=3, use_mitigation=True)

    def test_reset_drops_options(self):
        config = DeviceConfig(transpile_level=1, use_mitigation=True)
        assert config.reset("USE_MITIGATION") == DeviceConfig(transpile_level=1)

    def test_reset_rejects_options_that_are_not_set(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                reset_unknown_message("transpile_levl", "transpile_level")
            ),
        ):
            DeviceConfig(transpile_level=1).reset("transpile_levl")


@pytest.mark.parametrize(
    "config",
    [JobConfig(shots=7), DeviceConfig(transpile_level=1)],
    ids=["job_config", "device_config"],
)
def test_model_copy_without_update_is_an_equal_copy(config):
    assert config.model_copy() == config
