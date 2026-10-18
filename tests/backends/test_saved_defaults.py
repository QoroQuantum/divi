# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from divi.backends import DeviceConfig, JobConfig, MaestroConfig, SimulatorCluster
from divi.backends._saved_defaults import (
    defaults_path,
    load_defaults,
    save_defaults,
)


def test_round_trip_keeps_each_config(defaults_file):
    maestro_config = MaestroConfig(
        max_bond_dimension=64, simulation_type="MatrixProductState"
    )
    device_config = DeviceConfig(
        twirling_enable_gates=True, transpile_optimization_level=1
    )

    save_defaults(
        defaults_file,
        job_config=JobConfig(qpu_system="hw", shots=500),
        maestro_config=maestro_config,
        device_config=device_config,
    )
    job_config, loaded_maestro, loaded_device = load_defaults(defaults_file)

    assert loaded_maestro == maestro_config
    assert loaded_device == device_config
    assert job_config.model_fields_set == {"qpu_system", "shots"}
    assert job_config.qpu_system == "hw"
    assert job_config.shots == 500


def test_target_objects_are_saved_as_names_and_unset_fields_are_left_out(
    defaults_file,
):
    save_defaults(
        defaults_file,
        job_config=JobConfig(
            simulator_cluster=SimulatorCluster(name="eon_cpu"), tag=None
        ),
    )

    saved = json.loads(defaults_file.read_text())

    assert saved == {"job_config": {"simulator_cluster": "eon_cpu"}}


def test_the_default_path_is_qoro_json_in_the_user_config_folder(defaults_file):
    assert defaults_path() == defaults_file

    save_defaults(defaults_file, device_config=DeviceConfig(use_mitigation=True))

    assert load_defaults() == (None, None, DeviceConfig(use_mitigation=True))


def test_saving_leaves_no_temporary_file_behind(defaults_file):
    save_defaults(defaults_file, device_config=DeviceConfig(use_mitigation=True))

    assert [entry.name for entry in defaults_file.parent.iterdir()] == ["qoro.json"]


def test_a_failed_replace_keeps_the_old_file_and_leaves_nothing_behind(
    mocker, defaults_file
):
    save_defaults(defaults_file, device_config=DeviceConfig(use_mitigation=True))
    before = defaults_file.read_bytes()
    mocker.patch(
        "divi.backends._saved_defaults.os.replace", side_effect=OSError("disk full")
    )

    with pytest.raises(OSError, match="disk full"):
        save_defaults(defaults_file, device_config=DeviceConfig(use_mitigation=False))

    assert defaults_file.read_bytes() == before
    assert [entry.name for entry in defaults_file.parent.iterdir()] == ["qoro.json"]


def test_a_missing_file_loads_nothing(defaults_file):
    assert load_defaults(defaults_file) == (None, None, None)


def test_a_path_that_cannot_be_read_names_itself(defaults_file):
    defaults_file.mkdir(parents=True)

    with pytest.raises(ValueError, match="directory") as exc_info:
        load_defaults(defaults_file)

    assert str(exc_info.value).startswith(str(defaults_file))


def test_absent_sections_are_left_out(defaults_file):
    save_defaults(defaults_file, device_config=DeviceConfig(use_mitigation=True))

    assert load_defaults(defaults_file) == (
        None,
        None,
        DeviceConfig(use_mitigation=True),
    )


@pytest.mark.parametrize(
    "content, problem",
    [
        ("{not json", "Expecting property name"),
        (json.dumps({"simulator": {}}), "simulator"),
        (json.dumps({"job_config": {"shotz": 1}}), "shotz"),
        (json.dumps({"job_config": {"shots": "many"}}), "shots"),
        (json.dumps({"maestro_config": {"bogus_option": 1}}), "bogus_option"),
        (json.dumps({"maestro_config": {"max_bond_dimension": "wide"}}), "maestro"),
        ("[]", "JSON object"),
        (json.dumps({"job_config": []}), "job_config"),
        (json.dumps({"maestro_config": {"noise_model": [{"x": 1}]}}), "method"),
    ],
    ids=[
        "invalid_json",
        "unknown_section",
        "unknown_job_field",
        "wrong_typed_shots",
        "unknown_maestro",
        "wrong_typed_maestro_value",
        "not_an_object",
        "section_not_an_object",
        "noise_call_without_method",
    ],
)
def test_a_bad_file_names_itself_and_the_problem(defaults_file, content, problem):
    defaults_file.parent.mkdir(parents=True)
    defaults_file.write_text(content)

    with pytest.raises(ValueError, match=problem) as exc_info:
        load_defaults(defaults_file)

    assert str(exc_info.value).startswith(str(defaults_file))
