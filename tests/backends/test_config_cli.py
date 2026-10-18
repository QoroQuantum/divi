# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tomllib
from pathlib import Path
from types import SimpleNamespace

import maestro
import pytest
import requests
from rich.console import Console
from textual.widgets import DataTable, Input, Static

from divi.backends import (
    QPU,
    DeviceConfig,
    JobConfig,
    MaestroConfig,
    QoroService,
    QPUSystem,
    SimulatorCluster,
)
from divi.backends._config_cli import (
    _MAESTRO_TYPES,
    ConfigHelper,
    _ConfirmScreen,
    _device_options,
    _first_pass,
    _maestro_specs,
    _maestro_value,
    _merge,
    _option_label,
    _OptionsScreen,
    _outside,
    _parse_value,
    _python,
    _review,
    _TargetScreen,
    _TextEditor,
    main,
)
from divi.backends._saved_defaults import load_defaults, save_defaults

_TOGGLE = {"kind": "toggle", "choices": ["true", "false"]}

_BLUEPRINTS = {
    "ibm": {
        "label": "IBM",
        "credentials": {},
        "device": {
            "IBM_DEVICE": {"kind": "text", "ui_editable": True},
            "TRANSPILE_OPTIMIZATION_LEVEL": {
                "kind": "number",
                "ui_editable": True,
                "default": 2,
                "min": 0,
                "max": 3,
                "description": "How hard the transpiler optimises.",
            },
            "AGGRESSIVE_COMPILING": {**_TOGGLE, "ui_editable": True, "default": "true"},
            "TWIRLING_STRATEGY": {
                "kind": "choice",
                "choices": ["active", "all"],
                "ui_editable": False,
            },
            "TRANSPILE_INITIAL_LAYOUT": {"kind": "list", "item_kind": "number"},
        },
    },
    "iqm": {
        "label": "IQM",
        "credentials": {},
        "device": {
            "IQM_DEVICE_URL": {"kind": "text", "ui_editable": True},
            "AGGRESSIVE_COMPILING": {
                **_TOGGLE,
                "ui_editable": True,
                "default": "false",
            },
        },
    },
    "braket": {"label": "BRAKET", "credentials": {}, "device": {}},
}

_NOTHING = (None, None, None)
_HW_AT_LEVEL_1 = (
    JobConfig(qpu_system="hw"),
    None,
    DeviceConfig(transpile_optimization_level=1),
)
_KEY_NAMES = {",": "comma", "-": "minus", ".": "full_stop"}
_SAVE = ("s", "enter")
_LEVEL = "TRANSPILE_OPTIMIZATION_LEVEL"


def _system(*vendor_types):
    return QPUSystem(
        name="hw",
        qpus=[
            QPU(
                nickname=f"qpu{i}",
                q_bits=5,
                status="ONLINE",
                system_kind="SUPERCONDUCTING_QPU",
                vendor_type=vendor_type,
            )
            for i, vendor_type in enumerate(vendor_types)
        ],
    )


def _render(renderable):
    console = Console(width=120, record=True)
    console.print(renderable)
    return console.export_text()


class _Row(str):
    """A step that moves the table cursor to the row with this key."""


def _row(key):
    return _Row(key)


def _type(text):
    return [_KEY_NAMES.get(char, char) for char in text]


def _edit(key, text):
    return [_row(key), "enter", *_type(text), "enter"]


def _pick(target, *, edit_only=False):
    return [_row(target), "e" if edit_only else "enter"]


def _press(app, *steps, read=lambda screen: screen):
    """Run ``steps`` against ``app`` and return ``read`` of the screen showing at the end.

    ``read`` runs while the app is still up, so it can query the screen's widgets.
    """

    async def drive():
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            for step in steps:
                if isinstance(step, _Row):
                    table = app.screen.query_one(DataTable)
                    table.move_cursor(row=table.get_row_index(step))
                else:
                    await pilot.press(step)
            await pilot.pause()
            return read(app.screen)

    return asyncio.run(drive())


def _error(screen):
    return str(screen.query_one(".error", Static).content)


def _pane(screen):
    return str(screen.query_one(".describe", Static).content)


def _row_count(screen):
    return screen.query_one(DataTable).row_count


@pytest.fixture
def helper(defaults_file):
    """Build the app on targets ``hw`` (given vendors, IBM by default) and ``eon_cpu``."""

    def build(*vendors, saved=_NOTHING, advanced=False):
        targets = [_system(*(vendors or ("ibm",))), SimulatorCluster(name="eon_cpu")]
        return ConfigHelper(
            defaults_file, targets, _BLUEPRINTS, saved, show_advanced=advanced
        )

    return build


def test_a_mixed_system_merges_a_shared_key_and_names_each_vendors_default():
    spec = _device_options(_system("ibm", "iqm"), _BLUEPRINTS)["AGGRESSIVE_COMPILING"]

    assert spec["kind"] == "toggle"
    assert spec["defaults"] == "IBM on · IQM off"


def test_merged_bounds_are_the_tightest():
    blueprints = {
        "ibm": {"device": {"LEVEL": {"kind": "number", "min": 0, "max": 3}}},
        "iqm": {"device": {"LEVEL": {"kind": "number", "min": 1, "max": 5}}},
    }

    assert (
        _device_options(_system("ibm", "iqm"), blueprints)["LEVEL"]["allowed"] == "1–3"
    )


def test_device_options_are_those_of_the_systems_vendors_minus_selection_keys():
    assert list(_device_options(_system("iqm"), _BLUEPRINTS)) == [
        "AGGRESSIVE_COMPILING"
    ]
    assert _device_options(_system("braket"), _BLUEPRINTS) == {}
    assert (
        _device_options(_system("ibm", "iqm"), _BLUEPRINTS)
        .keys()
        .isdisjoint({"IBM_DEVICE", "IQM_DEVICE_URL"})
    )


def test_a_system_without_vendor_types_is_refused():
    with pytest.raises(ValueError, match="too old"):
        _device_options(_system(None), _BLUEPRINTS)


def test_the_first_pass_is_the_ui_editable_keys():
    options = _device_options(_system("ibm"), _BLUEPRINTS)

    assert _first_pass(options) == [_LEVEL, "AGGRESSIVE_COMPILING"]


@pytest.mark.parametrize(
    "text, spec, expected",
    [
        ("NO", _TOGGLE, False),
        ("on", _TOGGLE, True),
        ("auto", {"kind": "number", "choices": ["auto"]}, "auto"),
        ("0.5", {"kind": "float"}, 0.5),
        ("0, 1", {"kind": "list", "item_kind": "number"}, [0, 1]),
        ("3600", {"kind": "text"}, 3600),
        ("30m", {"kind": "text"}, "30m"),
    ],
)
def test_text_is_parsed_to_its_kind(text, spec, expected):
    assert _parse_value(text, spec) == expected


@pytest.mark.parametrize(
    "text, spec", [("high", {"kind": "number"}), ("maybe", _TOGGLE)]
)
def test_unparseable_text_raises(text, spec):
    with pytest.raises(ValueError):
        _parse_value(text, spec)


@pytest.mark.parametrize(
    "spec, allowed",
    [
        ({"kind": "number", "min": 1, "max": 9}, "1–9"),
        ({"kind": "number", "min": 1, "choices": ["auto"]}, "auto or ≥ 1"),
        ({"kind": "choice", "choices": ["a", "b"]}, "a | b"),
        ({"kind": "choice", "choices": list("abcde")}, "5 choices"),
        ({"kind": "list", "item_kind": "number"}, "list of number, comma-separated"),
    ],
)
def test_what_an_option_takes_is_spelled_out(spec, allowed):
    assert _merge({"vendor": spec})["allowed"] == allowed


@pytest.mark.parametrize(
    "spec, value, outside",
    [
        ({"kind": "number", "min": 1, "max": 3, "choices": []}, 9, True),
        ({"kind": "number", "min": 1, "max": 3, "choices": []}, 2, False),
        (
            {"kind": "choice", "min": None, "max": None, "choices": ["a", "b"]},
            "c",
            True,
        ),
    ],
)
def test_a_value_outside_the_stated_range_or_choices_is_flagged(spec, value, outside):
    assert _outside(spec, value) is outside


@pytest.mark.parametrize(
    "name, kind",
    [
        ("simulation_type", "choice"),
        ("max_bond_dimension", "number"),
        ("gate_fusion", "toggle"),
        ("mps_svd_solver", "choice"),
    ],
)
def test_maestro_specs_have_the_kind_each_option_takes(name, kind):
    assert _maestro_specs()[name]["kind"] == kind


def test_mapping_valued_maestro_options_are_not_offered():
    assert "distributed_options" not in _maestro_specs()


def test_the_kinds_table_names_only_options_maestro_has():
    assert _MAESTRO_TYPES.keys() <= set(maestro.SimulatorConfig._fields)


def test_maestro_accepts_what_every_options_spec_offers():
    for name, spec in _maestro_specs().items():
        offered = {
            "toggle": [True, False],
            "number": [1],
            "float": [0.5],
            "choice": spec["choices"],
            "text": [],
        }[spec["kind"]]
        for value in offered:
            _maestro_value(name, value)


def test_every_maestro_option_is_described_by_maestros_own_docstring():
    for name, spec in _maestro_specs().items():
        docstring = getattr(maestro.SimulatorConfig, name).__doc__

        assert spec["description"] == " ".join(docstring.split())


def test_a_blueprint_description_is_kept_and_the_first_vendors_wins():
    blueprints = {
        "ibm": {"device": {"X": {"kind": "text", "description": "Explains X."}}},
        "iqm": {"device": {"X": {"kind": "text", "description": "Other."}}},
    }

    spec = _device_options(_system("ibm", "iqm"), blueprints)["X"]

    assert spec["description"] == "Explains X."


def test_the_python_form_shows_every_saved_defaults():
    python = _python(
        {"qpu_system": "hw", "shots": 500},
        {"simulation_type": maestro.SimulationType.MatrixProductState},
        {"transpile_optimization_level": 2},
    )

    assert python == (
        "QoroService(job_config=JobConfig(qpu_system='hw', shots=500), "
        "maestro_config=MaestroConfig(simulation_type='MatrixProductState'), "
        "device_config=DeviceConfig(transpile_optimization_level=2))"
    )


def test_the_review_lists_each_kind_of_change_the_warnings_and_the_python_form():
    out = _render(
        _review(
            _system("ibm"),
            "eon",
            True,
            {_LEVEL: 1, "AGGRESSIVE_COMPILING": True, "SEED": 3},
            {_LEVEL: 9, "SEED": 3, "TWIRLING_STRATEGY": "all"},
            ["shots=500"],
            ["transpile_optimization_level = 9 (allowed: 0–3)"],
            "QoroService(device_config=DeviceConfig(transpile_optimization_level=9))",
        )
    )

    assert "1 → 9" in out
    assert "aggressive_compiling" in out and "cleared" in out
    assert "3  kept" in out
    assert "twirling_strategy" in out
    assert "(was eon)" in out
    assert "shots=500" in out
    assert "will reject jobs with" in out and "allowed: 0–3" in out
    assert (
        "QoroService(device_config=DeviceConfig(transpile_optimization_level=9))" in out
    )


def test_editing_options_only_leaves_the_default_target_out_of_the_review():
    out = _render(_review(_system("ibm"), "eon", False, {}, {"SEED": 1}, [], [], "x"))

    assert "eon  (unchanged)" in out and "editing hw" in out


@pytest.mark.parametrize(
    "key, label",
    [("simulation_type", "Simulation type"), ("mps_svd_solver", "MPS SVD solver")],
)
def test_option_labels_read_as_words(key, label):
    assert _option_label(key) == label


def test_a_qpu_target_and_a_typed_option_are_saved(helper, defaults_file):
    app = helper()

    _press(app, *_pick("hw"), *_edit(_LEVEL, "2"), *_SAVE)

    job_config, _, device_config = load_defaults(defaults_file)
    assert job_config.qpu_system == "hw"
    assert device_config == DeviceConfig(transpile_optimization_level=2)
    assert app.return_value == defaults_file


@pytest.mark.parametrize("choices_down, expected", [(0, True), (1, False)])
def test_a_toggle_is_picked_from_an_on_off_list(
    helper, defaults_file, choices_down, expected
):
    _press(
        helper(),
        *_pick("hw"),
        _row("AGGRESSIVE_COMPILING"),
        "enter",
        *["down"] * choices_down,
        "enter",
        *_SAVE,
    )

    assert load_defaults(defaults_file)[2] == DeviceConfig(
        aggressive_compiling=expected
    )


def test_a_target_alone_can_be_saved(helper, defaults_file):
    _press(helper(), *_pick("eon_cpu"), *_SAVE)

    job_config, maestro_config, device_config = load_defaults(defaults_file)
    assert job_config == JobConfig(simulator_cluster="eon_cpu")
    assert maestro_config is None and device_config is None


def test_the_advanced_options_start_collapsed_and_open_with_a_or_the_heading(helper):
    collapsed = _press(helper(), *_pick("hw"), read=_row_count)

    assert _press(helper(), *_pick("hw"), "a", read=_row_count) > collapsed
    assert (
        _press(helper(), *_pick("hw"), _row("advanced"), "enter", read=_row_count)
        > collapsed
    )
    assert _press(helper(advanced=True), *_pick("hw"), read=_row_count) > collapsed


def test_a_list_option_is_typed_as_comma_separated_values(helper, defaults_file):
    _press(
        helper(advanced=True),
        *_pick("hw"),
        *_edit("TRANSPILE_INITIAL_LAYOUT", "0,1"),
        *_SAVE,
    )

    assert load_defaults(defaults_file)[2] == DeviceConfig(
        transpile_initial_layout=[0, 1]
    )


def test_text_that_is_not_a_number_is_reported_in_the_editor(helper):
    screen, error = _press(
        helper(),
        *_pick("eon_cpu"),
        *_edit("max_bond_dimension", "abc"),
        read=lambda screen: (screen, _error(screen)),
    )

    assert isinstance(screen, _TextEditor)
    assert error == "Max bond dimension takes integer; 'abc' was not accepted."


def test_a_value_maestro_refuses_is_reported_in_the_editor(helper):
    error = _press(
        helper(advanced=True),
        *_pick("eon_cpu"),
        *_edit("gpu_device", "-1"),
        read=_error,
    )

    assert error == "GPU device takes integer; '-1' was not accepted."


def test_a_simulator_cluster_with_enum_and_number_options_is_saved(
    helper, defaults_file
):
    members = _maestro_specs()["simulation_type"]["choices"]

    _press(
        helper(),
        *_pick("eon_cpu"),
        _row("simulation_type"),
        "enter",
        "down",
        "enter",
        *_edit("max_bond_dimension", "64"),
        *_SAVE,
    )

    job_config, maestro_config, _ = load_defaults(defaults_file)
    assert job_config.simulator_cluster == "eon_cpu"
    assert maestro_config == MaestroConfig(
        simulation_type=members[1], max_bond_dimension=64
    )


def test_float_and_optional_toggle_options_are_set_in_the_advanced_view(
    helper, defaults_file
):
    _press(
        helper(advanced=True),
        *_pick("eon_cpu"),
        *_edit("singular_value_threshold", "0.5"),
        _row("gate_fusion"),
        "enter",
        "enter",
        *_SAVE,
    )

    assert load_defaults(defaults_file)[1] == MaestroConfig(
        singular_value_threshold=0.5, gate_fusion=True
    )


def test_editing_a_simulator_keeps_the_settings_maestro_options_do_not_cover(
    helper, defaults_file
):
    saved = MaestroConfig(
        noise_seed=7, max_bond_dimension=8, distributed_options={"distributed_x": "1"}
    )

    _press(
        helper(saved=(JobConfig(simulator_cluster="eon_cpu"), saved, None)),
        *_pick("eon_cpu"),
        _row("max_bond_dimension"),
        "enter",
        "backspace",
        *_type("64"),
        "enter",
        *_SAVE,
    )

    assert load_defaults(defaults_file)[1] == MaestroConfig(
        noise_seed=7, max_bond_dimension=64, distributed_options={"distributed_x": "1"}
    )


def test_choosing_another_kind_keeps_the_rest_of_the_file(helper, defaults_file):
    saved = (
        JobConfig(qpu_system="hw", shots=500),
        None,
        DeviceConfig(use_mitigation=True),
    )

    _press(
        helper(saved=saved),
        *_pick("eon_cpu"),
        *_edit("max_bond_dimension", "64"),
        *_SAVE,
    )

    job_config, maestro_config, device_config = load_defaults(defaults_file)
    assert job_config.simulator_cluster == "eon_cpu"
    assert job_config.qpu_system is None
    assert job_config.shots == 500
    assert device_config == DeviceConfig(use_mitigation=True)
    assert maestro_config == MaestroConfig(max_bond_dimension=64)


def test_options_can_be_edited_without_changing_the_default_target(
    helper, defaults_file
):
    saved = (JobConfig(simulator_cluster="eon_cpu"), None, None)

    _press(
        helper(saved=saved),
        *_pick("hw", edit_only=True),
        *_edit(_LEVEL, "1"),
        *_SAVE,
    )

    job_config, _, device_config = load_defaults(defaults_file)
    assert job_config.simulator_cluster == "eon_cpu"
    assert job_config.qpu_system is None
    assert device_config == DeviceConfig(transpile_optimization_level=1)


def test_delete_clears_the_highlighted_option(helper, defaults_file):
    _press(helper(saved=_HW_AT_LEVEL_1), *_pick("hw"), _row(_LEVEL), "delete", *_SAVE)

    assert load_defaults(defaults_file)[2] is None


def test_an_empty_text_box_clears_the_option(helper, defaults_file):
    _press(
        helper(saved=_HW_AT_LEVEL_1),
        *_pick("hw"),
        _row(_LEVEL),
        "enter",
        "backspace",
        "enter",
        *_SAVE,
    )

    assert load_defaults(defaults_file)[2] is None


def test_clearing_the_last_maestro_option_drops_the_section(helper, defaults_file):
    saved = (
        JobConfig(simulator_cluster="eon_cpu"),
        MaestroConfig(max_bond_dimension=8),
        None,
    )

    _press(
        helper(saved=saved),
        *_pick("eon_cpu"),
        _row("max_bond_dimension"),
        "delete",
        *_SAVE,
    )

    assert load_defaults(defaults_file)[1] is None


def test_the_review_warns_about_a_value_outside_the_stated_range(helper):
    screen = _press(helper(), *_pick("hw"), *_edit(_LEVEL, "9"), "s")

    out = _render(screen.review)
    assert "will reject jobs with" in out
    assert "transpile_optimization_level = 9 (allowed: 0–3)" in out


def test_the_review_lists_what_else_stays_in_the_file(helper):
    saved = (
        JobConfig(qpu_system="hw", shots=500),
        MaestroConfig(max_bond_dimension=8),
        None,
    )

    screen = _press(helper(saved=saved), *_pick("hw"), *_edit(_LEVEL, "2"), "s")

    out = _render(screen.review)
    assert "shots=500" in out
    assert "Maestro settings for simulator targets" in out


def test_saving_without_changes_says_so_and_writes_nothing(
    helper, mocker, defaults_file
):
    save_defaults(defaults_file, job_config=JobConfig(qpu_system="hw"))
    before = defaults_file.read_bytes()
    app = helper(saved=(JobConfig(qpu_system="hw"), None, None))
    notify = mocker.spy(app, "notify")

    screen = _press(app, *_pick("hw"), "s")

    assert isinstance(screen, _OptionsScreen)
    notify.assert_called_once_with("No changes to save.")
    assert defaults_file.read_bytes() == before


def test_backing_out_of_the_review_saves_nothing(helper, defaults_file):
    screen = _press(helper(), *_pick("hw"), *_edit(_LEVEL, "2"), "s", "escape")

    assert isinstance(screen, _OptionsScreen)
    assert not defaults_file.exists()


def test_a_failed_save_keeps_the_edits_on_screen(helper, mocker, defaults_file):
    defaults_file.parent.write_text("a file where the folder should be")
    app = helper()
    notify = mocker.spy(app, "notify")

    screen = _press(app, *_pick("hw"), *_edit(_LEVEL, "2"), *_SAVE)

    assert isinstance(screen, _OptionsScreen)
    assert screen.values == {_LEVEL: 2}
    assert app.return_value is None
    assert notify.call_args.args[0].startswith("Could not save: ")


def test_escape_on_the_options_returns_to_the_targets(helper):
    assert isinstance(_press(helper(), *_pick("hw"), "escape"), _TargetScreen)


def test_leaving_with_unsaved_edits_asks_first(helper):
    screen = _press(helper(), *_pick("hw"), *_edit(_LEVEL, "2"), "escape")

    assert isinstance(screen, _ConfirmScreen)
    assert screen.question == "Discard 1 unsaved change?"


@pytest.mark.parametrize(
    "answer, destination", [("y", _TargetScreen), ("n", _OptionsScreen)]
)
def test_the_discard_question_is_answered_with_keys(
    helper, defaults_file, answer, destination
):
    screen = _press(helper(), *_pick("hw"), *_edit(_LEVEL, "2"), "escape", answer)

    assert isinstance(screen, destination)
    assert not defaults_file.exists()


def test_quitting_with_unsaved_edits_asks_and_then_saves_nothing(helper, defaults_file):
    app = helper()

    _press(app, *_pick("hw"), *_edit(_LEVEL, "2"), "q", "y")

    assert app.return_value is None
    assert not defaults_file.exists()


def test_the_saved_target_is_highlighted_first(helper):
    saved = (JobConfig(simulator_cluster="eon_cpu"), None, None)

    screen = _press(helper(saved=saved), "enter")

    assert screen.target.name == "eon_cpu"


def test_a_target_with_no_options_says_so(helper):
    screen = _press(helper("braket"), *_pick("hw"))

    assert screen.notes == [
        "BRAKET QPUs take no device options; they keep their own settings.",
        "No options to set for this target.",
    ]


def test_a_system_without_vendor_types_returns_to_the_targets(helper):
    assert isinstance(_press(helper(None), *_pick("hw")), _TargetScreen)


def test_the_text_box_starts_with_the_saved_value(helper):
    screen, value = _press(
        helper(saved=_HW_AT_LEVEL_1),
        *_pick("hw"),
        _row(_LEVEL),
        "enter",
        read=lambda screen: (screen, screen.query_one(Input).value),
    )

    assert isinstance(screen, _TextEditor)
    assert value == "1"


def test_escape_cancels_an_editor_without_changing_anything(helper):
    screen = _press(helper(), *_pick("hw"), _row(_LEVEL), "enter", "escape")

    assert isinstance(screen, _OptionsScreen)
    assert screen.values == {}


def test_the_pane_describes_the_highlighted_option_and_follows_the_cursor(helper):
    specs = _maestro_specs()

    on_bond = _press(
        helper(), *_pick("eon_cpu"), _row("max_bond_dimension"), read=_pane
    )
    on_type = _press(helper(), *_pick("eon_cpu"), _row("simulation_type"), read=_pane)

    assert on_bond == specs["max_bond_dimension"]["description"]
    assert on_type == specs["simulation_type"]["description"]


def test_the_pane_shows_a_service_description_and_is_empty_without_one(helper):
    described = _press(helper(), *_pick("hw"), _row(_LEVEL), read=_pane)
    undescribed = _press(
        helper(), *_pick("hw"), _row("AGGRESSIVE_COMPILING"), read=_pane
    )
    heading = _press(helper(), *_pick("hw"), _row("advanced"), read=_pane)

    assert described == "How hard the transpiler optimises."
    assert undescribed == heading == ""


def test_the_editor_repeats_the_description(helper):
    labels = _press(
        helper(),
        *_pick("eon_cpu"),
        _row("simulation_type"),
        "enter",
        read=lambda screen: [
            str(each.content) for each in screen.query(".description")
        ],
    )

    assert labels == [_maestro_specs()["simulation_type"]["description"]]


@pytest.fixture
def served(mocker):
    service = mocker.create_autospec(QoroService, instance=True)
    service.fetch_vendor_blueprints.return_value = _BLUEPRINTS
    patched = "divi.backends._config_cli."
    app = mocker.patch(patched + "ConfigHelper")
    app.return_value.run.return_value = None
    mocker.patch(patched + "get_available_qpu_systems", return_value=[_system("ibm")])
    mocker.patch(
        patched + "get_available_simulator_clusters",
        return_value=[SimulatorCluster(name="eon_cpu")],
    )
    return SimpleNamespace(
        service=mocker.patch(patched + "QoroService", return_value=service),
        app=app,
    )


def test_main_fetches_the_targets_without_the_saved_defaults(served):
    main([])

    served.service.assert_called_once_with(use_saved_defaults=False)


def test_main_opens_the_app_on_the_saved_file_and_passes_the_advanced_flag(
    served, defaults_file
):
    save_defaults(defaults_file, job_config=JobConfig(qpu_system="hw"))

    main(["--advanced"])

    path, targets, blueprints, saved = served.app.call_args.args
    assert path == defaults_file
    assert [target.name for target in targets] == ["hw", "eon_cpu"]
    assert blueprints == _BLUEPRINTS
    assert saved == load_defaults(defaults_file)
    assert served.app.call_args.kwargs == {"show_advanced": True}


def test_a_broken_existing_file_is_reported_and_replaced(served, defaults_file, capsys):
    defaults_file.parent.mkdir(parents=True)
    defaults_file.write_text("{not json")

    main([])

    assert "It will be replaced if you save." in capsys.readouterr().out
    assert served.app.call_args.args[3] == _NOTHING


@pytest.mark.parametrize(
    "failure, message",
    [
        (requests.ConnectionError("down"), "Could not reach the Qoro Service: down"),
        (ValueError("no key"), "no key"),
    ],
    ids=["network", "service"],
)
def test_a_failure_to_load_the_targets_is_one_line_not_a_traceback(
    served, capsys, failure, message
):
    served.service.side_effect = failure

    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().out


def test_a_key_with_no_targets_is_told_so(served, mocker, capsys):
    mocker.patch("divi.backends._config_cli.get_available_qpu_systems", return_value=[])
    mocker.patch(
        "divi.backends._config_cli.get_available_simulator_clusters", return_value=[]
    )

    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 1
    assert "no QPU systems or simulator clusters" in capsys.readouterr().out


def test_the_console_script_is_registered():
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert (
        pyproject["project"]["scripts"]["divi-config-helper"]
        == "divi.backends._config_cli:main"
    )
