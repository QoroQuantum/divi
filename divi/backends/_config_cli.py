# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""``divi-config-helper``: choose a default Qoro Service target and its options."""

import argparse
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

import maestro
import requests
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Input, Label, OptionList, Static

from ._config import _DEVICE_SELECTION_KEYS, _TARGET_FIELDS, DeviceConfig, JobConfig
from ._saved_defaults import (
    SavedDefaults,
    defaults_path,
    load_defaults,
    save_defaults,
)
from ._systems import (
    QPUSystem,
    SimulatorCluster,
    get_available_qpu_systems,
    get_available_simulator_clusters,
)
from .runners._maestro import _SIMULATOR_OPTIONS, MaestroConfig
from .runners._qoro import QoroService

_console = Console()

_MAESTRO_FIRST_PASS = ("simulator_type", "simulation_type", "max_bond_dimension")

_TOGGLE_WORDS = {
    "true": True,
    "yes": True,
    "on": True,
    "false": False,
    "no": False,
    "off": False,
}
_SCALAR_KINDS = {"number": int, "float": float}
_KINDS_BY_TYPE = {bool: "toggle", int: "number", float: "float"}
_ACRONYMS = frozenset({"mps", "mpo", "svd", "gpu", "qpu", "pp", "ibm", "iqm", "dd"})
_ADVANCED_ROW = "advanced"
_CLEAR = object()

_SVD_SOLVERS = ["gesvd", "gesvdj", "gesvdp", "gesvdr"]
_MAESTRO_TYPES: dict[str, dict[str, Any]] = (
    {
        name: {"kind": "number"}
        for name in (
            "max_bond_dimension",
            "seed",
            "gpu_device",
            "pp_max_pauli_weight",
            "pp_gates_between_trims",
            "pp_gates_between_deduplications",
        )
    }
    | {
        name: {"kind": "float"}
        for name in (
            "singular_value_threshold",
            "pp_coefficient_threshold",
            "path_integral_threshold",
        )
    }
    | {"gate_fusion": {"kind": "toggle"}}
    | {
        name: {"kind": "choice", "choices": choices}
        for name, choices in {
            "truncation_mode": ["relative_max", "discarded_weight"],
            "precision": ["single", "double"],
            "mps_sampling": ["probabilities", "apply_measure"],
            "mps_svd_solver": _SVD_SOLVERS,
            "mpo_svd_solver": _SVD_SOLVERS,
            "tensor_network_svd_solver": _SVD_SOLVERS,
            "mpo_kraus_completeness_check": ["ignore", "warn", "strict"],
        }.items()
    }
)

_CONFIG_TYPES = {
    "job_config": JobConfig,
    "maestro_config": MaestroConfig,
    "device_config": DeviceConfig,
}

_PINK, _BLUE, _TEXT, _MUTED = "#FF93FA", "#2967A2", "#E9EEF6", "#94A0AA"
_GREEN, _AMBER, _RED = "#50fa7b", "#FFB86C", "#ff5555"
_THEME = Theme(
    name="divi",
    primary=_PINK,
    secondary=_BLUE,
    accent=_PINK,
    success=_GREEN,
    warning=_AMBER,
    error=_RED,
    foreground=_TEXT,
    background="#151616",
    surface="#1E1F20",
    panel="#1E1F20",
    variables={"muted": _MUTED},
)

Spec = dict[str, Any]
Normalise = Callable[[str, Any], Any]
Target = QPUSystem | SimulatorCluster
Plan = tuple[dict[str, Any], dict[str, Any], dict[str, Any]]


def _toggle(text: str) -> bool:
    try:
        return _TOGGLE_WORDS[text.lower()]
    except KeyError:
        raise ValueError(f"{text!r} is not true or false.") from None


def _value_text(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def _has_choices(spec: Mapping[str, Any]) -> bool:
    return spec.get("kind") == "choice" and bool(spec["choices"])


def _allowed_text(spec: Spec) -> str:
    kind, choices = spec.get("kind"), spec["choices"]
    low, high = spec["min"], spec["max"]
    if kind == "toggle":
        return "on/off"
    if _has_choices(spec):
        listed = " | ".join(choices)
        return (
            listed
            if len(choices) <= 4 and len(listed) <= 28
            else f"{len(choices)} choices"
        )
    if kind == "list":
        return f"list of {spec.get('item_kind') or 'text'}, comma-separated"
    if kind not in _SCALAR_KINDS:
        text = "text"
    elif low is not None and high is not None:
        text = f"{low}–{high}"
    elif low is not None:
        text = f"≥ {low}"
    elif high is not None:
        text = f"≤ {high}"
    else:
        text = "integer" if kind == "number" else "number"
    return " or ".join([*choices, text])


def _outside(spec: Spec, value: Any) -> bool:
    """Whether ``value`` breaks the range or choices the service states for the option."""
    if _has_choices(spec):
        return _value_text(value) not in spec["choices"]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return (spec["min"] is not None and value < spec["min"]) or (
            spec["max"] is not None and value > spec["max"]
        )
    return False


def _rejected(label: str, spec: Spec, typed: str) -> str:
    return f"{label} takes {spec['allowed']}; {typed!r} was not accepted."


def _merge(by_vendor: Mapping[str, Mapping[str, Any]]) -> Spec:
    """One spec for an option several vendors accept: every choice, the tightest bounds."""
    specs = list(by_vendor.values())
    spec = dict(specs[0]) | {
        "choices": list(
            dict.fromkeys(str(c) for one in specs for c in one.get("choices") or ())
        ),
        "min": max(
            (bound for one in specs if (bound := one.get("min")) is not None),
            default=None,
        ),
        "max": min(
            (bound for one in specs if (bound := one.get("max")) is not None),
            default=None,
        ),
        "common": any(one.get("ui_editable") for one in specs),
        "description": next(
            (text for one in specs if (text := one.get("description"))), None
        ),
    }
    defaults = {
        vendor: _value_text(
            _TOGGLE_WORDS.get(str(one.get("default")), one.get("default"))
        )
        for vendor, one in by_vendor.items()
    }
    if len(set(defaults.values())) == 1:
        default = next(iter(defaults.values()))
        spec["defaults"] = (
            "not set by default" if default == "not set" else f"default {default}"
        )
    else:
        spec["defaults"] = " · ".join(f"{v.upper()} {d}" for v, d in defaults.items())
    spec["allowed"] = _allowed_text(spec)
    return spec


def _device_options(
    system: QPUSystem, blueprints: Mapping[str, Mapping]
) -> dict[str, Spec]:
    """Each device key the system's vendors accept, with its spec merged across them.

    Raises:
        ValueError: If the service does not report the system's vendors.
    """
    by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
    for vendor in dict.fromkeys(qpu.vendor_type for qpu in system.qpus):
        if vendor is None:
            raise ValueError(
                f"The Qoro Service does not report which vendor the QPUs of "
                f"'{system.name}' use, so it is too old for divi-config-helper."
            )
        for key, spec in blueprints.get(vendor, {}).get("device", {}).items():
            if key not in _DEVICE_SELECTION_KEYS:
                by_key.setdefault(key, {})[vendor] = spec
    return {key: _merge(specs) for key, specs in by_key.items()}


def _maestro_specs() -> dict[str, Spec]:
    """Maestro's options as specs; mapping-valued ones are left out."""
    defaults = maestro.SimulatorConfig()
    specs: dict[str, Spec] = {}
    for name in maestro.SimulatorConfig._fields:
        default = getattr(defaults, name)
        if isinstance(default, Mapping):
            continue
        if name in _MAESTRO_TYPES:
            spec = dict(_MAESTRO_TYPES[name])
        elif isinstance(default, Enum):
            spec = {"kind": "choice", "choices": list(type(default).__members__)}
        else:
            spec = {"kind": _KINDS_BY_TYPE.get(type(default), "text")}
        spec |= {
            "default": default,
            "ui_editable": name in _MAESTRO_FIRST_PASS,
            "description": " ".join(
                getattr(maestro.SimulatorConfig, name).__doc__.split()
            ),
        }
        specs[name] = _merge({"maestro": spec})
    return specs


def _first_pass(options: Mapping[str, Spec]) -> list[str]:
    """The keys shown before the advanced ones."""
    return [key for key, spec in options.items() if spec["common"]]


def _parse_value(text: str, spec: Mapping[str, Any]) -> Any:
    """``text`` as a value of the spec's kind; ranges and choices are the service's to check."""
    text = text.strip()
    kind = spec.get("kind")
    if kind == "toggle":
        return _toggle(text)
    if kind == "choice" or text in spec.get("choices", ()):
        return text
    if kind == "list":
        return [
            _parse_value(part, {"kind": spec.get("item_kind")})
            for part in text.split(",")
        ]
    if kind in _SCALAR_KINDS:
        return _SCALAR_KINDS[kind](text)
    with suppress(ValueError):
        return int(text)
    with suppress(ValueError):
        return float(text)
    return text


def _option_label(key: str) -> str:
    """``key`` as words: ``simulation_type`` reads "Simulation type"."""
    first, *rest = key.lower().split("_")
    words = [first if first in _ACRONYMS else first.capitalize(), *rest]
    return " ".join(word.upper() if word in _ACRONYMS else word for word in words)


def _target_name(job_config: JobConfig | None) -> str | None:
    if job_config is None:
        return None
    name = job_config.qpu_system or job_config.simulator_cluster
    return name if isinstance(name, str) else None


def _job_kwargs(
    job_config: JobConfig | None, target: Target | None = None
) -> dict[str, Any]:
    """The fields ``job_config`` sets, with ``target`` replacing its target if given."""
    fields = job_config._set_fields() if job_config is not None else {}
    if target is not None:
        fields = {
            name: value for name, value in fields.items() if name not in _TARGET_FIELDS
        }
        fields[
            "qpu_system" if isinstance(target, QPUSystem) else "simulator_cluster"
        ] = target.name
    return fields


def _identity(key: str, value: Any) -> Any:
    return value


def _maestro_value(key: str, value: Any) -> Any:
    return MaestroConfig(**{key: value})._simulator_options()[key]


def _python_value(value: Any) -> str:
    if isinstance(value, Enum):
        return repr(value.name)
    if isinstance(value, maestro.NoiseModel):
        return f"<noise model: {len(value._call_log)} set_* calls>"
    return repr(value)


def _python(*plan: Mapping[str, Any]) -> str:
    """``QoroService(...)`` with the configs a plan saves."""
    arguments = []
    for (name, config_type), fields in zip(_CONFIG_TYPES.items(), plan):
        if fields:
            call = ", ".join(
                f"{key.lower()}={_python_value(value)}" for key, value in fields.items()
            )
            arguments.append(f"{name}={config_type.__name__}({call})")
    return f"QoroService({', '.join(arguments)})"


def _change_rows(before: Mapping[str, Any], after: Mapping[str, Any]) -> Table:
    rows = Table.grid(padding=(0, 2))
    for key in dict.fromkeys([*before, *after]):
        if key not in before:
            mark, text = Text("+", style=_GREEN), _value_text(after[key])
        elif key not in after:
            mark, text = Text("−", style=_RED), f"{_value_text(before[key])}  cleared"
        elif before[key] != after[key]:
            mark = Text("~", style=_AMBER)
            text = f"{_value_text(before[key])} → {_value_text(after[key])}"
        else:
            mark, text = Text(" "), f"{_value_text(after[key])}  kept"
        rows.add_row(mark, Text(key.lower(), style="bold"), text)
    return rows


def _review(
    target: Target,
    previous: str | None,
    make_default: bool,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    kept: list[str],
    warnings: list[str],
    python: str,
) -> Group:
    """What saving writes: the target, each option's change and the equivalent Python."""
    kind = "QPU system" if isinstance(target, QPUSystem) else "simulator cluster"
    if make_default:
        was = f"  (was {previous})" if previous and previous != target.name else ""
        heading = Text.assemble(
            ("Default target  ", "bold"), target.name, (f"  {kind}{was}", _MUTED)
        )
    else:
        unchanged = f"{previous or 'none'}  (unchanged)"
        heading = Text.assemble(
            ("Default target  ", "bold"),
            unchanged,
            (f"  ·  editing {target.name}", _MUTED),
        )
    parts: list[Any] = [
        heading,
        Text(),
        (
            _change_rows(before, after)
            if before or after
            else Text("No options set; the target's own settings apply.", style=_MUTED)
        ),
    ]
    if warnings:
        parts += [
            Text(),
            Text("The service will reject jobs with:", style=f"bold {_AMBER}"),
        ]
        parts += [Text(f"  {warning}", style=_AMBER) for warning in warnings]
    if kept:
        parts += [
            Text(),
            Text.assemble(("Also kept in the file  ", "bold"), " · ".join(kept)),
        ]
    parts += [
        Text(),
        Text("Equivalent in Python", style="bold"),
        Syntax(
            python,
            "python",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        ),
    ]
    return Group(*parts)


def _target_details(target: Target) -> str:
    if isinstance(target, SimulatorCluster):
        return f"{target.minimum_tier.replace('_', ' ').capitalize()} tier"
    qpus = target.qpus
    vendors = sorted({qpu.vendor_type.upper() for qpu in qpus if qpu.vendor_type})
    online = sum(qpu.status == "ONLINE" for qpu in qpus)
    largest = max((qpu.q_bits for qpu in qpus), default=0)
    return f"{len(qpus)} QPUs · up to {largest} qubits · {', '.join(vendors)} · {online}/{len(qpus)} online"


def _brand(context: str) -> Static:
    dot = f"  [{_MUTED}]·[/]  "
    title = f"[bold {_PINK}]Qoro[/]{dot}[bold {_TEXT}]divi-config-helper[/]{dot}[{_MUTED}]{escape(context)}[/]"
    return Static(title, classes="brand")


class _Table(DataTable):
    """A row table whose Enter shows in the footer."""

    BINDINGS = [Binding("enter", "select_cursor", "Select")]

    def __init__(self) -> None:
        super().__init__(cursor_type="row", cell_padding=2)


class _ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y,enter", "dismiss(True)", "Yes"),
        Binding("n,escape", "dismiss(False)", "No"),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="editor"):
            yield Label(Text(self.question, style="bold"))
        yield Footer()


class _TargetScreen(Screen[tuple[Target, bool] | None]):
    """Pick the target with the arrow keys; the result says whether it becomes the default."""

    BINDINGS = [
        Binding("e", "choose(False)", "Edit options only"),
        Binding("escape,q", "dismiss", "Quit"),
    ]

    def __init__(self, targets: list[Target], current: str | None) -> None:
        super().__init__()
        self.targets = targets
        self.current = current

    def compose(self) -> ComposeResult:
        yield _brand("Default target")
        yield Label(
            "Enter makes the highlighted target your default · e edits its options and leaves the default as it is.",
            classes="note",
        )
        yield _Table()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("", "Target", "Type", "Details")
        for target in self.targets:
            kind = (
                "QPU system" if isinstance(target, QPUSystem) else "Simulator cluster"
            )
            marker = Text("●", style=_GREEN) if target.name == self.current else ""
            table.add_row(
                marker,
                Text(target.name, style="bold"),
                Text(kind, style=_MUTED),
                Text(_target_details(target), style=_MUTED),
                key=target.name,
            )
        names = [target.name for target in self.targets]
        if self.current in names:
            table.move_cursor(row=names.index(self.current))
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_choose(True)

    def action_choose(self, make_default: bool) -> None:
        if self.targets:
            row = self.query_one(DataTable).cursor_row
            self.dismiss((self.targets[row], make_default))


class _ChoiceEditor(ModalScreen[Any]):
    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        title: Text,
        hint: str,
        description: str | None,
        choices: list[str],
        current: str,
        convert: Callable[[str], Any],
    ) -> None:
        super().__init__()
        self.title_text, self.hint, self.description = title, hint, description
        self.choices, self.current, self.convert = choices, current, convert

    def compose(self) -> ComposeResult:
        with Vertical(classes="editor"):
            yield Label(self.title_text)
            yield Label(self.hint, classes="hint", markup=False)
            if self.description:
                yield Static(self.description, classes="description", markup=False)
            yield OptionList(*self.choices)
        yield Footer()

    def on_mount(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = (
            self.choices.index(self.current) if self.current in self.choices else 0
        )
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        try:
            self.dismiss(self.convert(self.choices[event.option_index]))
        except ValueError as exc:
            self.notify(str(exc), severity="error", markup=False)


class _TextEditor(ModalScreen[Any]):
    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        title: Text,
        hint: str,
        description: str | None,
        current: str,
        convert: Callable[[str], Any],
    ) -> None:
        super().__init__()
        self.title_text, self.hint, self.description = title, hint, description
        self.current, self.convert = current, convert

    def compose(self) -> ComposeResult:
        with Vertical(classes="editor"):
            yield Label(self.title_text)
            yield Label(f"{self.hint} · empty clears", classes="hint", markup=False)
            if self.description:
                yield Static(self.description, classes="description", markup=False)
            yield Input(value=self.current)
            yield Static("", classes="error", markup=False)
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            self.dismiss(_CLEAR)
            return
        try:
            self.dismiss(self.convert(event.value))
        except ValueError as exc:
            self.query_one(".error", Static).update(str(exc))


class _ReviewScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("enter,y", "dismiss(True)", "Save"),
        Binding("escape", "dismiss", "Keep editing"),
    ]

    def __init__(self, review: Group, path: Path) -> None:
        super().__init__()
        self.review, self.path = review, path

    def compose(self) -> ComposeResult:
        panel = Panel(
            self.review, title="Review", subtitle=str(self.path), border_style=_BLUE
        )
        yield Static(panel, classes="review")
        yield Footer()


class _OptionsScreen(Screen[None]):
    """The target's options in one table; Enter edits the highlighted one."""

    BINDINGS = [
        Binding("delete,backspace", "clear", "Clear"),
        Binding("a", "toggle_advanced", "Advanced"),
        Binding("s", "review", "Review & save"),
        Binding("escape", "leave(False)", "Back"),
        Binding("q", "leave(True)", "Quit"),
    ]

    def __init__(
        self,
        target: Target,
        options: dict[str, Spec],
        before: Mapping[str, Any],
        normalise: Normalise,
        notes: list[str],
        save: Callable[[dict[str, Any]], None],
        show_advanced: bool,
    ) -> None:
        super().__init__()
        self.target, self.options, self.before = target, options, dict(before)
        self.normalise, self.notes, self.save = normalise, notes, save
        self.values = dict(before)
        self.common = _first_pass(options)
        self.advanced = [key for key in options if key not in self.common]
        self.show_advanced = show_advanced or not self.common

    def compose(self) -> ComposeResult:
        yield _brand(self.target.name)
        yield Label(
            f"Enter edits · Delete clears · [{_GREEN}]green[/] saved · "
            f"[bold {_AMBER}]*[/] changed · [italic {_MUTED}]dim[/] default",
            classes="note",
        )
        for note in self.notes:
            yield Label(note, classes="note", markup=False)
        yield _Table()
        yield Static("", classes="describe", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Option", "Value", "Allowed")
        self._fill()
        table.focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._describe(None if event.row_key is None else event.row_key.value)

    def _describe(self, key: str | None) -> None:
        description = (
            self.options[key].get("description") if key in self.options else None
        )
        self.query_one(".describe", Static).update(description or "")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "toggle_advanced":
            return bool(self.common and self.advanced)
        return True

    def _changed(self) -> int:
        return sum(
            self.before.get(key) != self.values.get(key)
            for key in {*self.before, *self.values}
        )

    def _value_cell(self, key: str) -> Text:
        if key not in self.values:
            if key in self.before:
                return Text(
                    f"* cleared  {self.options[key]['defaults']}",
                    style=f"bold {_AMBER}",
                )
            return Text(self.options[key]["defaults"], style=f"italic {_MUTED}")
        text = _value_text(self.values[key])
        if self.before.get(key) == self.values[key]:
            return Text(text, style=_GREEN)
        return Text(f"* {text}", style=f"bold {_AMBER}")

    def _add_option(self, table: DataTable, key: str) -> None:
        allowed = Text(self.options[key]["allowed"], style=_MUTED)
        table.add_row(
            Text(_option_label(key), style="bold"),
            self._value_cell(key),
            allowed,
            key=key,
        )

    def _fill(self) -> None:
        table = self.query_one(DataTable)
        row = table.cursor_row
        table.clear()
        for key in self.common:
            self._add_option(table, key)
        if self.advanced and self.common:
            arrow = "▾" if self.show_advanced else "▸"
            heading = Text(
                f"{arrow} {len(self.advanced)} advanced options", style=f"bold {_PINK}"
            )
            table.add_row(heading, "", "", key=_ADVANCED_ROW)
        for key in self.advanced if self.show_advanced else ():
            self._add_option(table, key)
        table.move_cursor(row=min(row, table.row_count - 1))
        self._describe(self._highlighted())

    def _highlighted(self) -> str | None:
        table = self.query_one(DataTable)
        if not table.row_count:
            return None
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value

    def _set(self, key: str, value: Any) -> None:
        if value is None:
            return
        if value is _CLEAR:
            self.values.pop(key, None)
        else:
            self.values[key] = value
        self._fill()

    def _edit_row(self, key: str) -> None:
        spec, current = self.options[key], self.values.get(key)
        label = _option_label(key)
        title = Text.assemble((label, "bold"), "  ", (key.lower(), _MUTED))
        hint = f"{spec['allowed']} · {spec['defaults']}"
        description = spec.get("description")

        def convert(typed: str) -> Any:
            try:
                return self.normalise(key, _parse_value(typed, spec))
            except ValueError:
                raise ValueError(_rejected(label, spec, typed)) from None

        callback = partial(self._set, key)
        if spec.get("kind") == "toggle" or _has_choices(spec):
            choices = ["on", "off"] if spec["kind"] == "toggle" else spec["choices"]
            editor = _ChoiceEditor(
                title, hint, description, choices, _value_text(current), convert
            )
        else:
            text = "" if current is None else _value_text(current)
            editor = _TextEditor(title, hint, description, text, convert)
        self.app.push_screen(editor, callback)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key.value
        if key == _ADVANCED_ROW:
            self.action_toggle_advanced()
        elif key is not None:
            self._edit_row(key)

    def action_clear(self) -> None:
        key = self._highlighted()
        if key in self.options:
            self._set(key, _CLEAR)

    def action_toggle_advanced(self) -> None:
        self.show_advanced = not self.show_advanced
        self._fill()

    def action_leave(self, exit_app: bool) -> None:
        def depart(discard: bool | None = True) -> None:
            if not discard:
                return
            if exit_app:
                self.app.exit(None)
            else:
                self.dismiss(None)

        changed = self._changed()
        if not changed:
            depart()
            return
        plural = "s" if changed != 1 else ""
        self.app.push_screen(
            _ConfirmScreen(f"Discard {changed} unsaved change{plural}?"), depart
        )

    def action_review(self) -> None:
        self.save(dict(self.values))


class ConfigHelper(App[Path | None]):
    """``divi-config-helper``'s screens: the target, then its options; returns the path saved to."""

    TITLE = "divi-config-helper"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    .note { color: $muted; padding: 1 2 0 2; }
    .hint { color: $muted; }
    .describe { padding: 1 2; height: auto; min-height: 4; max-height: 8; background: $surface; }
    .description { margin-top: 1; width: 100%; }
    .brand { dock: top; height: 3; background: $surface; color: $foreground; padding: 1 2; }
    DataTable { height: 1fr; margin: 1 2; background: $background; }
    DataTable > .datatable--header { background: $surface; color: $primary; text-style: bold; }
    DataTable > .datatable--cursor { background: $secondary 60%; color: $foreground; text-style: bold; }
    DataTable > .datatable--hover { background: $surface; }
    OptionList > .option-list--option-highlighted { background: $secondary 60%; color: $foreground; text-style: bold; }
    OptionList > .option-list--option-hover { background: $surface; }
    ModalScreen { align: center middle; background: $background 70%; }
    .editor { width: 72; height: auto; max-height: 90%; padding: 1 2; border: round $secondary; background: $surface; }
    .editor OptionList { height: auto; max-height: 20; margin-top: 1; border: none; background: $surface; }
    Input { background: $background; border: round $secondary; margin-top: 1; }
    Input:focus { border: round $primary; }
    .error { color: $error; height: auto; }
    .review { width: 100; height: auto; max-height: 90%; }
    """

    def __init__(
        self,
        path: Path,
        targets: list[Target],
        blueprints: Mapping[str, Mapping],
        saved: SavedDefaults,
        *,
        show_advanced: bool = False,
    ) -> None:
        super().__init__()
        self.register_theme(_THEME)
        self.theme = _THEME.name
        self.path, self.targets, self.blueprints = path, targets, blueprints
        self.job_config, self.maestro_config, self.device_config = saved
        self.previous = _target_name(self.job_config)
        self.show_advanced = show_advanced

    def on_mount(self) -> None:
        self._choose()

    def _choose(self, _: None = None) -> None:
        self.push_screen(_TargetScreen(self.targets, self.previous), self._open)

    def _plan(self, target: Target, make_default: bool, values: dict[str, Any]) -> Plan:
        """The keyword arguments of the job, Maestro and device configs saving would write."""
        job = _job_kwargs(self.job_config, target if make_default else None)
        saved_maestro = self.maestro_config._set_fields() if self.maestro_config else {}
        saved_device = self.device_config.to_payload() if self.device_config else {}
        if isinstance(target, QPUSystem):
            return job, saved_maestro, dict(values)
        rest = {
            name: value
            for name, value in saved_maestro.items()
            if name not in _SIMULATOR_OPTIONS
        }
        return job, rest | values, saved_device

    def _save(self, target: Target, make_default: bool, values: dict[str, Any]) -> None:
        job, maestro_kwargs, device_kwargs = self._plan(target, make_default, values)
        save_defaults(
            self.path,
            JobConfig(**job) if job else None,
            MaestroConfig(**maestro_kwargs) if maestro_kwargs else None,
            DeviceConfig(**device_kwargs) if device_kwargs else None,
        )
        self.exit(self.path)

    def _confirm(
        self,
        target: Target,
        make_default: bool,
        options: dict[str, Spec],
        before: dict[str, Any],
        kept: list[str],
        values: dict[str, Any],
    ) -> None:
        if values == before and (not make_default or target.name == self.previous):
            self.notify("No changes to save.")
            return
        warnings = [
            f"{key.lower()} = {_value_text(value)} (allowed: {options[key]['allowed']})"
            for key, value in values.items()
            if key in options and _outside(options[key], value)
        ]
        python = _python(*self._plan(target, make_default, values))
        review = _review(
            target, self.previous, make_default, before, values, kept, warnings, python
        )

        def saved(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                self._save(target, make_default, values)
            except (OSError, ValueError) as exc:
                self.notify(f"Could not save: {exc}", severity="error", markup=False)

        self.push_screen(_ReviewScreen(review, self.path), saved)

    def _open(self, choice: tuple[Target, bool] | None) -> None:
        if choice is None:
            self.exit(None)
            return
        target, make_default = choice
        kept = [
            f"{name}={value!r}"
            for name, value in _job_kwargs(self.job_config).items()
            if name not in _TARGET_FIELDS
        ]
        notes: list[str] = []
        if isinstance(target, QPUSystem):
            try:
                options = _device_options(target, self.blueprints)
            except ValueError as exc:
                self.notify(str(exc), severity="error", markup=False)
                self._choose()
                return
            for vendor in dict.fromkeys(qpu.vendor_type for qpu in target.qpus):
                if vendor and not self.blueprints.get(vendor, {}).get("device"):
                    notes.append(
                        f"{vendor.upper()} QPUs take no device options; they keep their own settings."
                    )
            before = self.device_config.to_payload() if self.device_config else {}
            normalise: Normalise = _identity
            if self.maestro_config is not None:
                kept.append("Maestro settings for simulator targets")
        else:
            options = _maestro_specs()
            before = (
                self.maestro_config._simulator_options() if self.maestro_config else {}
            )
            normalise = _maestro_value
            if self.device_config is not None:
                kept.append("device options for QPU targets")
        if not options:
            notes.append("No options to set for this target.")

        save = partial(self._confirm, target, make_default, options, before, kept)
        screen = _OptionsScreen(
            target, options, before, normalise, notes, save, self.show_advanced
        )
        self.push_screen(screen, self._choose)


def _fail(message: str) -> NoReturn:
    _console.print(message, style="red", markup=False)
    raise SystemExit(1)


def _load_saved(path: Path) -> SavedDefaults:
    try:
        return load_defaults(path)
    except ValueError as exc:
        _console.print(
            f"{exc}\nIt will be replaced if you save.", style="yellow", markup=False
        )
        return None, None, None


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``divi-config-helper`` console script."""
    parser = argparse.ArgumentParser(
        prog="divi-config-helper",
        description="Choose a default Qoro Service target and its options.",
    )
    parser.add_argument(
        "--advanced", action="store_true", help="Start with the advanced options shown."
    )
    args = parser.parse_args(argv)
    path = defaults_path()
    saved = _load_saved(path)

    try:
        with _console.status("Fetching targets from the Qoro Service…"):
            service = QoroService(use_saved_defaults=False)
            blueprints = service.fetch_vendor_blueprints()
    except requests.RequestException as exc:
        _fail(f"Could not reach the Qoro Service: {exc}")
    except ValueError as exc:
        _fail(str(exc))

    targets: list[Target] = [
        *get_available_qpu_systems(),
        *get_available_simulator_clusters(),
    ]
    if not targets:
        _fail("This API key can use no QPU systems or simulator clusters.")
    app = ConfigHelper(path, targets, blueprints, saved, show_advanced=args.advanced)
    if app.run() is None:
        _console.print("Nothing saved.")
    else:
        _console.print(f"[green]✓[/green] Saved to {escape(str(path))}")
