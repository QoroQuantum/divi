# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dry-run report as a value: immutability, serialization, validation."""

import copy
import dataclasses
import json
import pickle
import pydoc

import pytest

from divi.pipeline import DryRunReport, PipelineCadence, StageInfo
from divi.pipeline._dry_run import (
    _REQUIRED_CIRCUIT_STATS,
    _cost_headline,
)
from divi.pipeline._dry_run_format import (
    _format_factor,
)
from tests.pipeline._helpers import h2_vqe


@pytest.fixture
def h2_report(default_test_simulator, default_optimizer):
    """A real cost report, so the assertions hold for what a program produces."""
    return h2_vqe(default_test_simulator, default_optimizer).dry_run()["cost"]


def test_report_repr_is_a_one_line_summary(h2_report):
    """``print(program.dry_run())`` is the first thing anyone tries; the generated
    repr inlines every stage and Pauli string."""
    text = repr(h2_report)
    assert "\n" not in text
    assert len(text) < 200
    assert "cost" in text and "circuit" in text


def test_report_survives_a_round_trip(h2_report):
    """Snapshotting a report to JSON for a CI baseline is a primary use, so the
    whole thing has to encode — an enum or tuple that will not breaks it."""
    assert json.loads(json.dumps(dataclasses.asdict(h2_report)))
    assert copy.deepcopy(h2_report) == h2_report
    assert pickle.loads(pickle.dumps(h2_report)) == h2_report


@pytest.mark.parametrize("field", ["metadata", "env_artifacts", "circuit_stats"])
def test_a_report_owns_its_mappings(field):
    """The trace's dicts go on living after the report is built, so the report takes
    its own copies rather than aliasing them."""
    probe = "min_depth" if field == "circuit_stats" else "probe"
    source = {key: 1.0 for key in _REQUIRED_CIRCUIT_STATS} if probe != "probe" else {}
    source[probe] = 1.0

    if field == "metadata":
        holder = StageInfo(name="S", axis="a", factor=1.0, metadata=source)
    else:
        holder = DryRunReport(
            pipeline_name="cost", stages=(), total_circuits=1, **{field: source}
        )

    source[probe] = 2.0
    assert getattr(holder, field)[probe] == 1.0


def test_a_frozen_field_cannot_be_rebound(h2_report):
    """``frozen=True`` is what stops a report being repointed at other numbers."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        h2_report.total_circuits = 99


def test_report_fields_are_self_documenting(h2_report):
    """``help(DryRunReport)`` is where a user goes to learn what a field means. A
    tuple-based report answers "Alias for field number 3", so the report is a
    dataclass — and must stay one."""
    assert dataclasses.is_dataclass(h2_report)
    names = {f.name for f in dataclasses.fields(h2_report)}
    assert {
        "total_circuits",
        "total_shots",
        "cadence",
        "objective_fingerprint",
    } <= names
    for cls in (DryRunReport, StageInfo):
        assert "Alias for field number" not in pydoc.render_doc(cls)


def test_a_cadence_reloaded_from_json_is_coerced_back():
    """``cadence`` is compared by identity, so a plain string from ``json.loads``
    would be treated as neither cadence and dropped from any roll-up."""
    report = DryRunReport(
        pipeline_name="cost",
        stages=(),
        total_circuits=7,
        total_shots=700,
        cadence="per_evaluation",
    )
    assert report.cadence is PipelineCadence.PER_EVALUATION


def test_an_unknown_cadence_is_rejected_and_lists_the_accepted_values():
    """The JSON-reload path makes this a message users hit, so it names what is
    accepted rather than leaving them Python's bare enum error."""
    with pytest.raises(
        ValueError, match=r"must be one of \['per_evaluation', 'once'\]"
    ):
        DryRunReport(
            pipeline_name="cost", stages=(), total_circuits=1, cadence="sometimes"
        )


def test_partial_circuit_stats_are_rejected_before_rendering():
    """Renderers index these keys unguarded, so a partial mapping used to raise a
    bare KeyError mid-tree — after a header had already been written."""
    with pytest.raises(ValueError, match="circuit_stats is missing"):
        DryRunReport(
            pipeline_name="cost",
            stages=(),
            total_circuits=1,
            circuit_stats={"mean_depth": 24.0},
        )


def test_format_factor_zero_is_bare_not_divide_by_zero():
    line_token, total_token, total_op = _format_factor(0.0)
    assert line_token == "0"
    assert total_token == ""
    assert total_op is None


def test_cost_headline_drops_shots_when_unknown():
    assert _cost_headline(500, 2_500_000) == "500 circuits · 2,500,000 shots"
    assert _cost_headline(5, 0) == "5 circuits"
