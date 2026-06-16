# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PipelineSet named-pipeline registry."""

import pytest

from divi.pipeline import PipelineSet


def test_named_access_and_membership():
    cost, sample = object(), object()
    pipelines = PipelineSet({"cost": (cost, lambda: 1), "sample": (sample, lambda: 2)})

    assert pipelines["cost"] is cost
    assert "cost" in pipelines
    assert "metric" not in pipelines
    assert dict(pipelines.items()) == {"cost": cost, "sample": sample}


def test_spec_for_resolves_factory_lazily():
    calls = []

    def factory():
        calls.append(1)
        return "seed"

    pipelines = PipelineSet({"cost": (object(), factory)})
    # The factory is not called at construction time.
    assert calls == []
    assert pipelines.spec_for("cost") == "seed"
    assert calls == [1]


def test_missing_pipeline_raises_with_available_names():
    pipelines = PipelineSet({"cost": (object(), lambda: None)})
    with pytest.raises(KeyError, match=r"exposes \['cost'\]"):
        pipelines["metric"]


def test_spec_for_missing_pipeline_raises_with_available_names():
    pipelines = PipelineSet({"cost": (object(), lambda: None)})
    with pytest.raises(KeyError, match=r"exposes \['cost'\]"):
        pipelines.spec_for("metric")


def test_repr():
    pipelines = PipelineSet(
        {"cost": (object(), lambda: None), "sample": (object(), lambda: None)}
    )
    assert repr(pipelines) == "PipelineSet(['cost', 'sample'])"
