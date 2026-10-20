# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.abc: PipelineEnv, PipelineTrace, ExpansionResult."""

import numpy as np
import pytest

from divi.pipeline import CircuitPipeline, PipelineEnv, StageOutput
from divi.pipeline.abc import BundleStage, PipelineResult, _EstimatorSamplePlan
from divi.pipeline.stages import MeasurementStage
from tests._helpers import exact_match

from ._helpers import (
    DummySpecStage,
    FanoutAndSumStage,
    two_group_meta,
    two_group_pipeline_stages,
)


class TestPipelineTypes:
    """Spec: PipelineEnv, PipelineTrace, ExpansionResult have expected attributes."""

    def test_pipeline_env_has_backend_and_optional_attrs(self, dummy_expval_backend):
        env = PipelineEnv(backend=dummy_expval_backend)

        assert env.backend is dummy_expval_backend
        # One set with no free parameters — the default has to satisfy the 2D
        # contract that ParameterBindingStage enforces.
        assert np.asarray(env.param_sets).shape == (1, 0)

    def test_pipeline_trace_has_initial_final_batch_and_expansions(
        self, dummy_pipeline_env
    ):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        spec_circ_key = (("spec", "circ"),)

        assert set(trace.initial_batch.keys()) == {spec_circ_key}
        assert set(trace.final_batch.keys()) == {spec_circ_key}
        assert len(trace.stage_expansions) == 1
        assert len(trace.stage_tokens) == 2
        assert len(trace.stage_expansions) == len(trace.stage_tokens) - 1
        assert trace.stage_expansions[0].stage_name == "MeasurementStage"
        assert set(trace.stage_expansions[0].batch.keys()) == {spec_circ_key}


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(np.int64(5), (5, 5, 5), id="numpy-integer"),
        pytest.param(np.array(5), (5, 5, 5), id="zero-dimensional-array"),
        pytest.param(np.array([4, 5, 6]), (4, 5, 6), id="integer-array"),
        pytest.param(np.array([4, 5, 6], dtype=np.uint8), (4, 5, 6), id="uint8-array"),
        pytest.param((np.int64(4), 5, 6), (4, 5, 6), id="mixed-tuple"),
    ],
)
def test_estimator_sample_plan_accepts_numpy_budgets(value, expected):
    plan = _EstimatorSamplePlan.from_value(value, 3)

    assert plan.by_param_set == expected
    assert all(type(budget) is int for budget in plan.by_param_set)


@pytest.mark.parametrize(
    "value, message",
    [
        pytest.param(
            np.array([True, True, True]),
            "estimator_samples values must be positive integers.",
            id="bool-array",
        ),
        pytest.param(
            np.array([1.0, 2.0, 3.0]),
            "estimator_samples values must be positive integers.",
            id="float-array",
        ),
        pytest.param(
            np.array([1, 0, 3]),
            "estimator_samples values must be positive integers.",
            id="non-positive-array",
        ),
        pytest.param(
            np.array([1, 2]),
            "estimator_samples must contain one value per parameter set.",
            id="wrong-length-array",
        ),
        pytest.param(
            np.array(True),
            "estimator_samples must be a positive integer.",
            id="bool-0d",
        ),
    ],
)
def test_estimator_sample_plan_rejects_invalid_arrays(value, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        _EstimatorSamplePlan.from_value(value, 3)


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param([0.5], 0.5, id="list"),
        pytest.param(np.array([0.5]), 0.5, id="array"),
        pytest.param((0.5,), 0.5, id="tuple"),
    ],
)
def test_pipeline_result_value_squeezes_a_single_observable(raw, expected):
    assert PipelineResult({(): raw}).value == expected


def test_pipeline_result_value_leaves_probability_dicts_alone():
    histogram = {"00": 0.5, "11": 0.5}

    assert PipelineResult({(): histogram}).value is histogram


def test_effective_shots_is_a_plain_int_for_a_numpy_override(dummy_expval_backend):
    env = PipelineEnv(backend=dummy_expval_backend, shots_override=np.int64(7))

    assert env.effective_shots == 7
    assert type(env.effective_shots) is int


def test_plain_bundle_stages_pass():
    """Spec: stages without validate overrides do not block pipeline construction."""
    CircuitPipeline(
        stages=[
            DummySpecStage(meta=two_group_meta()),
            FanoutAndSumStage("x", 2),
            MeasurementStage(),
        ]
    )


class _EnvEchoStage(BundleStage):
    """Returns its input batch unchanged with the env as its token."""

    def __init__(self):
        super().__init__(name=type(self).__name__)

    def expand(self, batch, env):
        return StageOutput(batch=batch, token=env)


def test_default_dry_expand_is_the_real_expand(dummy_pipeline_env):
    batch = {(("spec", "circ"),): two_group_meta()}

    output = _EnvEchoStage().dry_expand(batch, dummy_pipeline_env)

    assert output.batch is batch
    assert output.token is dummy_pipeline_env
