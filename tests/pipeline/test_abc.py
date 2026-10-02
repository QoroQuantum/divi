# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.abc: PipelineEnv, PipelineTrace, ExpansionResult."""

import numpy as np

from divi.pipeline import CircuitPipeline, PipelineEnv, StageOutput
from divi.pipeline.abc import BundleStage
from divi.pipeline.stages import MeasurementStage

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
