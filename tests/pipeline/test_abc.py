# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.abc: PipelineEnv, PipelineTrace, ExpansionResult."""

from divi.pipeline import CircuitPipeline, PipelineEnv
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
        assert env.param_sets == ()

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

    def test_expansion_result_has_batch_and_stage_name(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        spec_circ_key = (("spec", "circ"),)
        exp = trace.stage_expansions[0]

        assert exp.stage_name == "MeasurementStage"
        assert set(exp.batch.keys()) == {spec_circ_key}


def test_plain_bundle_stages_pass():
    """Spec: stages without validate overrides do not block pipeline construction."""
    CircuitPipeline(
        stages=[
            DummySpecStage(meta=two_group_meta()),
            FanoutAndSumStage("x", 2),
            MeasurementStage(),
        ]
    )
