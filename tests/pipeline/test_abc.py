# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.abc: PipelineEnv, PipelineTrace, ExpansionResult."""

from divi.pipeline import CircuitPipeline, PipelineEnv

from .helpers import two_group_pipeline_stages


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
        assert hasattr(trace, "initial_batch")
        assert hasattr(trace, "final_batch")
        assert hasattr(trace, "stage_expansions")
        assert hasattr(trace, "stage_tokens")
        assert len(trace.stage_expansions) == len(trace.stage_tokens) - 1

    def test_expansion_result_has_batch_and_stage_name(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        exp = trace.stage_expansions[0]
        assert hasattr(exp, "batch")
        assert exp.stage_name is not None and isinstance(exp.stage_name, str)
