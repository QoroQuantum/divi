# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._parameter_binding_stage."""

import numpy as np
import pytest

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import MeasurementStage, ParameterBindingStage
from tests.pipeline.helpers import DummySpecStage, two_group_meta


class TestParameterBindingStage:
    """Spec: ParameterBindingStage expand binds env.param_sets into circuit body QASMs; reduce is identity."""

    def test_requires_2d_param_sets(self, dummy_pipeline_env):
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=[1.0, 2.0])
        with pytest.raises(ValueError, match="param_sets to be 2D"):
            pipeline.run_forward_pass("x", env)

    def test_passthrough_when_no_symbols(self, dummy_pipeline_env):
        meta = two_group_meta()
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[0.0]]),
        )
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        trace = pipeline.run_forward_pass("x", env)
        for node in trace.final_batch.values():
            assert node.circuit_body_qasms
