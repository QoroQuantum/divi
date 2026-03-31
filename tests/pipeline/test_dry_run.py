# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipeline dry-run tool."""

from divi.circuits.qem import _NoMitigation
from divi.pipeline import CircuitPipeline, dry_run_pipeline, format_dry_run
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.stages import MeasurementStage, QEMStage
from tests.pipeline.helpers import DummySpecStage, two_group_meta


class TestDryRunPipeline:
    def test_basic_pipeline(self, dummy_pipeline_env):
        """Spec + Measurement produces correct fan-out."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        report = dry_run_pipeline("test", trace, pipeline.stages)

        assert report.pipeline_name == "test"
        assert len(report.stages) == 2
        assert report.stages[0].name == "DummySpecStage"
        assert report.stages[1].name == "MeasurementStage"
        assert report.total_circuits > 0

    def test_total_matches_compile(self, dummy_pipeline_env):
        """Total circuits matches actual _compile_batch output."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                QEMStage(protocol=_NoMitigation()),
            ]
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        report = dry_run_pipeline("test", trace, pipeline.stages)
        compiled, _ = _compile_batch(trace.final_batch)
        assert report.total_circuits == len(compiled)

    def test_format_does_not_crash(self, dummy_pipeline_env):
        """format_dry_run prints without errors."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        report = dry_run_pipeline("test", trace, pipeline.stages)
        format_dry_run({"test": report})
