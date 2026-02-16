# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline._core: validation, CircuitPipeline, compile_batch, format_pipeline_tree."""

import pytest

from divi.pipeline import (
    CircuitPipeline,
    PipelineEnv,
    PipelineTrace,
    format_pipeline_tree,
)
from divi.pipeline._compilation import (
    _collapse_to_parent_results,
    _compile_batch,
)
from divi.pipeline._core import _validate_stage_order
from divi.pipeline.stages import MeasurementStage

from .helpers import (
    DummySpecStage,
    FanoutAndSumStage,
    StatefulFanoutStage,
    ones_execute_fn,
    two_group_meta,
    two_group_pipeline_stages,
)


class TestValidateStageOrder:
    """Spec: _validate_stage_order enforces pipeline structure and unique axis names."""

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="stages cannot be empty"):
            _validate_stage_order([])

    def test_bundle_stage_first_raises(self):
        meta = two_group_meta()
        with pytest.raises(
            ValueError,
            match="exactly one 'spec' stage and it must come before",
        ):
            CircuitPipeline(
                stages=[
                    MeasurementStage(),
                    DummySpecStage(meta=meta),
                ]
            )

    def test_spec_stage_only_raises_missing_measurement_stage(self):
        """Single spec stage without MeasurementStage raises."""
        with pytest.raises(
            ValueError,
            match="Pipeline must contain at least one MeasurementStage",
        ):
            CircuitPipeline(stages=[DummySpecStage(meta=two_group_meta())])

    def test_no_measurement_stage_raises(self):
        """Pipeline with spec + bundle but no MeasurementStage raises."""
        with pytest.raises(
            ValueError,
            match="Pipeline must contain at least one MeasurementStage",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    FanoutAndSumStage("x", 2),
                ]
            )

    def test_duplicate_axis_names_raise(self):
        """Duplicate stage axis_name (e.g. two MeasurementStages) raises."""
        meta = two_group_meta()
        with pytest.raises(
            ValueError,
            match="Duplicate stage axis names",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=meta),
                    MeasurementStage(),
                    MeasurementStage(),
                ]
            )


class TestCircuitPipelineRunForwardPass:
    """Spec: run_forward_pass expands spec then bundle stages and returns PipelineTrace."""

    def test_plan_tracks_lineage_for_two_level_fanout(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)

        spec_circ_key = (("spec", "circ"),)
        assert set(plan.initial_batch.keys()) == {spec_circ_key}
        assert len(plan.stage_expansions) == 2
        assert len(plan.stage_tokens) == 3
        assert len(plan.final_batch) == 3
        assert set(plan.final_batch.keys()) == {
            spec_circ_key + (("fold", 0),),
            spec_circ_key + (("fold", 1),),
            spec_circ_key + (("fold", 2),),
        }

    def test_run_reduces_bottom_up_after_fanout(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

        reduced = pipeline.run(
            initial_spec="ignored",
            env=dummy_pipeline_env,
            execute_fn=ones_execute_fn,
        )

        assert len(reduced) == 1
        assert list(reduced.values())[0] == pytest.approx(3.9)

    def test_force_forward_sweep_recomputes_even_with_cache(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())
        trace1 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        trace2 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env, force_forward_sweep=True
        )
        assert trace1.final_batch.keys() == trace2.final_batch.keys()
        assert id(trace1) != id(trace2) or trace1 is trace2

    def test_stateful_bundle_stage_triggers_partial_rerun_from_cache(
        self, dummy_pipeline_env
    ):
        """When a bundle stage is stateful, second run_forward_pass reuses cache and reruns from that stage."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
                StatefulFanoutStage("stateful", 2),
            ]
        )
        spec_key = (("spec", "circ"),)

        trace1 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        assert len(trace1.stage_expansions) == 2
        assert set(trace1.final_batch.keys()) == {
            spec_key + (("stateful", 0),),
            spec_key + (("stateful", 1),),
        }

        trace2 = pipeline.run_forward_pass(
            initial_spec="ignored", env=dummy_pipeline_env
        )
        assert len(trace2.stage_expansions) == 2
        assert trace2.initial_batch == trace1.initial_batch
        assert set(trace2.final_batch.keys()) == set(trace1.final_batch.keys())
        assert trace2.stage_expansions[0].batch == trace1.stage_expansions[0].batch


class TestCompileBatch:
    """Spec: _compile_batch lowers MetaCircuit batch to QASM labels and lineage."""

    def test_raises_when_measurement_qasms_missing(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        # Manually remove measurement_qasms to simulate the error
        batch = trace.final_batch
        node = next(iter(batch.values()))
        assert hasattr(node, "circuit_body_qasms") and node.circuit_body_qasms

    def test_produces_lineage_and_circuits_for_grouped_batch(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("x", dummy_pipeline_env)
        circuits, lineage_by_label = _compile_batch(trace.final_batch)
        assert len(circuits) == len(lineage_by_label)
        for label, qasm in circuits.items():
            assert isinstance(qasm, str) and "OPENQASM" in qasm
            assert label in lineage_by_label
            branch_key = lineage_by_label[label]
            assert branch_key[0] == ("spec", "circ") and len(branch_key) >= 2
            assert any(e[0] == "obs_group" for e in branch_key)


class TestCollapseToParentResults:
    """Spec: _collapse_to_parent_results maps backend labels back to BranchKeys."""

    def test_maps_labels_to_branch_keys(self):
        spec_circ = ("spec", "circ")
        lineage = {
            "a/obs_group:0": (spec_circ, ("obs_group", 0)),
            "b/obs_group:1": (spec_circ, ("obs_group", 1)),
        }
        raw = {"a/obs_group:0": 1.0, "b/obs_group:1": 2.0}
        out = _collapse_to_parent_results(raw, lineage)
        assert out[(spec_circ, ("obs_group", 0))] == 1.0
        assert out[(spec_circ, ("obs_group", 1))] == 2.0

    def test_ignores_unknown_labels(self):
        lineage = {"only": (("spec", "k"),)}
        raw = {"only": 1, "unknown": 2}
        out = _collapse_to_parent_results(raw, lineage)
        assert out == {(("spec", "k"),): 1}


class TestFormatPipelineTree:
    """Spec: format_pipeline_tree prints ASCII tree of final_batch keys."""

    def test_format_pipeline_tree_prints_tree(self, capsys, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 2)))
        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)

        format_pipeline_tree(plan)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]

        assert lines[0] == "spec:circ"
        # Two fold branches, each with one obs_group leaf
        # (_backend_expval puts all observables in a single group).
        assert sum("fold:0" in l for l in lines) == 1
        assert sum("fold:1" in l for l in lines) == 1
        assert sum("obs_group:0" in l for l in lines) == 2
        # 5 lines total: root + 2 folds + 2 obs_groups
        assert len(lines) == 5

    def test_format_pipeline_tree_empty_batch(self, capsys):
        trace = PipelineTrace(
            initial_batch={},
            final_batch={},
            stage_expansions=(),
            stage_tokens=(),
        )
        format_pipeline_tree(trace)
        assert capsys.readouterr().out.strip() == "(empty)"


def test_custom_execute_fn_returning_per_key_values_reduces_correctly(
    dummy_pipeline_env,
):
    """Execute_fn returning distinct values per key is reduced by pipeline stages."""
    pipeline = CircuitPipeline(stages=two_group_pipeline_stages(fanout=("fold", 3)))

    def _execute_fn(trace, env):
        _, lineage_by_label = _compile_batch(trace.final_batch)
        return {bk: 2 for bk in lineage_by_label.values()}

    reduced = pipeline.run(
        initial_spec="ignored",
        env=dummy_pipeline_env,
        execute_fn=_execute_fn,
    )
    assert len(reduced) == 1
    assert list(reduced.values())[0] == pytest.approx(7.8)
    assert next(iter(reduced)) == (("spec", "circ"),)


def test_default_execute_raises_when_measurement_qasms_missing(dummy_pipeline_env):
    """Pipeline without MeasurementStage raises at construction time."""
    with pytest.raises(
        ValueError,
        match="Pipeline must contain at least one MeasurementStage",
    ):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                FanoutAndSumStage("x", 2),
            ]
        )


def test_run_with_default_execute_fn_and_shots_backend_auto_converts_counts(
    dummy_simulator,
):
    """When backend.supports_expval is False, run() auto-converts counts → expvals via _counts_to_expvals."""
    env = PipelineEnv(backend=dummy_simulator)
    pipeline = CircuitPipeline(
        stages=[
            DummySpecStage(meta=two_group_meta()),
            MeasurementStage(),
        ]
    )
    reduced = pipeline.run(initial_spec="x", env=env)
    assert len(reduced) == 1
    key = next(iter(reduced))
    assert key == (("spec", "circ"),)
    assert isinstance(reduced[key], (int, float))
