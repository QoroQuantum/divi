# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._measurement_stage."""

import numpy as np
import pennylane as qml
import pytest

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.abc import ChildResults, ResultFormat
from divi.pipeline.stages import MeasurementStage
from divi.pipeline.stages._measurement_stage import OBS_GROUP_AXIS, MeasurementToken
from tests.pipeline.helpers import (
    DummySpecStage,
    ones_execute_fn,
    two_group_pipeline_stages,
)


class TestMeasurementStage:
    """Spec: MeasurementStage expand sets measurement groups; reduce applies postprocess."""

    def test_fanout_and_regroup(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(stages=two_group_pipeline_stages())

        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)
        spec_circ_key = (("spec", "circ"),)
        assert set(plan.final_batch.keys()) == {spec_circ_key}

        reduced = pipeline.run(
            initial_spec="ignored", env=dummy_pipeline_env, execute_fn=ones_execute_fn
        )
        assert len(reduced) == 1
        assert list(reduced.values())[0] == pytest.approx(1.3)


class TestMeasurementStageExpvalBackendReduce:
    """Tests for _reduce_expval with ham_ops_list (expval-native backend path).

    Replaces the removed VQA tests:
    - test_post_process_with_expectation_values_happy_path
    - test_post_process_with_expectation_values_missing_ham_ops
    """

    def test_expval_backend_pipeline_sets_ham_ops(self, dummy_expval_backend):
        """Full pipeline with expval-native backend sets ham_ops in env.artifacts."""
        hamiltonian = 0.5 * qml.Z(0) + (-0.3) * qml.X(0)
        qscript = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.expval(hamiltonian)],
        )
        meta = MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))

        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()],
        )

        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        assert "ham_ops" in env.artifacts

    def test_reduce_indexed_dicts_from_expval_backend(self):
        """_reduce_expval handles {int: float} dicts (normalised by _core.py).

        After _expval_dicts_to_indexed normalises backend results, reduce
        receives {obs_idx: float} dicts — same format as _counts_to_expvals.
        """
        stage = MeasurementStage()

        base_key = (("spec", "circ"),)
        # Single obs_group with indexed dict (already normalised)
        results: ChildResults = {
            base_key + ((OBS_GROUP_AXIS, 0),): {0: 0.5, 1: -0.3, 2: 0.2},
        }

        def _postprocess(values):
            total = 0.0
            for v in values:
                if isinstance(v, dict):
                    total += sum(v.values())
                else:
                    total += v
            return total

        token = MeasurementToken(
            postprocess_fn_by_spec={base_key: _postprocess},
        )

        env = PipelineEnv.__new__(PipelineEnv)

        reduced = stage.reduce(results, env, token)

        assert base_key in reduced
        # 0.5 + (-0.3) + 0.2 = 0.4
        assert reduced[base_key] == pytest.approx(0.4)

    def test_reduce_without_ham_ops_uses_standard_postprocess(
        self, dummy_expval_backend
    ):
        """When ham_ops_list is None, standard postprocessing is applied."""
        hamiltonian = 0.9 * qml.Z(0) + 0.4 * qml.X(0)
        qscript = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.expval(hamiltonian)],
        )
        meta = MetaCircuit(
            source_circuit=qscript,
            symbols=np.array([], dtype=object),
        )
        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()],
        )

        reduced = pipeline.run(
            initial_spec="ignored",
            env=env,
            execute_fn=ones_execute_fn,
        )

        assert len(reduced) == 1
        # With ones_execute_fn each obs group returns 1.0
        # Postprocessing applies coefficients: 0.9 * 1.0 + 0.4 * 1.0 = 1.3
        assert list(reduced.values())[0] == pytest.approx(1.3)


class TestMeasurementStageResultFormatOverride:
    """Tests for result_format_override on MeasurementStage."""

    def test_expand_applies_result_format_override(self, dummy_expval_backend):
        """When result_format_override is set, expand overrides env.result_format."""
        hamiltonian = 0.5 * qml.Z(0) + (-0.3) * qml.X(0)
        qscript = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.expval(hamiltonian)],
        )
        meta = MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))

        stage = MeasurementStage(result_format_override=ResultFormat.COUNTS)
        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), stage],
        )

        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        assert env.result_format is ResultFormat.COUNTS

    def test_reduce_returns_raw_for_counts_override(self):
        """With COUNTS override, reduce strips obs_group axis without postprocessing."""
        stage = MeasurementStage(result_format_override=ResultFormat.COUNTS)

        base_key = (("spec", "circ"),)
        results: ChildResults = {
            base_key + ((OBS_GROUP_AXIS, 0),): {"00": 50, "11": 50},
            base_key + ((OBS_GROUP_AXIS, 1),): {"01": 30, "10": 70},
        }

        # Token with postprocess_fn that should NOT be called
        token = MeasurementToken(
            postprocess_fn_by_spec={base_key: lambda _: 999.0},
        )

        env = PipelineEnv.__new__(PipelineEnv)
        env.result_format = ResultFormat.COUNTS

        reduced = stage.reduce(results, env, token)

        # Raw results: obs_group axis stripped, values passed through as-is.
        # Two groups map to the same base_key, so the last one wins (dict update).
        assert base_key in reduced
