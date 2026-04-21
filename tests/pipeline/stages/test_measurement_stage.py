# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._measurement_stage."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline._compilation import _compile_batch
from divi.pipeline._grouping import compute_measurement_groups
from divi.pipeline.abc import ChildResults, MetaCircuitBatch, ResultFormat, SpecStage
from divi.pipeline.stages import MeasurementStage
from divi.pipeline.stages._measurement_stage import (
    OBS_GROUP_AXIS,
    MeasurementToken,
    _allocate_per_group_shots,
)
from tests.conftest import DummySimulator
from tests.pipeline.helpers import (
    DummySpecStage,
    ones_execute_fn,
    two_group_pipeline_stages,
)


def _two_term_meta() -> MetaCircuit:
    """Hadamard circuit with 0.5*Z + -0.3*X observable."""
    qc = QuantumCircuit(1)
    qc.h(0)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=SparsePauliOp.from_list([("Z", 0.5), ("X", -0.3)]),
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
        meta = _two_term_meta()

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
        qc = QuantumCircuit(1)
        qc.h(0)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=SparsePauliOp.from_list([("Z", 0.9), ("X", 0.4)]),
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
        meta = _two_term_meta()

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


# --------------------------------------------------------------------------- #
# Adaptive shot allocation (Strategy A — group-level)
# --------------------------------------------------------------------------- #


def _three_group_meta() -> MetaCircuit:
    """A Hamiltonian that produces 3 QWC groups with skewed L1 norms.

    H = 10*Z(0) + 1*X(0) + 0.1*Y(0)

    Each Pauli operator on the same wire commutes with itself only, so QWC
    grouping yields 3 single-term groups with L1 norms [10, 1, 0.1].
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=SparsePauliOp.from_list([("Z", 10.0), ("X", 1.0), ("Y", 0.1)]),
    )


def _multi_term_qwc_meta() -> MetaCircuit:
    """Hamiltonian where two terms QWC-commute into one group, third in its own.

    H = 1.0*Z(0) + 1.0*Z(1) + 0.5*X(0)
    Z(0) and Z(1) live on different wires and commute -> one group.
    X(0) doesn't commute with Z(0) -> separate group.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        observable=SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 1.0), ("IX", 0.5)]),
    )


class TestMeasurementStageShotDistributionDefaultBehavior:
    """Spec: shot_distribution=None preserves existing behaviour exactly."""

    def test_no_per_group_shots_in_artifacts(self, dummy_simulator):
        env = PipelineEnv(backend=dummy_simulator)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=_three_group_meta()), MeasurementStage()],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)
        assert "per_group_shots" not in env.artifacts

    def test_no_zero_shot_groups_in_token(self, dummy_simulator):
        env = PipelineEnv(backend=dummy_simulator)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=_three_group_meta()), MeasurementStage()],
        )
        trace = pipeline.run_forward_pass(initial_spec="ignored", env=env)
        # MeasurementStage is index 1; its token is the second one.
        meas_token = trace.stage_tokens[1]
        assert isinstance(meas_token, MeasurementToken)
        assert meas_token.zero_shot_groups_by_spec == {}

    def test_no_warning_emitted(self, dummy_simulator):
        env = PipelineEnv(backend=dummy_simulator)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=_three_group_meta()), MeasurementStage()],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.run_forward_pass(initial_spec="ignored", env=env)


class TestMeasurementStageShotDistributionUniform:
    """Spec: 'uniform' splits backend.shots equally across groups."""

    def test_per_group_shots_equal_split(self):

        backend = DummySimulator(shots=300)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"]
        spec_key = (("spec", "circ"),)
        assert spec_key in per_group
        # 300 / 3 groups = 100 per group.
        assert per_group[spec_key] == {0: 100, 1: 100, 2: 100}

    def test_uniform_total_preserved_with_remainder(self):

        backend = DummySimulator(shots=10)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        assert sum(per_group.values()) == 10
        assert all(s in (3, 4) for s in per_group.values())


class TestMeasurementStageShotDistributionWeighted:
    """Spec: 'weighted' allocates shots proportional to per-group L1 norm."""

    def test_dominant_group_gets_most_shots(self):

        backend = DummySimulator(shots=1000)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        # L1 norms 10:1:0.1 -> total 11.1
        assert per_group[0] > per_group[1] > per_group[2]
        assert sum(per_group.values()) == 1000

    def test_qwc_group_l1_uses_combined_terms(self):
        """Two-term QWC group's L1 norm = sum of |c_i| within the group."""

        backend = DummySimulator(shots=1000)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_multi_term_qwc_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        # Two groups: {Z(0), Z(1)} with L1=2.0 and {X(0)} with L1=0.5.
        # Allocation 2.0 : 0.5 -> 800 : 200
        assert sum(per_group.values()) == 1000
        assert max(per_group.values()) == 800
        assert min(per_group.values()) == 200


class TestMeasurementStageShotDistributionWires:
    """Spec: 'wires' grouping also supports adaptive shot allocation."""

    def test_wires_strategy_per_group_shots(self):

        backend = DummySimulator(shots=300)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(
                    grouping_strategy="wires", shot_distribution="uniform"
                ),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        # All three observables share wire 0 -> one group per observable.
        assert sum(per_group.values()) == 300
        assert len(per_group) == 3


class TestMeasurementStageShotDistributionNoneStrategy:
    """Spec: grouping_strategy=None (one group per observable) supports allocation."""

    def test_none_strategy_per_group_shots(self):

        backend = DummySimulator(shots=1000)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(grouping_strategy=None, shot_distribution="weighted"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        assert len(per_group) == 3
        assert sum(per_group.values()) == 1000


class TestMeasurementStageShotDistributionDropZeroGroups:
    """Spec: groups with zero allocated shots are dropped + warn."""

    def test_warning_emitted_when_groups_dropped(self):

        backend = DummySimulator(shots=11)  # weighted 10:1:0.1 -> 10:1:0
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        with pytest.warns(UserWarning, match="zero shots"):
            pipeline.run_forward_pass(initial_spec="ignored", env=env)

    def test_dropped_groups_excluded_from_per_group_shots(self):

        backend = DummySimulator(shots=11)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        # Group 2 (norm 0.1) should be dropped; group 0 and 1 survive.
        assert 2 not in per_group
        assert 0 in per_group and 1 in per_group

    def test_zero_shot_groups_in_token(self):

        backend = DummySimulator(shots=11)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pipeline.run_forward_pass(initial_spec="ignored", env=env)

        meas_token = trace.stage_tokens[1]
        spec_key = (("spec", "circ"),)
        assert spec_key in meas_token.zero_shot_groups_by_spec
        # Group 2 has a single observable -> zero-fill dict size 1.
        assert meas_token.zero_shot_groups_by_spec[spec_key] == {2: {0: 0.0}}

    def test_measurement_qasms_skip_dropped_groups(self):

        backend = DummySimulator(shots=11)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pipeline.run_forward_pass(initial_spec="ignored", env=env)

        meta = next(iter(trace.final_batch.values()))
        # Only 2 measurement_qasms should remain (groups 0 and 1).
        assert len(meta.measurement_qasms) == 2
        # Original obs_group indices preserved (no renumbering).
        tags = [tag for tag, _ in meta.measurement_qasms]
        flat_indices = [t[0][1] for t in tags]
        assert flat_indices == [0, 1]

    def test_metacircuit_keeps_full_measurement_groups(self):
        """The MetaCircuit retains all groups so _counts_to_expvals can index by orig idx."""

        backend = DummySimulator(shots=11)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pipeline.run_forward_pass(initial_spec="ignored", env=env)

        meta = next(iter(trace.final_batch.values()))
        # All 3 groups present even though group 2 was dropped from submission.
        assert len(meta.measurement_groups) == 3


class TestMeasurementStageShotDistributionReducePath:
    """Spec: dropped groups contribute zero in the final postprocessed result."""

    @staticmethod
    def _counts_returning_plus_one(trace, env):
        """Return ``{"0": shots}`` per circuit. After basis rotation each
        measurement-axis Pauli has eigenvalue +1 on |0>, so each group's
        expval is exactly +1.0."""

        _, lineage = _compile_batch(trace.final_batch)
        return {bk: {"0": 100} for bk in lineage.values()}

    def test_reduce_injects_placeholders_and_yields_correct_energy(self):
        """End-to-end: with mocked backend results for surviving groups, the
        reduce path injects 0 for dropped groups and computes the right sum."""

        backend = DummySimulator(shots=11)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reduced = pipeline.run(
                initial_spec="ignored",
                env=env,
                execute_fn=self._counts_returning_plus_one,
            )

        # Surviving groups: 0 (coeff 10) and 1 (coeff 1) each return +1.0.
        # Dropped group 2 (coeff 0.1) contributes 0.
        # Final: 10*1 + 1*1 + 0.1*0 = 11.0
        assert list(reduced.values())[0] == pytest.approx(11.0)

    def test_reduce_no_drops_works_normally(self):

        backend = DummySimulator(shots=10000)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
        )

        reduced = pipeline.run(
            initial_spec="ignored",
            env=env,
            execute_fn=self._counts_returning_plus_one,
        )
        # All 3 terms contribute: 10 + 1 + 0.1 = 11.1
        assert list(reduced.values())[0] == pytest.approx(11.1)


class TestMeasurementStageShotDistributionBackendExpval:
    """Spec: shot_distribution is incompatible with _backend_expval strategy."""

    def test_explicit_backend_expval_raises(self, dummy_expval_backend):
        """Setting shot_distribution with explicit _backend_expval -> ValueError."""
        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(
                    grouping_strategy="_backend_expval",
                    shot_distribution="uniform",
                ),
            ],
        )
        with pytest.raises(ValueError, match="_backend_expval"):
            pipeline.run_forward_pass(initial_spec="ignored", env=env)

    def test_shot_distribution_suppresses_auto_fallback(self, dummy_expval_backend):
        """qwc + expval-supporting backend would normally auto-select
        _backend_expval, but shot_distribution declares sampling intent —
        the auto-fallback must be skipped so per-group shots are honoured."""
        env = PipelineEnv(backend=dummy_expval_backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(grouping_strategy="qwc", shot_distribution="uniform"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)
        assert "per_group_shots" in env.artifacts
        assert "ham_ops" not in env.artifacts

    def test_qwc_with_non_expval_backend_works(self):
        """qwc + non-expval backend doesn't auto-switch -> shot_distribution OK."""
        env = PipelineEnv(backend=DummySimulator(shots=300))
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(grouping_strategy="qwc", shot_distribution="uniform"),
            ],
        )
        # Should not raise or warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.run_forward_pass(initial_spec="ignored", env=env)
        assert "per_group_shots" in env.artifacts

    def test_no_ham_ops_when_shot_distribution_used(self):
        """env.artifacts should not have ham_ops set in QWC + shot_distribution mode."""
        env = PipelineEnv(backend=DummySimulator(shots=300))
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipeline.run_forward_pass(initial_spec="ignored", env=env)
        assert "ham_ops" not in env.artifacts


class TestMeasurementStageShotDistributionCallable:
    """Spec: shot_distribution accepts a callable for custom allocation."""

    def test_callable_strategy(self):

        def custom(norms, total):
            # Allocate everything to the first group.
            return [total] + [0] * (len(norms) - 1)

        backend = DummySimulator(shots=500)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution=custom),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"][(("spec", "circ"),)]
        assert per_group == {0: 500}


class TestMeasurementStageShotDistributionReproducibility:
    """Spec: weighted_random uses env.rng so seeded VQAs reproduce allocations."""

    def test_seeded_rng_produces_identical_allocations(self):
        backend = DummySimulator(shots=10000)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        pipeline_a = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted_random"),
            ],
        )
        pipeline_b = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted_random"),
            ],
        )
        env_a = PipelineEnv(backend=backend, rng=rng_a)
        env_b = PipelineEnv(backend=backend, rng=rng_b)
        pipeline_a.run_forward_pass(initial_spec="ignored", env=env_a)
        pipeline_b.run_forward_pass(initial_spec="ignored", env=env_b)
        assert env_a.artifacts["per_group_shots"] == env_b.artifacts["per_group_shots"]

    def test_unseeded_rng_can_drift_between_pipelines(self):
        """Sanity check: without an env.rng, allocations CAN differ — proving
        that the seeded test above is actually exercising the rng plumbing
        (rather than just hitting a deterministic code path)."""
        backend = DummySimulator(shots=10000)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="weighted_random"),
            ],
        )
        # No explicit rng -> falls back to a fresh default_rng each call.
        env_a = PipelineEnv(backend=backend)
        env_b = PipelineEnv(backend=backend)
        pipeline.run_forward_pass(initial_spec="ignored", env=env_a)
        pipeline.run_forward_pass(
            initial_spec="ignored", env=env_b, force_forward_sweep=True
        )
        # We can't assert inequality strictly (multinomial may collide), but
        # the shape and keys should match.
        assert (
            env_a.artifacts["per_group_shots"].keys()
            == env_b.artifacts["per_group_shots"].keys()
        )


class TestMeasurementStageShotDistributionMultiSpec:
    """Spec: per_group_shots / placeholders are keyed per spec_key."""

    def test_two_specs_both_get_their_own_allocations(self):
        """Use a custom spec stage emitting two distinct spec batch keys."""

        meta_a = _three_group_meta()
        meta_b = _multi_term_qwc_meta()  # 2 groups

        class TwoSpecStage(SpecStage[str]):
            def __init__(self):
                super().__init__(name=type(self).__name__)

            def expand(self, items, env):
                batch: MetaCircuitBatch = {
                    (("spec", "a"),): meta_a,
                    (("spec", "b"),): meta_b,
                }
                return batch, None

            def reduce(self, results, env, token):
                return results

        backend = DummySimulator(shots=1000)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[TwoSpecStage(), MeasurementStage(shot_distribution="weighted")],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)

        per_group = env.artifacts["per_group_shots"]
        assert (("spec", "a"),) in per_group
        assert (("spec", "b"),) in per_group
        # Each spec independently sums to 1000.
        assert sum(per_group[(("spec", "a"),)].values()) == 1000
        assert sum(per_group[(("spec", "b"),)].values()) == 1000


class TestMeasurementStageShotDistributionPipelineRerun:
    """Spec: re-running the pipeline doesn't accumulate stale per_group_shots."""

    def test_per_group_shots_replaced_on_rerun(self):
        """Re-running on a fresh env should populate per_group_shots, whether
        the allocation is served from cache (deterministic strategies) or
        recomputed (random strategies)."""
        backend = DummySimulator(shots=300)
        env = PipelineEnv(backend=backend)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        pipeline.run_forward_pass(initial_spec="ignored", env=env)
        first = dict(env.artifacts["per_group_shots"])

        env2 = PipelineEnv(backend=backend)
        pipeline.run_forward_pass(initial_spec="ignored", env=env2)
        second = dict(env2.artifacts["per_group_shots"])
        assert first == second

    def test_shots_change_between_runs_picked_up(self):
        """Bumping backend.shots between runs should produce a NEW allocation,
        not restore the cached one. Cache invalidation comes from
        ``MeasurementStage.cache_key_extras`` seeing the new shot count."""
        backend = DummySimulator(shots=300)
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        env_a = PipelineEnv(backend=backend)
        pipeline.run_forward_pass(initial_spec="ignored", env=env_a)
        first = dict(env_a.artifacts["per_group_shots"][(("spec", "circ"),)])
        assert sum(first.values()) == 300

        # Bump the backend's shot count and re-run — allocation must follow.
        backend._shots = 600
        env_b = PipelineEnv(backend=backend)
        pipeline.run_forward_pass(initial_spec="ignored", env=env_b)
        second = dict(env_b.artifacts["per_group_shots"][(("spec", "circ"),)])
        assert sum(second.values()) == 600

    def test_stateful_only_for_random_strategies(self):
        """Implementation detail: only random/callable strategies mark the
        stage stateful. Deterministic strategies ("uniform", "weighted") rely
        on cache_key_extras for invalidation and stay cacheable."""
        assert MeasurementStage().stateful is False
        assert MeasurementStage(shot_distribution="uniform").stateful is False
        assert MeasurementStage(shot_distribution="weighted").stateful is False
        assert MeasurementStage(shot_distribution="weighted_random").stateful is True
        assert (
            MeasurementStage(shot_distribution=lambda *a, **kw: None).stateful is True
        )

    def test_cache_key_extras_tracks_backend_shots(self, dummy_simulator):
        """Deterministic shot-distribution strategies must fold backend.shots
        into cache_key_extras so a shots change invalidates the cached trace."""
        env = PipelineEnv(backend=dummy_simulator)
        none_stage = MeasurementStage()
        uniform_stage = MeasurementStage(shot_distribution="uniform")
        weighted_stage = MeasurementStage(shot_distribution="weighted")

        assert none_stage.cache_key_extras(env) == ()
        assert uniform_stage.cache_key_extras(env) == (dummy_simulator.shots,)
        assert weighted_stage.cache_key_extras(env) == (dummy_simulator.shots,)

    def test_disable_then_run_clears_artifact(self):
        """If MeasurementStage runs once with shot_distribution and then a fresh
        env runs WITHOUT, per_group_shots should not appear."""

        backend = DummySimulator(shots=300)
        env_with = PipelineEnv(backend=backend)
        pipeline_with = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(shot_distribution="uniform"),
            ],
        )
        pipeline_with.run_forward_pass(initial_spec="ignored", env=env_with)
        assert "per_group_shots" in env_with.artifacts

        env_without = PipelineEnv(backend=backend)
        pipeline_without = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_three_group_meta()),
                MeasurementStage(),
            ],
        )
        pipeline_without.run_forward_pass(initial_spec="ignored", env=env_without)
        assert "per_group_shots" not in env_without.artifacts


class TestAllocatePerGroupShotsHelper:
    """Implementation detail: free helper _allocate_per_group_shots returns
    the expected ``(surviving, dropped, shots)`` triple based purely on its
    arguments — no MeasurementStage instance required."""

    def test_returns_full_indices_when_disabled(self):
        meta = _three_group_meta()
        groups, partition, _ = compute_measurement_groups(
            meta.observable, "qwc", meta.n_qubits
        )
        env = PipelineEnv(backend=DummySimulator(shots=100))

        surviving, dropped, shots = _allocate_per_group_shots(
            "spec_x",
            meta.observable,
            groups,
            partition,
            env,
            shot_distribution=None,
        )
        assert surviving == list(range(len(groups)))
        assert dropped == {}
        assert shots is None

    def test_returns_per_spec_shots_when_enabled(self):
        meta = _three_group_meta()
        groups, partition, _ = compute_measurement_groups(
            meta.observable, "qwc", meta.n_qubits
        )
        env = PipelineEnv(backend=DummySimulator(shots=300))

        surviving, dropped, shots = _allocate_per_group_shots(
            "spec_x",
            meta.observable,
            groups,
            partition,
            env,
            shot_distribution="uniform",
        )
        assert surviving == [0, 1, 2]
        assert dropped == {}
        assert shots == {0: 100, 1: 100, 2: 100}

    def test_drops_zero_shot_groups(self):
        meta = _three_group_meta()  # norms 10:1:0.1
        groups, partition, _ = compute_measurement_groups(
            meta.observable, "qwc", meta.n_qubits
        )
        env = PipelineEnv(backend=DummySimulator(shots=11))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surviving, dropped, shots = _allocate_per_group_shots(
                "spec_x",
                meta.observable,
                groups,
                partition,
                env,
                shot_distribution="weighted",
            )
        assert 2 not in shots
        assert dropped == {2: {0: 0.0}}
        assert surviving == [0, 1]
