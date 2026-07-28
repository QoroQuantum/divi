# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Dry-run mechanics at the pipeline and stage level: analytic-vs-real equivalence, QuEPP preview, safety fallbacks, circuit stats."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

import divi.circuits.quepp as _quepp_mod
import divi.pipeline.stages._pauli_twirl_stage as _pauli_twirl_mod
from divi.circuits import MetaCircuit
from divi.circuits.qem import _NoMitigation
from divi.circuits.quepp import QuEPP
from divi.pipeline import (
    CircuitPipeline,
    CircuitPreprocessor,
    PipelineCadence,
    dry_run_pipeline,
    format_dry_run,
)
from divi.pipeline._compilation import _compile_batch
from divi.pipeline._dry_run import (
    _aggregate_circuit_stats,
    _two_qubit_depth,
)
from divi.pipeline._preprocessor import sample_preprocessor
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    DiviPerformanceWarning,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageOutput,
    StageToken,
)
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    PreprocessStage,
    QEMStage,
)
from tests.pipeline._helpers import (
    DummySpecStage,
    assert_same_fanout,
    dry_run_stages,
    meta_with_observable,
    parametric_twirlable_meta,
    two_group_meta,
)


class TestDryRunPipeline:
    """Original dry-run report shape tests."""

    def test_basic_pipeline(self, dummy_pipeline_env):
        """Spec + Measurement produces correct fan-out."""
        _, report = dry_run_stages(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage()],
            dummy_pipeline_env,
            dry=False,
        )

        assert report.pipeline_name == "test"
        assert len(report.stages) == 2
        assert report.stages[0].name == "DummySpecStage"
        assert report.stages[1].name == "MeasurementStage"
        assert report.total_circuits > 0

    def test_total_matches_compile(self, dummy_pipeline_env):
        """Total circuits matches actual _compile_batch output (full-generation mode)."""
        trace, report = dry_run_stages(
            [
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
                QEMStage(protocol=_NoMitigation()),
            ],
            dummy_pipeline_env,
            dry=False,
        )
        compiled, _ = _compile_batch(trace.final_batch)
        assert report.total_circuits == len(compiled)

    def test_format_does_not_raise(self, dummy_pipeline_env):
        """format_dry_run prints without errors."""
        _, report = dry_run_stages(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage()],
            dummy_pipeline_env,
            dry=False,
        )
        format_dry_run({"test": report})

    def test_dry_run_pipeline_threads_cadence(self, dummy_pipeline_env):
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env, dry=False)
        report = dry_run_pipeline(
            "x",
            trace,
            pipeline.stages,
            dummy_pipeline_env,
            cadence=PipelineCadence.ONCE,
        )
        assert report.cadence is PipelineCadence.ONCE

    def test_total_shots_weights_circuits_by_backend_shots(self, dummy_pipeline_env):
        # dummy_expval_backend runs 100 shots; no shot_distribution, so every
        # circuit is billed the same.
        _, report = dry_run_stages(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage()],
            dummy_pipeline_env,
            dry=False,
        )
        assert report.total_shots == report.total_circuits * 100

    def test_a_broken_introspect_does_not_break_the_dry_run(
        self, dummy_pipeline_env, mocker
    ):
        """``introspect`` is optional colour, so a bug in one must be reported in
        place rather than taking down the instrument being used to debug."""
        spec = DummySpecStage(meta=two_group_meta())
        mocker.patch.object(
            spec, "introspect", side_effect=RuntimeError("my introspect has a bug")
        )
        _, report = dry_run_stages([spec, MeasurementStage()], dummy_pipeline_env)

        assert report.total_circuits > 0  # the analysis still completed
        assert "RuntimeError" in report.stages[0].metadata["introspect failed"]

    def test_objective_fingerprint_tracks_the_observable_coefficients(
        self, dummy_pipeline_env
    ):
        """The fingerprint has to come off the real observable, not be hand-set:
        two Hamiltonians differing only in a coefficient must not share it."""
        a = dry_run_stages(
            [
                DummySpecStage(meta=meta_with_observable(SparsePauliOp(["ZZ"], [1.0]))),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
        )[1]
        b = dry_run_stages(
            [
                DummySpecStage(meta=meta_with_observable(SparsePauliOp(["ZZ"], [2.0]))),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
        )[1]
        assert a.total_circuits == b.total_circuits
        assert a.objective_fingerprint != b.objective_fingerprint

    def test_objective_fingerprint_survives_float_noise(self, dummy_pipeline_env):
        """Coefficients are rounded, so recomputing a Hamiltonian must not split two
        otherwise-identical objectives on the last few bits of a float."""

        def fingerprint(coefficient):
            return dry_run_stages(
                [
                    DummySpecStage(
                        meta=meta_with_observable(SparsePauliOp(["ZZ"], [coefficient]))
                    ),
                    MeasurementStage(),
                ],
                dummy_pipeline_env,
            )[1].objective_fingerprint

        # Differing past the rounding precision: the same objective.
        assert fingerprint(1.0) == fingerprint(1.0 + 1e-12)
        # Differing within it: distinct objectives.
        assert fingerprint(1.0) != fingerprint(1.000001)


@pytest.mark.filterwarnings("ignore:shot_distribution is set but backend")
class TestAnalyticDryRun:
    """Analytic dry path (dry=True) must produce identical counts to real expand."""

    def test_pauli_twirl_counts_match(self, dummy_pipeline_env):
        """PauliTwirl fan-out is known analytically — dry and real counts must match."""
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2], [0.3, 0.4]])

        def stages():
            return [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=7, seed=0),
                MeasurementStage(),
            ]

        _, real = dry_run_stages(stages(), dummy_pipeline_env, dry=False, name="real")
        _, dry = dry_run_stages(stages(), dummy_pipeline_env, dry=True, name="dry")
        assert_same_fanout(real, dry)

    def test_param_binding_fast_path_counts_correctly(
        self, dummy_sampling_pipeline_env
    ):
        """Fast-path ParameterBindingStage populates qasm_bodies; dry-run
        counter must read from there (not from the untouched circuit_bodies)."""
        meta = parametric_twirlable_meta()
        dummy_sampling_pipeline_env.param_sets = np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        _, dry_report = dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_sampling_pipeline_env,
        )

        # 1 body × 3 param sets × 2 obs groups.
        assert dry_report.total_circuits == 6
        # ZZ+XX don't QWC-commute, so grouping saves nothing: factor == 1.
        meas_stage = next(s for s in dry_report.stages if s.name == "MeasurementStage")
        assert meas_stage.factor == 1.0

    def test_dry_preprocess_isolates_each_entry_from_the_others(
        self, dummy_pipeline_env
    ):
        """An analytic upstream stage hands the same DAG object to every batch
        entry, so a transform that mutates in place would reach all of them and
        the caller's own copy. Each entry is given a private one first."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        shared = circuit_to_dag(qc)
        original_size = shared.size()

        def meta():
            return MetaCircuit(
                circuit_bodies=(((), shared),),
                parameters=(theta,),
                observable=SparsePauliOp.from_list([("Z", 1.0)]),
            )

        def mutate_in_place(m):
            _, dag = m.circuit_bodies[0]
            dag.apply_operation_back(XGate(), (dag.qubits[0],))
            return m

        stage = PreprocessStage(
            CircuitPreprocessor("mutating", preprocess=mutate_in_place)
        )
        out = stage.dry_expand({"a": meta(), "b": meta()}, dummy_pipeline_env)

        # Each entry absorbed exactly its own mutation, not the other's.
        for key in ("a", "b"):
            assert out.batch[key].circuit_bodies[0][1].size() == original_size + 1
        assert shared.size() == original_size

    def test_dry_skips_pauli_twirl_deepcopy(self, dummy_pipeline_env, mocker):
        """Dry PauliTwirl must not invoke the twirl-substitute DAG surgery."""
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=50, seed=0),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
        )
        assert spy.call_count == 0, "dry path must skip twirl DAG substitution"

    def test_real_path_uses_pauli_twirl_deepcopy(self, dummy_pipeline_env, mocker):
        """Sanity check: the real path still invokes the twirl DAG surgery."""
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=5, seed=0),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            dry=False,
        )
        assert spy.call_count > 0, "real path must apply twirl DAG substitution"

    def test_dry_preserves_per_group_shots_artifact(self, dummy_sampling_pipeline_env):
        """Dry MeasurementStage must still populate per_group_shots via shot allocation."""
        trace, _ = dry_run_stages(
            [
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_sampling_pipeline_env,
        )
        assert "per_group_shots" in trace.env_artifacts

    def test_introspect_metadata_survives_dry(self, dummy_sampling_pipeline_env):
        """Each stage's ``introspect()`` feeds ``DryRunReport.stages[i].metadata``.
        If ``introspect`` were silently skipped in dry mode, or stages swapped
        for ones that return degenerate metadata, the fan-out counts would
        still match — so the payload itself needs its own lock-in."""
        meta = parametric_twirlable_meta()
        dummy_sampling_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        _, report = dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=4, seed=0),
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_sampling_pipeline_env,
        )

        by_name = {s.name: s for s in report.stages}

        # ParameterBindingStage surfaces its param-set count and path choice.
        # With PauliTwirlStage (consumes DAG bodies) downstream, ParamBind
        # is forced onto the slow DAG-binding path.
        pb = by_name["ParameterBindingStage"]
        assert pb.metadata["n_param_sets"] == 1
        assert pb.metadata["n_bound_params"] == 2
        assert pb.metadata["fast_path"] is False

        # PauliTwirlStage surfaces its configured twirl count and path choice.
        # With ParamBind upstream (param_set axis) and MeasurementStage
        # downstream (no DAG consumer), PauliTwirl lands on its fast path.
        pt = by_name["PauliTwirlStage"]
        assert pt.metadata["n_twirls"] == 4
        assert pt.metadata["fast_path"] is True

        # MeasurementStage surfaces the observable grouping outcome — ZZ
        # and XX don't QWC-commute, so each gets its own group.
        meas = by_name["MeasurementStage"]
        n_pauli_terms = sum(len(o) for o in meta.observable)
        assert meas.metadata["n_groups"] == n_pauli_terms
        assert meas.metadata["n_pauli_terms"] == n_pauli_terms
        assert meas.factor == 1.0

    def test_env_artifacts_surface_on_dry_run_report(self, dummy_sampling_pipeline_env):
        """``DryRunReport.env_artifacts`` is the canonical introspection surface
        for stage-produced state — callers should not need to drop into
        ``_build_pipeline_env`` or a pipeline's private spec factory to read
        shot allocations or other forward-pass artifacts."""
        trace, report = dry_run_stages(
            [
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_sampling_pipeline_env,
        )
        assert "per_group_shots" in report.env_artifacts
        assert (
            report.env_artifacts["per_group_shots"]
            == trace.env_artifacts["per_group_shots"]
        )

    def test_dry_trace_not_cached(self, dummy_pipeline_env, mocker):
        """Dry traces must never be cached nor served from the forward-pass cache."""
        meta = two_group_meta()
        spec_stage = DummySpecStage(meta=meta)
        pipeline = CircuitPipeline(stages=[spec_stage, MeasurementStage()])
        expand_spy = mocker.spy(spec_stage, "expand")

        pipeline.run_forward_pass("ignored", dummy_pipeline_env, dry=True)
        pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        pipeline.run_forward_pass("ignored", dummy_pipeline_env)

        assert expand_spy.call_count == 2

    def test_real_run_after_dry_writes_real_trace(self, dummy_pipeline_env):
        """A real pass after a dry pass never inherits placeholder bodies."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )

        pipeline.run_forward_pass("ignored", dummy_pipeline_env, dry=True)

        real_trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)

        # Dry mode's measurement placeholders are empty strings; the real
        # pass must emit actual ``measure q[...] -> c[...]`` QASM.
        real_meta = next(iter(real_trace.final_batch.values()))
        real_meas = real_meta.measurement_qasms
        assert real_meas, "real measurement_qasms must be populated"
        assert all("measure" in qasm for _tag, qasm in real_meas), (
            "real run produced placeholder measurement QASMs — "
            "cache leaked dry state"
        )


@pytest.mark.filterwarnings("ignore:shot_distribution is set but backend")
class TestMeasurementStageReduction:
    """MeasurementStage must be reported as a reduction when observable grouping
    collapses multiple Pauli terms into fewer commuting groups. The spec stage's
    logical count uses one circuit per term; the measurement stage's logical
    count uses one circuit per group."""

    @staticmethod
    def _qwc_single_group_meta() -> MetaCircuit:
        """0.5*ZZ + 0.5*ZI — both QWC-commute, so qwc grouping yields 1 group."""
        qc = QuantumCircuit(2)
        qc.h(0)
        observable = SparsePauliOp.from_list([("ZZ", 0.5), ("ZI", 0.5)])
        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=observable,
        )

    def test_qwc_grouping_shows_reduction(self, dummy_pipeline_env):
        """Two QWC-commuting Pauli terms collapse to one group — the spec stage
        reports one circuit per term (×2 baseline) and MeasurementStage reports
        the grouping as a reduction (``factor = 0.5``)."""
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=self._qwc_single_group_meta()),
                # Pin shot_distribution so MeasurementStage stays on the qwc
                # branch (the dummy backend supports expval and would otherwise
                # auto-promote to _backend_expval).
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_pipeline_env,
        )

        spec, meas = report.stages
        assert spec.factor == 2.0  # one per Pauli term
        assert meas.factor == 0.5  # 2 terms -> 1 group
        assert meas.metadata["n_groups"] == 1
        assert report.total_circuits == 1

    def test_reduction_scales_with_upstream_fanout(self, dummy_pipeline_env):
        """Upstream body fan-out (e.g. ParameterBindingStage) scales the logical
        baseline. MeasurementStage still reports the same reduction factor."""
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(phi, 1)
        qc.cx(0, 1)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=(theta, phi),
            observable=SparsePauliOp.from_list([("ZZ", 0.5), ("ZI", 0.5)]),
        )

        _, report = dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(shot_distribution="weighted"),
            ],
            dummy_pipeline_env,
        )

        spec, param, meas = report.stages
        assert spec.factor == 2.0  # 1 body × 2 Pauli terms
        assert param.factor == 3.0  # 3 param sets
        assert meas.factor == 0.5  # 2 terms -> 1 group
        assert report.total_circuits == 3

    def test_backend_expval_shows_full_reduction(self, dummy_pipeline_env):
        """When MeasurementStage auto-promotes to ``_backend_expval`` (all-same
        observable, expval-native backend), the whole observable collapses to
        one backend-evaluated expval — ``factor = 1 / n_terms``."""
        # Four distinct Pauli terms fed to an expval backend without
        # shot_distribution, so strategy auto-promotes to ``_backend_expval``.
        qc = QuantumCircuit(2)
        qc.h(0)
        observable = SparsePauliOp.from_list(
            [("ZZ", 0.1), ("ZI", 0.2), ("IZ", 0.3), ("II", 0.4)]
        )
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=observable,
        )
        _, report = dry_run_stages(
            [DummySpecStage(meta=meta), MeasurementStage()], dummy_pipeline_env
        )

        spec, meas = report.stages
        # All 4 Paulis QWC-commute, so plain qwc would also yield 1 group —
        # pin the strategy so this test doesn't silently pass if auto-promotion
        # were removed.
        assert meas.metadata["strategy"] == "_backend_expval"
        assert spec.factor == 4.0  # one per Pauli term
        assert meas.factor == pytest.approx(0.25)  # 4 terms -> 1 backend expval
        assert report.total_circuits == 1

    def test_probs_path_has_unit_factor(self, dummy_pipeline_env):
        """Probs circuits (``measured_wires`` instead of observable) have no
        observable to group — MeasurementStage must report ``factor = 1`` and
        the total matches the body count."""
        qc = QuantumCircuit(2)
        qc.h(0)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            measured_wires=(0, 1),
        )
        _, report = dry_run_stages(
            [DummySpecStage(meta=meta), MeasurementStage()], dummy_pipeline_env
        )

        spec, meas = report.stages
        assert spec.factor == 1.0  # 1 body, no observable
        assert meas.factor == 1.0
        assert report.total_circuits == 1

    def test_counts_override_on_observable_counts_per_term(self, dummy_pipeline_env):
        """A raw COUNTS readout of a grouped observable still measures it per term —
        the observable survives onto the final batch — so the per-term baseline must
        apply. The terminal ResultFormat is COUNTS for both this and a
        dropped-observable sample, so it cannot be the signal."""
        qc = QuantumCircuit(2)
        qc.h(0)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=SparsePauliOp.from_list([("ZZ", 0.5), ("ZI", 0.5)]),
        )
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=meta),
                MeasurementStage(result_format_override=ResultFormat.COUNTS),
            ],
            dummy_pipeline_env,
        )
        # Per-term baseline retained (2 terms), not collapsed to 1 as it would be
        # if COUNTS were misread as sampling.
        assert report.stages[0].factor == 2.0
        assert report.total_circuits == 1

    def test_sampling_dropped_observable_is_not_a_phantom_reduction(
        self, dummy_pipeline_env
    ):
        """A sampling routine (``ResultFormat.PROBS``) drops the observable for a
        computational-basis measurement, so the naive per-Pauli-term baseline must
        not apply — otherwise the drop renders as a phantom ``÷N`` grouping that
        never happened. Uses the real ``sample_preprocessor`` transform."""
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=parametric_twirlable_meta()),  # 2-term observable
                PreprocessStage(sample_preprocessor()),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            dry=False,
        )
        spec, preprocess, meas = report.stages
        assert spec.factor == 1.0  # bodies, not the 2 observable terms
        assert preprocess.factor == 1.0  # no phantom reduction where obs is dropped
        assert meas.factor == 1.0


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestQuEPPDryExpand:
    """QuEPP dry path must skip Clifford simulation while preserving fan-out."""

    def test_quepp_dry_matches_real_counts(self, dummy_pipeline_env):
        """QuEPP's analytic path must produce the same ``1 + n_paths`` fan-out."""
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        def stages():
            return [
                DummySpecStage(meta=meta),
                QEMStage(
                    protocol=QuEPP(
                        truncation_order=1, n_twirls=0, sampling="exhaustive"
                    )
                ),
                PauliTwirlStage(n_twirls=3, seed=0),
                MeasurementStage(),
            ]

        _, real_report = dry_run_stages(
            stages(),
            dummy_pipeline_env,
            dry=False,
            name="r",
            suppress_performance_warnings=True,
        )
        _, dry_report = dry_run_stages(
            stages(),
            dummy_pipeline_env,
            dry=True,
            name="d",
            suppress_performance_warnings=True,
        )
        assert_same_fanout(real_report, dry_report)

        # QEMStage's introspect metadata must also survive the dry path —
        # n_rotations / n_paths are populated in QuEPP.dry_expand's context
        # and must match the real path's counts.
        real_qem = next(s for s in real_report.stages if s.name == "QEMStage")
        dry_qem = next(s for s in dry_report.stages if s.name == "QEMStage")
        assert dry_qem.metadata["protocol"] == "quepp"
        assert dry_qem.metadata["n_rotations"] == real_qem.metadata["n_rotations"]
        assert dry_qem.metadata["n_paths"] == real_qem.metadata["n_paths"]

    def _concrete_montecarlo_meta(self) -> MetaCircuit:
        """A bound circuit: Monte Carlo path sampling only runs on concrete angles."""
        qc = QuantumCircuit(2)
        qc.rx(0.31, 0)
        qc.cx(0, 1)
        qc.rz(0.47, 1)
        qc.ry(0.19, 0)
        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=(),
            observable=SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)]),
        )

    def test_montecarlo_preview_does_not_advance_the_protocol_rng(
        self, dummy_pipeline_env
    ):
        """Previewing must not change what a later run samples.

        The protocol owns its generator, so drawing preview paths from it left the
        real run further along the stream — previewing a seeded program changed its
        circuit count.
        """
        protocol = QuEPP(truncation_order=2, n_twirls=0, seed=11)
        meta = self._concrete_montecarlo_meta()
        state_before = protocol._rng.bit_generator.state

        dags, context = protocol.dry_expand(
            dag=meta.circuit_bodies[0][1], observable=meta.observable
        )

        assert protocol._rng.bit_generator.state == state_before
        assert context["sampled_paths"] is True
        assert len(dags) == 1 + context["n_paths"]

    def test_montecarlo_preview_does_not_depend_on_run_position(self):
        """One protocol instance is often shared across an ensemble's programs, and
        each program's expand advances its generator. A preview that read the
        generator's *position* therefore described only the first program — the
        rest were quoted a count drawn from a state their run had already left.
        """
        protocol = QuEPP(truncation_order=2, n_twirls=0, seed=11)
        meta = self._concrete_montecarlo_meta()
        dag, observable = meta.circuit_bodies[0][1], meta.observable

        previews = []
        for _ in range(4):
            previews.append(len(protocol.dry_expand(dag=dag, observable=observable)[0]))
            # Interleave real expands, as running one program of an ensemble does.
            protocol.expand(dag=dag, observable=observable)
        assert len(set(previews)) == 1, previews

    def test_montecarlo_preview_is_reproducible_and_discloses_sampling(
        self, dummy_pipeline_env
    ):
        """A sampled fan-out varies per evaluation, so the stage says so — and
        repeated previews of one program agree rather than drifting."""
        protocol = QuEPP(truncation_order=2, n_twirls=0, seed=11)
        meta = self._concrete_montecarlo_meta()
        previews = [
            protocol.dry_expand(
                dag=meta.circuit_bodies[0][1], observable=meta.observable
            )
            for _ in range(4)
        ]
        assert len({len(dags) for dags, _ in previews}) == 1
        assert all(ctx.get("sampled_paths") for _, ctx in previews)

        stage = QEMStage(protocol=protocol)
        info = stage.introspect({}, env=dummy_pipeline_env, token={(): previews[0][1]})
        assert info["path_count"] == "sampled (an estimate, not an exact count)"

    def test_exhaustive_enumeration_is_not_marked_sampled(self, dummy_pipeline_env):
        """Deterministic enumeration must not carry the sampled-count caveat."""
        protocol = QuEPP(truncation_order=1, n_twirls=0, sampling="exhaustive")
        meta = self._concrete_montecarlo_meta()
        _, context = protocol.dry_expand(
            dag=meta.circuit_bodies[0][1], observable=meta.observable
        )
        assert "sampled_paths" not in context

    def test_quepp_dry_skips_clifford_simulation(self, dummy_pipeline_env, mocker):
        """Dry QuEPP must not invoke the Clifford ensemble simulator."""
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_quepp_mod, "_simulate_clifford_ensemble")

        dry_run_stages(
            [
                DummySpecStage(meta=meta),
                QEMStage(
                    protocol=QuEPP(
                        truncation_order=1, n_twirls=0, sampling="exhaustive"
                    )
                ),
                PauliTwirlStage(n_twirls=1, seed=0),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            suppress_performance_warnings=True,
        )
        assert spy.call_count == 0, "dry QuEPP must skip Clifford simulation"


class _NonDryDagConsumerStage(BundleStage):
    """Test double: a third-party-style DAG consumer lacking ``dry_expand``.

    Declares ``consumes_dag_bodies=True`` to advertise the intent of
    reading / mutating body DAGs, so the pipeline will recognise it as an
    unsafe downstream neighbour under dry mode. ``expand`` itself is a
    passthrough — the test cares only about the fallback decision, not
    about actual DAG mutation.

    ``axis_name`` is parameterised so several instances can coexist in the
    same pipeline (pipeline validation rejects duplicate axis names).
    """

    def __init__(self, axis_name: str = "non_dry_consumer") -> None:
        super().__init__(name=type(self).__name__)
        self._axis_name = axis_name

    @property
    def consumes_dag_bodies(self) -> bool:
        return True

    @property
    def axis_name(self) -> str:
        return self._axis_name

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        return StageOutput(batch=dict(batch))

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results


class _SecondNonDryDagConsumerStage(_NonDryDagConsumerStage):
    """Sibling of :class:`_NonDryDagConsumerStage` — used to exercise the
    multi-culprit path in :func:`_warn_dry_fallback`, where the warning's
    culprit list comma-joins several distinct class names."""

    def __init__(self) -> None:
        super().__init__(axis_name="second_non_dry_consumer")


class TestDrySafetyFallback:
    """Dry runs demote upstream stages to real ``expand`` when a downstream
    stage would mutate shared placeholder DAGs, warning the user and keeping
    the circuit count correct."""

    def _stages(self, meta):
        return [
            DummySpecStage(meta=meta),
            PauliTwirlStage(n_twirls=4, seed=0),
            _NonDryDagConsumerStage(),
            MeasurementStage(),
        ]

    def test_fallback_warning_names_upstream_and_culprit(self, dummy_pipeline_env):
        """The emitted warning names both the upstream dry-aware stage and
        the downstream non-dry-aware DAG consumer(s)."""
        with pytest.warns(DiviPerformanceWarning) as record:
            dry_run_stages(
                self._stages(parametric_twirlable_meta()), dummy_pipeline_env
            )

        messages = [str(w.message) for w in record.list]
        assert any(
            "PauliTwirlStage" in msg and "_NonDryDagConsumerStage" in msg
            for msg in messages
        ), (
            "Expected a DiviPerformanceWarning naming both PauliTwirlStage "
            f"and _NonDryDagConsumerStage. Got: {messages}"
        )

    def test_fallback_warning_lists_multiple_culprits(self, dummy_pipeline_env):
        """When two or more downstream stages are unsafe, the warning's
        ``culprits`` list must name all of them — comma-joined, in pipeline
        order — so users can see every stage they need to fix."""
        stages = [
            DummySpecStage(meta=parametric_twirlable_meta()),
            PauliTwirlStage(n_twirls=3, seed=0),
            _NonDryDagConsumerStage(),
            _SecondNonDryDagConsumerStage(),
            MeasurementStage(),
        ]
        with pytest.warns(DiviPerformanceWarning) as record:
            dry_run_stages(stages, dummy_pipeline_env)

        messages = [str(w.message) for w in record.list]
        # Both distinct culprit class names must appear in a single warning,
        # and in pipeline order (comma-joined by :func:`_warn_dry_fallback`).
        assert any(
            "_NonDryDagConsumerStage, _SecondNonDryDagConsumerStage" in msg
            for msg in messages
        ), (
            "Expected a warning listing both culprits comma-joined in "
            f"pipeline order. Got: {messages}"
        )

    def test_fallback_invokes_real_expand(self, dummy_pipeline_env, mocker):
        """The demoted stage's real expand ran (not the dry placeholder path)."""
        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        with pytest.warns(DiviPerformanceWarning):
            dry_run_stages(
                self._stages(parametric_twirlable_meta()), dummy_pipeline_env
            )
        assert spy.call_count > 0, (
            "Fallback should have run the real PauliTwirl expand, which "
            "invokes _apply_twirl_substitute"
        )

    def test_fallback_preserves_circuit_count(self, dummy_pipeline_env):
        """The analytic+fallback dry run must report the same count as a
        fully-real forward pass."""
        meta = parametric_twirlable_meta()

        with pytest.warns(DiviPerformanceWarning):
            _, dry_report = dry_run_stages(
                self._stages(meta), dummy_pipeline_env, name="dry"
            )
        _, real_report = dry_run_stages(
            self._stages(meta), dummy_pipeline_env, dry=False, name="real"
        )
        assert dry_report.total_circuits == real_report.total_circuits

    def test_no_warning_when_all_downstream_dry_aware(self, dummy_pipeline_env):
        """Safe pipelines (every downstream stage overrides ``dry_expand``)
        must not emit the fallback warning."""
        stages = [
            DummySpecStage(meta=parametric_twirlable_meta()),
            PauliTwirlStage(n_twirls=4, seed=0),
            MeasurementStage(),
        ]
        # Any ``DiviPerformanceWarning`` fired here would mean the pipeline
        # spuriously demoted a stage — promote it to an exception so the
        # test fails loudly rather than needing to inspect a record list.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            dry_run_stages(stages, dummy_pipeline_env)


class TestTwoQubitDepth:
    """Spec: ``_two_qubit_depth`` returns the longest chain of 2q gates."""

    def test_zero_when_no_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.s(0)
        assert _two_qubit_depth(circuit_to_dag(qc)) == 0

    def test_counts_chained_2q_via_shared_qubit(self):
        # Three CXs all touching qubit 0 → chain of 3 (each follows the previous).
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        assert _two_qubit_depth(circuit_to_dag(qc)) == 3

    def test_independent_2q_gates_dont_chain(self):
        # CX(0,1) and CX(2,3) share no qubits → each contributes a chain of 1.
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        assert _two_qubit_depth(circuit_to_dag(qc)) == 1

    def test_ignores_single_qubit_gates_between_2q(self):
        # Single-qubit gates extend overall depth but not 2q-depth.
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        # depth() includes the H layers; _two_qubit_depth only counts the CXs.
        assert circuit_to_dag(qc).depth() > 2
        assert _two_qubit_depth(circuit_to_dag(qc)) == 2


class TestCircuitStatsAggregate:
    """Spec: ``DryRunReport.circuit_stats`` aggregates depth/width over the
    final batch's DAG bodies, mirroring ``CircuitRunner.depth_history`` semantics.
    """

    def test_constant_depth_yields_zero_std(self, dummy_pipeline_env):
        # Single body, fanned out only by Pauli twirling → all variants share
        # the same parametric structure, so depth/width stats are constant.
        meta = parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])
        _, report = dry_run_stages(
            [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=3, seed=0),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            name="t",
        )
        stats = report.circuit_stats
        assert stats, "circuit_stats should be populated when DAG bodies exist"
        # Catch zeroing bugs: the parametric meta has gates, so mean_depth > 0.
        assert stats["mean_depth"] > 0
        assert stats["std_depth"] == 0.0
        assert stats["min_depth"] == stats["max_depth"]
        assert stats["min_width"] == stats["max_width"] == 2

    def test_varying_depth_populates_range_stats(self):
        # Two MetaCircuits with different depths exercise the spread branches
        # of every stat (min/max/mean/std/2q_depth) — the constant case
        # collapses all of them to the same number.
        qc_shallow = QuantumCircuit(2)
        qc_shallow.cx(0, 1)
        qc_deep = QuantumCircuit(2)
        qc_deep.cx(0, 1)
        qc_deep.cx(0, 1)
        batch = {
            (("circuit", 0),): MetaCircuit(
                circuit_bodies=(((), circuit_to_dag(qc_shallow)),)
            ),
            (("circuit", 1),): MetaCircuit(
                circuit_bodies=(((), circuit_to_dag(qc_deep)),)
            ),
        }
        stats = _aggregate_circuit_stats(batch)
        assert stats["min_depth"] == 1
        assert stats["max_depth"] == 2
        assert stats["mean_depth"] == 1.5
        assert stats["std_depth"] > 0
        assert stats["mean_2q_depth"] == 1.5
        # 1 + 2 CX gates summed across the two bodies.
        assert stats["total_2q_gates"] == 3
        # Width is constant at 2 across both circuits.
        assert stats["min_width"] == stats["max_width"] == 2
        assert stats["std_width"] == 0.0

    def test_empty_dict_for_empty_batch(self):
        # Aggregator returns {} when no MetaCircuits have DAG bodies to read.
        assert _aggregate_circuit_stats({}) == {}
