# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipeline dry-run tool."""

import warnings

import networkx as nx
import numpy as np
import pennylane as qp
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate, XGate
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
from divi.pipeline._dry_run import _aggregate_circuit_stats, _two_qubit_depth
from divi.pipeline._preprocessor import sample_preprocessor
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ContractViolation,
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
from divi.qprog import QAOA, VQE, HartreeFockAnsatz, QuantumProgram
from divi.qprog._metrics import (
    METRIC_ROUTINE,
    FubiniStudyMetricEstimator,
    MetricEstimator,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
)
from divi.qprog.algorithms import PCE, GenericLayerAnsatz, TimeEvolution
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    QNGOptimizer,
    QNSPSAOptimizer,
    ScipyMethod,
    ScipyOptimizer,
    SPSAOptimizer,
)
from divi.qprog.problems import BinaryOptimizationProblem, MaxCutProblem
from tests.pipeline._helpers import (
    DummySpecStage,
    FanoutAndSumStage,
    h2_vqe,
    meta_with_observable,
    two_group_meta,
)


def _metric_compatible_vqe(backend, optimizer, n_layers: int = 1) -> VQE:
    """A VQE whose GenericLayerAnsatz is compatible with all three metric
    estimators (expval cost, invertible, FS-supported RY/RZ gates)."""
    return VQE(
        molecule=qp.qchem.Molecule(
            symbols=["H", "H"],
            coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]),
        ),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=n_layers,
        backend=backend,
        optimizer=optimizer,
    )


def _h2_vqe_for_totals(backend, optimizer) -> VQE:
    """A circuit-seeded program for the submitted-circuit oracle."""
    return VQE(
        molecule=qp.qchem.Molecule(
            symbols=["H", "H"],
            coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]),
        ),
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        max_iterations=1,
    )


def _maxcut_qaoa_for_totals(backend, optimizer) -> QAOA:
    """A Hamiltonian-seeded program for the submitted-circuit oracle: its seed is a
    ``SparsePauliOp``, which declares no parameters of its own."""
    return QAOA(
        MaxCutProblem(nx.random_regular_graph(3, 6, seed=1)),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
        max_iterations=1,
    )


def _parametric_twirlable_meta() -> MetaCircuit:
    """MetaCircuit with CX gates (twirlable) and free parameters (bindable)."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.ry(phi, 1)
    qc.cx(1, 0)
    observable = SparsePauliOp.from_list([("ZZ", 0.9), ("XX", 0.4)])
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=(theta, phi),
        observable=observable,
    )


def _dry_run(
    stages,
    env,
    *,
    dry=True,
    name="test",
    cadence=PipelineCadence.PER_EVALUATION,
    **pipeline_kwargs,
):
    """Build a pipeline over ``stages``, run one forward pass, and return
    ``(trace, report)`` for that pass.

    These are bare pipelines with no routine behind them, so ``cadence`` is the
    helper's own choice rather than a declared one.

    A fresh ``CircuitPipeline`` is built per call, so a real and a dry pass
    over equivalent ``stages`` never share a forward-pass cache.
    """
    pipeline = CircuitPipeline(stages=stages, **pipeline_kwargs)
    trace = pipeline.run_forward_pass("ignored", env, dry=dry)
    report = dry_run_pipeline(name, trace, pipeline.stages, env, cadence)
    return trace, report


def _assert_same_fanout(report_a, report_b):
    """Two reports agree on total circuit count and per-stage fan-out factor."""
    assert report_a.total_circuits == report_b.total_circuits
    for stage_a, stage_b in zip(report_a.stages, report_b.stages):
        assert (
            stage_a.factor == stage_b.factor
        ), f"{stage_a.name}: {stage_a.factor} != {stage_b.factor}"


def _assert_report_dicts_match(reports_a, reports_b):
    """Per-pipeline report dicts (``QuantumProgram.dry_run`` output) agree
    pipeline-by-pipeline on totals and per-stage fan-out."""
    assert set(reports_a) == set(reports_b)
    for name in reports_a:
        _assert_same_fanout(reports_a[name], reports_b[name])


class TestDryRunPipeline:
    """Original dry-run report shape tests."""

    def test_basic_pipeline(self, dummy_pipeline_env):
        """Spec + Measurement produces correct fan-out."""
        _, report = _dry_run(
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
        trace, report = _dry_run(
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
        _, report = _dry_run(
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
        _, report = _dry_run(
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
        _, report = _dry_run([spec, MeasurementStage()], dummy_pipeline_env)

        assert report.total_circuits > 0  # the analysis still completed
        assert "RuntimeError" in report.stages[0].metadata["introspect failed"]

    def test_objective_key_tracks_the_observable_coefficients(self, dummy_pipeline_env):
        """The fingerprint has to come off the real observable, not be hand-set:
        two Hamiltonians differing only in a coefficient must not share it."""
        a = _dry_run(
            [
                DummySpecStage(meta=meta_with_observable(SparsePauliOp(["ZZ"], [1.0]))),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
        )[1]
        b = _dry_run(
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
            return _dry_run(
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
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2], [0.3, 0.4]])

        def stages():
            return [
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=7, seed=0),
                MeasurementStage(),
            ]

        _, real = _dry_run(stages(), dummy_pipeline_env, dry=False, name="real")
        _, dry = _dry_run(stages(), dummy_pipeline_env, dry=True, name="dry")
        _assert_same_fanout(real, dry)

    def test_param_binding_fast_path_counts_correctly(
        self, dummy_sampling_pipeline_env
    ):
        """Fast-path ParameterBindingStage populates qasm_bodies; dry-run
        counter must read from there (not from the untouched circuit_bodies)."""
        meta = _parametric_twirlable_meta()
        dummy_sampling_pipeline_env.param_sets = np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        _, dry_report = _dry_run(
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
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        _dry_run(
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
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        _dry_run(
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
        trace, _ = _dry_run(
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
        meta = _parametric_twirlable_meta()
        dummy_sampling_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        _, report = _dry_run(
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
        trace, report = _dry_run(
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
        _, report = _dry_run(
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

        _, report = _dry_run(
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
        _, report = _dry_run(
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
        _, report = _dry_run(
            [DummySpecStage(meta=meta), MeasurementStage()], dummy_pipeline_env
        )

        spec, meas = report.stages
        assert spec.factor == 1.0  # 1 body, no observable
        assert meas.factor == 1.0
        assert report.total_circuits == 1


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestQuEPPDryExpand:
    """QuEPP dry path must skip Clifford simulation while preserving fan-out."""

    def test_quepp_dry_matches_real_counts(self, dummy_pipeline_env):
        """QuEPP's analytic path must produce the same ``1 + n_paths`` fan-out."""
        meta = _parametric_twirlable_meta()
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

        _, real_report = _dry_run(
            stages(),
            dummy_pipeline_env,
            dry=False,
            name="r",
            suppress_performance_warnings=True,
        )
        _, dry_report = _dry_run(
            stages(),
            dummy_pipeline_env,
            dry=True,
            name="d",
            suppress_performance_warnings=True,
        )
        _assert_same_fanout(real_report, dry_report)

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
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_quepp_mod, "_simulate_clifford_ensemble")

        _dry_run(
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


class _ParametricProgram(QuantumProgram):
    """A direct ``QuantumProgram`` subclass with a parametric seed — the shape the
    custom-program docs describe, and the one a VQA-only test never exercises
    (``VariationalQuantumAlgorithm`` overrides the env construction involved)."""

    def __init__(self, backend):
        super().__init__(backend=backend)
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        self._meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=SparsePauliOp.from_list([("Z", 1.0)]),
            parameters=(theta,),
        )

    def has_results(self) -> bool:
        return False

    def _initial_spec(self):
        return self._meta

    def _preprocessors(self):
        return (CircuitPreprocessor("cost", terminal_stage=MeasurementStage()),)

    def _assemble_pipeline(
        self, spec_stage, terminal_stage, *, result_format, extra_stages=()
    ):
        return CircuitPipeline(
            stages=[
                spec_stage,
                *extra_stages,
                *self._mitigation_stages(result_format),
                terminal_stage,
                ParameterBindingStage(),
            ]
        )

    def run(self):
        pass


@pytest.mark.parametrize("force", [False, True])
def test_direct_subclass_previews_its_real_parameter_width(dummy_simulator, force):
    """A direct subclass must get a parameter vector of its seed's width in both
    modes: the analytic path otherwise reports ``n_params: 0`` for a parametric
    circuit, and the forced path fails outright."""
    program = _ParametricProgram(dummy_simulator)
    report = program.dry_run(force_circuit_generation=force)["cost"]
    binding = next(s for s in report.stages if s.name == "ParameterBindingStage")
    assert binding.metadata["n_bound_params"] == 1


def test_routines_sharing_a_name_are_all_reported(dummy_simulator, mocker):
    """The routine name is the report key, so a second routine claiming it would
    replace the first — reporting fewer pipelines than the program runs."""
    program = _ParametricProgram(dummy_simulator)
    duplicate = CircuitPreprocessor("cost", terminal_stage=MeasurementStage())
    mocker.patch.object(
        program, "_preprocessors", return_value=(duplicate, duplicate, duplicate)
    )

    with pytest.warns(UserWarning, match="are both named 'cost'"):
        reports = program.dry_run()

    assert set(reports) == {"cost", "cost#2", "cost#3"}


@pytest.mark.usefixtures("suppress_quepp_warnings")
@pytest.mark.filterwarnings("ignore::UserWarning:divi.qprog.algorithms._vqe")
@pytest.mark.filterwarnings("ignore:shot_distribution is set but backend")
class TestQuantumProgramDryRun:
    """``QuantumProgram.dry_run`` analytic default + ``force_circuit_generation`` escape hatch."""

    @pytest.fixture
    def time_evolution_program(self, default_test_simulator):
        return TimeEvolution(
            hamiltonian=qp.PauliX(0) + qp.PauliZ(0),
            observable=qp.PauliZ(0),
            backend=default_test_simulator,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=3),
        )

    def _h2_vqe(self, backend, optimizer, **kwargs):
        return h2_vqe(backend, optimizer, **kwargs)

    def test_default_and_forced_match(self, time_evolution_program):
        """``dry_run()`` (analytic) and ``dry_run(force_circuit_generation=True)``
        must produce the same total circuit count and per-stage fan-out."""
        _assert_report_dicts_match(
            time_evolution_program.dry_run(),
            time_evolution_program.dry_run(force_circuit_generation=True),
        )

    def test_env_artifacts_exposed_via_dry_run(
        self, sampling_test_simulator, default_optimizer
    ):
        """Callers can read ``per_group_shots`` (and any other stage artifact)
        straight off the :class:`DryRunReport` — no private hooks needed."""
        vqe = self._h2_vqe(
            sampling_test_simulator,
            default_optimizer,
            grouping_strategy="qwc",
            shot_distribution="weighted",
        )
        cost_report = vqe.dry_run()["cost"]
        assert "per_group_shots" in cost_report.env_artifacts
        # The allocation is a dict {spec_key: {group_idx: shots}}; at least
        # one spec, at least one group with a non-zero shot budget.
        spec_alloc = next(iter(cost_report.env_artifacts["per_group_shots"].values()))
        assert any(shots > 0 for shots in spec_alloc.values())

    def test_qng_surfaces_metric_terms_pipeline(
        self, default_test_simulator, default_optimizer
    ):
        vqe = self._h2_vqe(default_test_simulator, QNGOptimizer())
        assert METRIC_ROUTINE in vqe.dry_run()

    def test_plain_optimizer_has_no_metric_pipeline(
        self, default_test_simulator, default_optimizer
    ):
        reports = self._h2_vqe(default_test_simulator, default_optimizer).dry_run()
        assert set(reports) == {"cost", "sample"}

    def test_an_incompatible_optimizer_refuses_the_preview(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """A preview of a pairing ``run()`` will reject is worth nothing: the report
        would omit exactly the routines that make it invalid, so it costs a program
        that cannot run. The refusal comes from the same check ``run()`` makes."""
        vqe = self._h2_vqe(default_test_simulator, QNSPSAOptimizer())
        mocker.patch.object(
            vqe.optimizer,
            "preprocessors",
            side_effect=ContractViolation("nope"),
        )
        with pytest.raises(ContractViolation, match="nope"):
            vqe.dry_run()

    def test_a_real_incompatible_pairing_refuses_with_the_estimators_reason(
        self, dummy_simulator, default_optimizer
    ):
        """PCE's cost is a classical (COUNTS) objective, which the pullback metric
        rejects. The estimator's own message names the alternative, so previewing
        surfaces it before any circuit is built."""
        pce = PCE(
            problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
            ansatz=GenericLayerAnsatz([RYGate, RZGate]),
            optimizer=QNGOptimizer(),  # default pullback metric
            backend=dummy_simulator,
        )
        with pytest.raises(ContractViolation, match="Fubini–Study estimator"):
            pce.dry_run()

    def test_fubini_study_surfaces_one_pipeline_per_block(
        self, default_test_simulator, default_optimizer
    ):
        """A block-diagonal Fubini-Study metric drives one prefix pipeline per
        commuting-gate block; each must appear with a unique report key (no
        same-named pipeline silently overwriting another)."""
        vqe = VQE(
            molecule=qp.qchem.Molecule(
                symbols=["H", "H"],
                coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]),
            ),
            ansatz=GenericLayerAnsatz([RYGate, RZGate]),
            n_layers=2,
            backend=default_test_simulator,
            optimizer=QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
        )
        reports = vqe.dry_run()
        prefix_keys = [k for k in reports if k.startswith(f"{METRIC_ROUTINE}[block")]
        n_blocks = len(vqe.optimizer.preprocessors(vqe))
        assert len(prefix_keys) == n_blocks > 1  # one per block, none dropped
        assert len(set(prefix_keys)) == len(prefix_keys)  # keys are unique

    def _metric_vqe(self, backend, optimizer):
        return _metric_compatible_vqe(backend, optimizer)

    def test_stage_report_exposes_class_label_and_axis_separately(
        self, default_test_simulator, default_optimizer
    ):
        """Three identifiers, each needed: the class (composition assertions), the
        constructor ``name=`` (telling two instances of one class apart), and the
        axis (what the rendered tree shows in brackets)."""
        stages = (
            self._h2_vqe(default_test_simulator, default_optimizer)
            .dry_run()["cost"]
            .stages
        )
        measurement = next(s for s in stages if s.name == "MeasurementStage")
        assert measurement.axis == "obs_group"
        assert measurement.label == "MeasurementStage"

    def test_label_and_name_differ_for_a_renamed_stage(self, dummy_pipeline_env):
        """The two coincide for built-in stages, so a stage whose constructor was
        given a different name is what proves they are separate identifiers."""
        _, report = _dry_run(
            [
                DummySpecStage(meta=two_group_meta()),
                FanoutAndSumStage(branch_prefix="b", n_children=2),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            dry=False,
        )
        info = next(s for s in report.stages if s.name == "FanoutAndSumStage")
        assert info.label == "FanoutAndSumStage:b"

    def test_sample_pipeline_is_once_cadence(
        self, default_test_simulator, default_optimizer
    ):
        reports = self._h2_vqe(default_test_simulator, default_optimizer).dry_run()
        assert reports["sample"].cadence is PipelineCadence.ONCE
        assert reports["cost"].cadence is PipelineCadence.PER_EVALUATION

    @pytest.mark.parametrize(
        "optimizer_factory, qem",
        [
            (MonteCarloOptimizer, None),
            (lambda: ScipyOptimizer(method=ScipyMethod.COBYLA), None),
            (SPSAOptimizer, None),
            (
                lambda: ScipyOptimizer(method=ScipyMethod.COBYLA),
                lambda: QuEPP(truncation_order=1, n_twirls=3),
            ),
        ],
        ids=["population", "gradient-free", "spsa", "mitigated"],
    )
    def test_two_qubit_total_matches_the_submitted_circuit_count(
        self, default_test_simulator, optimizer_factory, qem
    ):
        """Asserted as the invariant ``total × per-circuit == total_2q_gates`` rather
        than against fixed numbers. The stats walk DAG bodies and never sees
        ``qasm_bodies``, where parameter binding fans out — so a population
        optimizer's multiplicity went missing while mitigation's was counted, and a
        spot-check on one optimizer could not tell the difference."""
        kwargs = {"qem_protocol": qem()} if qem else {}
        vqe = self._h2_vqe(default_test_simulator, optimizer_factory(), **kwargs)
        for forced in (False, True):
            report = vqe.dry_run(force_circuit_generation=forced)["cost"]
            per_circuit = next(
                s.metadata["n_2q_gates"]
                for s in report.stages
                if "n_2q_gates" in s.metadata
            )
            assert report.circuit_stats["total_2q_gates"] == (
                report.total_circuits * per_circuit
            )

    def test_two_previews_of_one_program_compare_equal(self, sampling_test_simulator):
        """Nothing about a preview depends on when it ran, so a report is a value:
        two previews of an unchanged program are the same report."""
        vqe = self._h2_vqe(
            sampling_test_simulator,
            SPSAOptimizer(),
            shot_distribution="weighted",
        )
        report, twin = vqe.dry_run()["cost"], vqe.dry_run()["cost"]
        assert report == twin
        # The nested allocation the user guide teaches is readable either way.
        allocations = report.env_artifacts["per_group_shots"]
        assert next(iter(next(iter(allocations.values())).values())) is not None

    @pytest.mark.parametrize(
        "estimator",
        [
            PullbackMetricEstimator(),
            FubiniStudyMetricEstimator(),
            StochasticFidelityMetricEstimator(),
        ],
    )
    def test_every_estimator_validates_before_enumerating(
        self, default_test_simulator, default_optimizer, estimator, mocker
    ):
        """Every estimator's ``preprocessors`` must call ``check_compatible`` first,
        so enumerating routines cannot describe a pairing ``run()`` would reject —
        the contract can't hold for one estimator and silently lapse in another. It
        must also validate exactly once per call, not per routine returned."""
        vqe = self._metric_vqe(default_test_simulator, default_optimizer)
        spy = mocker.spy(estimator, "check_compatible")
        estimator.preprocessors(vqe)
        spy.assert_called_once_with(vqe)

    def test_qng_rejects_a_fidelity_sampling_estimator(self):
        """QNG needs a closed-form metric; the stochastic-fidelity estimator only
        supplies a fidelity function, so the pairing fails at construction rather
        than with a bare ValueError deep inside optimize()."""
        with pytest.raises(ValueError, match="supplies no closed-form metric"):
            QNGOptimizer(metric_estimator=StochasticFidelityMetricEstimator())

    def test_a_metric_the_ansatz_rejects_refuses_the_preview(
        self, default_test_simulator, default_optimizer
    ):
        """The ansatz's angles are expressions rather than bare parameters, which
        Fubini–Study cannot differentiate. Refusing names the gate and the angle;
        reporting the cost routine alone would price a run that cannot happen."""
        vqe = self._h2_vqe(
            default_test_simulator,
            QNSPSAOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
        )
        with pytest.raises(ContractViolation, match="not a bare trainable parameter"):
            vqe.dry_run()

    # The forced path runs QuEPP's real expand, which warns that Monte Carlo
    # sampling needs concrete angles — expected when comparing against it.
    @pytest.mark.filterwarnings("ignore:QuEPP")
    def test_analytic_path_matches_counts_but_not_mitigated_depth(
        self, default_test_simulator, default_optimizer
    ):
        """The documented split: circuit counts are exact on the analytic path,
        while the shape figures predate a rewriting stage (QuEPP + twirls here),
        because it is previewed as placeholders rather than real circuits."""
        vqe = self._h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
        )
        analytic = vqe.dry_run()["cost"]
        forced = vqe.dry_run(force_circuit_generation=True)["cost"]

        assert analytic.total_circuits == forced.total_circuits
        assert analytic.circuit_stats["mean_depth"] < forced.circuit_stats["mean_depth"]

    def test_once_pipeline_previews_a_single_parameter_set(
        self, default_test_simulator, default_optimizer
    ):
        """A one-time readout runs after optimization at the trained parameters —
        one set — not over the optimizer's working set. Previewing it with the
        population would report circuits the run never submits."""
        population = 6
        vqe = self._metric_vqe(
            default_test_simulator,
            MonteCarloOptimizer(population_size=population, n_best_sets=1),
        )
        reports = vqe.dry_run()
        assert reports["cost"].total_circuits == population
        assert reports["sample"].total_circuits == 1

    def test_sampling_readout_reports_no_observable_instead_of_zeros(
        self, default_test_simulator, default_optimizer
    ):
        """A computational-basis readout has no observable to partition, so the
        group/term counts are omitted rather than printed as a misleading ``0``."""
        vqe = self._metric_vqe(default_test_simulator, default_optimizer)
        sample_meta = next(
            s.metadata
            for s in vqe.dry_run()["sample"].stages
            if s.name == "MeasurementStage"
        )
        assert "readout" in sample_meta
        assert "n_groups" not in sample_meta
        assert "n_pauli_terms" not in sample_meta

    def test_each_metric_block_gets_its_own_report_entry(
        self, default_test_simulator, dummy_expval_backend
    ):
        """The pipeline name is the report key, so blocks sharing one would collapse
        into a single entry showing only the last block's circuits."""
        vqe = _metric_compatible_vqe(
            default_test_simulator,
            QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
        )
        prefix_keys = [
            k for k in vqe.dry_run() if k.startswith(f"{METRIC_ROUTINE}[block")
        ]
        assert len(prefix_keys) > 1
        assert len(set(prefix_keys)) == len(prefix_keys)
        assert all("block" in k for k in prefix_keys)

    @pytest.mark.parametrize(
        "optimizer_factory",
        [
            QNSPSAOptimizer,
            QNGOptimizer,
            lambda: QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()),
            MonteCarloOptimizer,
            lambda: ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        ],
        ids=["qnspsa", "qng", "qng-fs", "montecarlo", "lbfgsb"],
    )
    def test_forced_generation_works_for_every_auxiliary_pipeline(
        self, default_test_simulator, optimizer_factory
    ):
        """The report recommends ``force_circuit_generation=True`` for real shapes,
        so it must work on the pipelines the report shows. An auxiliary routine may
        bind a different parameter width than the cost one — an overlap circuit
        carries two concatenated vectors — which the env has to supply per routine.
        """
        vqe = _metric_compatible_vqe(default_test_simulator, optimizer_factory())
        lazy = vqe.dry_run()
        forced = vqe.dry_run(force_circuit_generation=True)
        assert set(lazy) == set(forced)
        for name, report in lazy.items():
            assert report.total_circuits == forced[name].total_circuits

    def test_forced_shape_is_reproducible_under_twirling(
        self, default_test_simulator, default_optimizer
    ):
        """Twirl labels are sampled, so an unseeded stage makes the one figure a
        hardware budget quotes — max depth — drift between calls on the same
        program, silently and with nothing marking it as a sample."""
        vqe = self._h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
            seed=1234,
        )
        first, second = (
            vqe.dry_run(force_circuit_generation=True)["cost"].circuit_stats
            for _ in range(2)
        )
        assert first == second

        twin = self._h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=4),
            seed=1234,
        )
        assert (
            twin.dry_run(force_circuit_generation=True)["cost"].circuit_stats == first
        )

    def test_analytic_shape_predates_a_rewriting_stage(
        self, default_test_simulator, default_optimizer
    ):
        """The analytic path measures the circuits *entering* a rewriting stage, so
        its depth understates the submitted one — while the counts stay exact.
        ``force_circuit_generation=True`` expands and measures for real."""
        vqe = self._h2_vqe(
            default_test_simulator,
            default_optimizer,
            qem_protocol=QuEPP(truncation_order=1, n_twirls=3),
        )
        lazy = vqe.dry_run()["cost"]
        forced = vqe.dry_run(force_circuit_generation=True)["cost"]

        assert lazy.total_circuits == forced.total_circuits
        assert lazy.circuit_stats["max_depth"] < forced.circuit_stats["max_depth"]

    def test_widest_basis_change_spans_all_groups(
        self, default_test_simulator, default_optimizer
    ):
        """With grouping off every group holds one term, so reading the width off
        the largest-by-size group degenerates to 1 — it must be the widest basis
        change any group needs (H2 carries width-4 terms such as ``YXXY``)."""
        vqe = self._h2_vqe(
            default_test_simulator, default_optimizer, grouping_strategy=None
        )
        meta = next(
            s.metadata
            for s in vqe.dry_run()["cost"].stages
            if s.name == "MeasurementStage"
        )
        assert meta["largest_group_size"] == 1
        assert meta["largest_group_width"] == 4

    def test_stochastic_bind_and_dry_run_share_overlap_preprocessor(
        self, default_test_simulator, default_optimizer
    ):
        """``bind`` and ``preprocessors`` build the same overlap routine (single
        source), so what is reported can't drift from what runs."""
        vqe = self._metric_vqe(default_test_simulator, default_optimizer)
        est = StochasticFidelityMetricEstimator()
        (preview_pp,) = est.preprocessors(vqe)
        run_pp = est._build_overlap_preprocessor(vqe)
        assert preview_pp.name == run_pp.name == METRIC_ROUTINE
        assert preview_pp.cache_key == run_pp.cache_key == METRIC_ROUTINE
        assert preview_pp.result_format == run_pp.result_format

    def test_dry_run_does_not_advance_program_rng(
        self, sampling_test_simulator, default_optimizer
    ):
        """A weighted-random preview draws from a throwaway RNG, so it neither
        advances the program's live generator (a later run() is unaffected) nor
        varies between repeated previews."""
        vqe = self._h2_vqe(
            sampling_test_simulator,
            default_optimizer,
            grouping_strategy="qwc",
            shot_distribution="weighted_random",
            seed=1234,
        )
        before = vqe._serialized_rng_state
        reports_a = vqe.dry_run()
        assert vqe._serialized_rng_state == before  # live RNG untouched

        reports_b = vqe.dry_run()
        assert (
            reports_a["cost"].env_artifacts["per_group_shots"]
            == reports_b["cost"].env_artifacts["per_group_shots"]
        )

    def test_dry_run_builds_env_without_reporter(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """The preview must stay silent — no progress reporter is wired into the
        forward-pass env, so nothing bleeds into stdout."""
        vqe = self._h2_vqe(default_test_simulator, default_optimizer)
        spy = mocker.spy(vqe, "_build_pipeline_env")
        vqe.dry_run()
        assert spy.call_count > 0
        assert all(call.kwargs.get("reporter") is None for call in spy.call_args_list)

    @pytest.mark.filterwarnings(
        "ignore:Backend supports analytic expectation values:UserWarning"
    )
    def test_default_and_forced_match_non_qem(
        self, default_test_simulator, default_optimizer
    ):
        """Parity between analytic and ``force_circuit_generation=True`` must
        also hold for programs without any QEM stage — the ``TimeEvolution``
        fixture above brings QuEPP, so the non-QEM path needs its own lock."""
        vqe = self._h2_vqe(
            default_test_simulator, default_optimizer, grouping_strategy="qwc"
        )
        _assert_report_dicts_match(
            vqe.dry_run(), vqe.dry_run(force_circuit_generation=True)
        )

    def test_dry_run_skips_initial_spec_without_preprocessors(
        self, default_test_simulator, default_optimizer, mocker
    ):
        """A program exposing no named pipelines returns ``{}`` from
        ``dry_run()`` without invoking ``_initial_spec()`` — honoring the
        documented opt-out for programs that never call ``evaluate()``."""
        vqe = self._h2_vqe(default_test_simulator, default_optimizer)
        mocker.patch.object(vqe, "_preprocessors", return_value=())
        spec = mocker.patch.object(
            vqe, "_initial_spec", side_effect=NotImplementedError
        )

        assert vqe.dry_run() == {}
        spec.assert_not_called()


def test_measurement_reports_which_qubits_it_reads(dummy_simulator):
    """``measure_all`` narrows the outcome space below the register width, which no
    other figure in the report reflects. A sampling backend, since a backend that
    evaluates the observable itself builds no basis-change circuits to narrow."""
    env = PipelineEnv(backend=dummy_simulator)

    def measured(flag):
        _, report = _dry_run(
            [DummySpecStage(meta=two_group_meta()), MeasurementStage(measure_all=flag)],
            env,
        )
        return report.stages[-1].metadata["measured_qubits"]

    assert measured(False) == "per group (observable support only)"
    assert measured(True) == "all"


@pytest.mark.parametrize(
    "make_program",
    [
        pytest.param(_h2_vqe_for_totals, id="vqe-circuit-seed"),
        pytest.param(_maxcut_qaoa_for_totals, id="qaoa-hamiltonian-seed"),
    ],
)
def test_cost_evaluation_submits_what_the_report_says(
    make_program, default_test_simulator, mocker
):
    """Oracle for the one claim the report makes: what a single evaluation submits.

    Measured against real submissions rather than restated arithmetic, and across
    both seed kinds — a Hamiltonian-seeded program declares no parameters on its
    own seed.
    """
    program = make_program(
        default_test_simulator, MonteCarloOptimizer(population_size=4, n_best_sets=1)
    )
    expected = program.dry_run()["cost"].total_circuits

    spy = mocker.spy(program.backend, "submit_circuits")
    params = np.zeros((program.optimizer.n_param_sets, program.n_params))
    program.evaluate(params, program.cost_preprocessor())

    submitted = sum(len(call.args[0]) for call in spy.call_args_list)
    assert submitted == expected


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
            _dry_run(self._stages(_parametric_twirlable_meta()), dummy_pipeline_env)

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
            DummySpecStage(meta=_parametric_twirlable_meta()),
            PauliTwirlStage(n_twirls=3, seed=0),
            _NonDryDagConsumerStage(),
            _SecondNonDryDagConsumerStage(),
            MeasurementStage(),
        ]
        with pytest.warns(DiviPerformanceWarning) as record:
            _dry_run(stages, dummy_pipeline_env)

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
            _dry_run(self._stages(_parametric_twirlable_meta()), dummy_pipeline_env)
        assert spy.call_count > 0, (
            "Fallback should have run the real PauliTwirl expand, which "
            "invokes _apply_twirl_substitute"
        )

    def test_fallback_preserves_circuit_count(self, dummy_pipeline_env):
        """The analytic+fallback dry run must report the same count as a
        fully-real forward pass."""
        meta = _parametric_twirlable_meta()

        with pytest.warns(DiviPerformanceWarning):
            _, dry_report = _dry_run(self._stages(meta), dummy_pipeline_env, name="dry")
        _, real_report = _dry_run(
            self._stages(meta), dummy_pipeline_env, dry=False, name="real"
        )
        assert dry_report.total_circuits == real_report.total_circuits

    def test_no_warning_when_all_downstream_dry_aware(self, dummy_pipeline_env):
        """Safe pipelines (every downstream stage overrides ``dry_expand``)
        must not emit the fallback warning."""
        stages = [
            DummySpecStage(meta=_parametric_twirlable_meta()),
            PauliTwirlStage(n_twirls=4, seed=0),
            MeasurementStage(),
        ]
        # Any ``DiviPerformanceWarning`` fired here would mean the pipeline
        # spuriously demoted a stage — promote it to an exception so the
        # test fails loudly rather than needing to inspect a record list.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            _dry_run(stages, dummy_pipeline_env)


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
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])
        _, report = _dry_run(
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


def test_quepp_montecarlo_dry_run_suppresses_fallback_warning(dummy_pipeline_env):
    """A dry preview always sees symbolic angles, so montecarlo QuEPP would warn
    it is falling back to exhaustive enumeration. That is execution-path noise
    for a nothing-executes preview and must stay silent."""
    meta = _parametric_twirlable_meta()
    dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _dry_run(
            [
                DummySpecStage(meta=meta),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0, n_samples=4)),
                MeasurementStage(),
            ],
            dummy_pipeline_env,
            suppress_performance_warnings=True,
        )
    assert not any("Monte Carlo sampling" in str(w.message) for w in record)


class _IncompleteMetricEstimator(MetricEstimator):
    """A metric estimator missing ``preprocessors`` — must stay abstract."""

    def check_compatible(self, program):
        pass

    def bind(self, program):
        return {}

    # deliberately omits preprocessors


def test_metric_estimator_must_declare_its_routines():
    """The abstract contract that prevents silent omission: a new metric estimator
    cannot be instantiated without declaring the routines it measures through, so
    they cannot be left out of what a program reports."""
    with pytest.raises(TypeError):
        _IncompleteMetricEstimator()


def test_counts_override_on_observable_counts_per_term(dummy_pipeline_env):
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
    _, report = _dry_run(
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


def test_sampling_dropped_observable_is_not_a_phantom_reduction(dummy_pipeline_env):
    """A sampling routine (``ResultFormat.PROBS``) drops the observable for a
    computational-basis measurement, so the naive per-Pauli-term baseline must
    not apply — otherwise the drop renders as a phantom ``÷N`` grouping that
    never happened. Uses the real ``sample_preprocessor`` transform."""
    _, report = _dry_run(
        [
            DummySpecStage(meta=_parametric_twirlable_meta()),  # 2-term observable
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
    assert report.total_circuits == 1


def test_pce_cost_pipeline_has_no_phantom_reduction(dummy_simulator, default_optimizer):
    """PCE reads raw bitstrings — one circuit per spec, no per-Pauli-term
    grouping — so its cost pipeline must not show a per-term baseline or a
    phantom reduction from the placeholder Hamiltonian it never measures."""
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        optimizer=default_optimizer,
        backend=dummy_simulator,
    )
    cost = pce.dry_run()["cost"]
    assert all(
        stage.factor in (1.0, float(cost.total_circuits)) for stage in cost.stages
    )
    assert not any(stage.factor < 1.0 for stage in cost.stages)  # no phantom ÷N
