# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipeline dry-run tool."""

import warnings

import numpy as np
import pennylane as qp
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

import divi.circuits.quepp as _quepp_mod
import divi.pipeline.stages._pauli_twirl_stage as _pauli_twirl_mod
from divi.circuits import MetaCircuit
from divi.circuits.qem import _NoMitigation
from divi.circuits.quepp import QuEPP
from divi.pipeline import CircuitPipeline, dry_run_pipeline, format_dry_run
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    DiviPerformanceWarning,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.algorithms import TimeEvolution
from tests.pipeline.helpers import DummySpecStage, two_group_meta


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


class TestDryRunPipeline:
    """Original dry-run report shape tests."""

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
        """Total circuits matches actual _compile_batch output (full-generation mode)."""
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

    def test_format_does_not_raise(self, dummy_pipeline_env):
        """format_dry_run prints without errors."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )
        trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        report = dry_run_pipeline("test", trace, pipeline.stages)
        format_dry_run({"test": report})


class TestAnalyticDryRun:
    """Analytic dry path (dry=True) must produce identical counts to real expand."""

    def test_pauli_twirl_counts_match(self, dummy_pipeline_env):
        """PauliTwirl fan-out is known analytically — dry and real counts must match."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2], [0.3, 0.4]])

        def _build():
            return CircuitPipeline(
                stages=[
                    DummySpecStage(meta=meta),
                    ParameterBindingStage(),
                    PauliTwirlStage(n_twirls=7, seed=0),
                    MeasurementStage(),
                ]
            )

        real_trace = _build().run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True
        )
        dry_trace = _build().run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )

        real_report = dry_run_pipeline(
            "real", real_trace, _build().stages, dummy_pipeline_env
        )
        dry_report = dry_run_pipeline(
            "dry", dry_trace, _build().stages, dummy_pipeline_env
        )
        assert real_report.total_circuits == dry_report.total_circuits
        # Fan-outs per stage must match too.
        for real_stage, dry_stage in zip(real_report.stages, dry_report.stages):
            assert real_stage.fan_out == dry_stage.fan_out

    def test_param_binding_fast_path_counts_correctly(self, dummy_pipeline_env):
        """Fast-path ParameterBindingStage populates bound_circuit_bodies; dry-run
        counter must read from there (not from the untouched circuit_bodies)."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(shot_distribution="weighted"),
            ]
        )
        dry_trace = pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        dry_report = dry_run_pipeline(
            "dry", dry_trace, pipeline.stages, dummy_pipeline_env
        )

        # 1 body × 3 param sets × 2 obs groups.
        assert dry_report.total_circuits == 6

    def test_dry_skips_pauli_twirl_deepcopy(self, dummy_pipeline_env, mocker):
        """Dry PauliTwirl must not invoke the twirl-substitute DAG surgery."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=50, seed=0),
                MeasurementStage(),
            ]
        )
        pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        assert spy.call_count == 0, "dry path must skip twirl DAG substitution"

    def test_real_path_uses_pauli_twirl_deepcopy(self, dummy_pipeline_env, mocker):
        """Sanity check: the real path still invokes the twirl DAG surgery."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_pauli_twirl_mod, "_apply_twirl_substitute")

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=5, seed=0),
                MeasurementStage(),
            ]
        )
        pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=False
        )
        assert spy.call_count > 0, "real path must apply twirl DAG substitution"

    def test_dry_preserves_per_group_shots_artifact(self, dummy_pipeline_env):
        """Dry MeasurementStage must still populate per_group_shots via shot allocation."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(shot_distribution="weighted"),
            ]
        )
        trace = pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        assert "per_group_shots" in trace.env_artifacts

    def test_introspect_metadata_survives_dry(self, dummy_pipeline_env):
        """Each stage's ``introspect()`` feeds ``DryRunReport.stages[i].metadata``.
        If ``introspect`` were silently skipped in dry mode, or stages swapped
        for ones that return degenerate metadata, the fan-out counts would
        still match — so the payload itself needs its own lock-in."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=4, seed=0),
                # Pin shot_distribution so MeasurementStage stays on the
                # qwc branch rather than auto-promoting to _backend_expval
                # on the dummy expval backend — the qwc grouping is what
                # we want to inspect for n_groups / n_terms.
                MeasurementStage(shot_distribution="weighted"),
            ]
        )
        trace = pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        report = dry_run_pipeline("test", trace, pipeline.stages, dummy_pipeline_env)

        by_name = {s.name: s for s in report.stages}

        # ParameterBindingStage surfaces its param-set count and path choice.
        # With PauliTwirlStage (consumes DAG bodies) downstream, ParamBind
        # is forced onto the slow DAG-binding path.
        pb = by_name["ParameterBindingStage"]
        assert pb.metadata["n_param_sets"] == 1
        assert pb.metadata["n_params"] == 2
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
        assert meas.metadata["n_groups"] == len(meta.observable)
        assert meas.metadata["n_terms"] == len(meta.observable)

    def test_env_artifacts_surface_on_dry_run_report(self, dummy_pipeline_env):
        """``DryRunReport.env_artifacts`` is the canonical introspection surface
        for stage-produced state — callers should not need to drop into
        ``_build_pipeline_env`` or private ``_get_initial_spec`` hooks to read
        shot allocations or other forward-pass artifacts."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(shot_distribution="weighted"),
            ]
        )
        trace = pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        report = dry_run_pipeline("test", trace, pipeline.stages, dummy_pipeline_env)
        assert "per_group_shots" in report.env_artifacts
        assert (
            report.env_artifacts["per_group_shots"]
            == trace.env_artifacts["per_group_shots"]
        )

    def test_dry_trace_not_cached(self, dummy_pipeline_env):
        """Dry traces must never be cached nor served from the forward-pass cache."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )
        pipeline.run_forward_pass("ignored", dummy_pipeline_env, dry=True)
        assert (
            pipeline._forward_cache == {}
        ), "dry traces must not populate the real forward-pass cache"

    def test_real_run_after_dry_writes_real_trace(self, dummy_pipeline_env):
        """The critical invariant beyond ``_forward_cache == {}``: a real
        forward pass that follows a dry pass on the same pipeline object
        must (a) actually populate the cache, and (b) store a real trace —
        never inherit the dry pass's placeholder bodies."""
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )

        pipeline.run_forward_pass("ignored", dummy_pipeline_env, dry=True)
        assert pipeline._forward_cache == {}

        real_trace = pipeline.run_forward_pass("ignored", dummy_pipeline_env)
        assert len(pipeline._forward_cache) == 1
        cached = next(iter(pipeline._forward_cache.values()))
        assert cached is real_trace

        # Dry mode's measurement placeholders are empty strings; the real
        # pass must emit actual ``measure q[...] -> c[...]`` QASM.
        real_meta = next(iter(real_trace.final_batch.values()))
        real_meas = real_meta.measurement_qasms
        assert real_meas, "real measurement_qasms must be populated"
        assert all("measure" in qasm for _tag, qasm in real_meas), (
            "real run produced placeholder measurement QASMs — "
            "cache leaked dry state"
        )


@pytest.mark.usefixtures("suppress_quepp_warnings")
class TestQuEPPDryExpand:
    """QuEPP dry path must skip Clifford simulation while preserving fan-out."""

    def test_quepp_dry_matches_real_counts(self, dummy_pipeline_env):
        """QuEPP's analytic path must produce the same ``1 + n_paths`` fan-out."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        def _build():
            return CircuitPipeline(
                stages=[
                    DummySpecStage(meta=meta),
                    QEMStage(
                        protocol=QuEPP(
                            truncation_order=1, n_twirls=0, sampling="exhaustive"
                        )
                    ),
                    PauliTwirlStage(n_twirls=3, seed=0),
                    MeasurementStage(),
                ],
                suppress_performance_warnings=True,
            )

        real_pipeline = _build()
        dry_pipeline = _build()
        real_trace = real_pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True
        )
        dry_trace = dry_pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        real_report = dry_run_pipeline(
            "r", real_trace, real_pipeline.stages, dummy_pipeline_env
        )
        dry_report = dry_run_pipeline(
            "d", dry_trace, dry_pipeline.stages, dummy_pipeline_env
        )
        assert real_report.total_circuits == dry_report.total_circuits

        # QEMStage's introspect metadata must also survive the dry path —
        # n_rotations / n_paths are populated in QuEPP.dry_expand's context
        # and must match the real path's counts.
        real_qem = next(s for s in real_report.stages if s.name == "QEMStage")
        dry_qem = next(s for s in dry_report.stages if s.name == "QEMStage")
        assert dry_qem.metadata["protocol"] == "quepp"
        assert dry_qem.metadata["n_rotations"] == real_qem.metadata["n_rotations"]
        assert dry_qem.metadata["n_paths"] == real_qem.metadata["n_paths"]

    def test_quepp_dry_skips_clifford_simulation(self, dummy_pipeline_env, mocker):
        """Dry QuEPP must not invoke the Clifford ensemble simulator."""
        meta = _parametric_twirlable_meta()
        dummy_pipeline_env.param_sets = np.asarray([[0.1, 0.2]])

        spy = mocker.spy(_quepp_mod, "_simulate_clifford_ensemble")

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                QEMStage(
                    protocol=QuEPP(
                        truncation_order=1, n_twirls=0, sampling="exhaustive"
                    )
                ),
                PauliTwirlStage(n_twirls=1, seed=0),
                MeasurementStage(),
            ],
            suppress_performance_warnings=True,
        )
        pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
        )
        assert spy.call_count == 0, "dry QuEPP must skip Clifford simulation"


@pytest.mark.usefixtures("suppress_quepp_warnings")
@pytest.mark.filterwarnings("ignore::UserWarning:divi.qprog.algorithms._vqe")
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

    def test_default_and_forced_match(self, time_evolution_program):
        """``dry_run()`` (analytic) and ``dry_run(force_circuit_generation=True)``
        must produce the same total circuit count."""
        analytic = time_evolution_program.dry_run()
        forced = time_evolution_program.dry_run(force_circuit_generation=True)
        assert set(analytic) == set(forced)
        for name in analytic:
            assert analytic[name].total_circuits == forced[name].total_circuits
            # Per-stage fan-outs must also agree.
            for a_stage, f_stage in zip(analytic[name].stages, forced[name].stages):
                assert a_stage.fan_out == f_stage.fan_out, (
                    f"{name}/{a_stage.name}: analytic={a_stage.fan_out} "
                    f"forced={f_stage.fan_out}"
                )

    def test_env_artifacts_exposed_via_dry_run(self, default_test_simulator):
        """Callers can read ``per_group_shots`` (and any other stage artifact)
        straight off the :class:`DryRunReport` — no private hooks needed."""
        vqe = VQE(
            molecule=qp.qchem.Molecule(
                symbols=["H", "H"],
                coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]),
            ),
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
            backend=default_test_simulator,
            grouping_strategy="qwc",
            shot_distribution="weighted",
        )
        cost_report = vqe.dry_run()["cost"]
        assert "per_group_shots" in cost_report.env_artifacts
        # The allocation is a dict {spec_key: {group_idx: shots}}; at least
        # one spec, at least one group with a non-zero shot budget.
        spec_alloc = next(iter(cost_report.env_artifacts["per_group_shots"].values()))
        assert any(shots > 0 for shots in spec_alloc.values())

    def test_default_and_forced_match_non_qem(self, default_test_simulator):
        """Parity between analytic and ``force_circuit_generation=True`` must
        also hold for programs without any QEM stage — the ``TimeEvolution``
        fixture above brings QuEPP, so the non-QEM path needs its own lock."""
        vqe = VQE(
            molecule=qp.qchem.Molecule(
                symbols=["H", "H"],
                coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]),
            ),
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
            backend=default_test_simulator,
            grouping_strategy="qwc",
        )
        analytic = vqe.dry_run()
        forced = vqe.dry_run(force_circuit_generation=True)
        assert set(analytic) == set(forced)
        for name in analytic:
            assert analytic[name].total_circuits == forced[name].total_circuits
            for a_stage, f_stage in zip(analytic[name].stages, forced[name].stages):
                assert a_stage.fan_out == f_stage.fan_out, (
                    f"{name}/{a_stage.name}: analytic={a_stage.fan_out} "
                    f"forced={f_stage.fan_out}"
                )


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
    ) -> tuple[ExpansionResult, StageToken]:
        return ExpansionResult(batch=dict(batch)), None

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

    def _build(self, meta):
        return CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                PauliTwirlStage(n_twirls=4, seed=0),
                _NonDryDagConsumerStage(),
                MeasurementStage(),
            ]
        )

    def test_fallback_warning_names_upstream_and_culprit(self, dummy_pipeline_env):
        """The emitted warning names both the upstream dry-aware stage and
        the downstream non-dry-aware DAG consumer(s)."""
        pipeline = self._build(_parametric_twirlable_meta())
        with pytest.warns(DiviPerformanceWarning) as record:
            pipeline.run_forward_pass(
                "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
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
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_parametric_twirlable_meta()),
                PauliTwirlStage(n_twirls=3, seed=0),
                _NonDryDagConsumerStage(),
                _SecondNonDryDagConsumerStage(),
                MeasurementStage(),
            ]
        )
        with pytest.warns(DiviPerformanceWarning) as record:
            pipeline.run_forward_pass(
                "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
            )

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

        pipeline = self._build(_parametric_twirlable_meta())
        with pytest.warns(DiviPerformanceWarning):
            pipeline.run_forward_pass(
                "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
            )
        assert spy.call_count > 0, (
            "Fallback should have run the real PauliTwirl expand, which "
            "invokes _apply_twirl_substitute"
        )

    def test_fallback_preserves_circuit_count(self, dummy_pipeline_env):
        """The analytic+fallback dry run must report the same count as a
        fully-real forward pass."""
        meta = _parametric_twirlable_meta()

        dry_pipeline = self._build(meta)
        real_pipeline = self._build(meta)

        with pytest.warns(DiviPerformanceWarning):
            dry_trace = dry_pipeline.run_forward_pass(
                "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
            )
        real_trace = real_pipeline.run_forward_pass(
            "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=False
        )

        dry_report = dry_run_pipeline(
            "dry", dry_trace, dry_pipeline.stages, dummy_pipeline_env
        )
        real_report = dry_run_pipeline(
            "real", real_trace, real_pipeline.stages, dummy_pipeline_env
        )
        assert dry_report.total_circuits == real_report.total_circuits

    def test_no_warning_when_all_downstream_dry_aware(self, dummy_pipeline_env):
        """Safe pipelines (every downstream stage overrides ``dry_expand``)
        must not emit the fallback warning."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=_parametric_twirlable_meta()),
                PauliTwirlStage(n_twirls=4, seed=0),
                MeasurementStage(),
            ]
        )
        # Any ``DiviPerformanceWarning`` fired here would mean the pipeline
        # spuriously demoted a stage — promote it to an exception so the
        # test fails loudly rather than needing to inspect a record list.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            pipeline.run_forward_pass(
                "ignored", dummy_pipeline_env, force_forward_sweep=True, dry=True
            )
