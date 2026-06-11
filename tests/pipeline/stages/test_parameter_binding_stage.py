# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._parameter_binding_stage."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _format_bound_param as _format_param
from divi.circuits.quepp import QuEPP
from divi.circuits.zne import ZNE
from divi.pipeline import CircuitPipeline, DiviPerformanceWarning, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.pipeline.stages._parameter_binding_stage import _validate_param_sets
from tests.pipeline._helpers import (
    DummySpecStage,
    run_binding_pipeline,
    two_group_meta,
)


def _parametric_meta(symbol_names: tuple[str, ...] = ("theta", "phi")) -> MetaCircuit:
    """Build a MetaCircuit whose DAG bodies reference Qiskit Parameters."""
    params = tuple(Parameter(name) for name in symbol_names)
    qc = QuantumCircuit(1)
    qc.rx(params[0], 0)
    qc.rz(params[1], 0)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=params,
        observable=SparsePauliOp.from_list([("Z", 1.0)]),
    )


class TestParameterBindingStage:
    """Spec: ParameterBindingStage expand binds env.param_sets into circuit body QASMs; reduce is identity."""

    def test_requires_2d_param_sets(self, dummy_pipeline_env):
        with pytest.raises(ValueError, match="param_sets to be 2D"):
            run_binding_pipeline(
                two_group_meta(),
                backend=dummy_pipeline_env.backend,
                param_sets=[1.0, 2.0],
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_param_sets(self, bad):
        """Non-finite weights are rejected at the binding boundary, before any
        render path runs."""
        env = PipelineEnv(backend=None, param_sets=[[1.0, bad]])
        with pytest.raises(ValueError, match="non-finite gate parameters"):
            _validate_param_sets(env)

    def test_passthrough_when_no_symbols(self, dummy_pipeline_env):
        trace = run_binding_pipeline(
            two_group_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[0.0]]),
        )
        for node in trace.final_batch.values():
            # Non-parametric pass-through still serialises each DAG body once.
            assert node.qasm_bodies

    def test_binds_parameters_into_qasm(self, dummy_pipeline_env):
        """Core spec: parameter names in the template are replaced by formatted values."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )

        # All bound bodies should contain the formatted values, not the param names.
        for node in trace.final_batch.values():
            for _tag, body in node.qasm_bodies:
                assert "theta" not in body
                assert "phi" not in body
                assert "1.5" in body
                assert "2.7" in body

    def test_multiple_param_sets_produce_multiple_bodies(self, dummy_pipeline_env):
        """Each param set produces a separate body variant tagged with param_set axis."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

        # The param_set axis appears on the bound-body tags.
        for node in trace.final_batch.values():
            param_set_indices = set()
            for tag, _body in node.qasm_bodies:
                for axis_name, axis_value in tag:
                    if axis_name == "param_set":
                        param_set_indices.add(axis_value)
            assert param_set_indices == {0, 1, 2}

    def test_fast_path_consumes_pre_populated_qasm_bodies(self, dummy_pipeline_env):
        """If ``qasm_bodies`` is set, the fast path uses it
        instead of deriving a body from the DAG.

        We construct a MetaCircuit whose ``qasm_bodies`` entry
        contains a marker (``"// SENTINEL\\n"`` plus an ``rx(theta) q[0];``
        gate) that the DAG itself does NOT emit. If PB consults the
        pre-populated body, the marker survives into ``qasm_bodies``.
        """
        # Real parametric DAG: 2 params, RX + RZ.
        meta = _parametric_meta()
        sentinel_body = "// SENTINEL\nrx(theta) q[0];\nrz(phi) q[0];\n"
        meta_with_pre = meta.set_qasm_bodies(((((), sentinel_body),)))

        trace = run_binding_pipeline(
            meta_with_pre,
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )

        for node in trace.final_batch.values():
            for _tag, body in node.qasm_bodies:
                assert "// SENTINEL" in body, (
                    "PB fast path did not consume the pre-populated "
                    "qasm_bodies; sentinel marker missing."
                )
                # Weight substitution should still have happened.
                assert "theta" not in body
                assert "phi" not in body

    def test_zero_param_fast_path_consumes_parked_data_bodies(self, dummy_pipeline_env):
        """A weight-less circuit (every parameter bound by DataBindingStage)
        must still emit the per-sample data bodies parked in
        ``qasm_bodies`` — not re-serialise the shared DAG, which
        would silently drop the feature batch (identical bodies)."""
        qc = QuantumCircuit(1)
        qc.h(0)
        meta = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=(),
            observable=SparsePauliOp.from_list([("Z", 1.0)]),
        )
        sentinel_body = "// DATA-BAKED\nrx(0.5) q[0];\n"
        meta_with_pre = meta.set_qasm_bodies(((((), sentinel_body),)))

        trace = run_binding_pipeline(
            meta_with_pre,
            backend=dummy_pipeline_env.backend,
            param_sets=np.zeros((1, 0)),
        )

        for node in trace.final_batch.values():
            assert node.qasm_bodies
            for _tag, body in node.qasm_bodies:
                assert "// DATA-BAKED" in body, (
                    "zero-weight fast path ignored parked data bodies; "
                    "the feature batch would be silently dropped."
                )

    def test_fast_path_falls_back_to_dag_when_tag_missing(self, dummy_pipeline_env):
        """A ``qasm_bodies`` entry for a different tag does not
        affect bodies whose tags are not in the lookup — those fall back
        to ``_qasm_body_cached(dag, ...)``."""
        meta = _parametric_meta()
        # Pre-populated entry uses a tag that doesn't match the body's tag.
        meta_with_pre = meta.set_qasm_bodies(
            (((("unrelated_axis", 0),), "// WRONG\n"),)
        )
        trace = run_binding_pipeline(
            meta_with_pre,
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )

        for node in trace.final_batch.values():
            for _tag, body in node.qasm_bodies:
                assert "// WRONG" not in body
                # Bodies derived from the DAG retain rx / rz instructions.
                assert "rx(" in body
                assert "rz(" in body

    def test_param_count_mismatch_raises(self, dummy_pipeline_env):
        """Providing wrong number of parameters for a circuit raises ValueError."""
        with pytest.raises(ValueError, match="expected 2 parameters"):
            run_binding_pipeline(
                _parametric_meta(),  # expects 2 symbols
                backend=dummy_pipeline_env.backend,
                param_sets=np.array([[1.0]]),  # only 1 value for 2 symbols
            )

    def test_reduce_is_identity(self):
        """Reduce returns its input unchanged."""
        stage = ParameterBindingStage()
        sentinel = {(("spec", "circ"),): 42.0}
        assert stage.reduce(sentinel, None, None) is sentinel

    def test_axis_name_is_param_set(self):
        assert ParameterBindingStage().axis_name == "param_set"

    def test_stateful_is_true(self):
        assert ParameterBindingStage().stateful is True

    def test_does_not_force_upstream_dag_materialization(self):
        assert ParameterBindingStage().consumes_dag_bodies is False


class TestFormatParam:
    """Spec: _format_param formats floats for QASM, strips trailing zeros, normalises negative zero."""

    def test_basic_formatting(self):
        assert _format_param(1.5, 10) == "1.5"

    def test_integer_value_strips_trailing_zeros(self):
        assert _format_param(3.0, 10) == "3"

    def test_negative_zero_normalised(self):
        assert _format_param(-0.0, 10) == "0"

    def test_precision_respected(self):
        result = _format_param(1.123456789, 4)
        assert result == "1.1235"

    def test_small_value(self):
        result = _format_param(0.001, 10)
        assert result == "0.001"

    def test_negative_value(self):
        result = _format_param(-2.5, 10)
        assert result == "-2.5"

    def test_zero(self):
        assert _format_param(0.0, 10) == "0"

    def test_very_small_rounds_to_zero(self):
        """A value that rounds to 0.000...0 at the given precision becomes '0'."""
        assert _format_param(1e-20, 10) == "0"


class _TemplateCapableBackend:
    """Minimal expval-supporting backend that implements the
    :class:`~divi.backends.SupportsCircuitTemplates` capability protocol."""

    is_async = False
    supports_expval = True
    shots = 100

    def submit_circuits(self, circuits, **kwargs):  # pragma: no cover - unused
        raise AssertionError(
            "submit_circuits should not be called on the template path"
        )

    def submit_circuit_templates(
        self, templates, **kwargs
    ):  # pragma: no cover - unused
        raise AssertionError(
            "submit_circuit_templates is not exercised by these stage tests"
        )


class TestParameterBindingStageTemplatePath:
    """Spec: when env.backend.supports_circuit_templates is True, the fast
    path defers parameter binding by parking parametric QASM in
    ``qasm_bodies`` instead of pre-rendering per param set."""

    def test_template_carrier_populated_when_backend_supports_templates(self):
        """Backend opts in → qasm_bodies carries parametric QASM."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=_TemplateCapableBackend(),
            param_sets=np.array([[1.5, 2.7], [3.0, 4.0]]),
        )

        for node in trace.final_batch.values():
            assert (
                node.qasm_bodies
            ), "qasm_bodies should be populated when backend supports templates."
            assert (
                node.parameters
            ), "Parameters must remain unbound so compile takes the template path."
            # Symbols survive: substitution is deferred to the backend.
            for _tag, body in node.qasm_bodies:
                assert "theta" in body
                assert "phi" in body

    def test_template_path_skipped_when_backend_does_not_support(
        self, dummy_pipeline_env
    ):
        """Non-template backend → fast path renders bound QASM as before."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )
        for node in trace.final_batch.values():
            assert node.qasm_bodies
            assert node.parameters == ()

    def test_template_path_skipped_when_slow_path_required(
        self, suppress_pipeline_perf_warnings
    ):
        """Slow path (QEM enabled) must bind locally even on template-capable backend."""
        meta = _parametric_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                QEMStage(ZNE(scale_factors=[1.0, 3.0])),  # active QEM → slow path
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(
            backend=_TemplateCapableBackend(),
            param_sets=np.array([[1.5, 2.7]]),
        )
        trace = pipeline.run_forward_pass("x", env)
        for node in trace.final_batch.values():
            assert node.parameters == ()  # bound locally, not deferred
            # Slow path binds into DAGs, so the QASM-string slot stays empty;
            # a template firing here would populate it.
            assert node.qasm_bodies == ()

    def test_non_parametric_falls_back_to_bound_emission(self):
        """No parameters → no template needed; emit bound bodies as the fast path does."""
        trace = run_binding_pipeline(
            two_group_meta(),
            backend=_TemplateCapableBackend(),
            param_sets=np.array([[0.0]]),
        )
        for node in trace.final_batch.values():
            assert node.qasm_bodies
            assert node.parameters == ()

    def test_template_path_disabled_when_per_group_shots_active(self):
        """Per-group shot allocation attaches shots to concrete flat circuits,
        which the deferred template payload can't express — so the template
        path is disabled even on a template-capable backend."""
        stage = ParameterBindingStage()
        stage._fast_path = True
        env = PipelineEnv(backend=_TemplateCapableBackend())
        assert stage._template_path_enabled(env) is True

        env.artifacts["per_group_shots"] = {(("spec", "c"),): {0: 100}}
        assert stage._template_path_enabled(env) is False


class TestParameterBindingStageOrdering:
    """ParameterBindingStage can appear in any order relative to QEMStage."""

    def test_param_binding_before_qem(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                ParameterBindingStage(),
                QEMStage(),
                MeasurementStage(),
            ]
        )

    def test_param_binding_after_qem(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                QEMStage(),
                ParameterBindingStage(),
                MeasurementStage(),
            ]
        )


class TestParamBindBeforeQEMWarning:
    """Spec: ParameterBindingStage placed before QEMStage emits DiviPerformanceWarning."""

    def test_param_bind_before_qem_warns(self):
        with pytest.warns(DiviPerformanceWarning, match="ParameterBindingStage"):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    ParameterBindingStage(),
                    QEMStage(
                        protocol=QuEPP(
                            sampling="montecarlo",
                            truncation_order=1,
                            n_twirls=1,
                        )
                    ),
                    PauliTwirlStage(n_twirls=1, seed=0),
                    MeasurementStage(),
                ]
            )

    def test_param_bind_after_qem_does_not_warn(self):
        stages = [
            DummySpecStage(meta=two_group_meta()),
            QEMStage(
                protocol=QuEPP(
                    sampling="montecarlo",
                    truncation_order=1,
                    n_twirls=1,
                )
            ),
            PauliTwirlStage(n_twirls=1, seed=0),
            ParameterBindingStage(),
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)

    def test_no_mitigation_qem_does_not_warn(self):
        stages = [
            DummySpecStage(meta=two_group_meta()),
            ParameterBindingStage(),
            QEMStage(),
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages)

    def test_suppress_performance_warnings_kwarg_silences_ordering(self):
        """``suppress_performance_warnings=True`` silences the ordering warning."""
        stages = [
            DummySpecStage(meta=two_group_meta()),
            ParameterBindingStage(),
            QEMStage(
                protocol=QuEPP(
                    sampling="montecarlo",
                    truncation_order=1,
                    n_twirls=1,
                )
            ),
            PauliTwirlStage(n_twirls=1, seed=0),
            MeasurementStage(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DiviPerformanceWarning)
            CircuitPipeline(stages=stages, suppress_performance_warnings=True)
