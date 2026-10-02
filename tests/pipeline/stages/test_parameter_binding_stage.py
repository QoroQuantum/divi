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
from divi.circuits.quepp import QuEPP
from divi.circuits.zne import ZNE
from divi.pipeline import (
    CircuitPipeline,
    DiviPerformanceWarning,
    PipelineEnv,
)
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from divi.pipeline.stages._parameter_binding_stage import _validate_param_sets
from tests._helpers import exact_match
from tests.pipeline._helpers import (
    DummySpecStage,
    FakeBackend,
    run_binding_pipeline,
    stage_body_tags,
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
        observable=SparsePauliOp("Z"),
    )


class TestParameterBindingStage:
    """Spec: ParameterBindingStage expand binds env.param_sets into circuit body QASMs; reduce is identity."""

    def test_requires_2d_param_sets(self, dummy_pipeline_env):
        with pytest.raises(
            ValueError,
            match=exact_match("ParameterBindingStage expects env.param_sets to be 2D."),
        ):
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
        with pytest.raises(
            ValueError,
            match=exact_match(
                "Cannot bind non-finite gate parameters: env.param_sets contains "
                "NaN or Inf. Check the feature batch / parameter values for "
                "missing data, divide-by-zero, or overflow in preprocessing."
            ),
        ):
            _validate_param_sets(env)

    def test_passthrough_when_no_symbols(self, dummy_pipeline_env):
        trace = run_binding_pipeline(
            two_group_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[0.0]]),
        )
        for node in trace.final_batch.values():
            assert len(node.qasm_bodies) == 1
            assert node.parameters == ()

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
            observable=SparsePauliOp("Z"),
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
        with pytest.raises(
            ValueError,
            match=exact_match(
                "ParameterBindingStage expected 2 parameters, got 1 in param set 0."
            ),
        ):
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

    def test_is_volatile(self):
        assert ParameterBindingStage().volatile is True

    def test_does_not_force_upstream_dag_materialization(self):
        assert ParameterBindingStage().consumes_dag_bodies is False


def _qasm_payload_backend() -> FakeBackend:
    """Expval backend that resolves parameters from QASM-encoded payloads."""
    return FakeBackend(resolves_parameters=True)


class TestParameterBindingStageDeferredBinding:
    """Spec: when the backend resolves parameters, the fast path defers
    binding by parking parametric QASM in ``qasm_bodies`` instead of
    pre-rendering per param set."""

    def test_parametric_qasm_parked_when_backend_resolves_parameters(self):
        """Backend opts in → qasm_bodies carries parametric QASM."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=_qasm_payload_backend(),
            param_sets=np.array([[1.5, 2.7], [3.0, 4.0]]),
        )

        for node in trace.final_batch.values():
            assert (
                node.qasm_bodies
            ), "qasm_bodies should be populated when the backend resolves parameters."
            assert (
                node.parameters
            ), "Parameters must remain unbound so compile emits a parametric CircuitPayload."
            # Symbols survive: substitution is deferred to the backend.
            for _tag, body in node.qasm_bodies:
                assert "theta" in body
                assert "phi" in body

    def test_binding_happens_locally_when_backend_does_not_resolve(
        self, dummy_pipeline_env
    ):
        """Backend resolves nothing → fast path renders bound QASM as before."""
        trace = run_binding_pipeline(
            _parametric_meta(),
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )
        for node in trace.final_batch.values():
            assert node.qasm_bodies
            assert node.parameters == ()

    def test_binding_happens_locally_when_slow_path_required(
        self, suppress_pipeline_perf_warnings
    ):
        """Slow path (QEM enabled) must bind locally even on a resolving backend."""
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
            backend=_qasm_payload_backend(),
            param_sets=np.array([[1.5, 2.7]]),
        )
        trace = pipeline.run_forward_pass("x", env)
        for node in trace.final_batch.values():
            assert node.parameters == ()  # bound locally, not deferred
            # Slow path binds into DAGs, so the QASM-string slot stays empty;
            # a deferred binding firing here would populate it.
            assert node.qasm_bodies == ()

    def test_non_parametric_falls_back_to_bound_emission(self):
        """No parameters → nothing to defer; emit bound bodies as the fast path does."""
        trace = run_binding_pipeline(
            two_group_meta(),
            backend=_qasm_payload_backend(),
            param_sets=np.array([[0.0]]),
        )
        for node in trace.final_batch.values():
            assert node.qasm_bodies
            assert node.parameters == ()

    def test_binding_is_not_deferred_when_per_group_shots_active(self):
        """Per-group shot allocation attaches shots to concrete flat circuits,
        which a CircuitPayload can't express — so binding happens locally even on a
        backend that would otherwise resolve the parameters itself."""
        stage = ParameterBindingStage()
        stage._fast_path = True
        env = PipelineEnv(backend=_qasm_payload_backend())
        meta = two_group_meta()
        batch = {(("spec", "c"),): meta}
        assert stage._defers_binding(batch, env) is True

        batch_with_shots = {(("spec", "c"),): meta.set_group_shots({0: 100})}
        assert stage._defers_binding(batch_with_shots, env) is False

        batch_with_param_shots = {
            (("spec", "c"),): meta.set_param_group_shots({0: {0: 100}})
        }
        assert stage._defers_binding(batch_with_param_shots, env) is False

    def test_introspect_agrees_with_the_path_actually_taken(self):
        """A dry run must not claim binding is deferred when it is not: the
        reported flag and ``_defers_binding`` read the same predicate."""
        stage = ParameterBindingStage()
        stage._fast_path = True
        env = PipelineEnv(backend=_qasm_payload_backend(), param_sets=np.array([[0.0]]))
        meta = two_group_meta()

        for batch in (
            {(("spec", "c"),): meta},
            {(("spec", "c"),): meta.set_group_shots({0: 100})},
        ):
            report = stage.introspect(batch, env, token=None)
            assert report["deferred_binding"] is stage._defers_binding(batch, env)


class TestParamBindBeforeQEMWarning:
    """Spec: ParameterBindingStage placed before QEMStage emits DiviPerformanceWarning."""

    def test_param_bind_before_qem_warns(self):
        with pytest.warns(
            DiviPerformanceWarning,
            match=exact_match(
                "ParameterBindingStage is placed before QEMStage. This forces QEM "
                "to re-expand on every bound parameter variant (one full QEM pass "
                "per param set). Consider placing ParameterBindingStage after "
                "QEMStage."
            ),
        ):
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


def _constant_angle_meta(*, parametric: bool, **kwargs) -> MetaCircuit:
    """One-qubit circuit with a fixed ``rz`` angle, optionally preceded by ``ry(theta)``."""
    qc = QuantumCircuit(1)
    qc.h(0)
    params = (Parameter("theta"),) if parametric else ()
    if parametric:
        qc.ry(params[0], 0)
    qc.rz(0.123456789, 0)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=params,
        observable=SparsePauliOp("Z"),
        **kwargs,
    )


def _param_sets_for(parametric: bool) -> np.ndarray:
    return np.full((2, 1 if parametric else 0), 0.3)


_FAST_STAGES = (ParameterBindingStage, MeasurementStage)
_SLOW_STAGES = (
    ParameterBindingStage,
    lambda: PauliTwirlStage(n_twirls=1, seed=0),
    MeasurementStage,
)


def _binding_pipeline(spec_stage, stage_factories) -> CircuitPipeline:
    return CircuitPipeline(stages=[spec_stage, *(make() for make in stage_factories)])


@pytest.mark.parametrize("parametric", [False, True], ids=["parameter-free", "bound"])
def test_fast_path_renders_circuit_constants_at_the_circuit_precision(
    dummy_pipeline_env, parametric
):
    trace = run_binding_pipeline(
        _constant_angle_meta(parametric=parametric, precision=4),
        backend=dummy_pipeline_env.backend,
        param_sets=_param_sets_for(parametric),
    )
    bodies = [
        body for node in trace.final_batch.values() for _, body in node.qasm_bodies
    ]
    assert bodies
    assert all("rz(0.1235) q[0];" in body for body in bodies)


@pytest.mark.parametrize(
    "stage_factories", [_FAST_STAGES, _SLOW_STAGES], ids=["fast", "slow"]
)
def test_a_parameter_free_entry_does_not_drop_later_entries(
    dummy_pipeline_env, stage_factories
):
    pipeline = _binding_pipeline(CircuitSpecStage(), stage_factories)
    env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=[[0.3]])
    trace = pipeline.run_forward_pass(
        {
            "free": _constant_angle_meta(parametric=False),
            "bound": _constant_angle_meta(parametric=True),
        },
        env,
    )
    assert set(trace.final_batch) == {(("circuit", "free"),), (("circuit", "bound"),)}


@pytest.mark.parametrize(
    "stage_factories", [_FAST_STAGES, _SLOW_STAGES], ids=["fast", "slow"]
)
@pytest.mark.parametrize("parametric", [False, True], ids=["parameter-free", "bound"])
def test_dry_expand_labels_match_real_expand(
    dummy_pipeline_env, stage_factories, parametric
):
    pipeline = _binding_pipeline(
        DummySpecStage(meta=_constant_angle_meta(parametric=parametric)),
        stage_factories,
    )
    env = PipelineEnv(
        backend=dummy_pipeline_env.backend, param_sets=_param_sets_for(parametric)
    )
    real = pipeline.run_forward_pass("x", env)
    dry = pipeline.run_forward_pass("x", env, dry=True)
    assert stage_body_tags(dry, "ParameterBindingStage") == stage_body_tags(
        real, "ParameterBindingStage"
    )


def test_dry_expand_accepts_non_finite_param_sets(dummy_pipeline_env):
    pipeline = _binding_pipeline(
        DummySpecStage(meta=_constant_angle_meta(parametric=True)), _FAST_STAGES
    )
    env = PipelineEnv(
        backend=dummy_pipeline_env.backend, param_sets=np.array([[np.nan], [np.inf]])
    )
    dry = pipeline.run_forward_pass("x", env, dry=True)
    assert len(stage_body_tags(dry, "ParameterBindingStage")) == 2
