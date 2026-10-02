# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qem_stage."""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits.qem import (
    QEMContext,
    QEMProtocol,
    _NoMitigation,
)
from divi.circuits.quepp import QuEPP, _ObservableCPT
from divi.circuits.zne import ZNE, LinearExtrapolator
from divi.pipeline import CircuitPipeline, ContractViolation, DiviPerformanceWarning
from divi.pipeline._compilation import batch_lineage
from divi.pipeline._result_keys_operations import FOREIGN_KEY_ATTR
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from tests._helpers import exact_match
from tests.pipeline._helpers import (
    DummySpecStage,
    ones_execute_fn,
    two_group_meta,
)


def _missing_twirl_stage_message(n_twirls: int) -> str:
    return exact_match(
        f"QEMStage with n_twirls={n_twirls} requires a PauliTwirlStage after it "
        "in the pipeline."
    )


class _DummyQEMProtocol(QEMProtocol):
    """Minimal QEM protocol for tests (used only in this module)."""

    @property
    def name(self) -> str:
        return "dummy-qem"

    def expand(
        self, dag: DAGCircuit, observable: Any | None = None
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        return (dag,), QEMContext()

    def reduce(
        self, quantum_results: Sequence[Any], context: QEMContext
    ) -> list[float]:
        if not quantum_results:
            return []
        sample = quantum_results[0]
        if isinstance(sample, (list, tuple)):
            n_obs = len(sample)
            return [float(sum(row[i] for row in quantum_results)) for i in range(n_obs)]
        return [float(sum(quantum_results))]


@pytest.fixture
def default_zne_protocol():
    # Linear extrapolator is deterministic (doesn't need scipy fits) and
    # matches the old ExpFactory contract closely enough for structural
    # assertions.
    return ZNE(scale_factors=[1, 3, 5], extrapolator=LinearExtrapolator())


@pytest.fixture
def parametric_meta() -> MetaCircuit:
    """A 4-qubit parametric MetaCircuit (proxy for the old sample_circuit)."""
    params = tuple(Parameter(f"w_{i}") for i in range(4))
    qc = QuantumCircuit(4)
    for i, p in enumerate(params):
        qc.ry(p, i)
    for i, p in enumerate(params):
        qc.rx(p, i)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=params,
        measured_wires=(0, 1, 2, 3),
    )


_BASE_KEY = (("spec", "circ"),)


def _symbolic_quepp_context() -> dict:
    """A QuEPP context whose weights are still ``cos(theta)`` / ``sin(theta)``."""
    theta = Parameter("theta")
    return {
        "per_obs": [
            _ObservableCPT(
                weights=np.array([theta.cos(), theta.sin()], dtype=object),
                classical_values=np.array([1.0, 0.0]),
                dag_indices=[0, 1, 2],
                entry_slots=[0, 0],
                target_slots=[0],
            )
        ],
        "symbolic": True,
        "weight_symbols": [theta],
        "target_idx": 0,
        "ensemble_start": 1,
        "n_rotations": 1,
        "n_paths": 2,
    }


def _observable_cpt(weights: list[float], classical: list[float]) -> _ObservableCPT:
    return _ObservableCPT(
        weights=np.array(weights),
        classical_values=np.array(classical),
        dag_indices=list(range(len(weights) + 1)),
        entry_slots=[0] * len(weights),
        target_slots=[0],
    )


_QUEPP_COUNTS = {
    "protocol": "quepp",
    "n_rotations": 2,
    "n_paths": 3,
    "n_clifford_sims": 3,
}


@pytest.mark.parametrize(
    "per_obs, expected",
    [
        pytest.param(
            [
                _observable_cpt([0.12345678, -0.65432109], [1.0, 0.5]),
                _observable_cpt([2.0], [3.0]),
            ],
            {
                **_QUEPP_COUNTS,
                "n_observables": 2,
                "weight_sum": -0.5309,
                "weight_l1_norm": 0.7778,
                "weight_range": [-0.6543, 0.1235],
                "classical_estimate": -0.203704,
            },
            id="mixed-sign-weights-first-observable",
        ),
        pytest.param(
            [_observable_cpt([0.25], [2.0])],
            {
                **_QUEPP_COUNTS,
                "n_observables": 1,
                "weight_sum": 0.25,
                "weight_l1_norm": 0.25,
                "weight_range": [0.25, 0.25],
                "classical_estimate": 0.5,
            },
            id="single-weight",
        ),
        pytest.param(
            [_observable_cpt([], [])],
            {**_QUEPP_COUNTS, "n_observables": 1},
            id="no-weights",
        ),
    ],
)
def test_introspect_reports_weight_statistics(per_obs, expected, dummy_pipeline_env):
    stage = QEMStage(QuEPP(truncation_order=1, n_twirls=0))
    ctx = {
        "n_rotations": 2,
        "n_paths": 3,
        "target_idx": 0,
        "ensemble_start": 1,
        "per_obs": per_obs,
    }
    info = stage.introspect({}, env=dummy_pipeline_env, token={_BASE_KEY: ctx})
    assert info == expected


def test_introspect_leaves_symbolic_weights_unbound(dummy_pipeline_env):
    stage = QEMStage(QuEPP(truncation_order=1, n_twirls=0))
    info = stage.introspect(
        {}, env=dummy_pipeline_env, token={_BASE_KEY: _symbolic_quepp_context()}
    )
    assert info == {
        "protocol": "quepp",
        "n_rotations": 1,
        "n_paths": 2,
        "n_clifford_sims": 2,
        "weights": "unbound (run after parameter binding)",
    }


def test_reused_quepp_context_does_not_carry_eta_rejection_forward(
    dummy_pipeline_env,
):
    """A cached forward trace hands the same concrete contexts to every reduce."""
    stage = QEMStage(QuEPP(truncation_order=1, n_twirls=0))
    ctx = {
        "per_obs": [_observable_cpt([0.5, 0.5], [1.0, 0.5])],
        "target_idx": 0,
        "ensemble_start": 1,
        "n_rotations": 1,
        "n_paths": 2,
    }

    def run(values):
        results = {
            (*_BASE_KEY, (stage.axis_name, i)): value for i, value in enumerate(values)
        }
        return stage.reduce(results, dummy_pipeline_env, token={_BASE_KEY: ctx})

    with pytest.warns(UserWarning, match="signal destroyed"):
        run([0.3, 0.0, 0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reduced = run([0.3, 1.0, 0.5])

    assert reduced == {_BASE_KEY: pytest.approx([0.3])}


def test_reduce_grouped_zne_extrapolates_each_observable(
    default_zne_protocol, dummy_pipeline_env
):
    stage = QEMStage(default_zne_protocol)
    group_key = (*_BASE_KEY, ("obs_group", 0))
    results = {
        (*group_key, ("qem_zne", i)): {0: 1 + s, 1: 10 - s}
        for i, s in enumerate((1, 2, 3))
    }
    token = {group_key: {"effective_scales": (1.0, 2.0, 3.0)}}

    reduced = stage.reduce(results, dummy_pipeline_env, token=token)

    assert reduced == {group_key: {0: pytest.approx([1.0]), 1: pytest.approx([10.0])}}


def test_reduce_single_variant_per_observable(dummy_pipeline_env):
    stage = QEMStage(_NoMitigation())
    results = {(*_BASE_KEY, ("qem_NoMitigation", 0)): {0: 1.5, 1: -0.5}}
    assert stage.reduce(results, dummy_pipeline_env, token=None) == {
        _BASE_KEY: {0: [1.5], 1: [-0.5]}
    }


def test_reduce_empty_results(dummy_pipeline_env):
    assert QEMStage(_NoMitigation()).reduce({}, dummy_pipeline_env, token=None) == {}


@pytest.mark.parametrize("use_zne", [True, False], ids=["zne", "no-mitigation"])
def test_reduce_rejects_probability_dicts(
    use_zne, default_zne_protocol, dummy_pipeline_env
):
    protocol = default_zne_protocol if use_zne else _NoMitigation()
    stage = QEMStage(protocol)
    results = {(*_BASE_KEY, (stage.axis_name, 0)): {"00": 0.5, "11": 0.5}}
    with pytest.raises(
        TypeError,
        match=exact_match(
            "QEMStage expects scalar expectation values, but received probability "
            f"dicts. {type(protocol).__name__} is not supported for "
            "probability-based measurements."
        ),
    ):
        stage.reduce(results, dummy_pipeline_env, token=None)


class TestObservableOverrideAgreement:
    """All bodies of one MetaCircuit share a measurement fan-out, so a protocol
    that refines the observable must declare the same refinement for each.
    A partial or conflicting declaration would emit one set of measurement
    groups that only some bodies' contexts could interpret.
    """

    ZI = SparsePauliOp.from_list([("ZI", 1.0)])
    IZ = SparsePauliOp.from_list([("IZ", 1.0)])

    @staticmethod
    def _stage() -> QEMStage:
        return QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0))

    def test_agreeing_overrides_resolve_to_the_shared_value(self):
        stage = self._stage()
        ctxs = [{"observable_override": (self.ZI,)} for _ in range(2)]
        assert stage._resolve_observable_override(ctxs) == (self.ZI,)

    def test_absent_override_resolves_to_none(self):
        assert self._stage()._resolve_observable_override([{}, {}]) is None

    def test_partial_declaration_is_rejected(self):
        stage = self._stage()
        ctxs = [{"observable_override": (self.ZI,)}, {}]
        with pytest.raises(ContractViolation, match="Either every body overrides"):
            stage._resolve_observable_override(ctxs)

    def test_conflicting_declarations_are_rejected(self):
        stage = self._stage()
        ctxs = [
            {"observable_override": (self.ZI,)},
            {"observable_override": (self.IZ,)},
        ]
        with pytest.raises(ContractViolation, match="conflicting observable overrides"):
            stage._resolve_observable_override(ctxs)


class TestQEMStage:
    """Spec: QEMStage expand applies protocol to body QASMs (fan-out); reduce postprocesses."""

    def test_no_mitigation_declares_no_dag_consumption(self):
        stage = QEMStage(protocol=_NoMitigation())
        assert stage.consumes_dag_bodies is False

    def test_active_protocol_declares_dag_consumption(self, default_zne_protocol):
        assert QEMStage(protocol=default_zne_protocol).consumes_dag_bodies is True

    def test_qem_fanout_and_reduce(self, dummy_pipeline_env):
        class _ScaleFactorProtocol(_DummyQEMProtocol):
            def __init__(self, scale_factors: tuple[float, ...]) -> None:
                self.scale_factors = scale_factors

            def expand(
                self, dag: DAGCircuit, observable: Any | None = None
            ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
                return tuple(dag for _ in self.scale_factors), QEMContext()

        protocol = _ScaleFactorProtocol((1.0, 2.0, 3.0))
        meta = two_group_meta()

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                QEMStage(protocol=protocol),
                MeasurementStage(),
            ],
        )

        plan = pipeline.run_forward_pass(initial_spec="ignored", env=dummy_pipeline_env)
        spec_circ_key = (("spec", "circ"),)
        assert set(plan.final_batch.keys()) == {spec_circ_key}

        reduced = pipeline.run(
            initial_spec="ignored",
            env=dummy_pipeline_env,
            execute_fn=ones_execute_fn,
        )
        assert len(reduced) == 1
        assert list(reduced.values())[0] == pytest.approx([3.9])

    def test_reduce_binds_symbolic_weights_from_param_set_foreign_key(
        self, dummy_pipeline_env
    ):
        stage = QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0))
        base_key = (("spec", "circ"),)
        bound_key = (("spec", "bound"),)
        contexts = {
            base_key: _symbolic_quepp_context(),
            bound_key: {"n_rotations": 0},
            FOREIGN_KEY_ATTR: (("param_set", 1),),
        }
        env = dummy_pipeline_env
        env.param_sets = np.array([[np.pi / 2], [0.0]])
        results = {
            (("spec", "circ"), ("qem_quepp", 0)): 0.5,
            (("spec", "circ"), ("qem_quepp", 1)): 1.0,
            (("spec", "circ"), ("qem_quepp", 2)): 0.0,
        }

        reduced = stage.reduce(results, env, token=contexts)

        assert reduced[base_key] == pytest.approx([0.5])
        assert contexts[base_key]["symbolic"] is False
        bound_weights = contexts[base_key]["per_obs"][0].weights
        assert bound_weights[0] == pytest.approx(1.0)
        assert bound_weights[1] == pytest.approx(0.0)
        assert contexts[bound_key] == {"n_rotations": 0}


class TestPipelineOutputMetaCircuitWithQEM:
    """Spec: Pipeline with ObservableGroupingStage + QEMStage produces MetaCircuits with correct structure."""

    def test_zne_fanout_produces_expected_structure(
        self, parametric_meta, dummy_pipeline_env, default_zne_protocol
    ):
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=parametric_meta),
                QEMStage(protocol=default_zne_protocol),
                MeasurementStage(),
            ],
        )
        trace = pipeline.run_forward_pass(42, dummy_pipeline_env)
        assert len(trace.final_batch) == 1
        meta = next(iter(trace.final_batch.values()))
        # Exactly one DAG body per scale factor — a duplicate append would pass a
        # lower bound.
        assert len(meta.circuit_bodies) == len(default_zne_protocol.scale_factors)

    def test_no_mitigation_single_body_per_key(
        self, parametric_meta, dummy_pipeline_env
    ):
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=parametric_meta),
                MeasurementStage(),
                QEMStage(protocol=_NoMitigation()),
            ],
        )
        trace = pipeline.run_forward_pass(42, dummy_pipeline_env)
        for key, meta in trace.final_batch.items():
            assert len(meta.circuit_bodies) == 1
            assert len(meta.measurement_qasms) == 1


def test_quepp_before_measurement_passes():
    """Spec: QEMStage.validate enforces QuEPP-before-measurement and twirl-after constraints."""
    CircuitPipeline(
        stages=[
            DummySpecStage(meta=two_group_meta()),
            QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0)),
            MeasurementStage(),
        ]
    )


class TestQuEPPLocalEffectiveness:
    @staticmethod
    def _single_rx_meta(angle: float) -> MetaCircuit:
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)
        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            observable=SparsePauliOp("Z"),
        )

    @staticmethod
    def _get_quepp_contexts(trace):
        qem_idx = next(
            i
            for i, exp in enumerate(trace.stage_expansions, start=1)
            if exp.stage_name == "QEMStage"
        )
        return trace.stage_tokens[qem_idx]

    @staticmethod
    def _find_context_for_branch(branch_key, contexts):
        for key, ctx in contexts.items():
            if key == FOREIGN_KEY_ATTR:
                continue
            if tuple(branch_key[: len(key)]) == key:
                return ctx
        raise AssertionError(f"No QEM context for branch key {branch_key!r}")

    @staticmethod
    def _axis_value(branch_key, axis_prefix: str) -> int | None:
        for axis, value in branch_key:
            if axis == axis_prefix:
                return int(value)
        return None

    @pytest.mark.usefixtures(
        "suppress_pipeline_perf_warnings", "suppress_quepp_warnings"
    )
    def test_local_relative_effectiveness_with_and_without_twirling(
        self,
        dummy_pipeline_env,
    ):
        angle = 0.8
        exact = float(np.cos(angle))
        noise_scale = 0.8
        target_bias = 0.01
        tolerance = 0.03
        meta = self._single_rx_meta(angle)

        noisy_target = exact * noise_scale + target_bias

        noisy_pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=meta), MeasurementStage()]
        )

        def noisy_execute_fn(trace, env):
            lineage_by_label = batch_lineage(trace.final_batch)
            return {
                branch_key: noisy_target for branch_key in lineage_by_label.values()
            }

        noisy_result = list(
            noisy_pipeline.run(
                initial_spec="ignored",
                env=dummy_pipeline_env,
                execute_fn=noisy_execute_fn,
            ).values()
        )[0][0]

        def build_quepp_execute_fn(with_twirls: bool):
            def execute_fn(trace, env):
                lineage_by_label = batch_lineage(trace.final_batch)
                contexts = self._get_quepp_contexts(trace)
                out = {}
                for branch_key in lineage_by_label.values():
                    ctx = self._find_context_for_branch(branch_key, contexts)
                    qem_idx = self._axis_value(branch_key, "qem_quepp")
                    twirl_idx = self._axis_value(branch_key, "twirl")
                    twirl_jitter = 0.0
                    if with_twirls and twirl_idx is not None:
                        twirl_jitter = (-0.02, 0.0, 0.02)[twirl_idx % 3]

                    if qem_idx == 0:
                        out[branch_key] = noisy_target + twirl_jitter
                    else:
                        path_idx = int(qem_idx) - 1
                        per_obs_entry = ctx["per_obs"][0]
                        out[branch_key] = (
                            float(per_obs_entry.classical_values[path_idx])
                            * noise_scale
                            + twirl_jitter
                        )
                return out

            return execute_fn

        quepp_pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                QEMStage(
                    protocol=QuEPP(
                        sampling="exhaustive", truncation_order=5, n_twirls=0
                    )
                ),
                MeasurementStage(),
            ],
            suppress_performance_warnings=True,
        )
        quepp_result = list(
            quepp_pipeline.run(
                initial_spec="ignored",
                env=dummy_pipeline_env,
                execute_fn=build_quepp_execute_fn(with_twirls=False),
            ).values()
        )[0][0]

        twirl_pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                QEMStage(
                    protocol=QuEPP(
                        sampling="exhaustive", truncation_order=5, n_twirls=3
                    )
                ),
                PauliTwirlStage(n_twirls=3, seed=11),
                MeasurementStage(),
            ],
            suppress_performance_warnings=True,
        )
        twirl_result = list(
            twirl_pipeline.run(
                initial_spec="ignored",
                env=dummy_pipeline_env,
                execute_fn=build_quepp_execute_fn(with_twirls=True),
            ).values()
        )[0][0]

        assert abs(noisy_result - exact) > tolerance
        assert quepp_result == pytest.approx(exact, abs=tolerance)
        assert twirl_result == pytest.approx(exact, abs=tolerance)

    @pytest.mark.parametrize(
        "trailing_stages",
        [
            pytest.param([], id="nothing-after"),
            pytest.param([PauliTwirlStage(n_twirls=1, seed=0)], id="twirl-after"),
        ],
    )
    def test_quepp_after_measurement_raises(self, trailing_stages):
        n_twirls = 1 if trailing_stages else 0
        with pytest.raises(
            ContractViolation,
            match=exact_match(
                "QEMStage with QuEPP requires a measurement-handling stage after "
                "it so that observable groups are recombined before QEM reduction."
            ),
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    MeasurementStage(),
                    QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=n_twirls)),
                    *trailing_stages,
                ]
            )

    def test_non_quepp_after_measurement_passes(self):
        """Non-QuEPP protocols (like ZNE) work in any position."""
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
                QEMStage(protocol=_DummyQEMProtocol()),
            ]
        )

    def test_no_mitigation_after_measurement_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                MeasurementStage(),
                QEMStage(protocol=_NoMitigation()),
            ]
        )

    def test_twirls_with_twirl_stage_after_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=10)),
                PauliTwirlStage(n_twirls=10),
                MeasurementStage(),
            ]
        )

    @pytest.mark.parametrize(
        "n_twirls, twirl_before_qem",
        [
            pytest.param(1, False, id="missing-1"),
            pytest.param(10, False, id="missing-10"),
            pytest.param(10, True, id="before-qem"),
        ],
    )
    def test_twirls_require_a_twirl_stage_after_qem(self, n_twirls, twirl_before_qem):
        leading = [PauliTwirlStage(n_twirls=n_twirls)] if twirl_before_qem else []
        with pytest.raises(
            ContractViolation,
            match=_missing_twirl_stage_message(n_twirls),
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    *leading,
                    QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=n_twirls)),
                    MeasurementStage(),
                ]
            )


class TestExhaustiveQuEPPWarning:
    """Spec: QuEPP with sampling='exhaustive' emits DiviPerformanceWarning."""

    def test_exhaustive_sampling_warns(self):
        with pytest.warns(
            DiviPerformanceWarning,
            match=exact_match(
                "QuEPP with sampling='exhaustive' enumerates all Pauli paths and "
                "scales poorly with truncation_order and circuit depth. Consider "
                "the default sampling='auto' unless you specifically need "
                "deterministic enumeration. To suppress this warning, pass "
                "suppress_performance_warnings=True to CircuitPipeline, or filter "
                "DiviPerformanceWarning via warnings.filterwarnings (import it "
                "from divi.pipeline)."
            ),
        ) as record:
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    QEMStage(
                        protocol=QuEPP(
                            sampling="exhaustive",
                            truncation_order=1,
                            n_twirls=1,
                        )
                    ),
                    PauliTwirlStage(n_twirls=1, seed=0),
                    ParameterBindingStage(),
                    MeasurementStage(),
                ]
            )
        exhaustive = [w for w in record if "exhaustive" in str(w.message)]
        assert [w.filename for w in exhaustive] == [__file__]

    def test_montecarlo_sampling_does_not_warn(self):
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

    def test_suppress_performance_warnings_kwarg_silences_exhaustive(self):
        """``suppress_performance_warnings=True`` silences the exhaustive-sampling warning."""
        stages = [
            DummySpecStage(meta=two_group_meta()),
            QEMStage(
                protocol=QuEPP(
                    sampling="exhaustive",
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
            CircuitPipeline(stages=stages, suppress_performance_warnings=True)


def test_exhaustive_and_param_bind_before_qem_warns_both():
    """When both footguns are present, both DiviPerformanceWarnings fire in
    a single ``CircuitPipeline.__init__`` — neither short-circuits the other."""
    stages = [
        DummySpecStage(meta=two_group_meta()),
        ParameterBindingStage(),
        QEMStage(
            protocol=QuEPP(
                sampling="exhaustive",
                truncation_order=1,
                n_twirls=1,
            )
        ),
        PauliTwirlStage(n_twirls=1, seed=0),
        MeasurementStage(),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DiviPerformanceWarning)
        CircuitPipeline(stages=stages)

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DiviPerformanceWarning)
    ]
    assert any("exhaustive" in m for m in messages)
    assert any("ParameterBindingStage" in m for m in messages)


def test_pauli_twirl_sample_unique_labels_deduplicates_repeated_vectors(mocker):
    stage = PauliTwirlStage(n_twirls=5, seed=123)
    sampled = [
        [0, 5],
        [10, 15],
        [0, 5],
        [10, 15],
        [3, 12],
    ]
    mocker.patch.object(stage, "_sample_labels", side_effect=sampled)

    unique_labels, twirl_to_unique = stage._sample_unique_labels(n_positions=2)

    assert unique_labels == [
        [0, 5],
        [10, 15],
        [3, 12],
    ]
    assert twirl_to_unique == [0, 1, 0, 1, 2]
