# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qem_stage."""

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
    ZNE,
    LinearExtrapolator,
    QEMContext,
    QEMProtocol,
    _NoMitigation,
)
from divi.circuits.quepp import QuEPP
from divi.pipeline import CircuitPipeline, ContractViolation
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.stages import MeasurementStage, PauliTwirlStage, QEMStage
from divi.pipeline.transformations import FOREIGN_KEY_ATTR
from tests.pipeline.helpers import DummySpecStage, ones_execute_fn, two_group_meta


class _DummyQEMProtocol(QEMProtocol):
    """Minimal QEM protocol for tests (used only in this module)."""

    @property
    def name(self) -> str:
        return "dummy-qem"

    def expand(
        self, dag: DAGCircuit, observable: Any | None = None
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        return (dag,), QEMContext()

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        return float(sum(quantum_results))


@pytest.fixture
def default_zne_protocol():
    # Odd-integer scales only (GlobalFoldPass constraint).  Linear extrapolator
    # is deterministic (doesn't need scipy fits) and matches the old ExpFactory
    # contract closely enough for structural assertions.
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


class TestQEMStage:
    """Spec: QEMStage expand applies protocol to body QASMs (fan-out); reduce postprocesses."""

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
        assert list(reduced.values())[0] == pytest.approx(3.9)

    def test_reduce_handles_multi_obs_expval_dicts(self, dummy_pipeline_env):
        """QEM reduce applies postprocessing per observable when values are {int: float} dicts."""

        class _ScaleFactorProtocol(_DummyQEMProtocol):
            """Produces 3 body variants (like 3 scale factors) and sums during reduce."""

            def __init__(self) -> None:
                self.scale_factors = (1.0, 2.0, 3.0)

            def expand(
                self, dag: DAGCircuit, observable: Any | None = None
            ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
                return tuple(dag for _ in self.scale_factors), QEMContext()

        protocol = _ScaleFactorProtocol()
        stage = QEMStage(protocol=protocol)

        # Simulate results with {int: float} dicts, as _counts_to_expvals produces
        # for multi-observable measurement groups.
        # Three QEM scale variants (indices 0, 1, 2), each with two obs values.
        results = {
            (("spec", "circ"), ("obs_group", 0), ("qem_dummy-qem", 0)): {
                0: 1.0,
                1: 10.0,
            },
            (("spec", "circ"), ("obs_group", 0), ("qem_dummy-qem", 1)): {
                0: 2.0,
                1: 11.0,
            },
            (("spec", "circ"), ("obs_group", 0), ("qem_dummy-qem", 2)): {
                0: 3.0,
                1: 12.0,
            },
        }

        reduced = stage.reduce(results, dummy_pipeline_env, token=None)

        # _DummyQEMProtocol.reduce sums values across scale factors.
        # obs 0: sum(1.0, 2.0, 3.0) = 6.0
        # obs 1: sum(10.0, 11.0, 12.0) = 33.0
        assert len(reduced) == 1
        key = (("spec", "circ"), ("obs_group", 0))
        assert key in reduced
        assert reduced[key] == {0: pytest.approx(6.0), 1: pytest.approx(33.0)}

    def test_reduce_binds_symbolic_weights_from_param_set_foreign_key(
        self, dummy_pipeline_env
    ):
        theta = Parameter("theta")
        stage = QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0))
        base_key = (("spec", "circ"),)
        contexts = {
            base_key: {
                "classical_values": np.array([1.0, 0.0]),
                "weights": np.array([theta.cos(), theta.sin()], dtype=object),
                "symbolic": True,
                "weight_symbols": [theta],
                "target_idx": 0,
                "ensemble_start": 1,
                "n_rotations": 1,
                "n_paths": 2,
            },
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

        assert reduced[base_key] == pytest.approx(0.5)
        assert contexts[base_key]["symbolic"] is False
        assert contexts[base_key]["weights"][0] == pytest.approx(1.0)
        assert contexts[base_key]["weights"][1] == pytest.approx(0.0)


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
        assert meta.circuit_bodies
        # One DAG body per scale factor.
        assert len(meta.circuit_bodies) >= len(default_zne_protocol.scale_factors)

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
            assert meta.measurement_qasms


class TestQEMStageValidate:
    """Spec: QEMStage.validate enforces QuEPP-before-measurement and twirl-after constraints."""

    def test_quepp_before_measurement_passes(self):
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
            observable=SparsePauliOp.from_list([("Z", 1.0)]),
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

    def test_local_relative_effectiveness_with_and_without_twirling(
        self,
        dummy_pipeline_env,
        suppress_pipeline_perf_warnings,
        suppress_quepp_warnings,
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
            _, lineage_by_label = _compile_batch(trace.final_batch)
            return {
                branch_key: noisy_target for branch_key in lineage_by_label.values()
            }

        noisy_result = list(
            noisy_pipeline.run(
                initial_spec="ignored",
                env=dummy_pipeline_env,
                execute_fn=noisy_execute_fn,
            ).values()
        )[0]

        def build_quepp_execute_fn(with_twirls: bool):
            def execute_fn(trace, env):
                _, lineage_by_label = _compile_batch(trace.final_batch)
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
                        out[branch_key] = (
                            float(ctx["classical_values"][path_idx]) * noise_scale
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
        )[0]

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
        )[0]

        assert np.isfinite(noisy_result)
        assert np.isfinite(quepp_result)
        assert np.isfinite(twirl_result)

        noisy_err = abs(noisy_result - exact)
        quepp_err = abs(quepp_result - exact)
        twirl_err = abs(twirl_result - exact)
        assert quepp_err <= noisy_err + tolerance
        assert twirl_err <= noisy_err + tolerance

    def test_quepp_after_measurement_raises(self):
        with pytest.raises(
            ContractViolation,
            match="requires a measurement-handling stage after it",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    MeasurementStage(),
                    QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0)),
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

    def test_twirls_missing_twirl_stage_raises(self):
        with pytest.raises(
            ContractViolation,
            match=r"n_twirls=10 requires a PauliTwirlStage after it",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=10)),
                    MeasurementStage(),
                ]
            )

    def test_twirls_twirl_before_qem_raises(self):
        with pytest.raises(
            ContractViolation,
            match=r"n_twirls=10 requires a PauliTwirlStage after it",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    PauliTwirlStage(n_twirls=10),
                    QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=10)),
                    MeasurementStage(),
                ]
            )

    def test_quepp_with_twirls_full_pipeline_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=10)),
                PauliTwirlStage(n_twirls=10),
                MeasurementStage(),
            ]
        )

    def test_no_twirls_no_twirl_stage_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                QEMStage(protocol=QuEPP(truncation_order=1, n_twirls=0)),
                MeasurementStage(),
            ]
        )


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
