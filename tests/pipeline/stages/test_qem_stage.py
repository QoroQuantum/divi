# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qem_stage."""

from collections.abc import Sequence
from functools import partial
from typing import Any

import pennylane as qml
import pytest
import sympy as sp
from cirq.circuits.circuit import Circuit
from mitiq.zne.inference import ExpFactory
from mitiq.zne.scaling import fold_global

from divi.circuits import MetaCircuit
from divi.circuits.qem import ZNE, QEMContext, QEMProtocol, _NoMitigation
from divi.circuits.quepp import QuEPP
from divi.pipeline import CircuitPipeline, ContractViolation
from divi.pipeline.stages import MeasurementStage, PauliTwirlStage, QEMStage
from tests.pipeline.helpers import DummySpecStage, ones_execute_fn, two_group_meta


class _DummyQEMProtocol(QEMProtocol):
    """Minimal QEM protocol for tests (used only in this module)."""

    @property
    def name(self) -> str:
        return "dummy-qem"

    def expand(
        self, cirq_circuit: Circuit, observable: Any | None = None
    ) -> tuple[tuple[Circuit, ...], QEMContext]:
        return (cirq_circuit,), QEMContext()

    def reduce(self, quantum_results: Sequence[float], context: QEMContext) -> float:
        return float(sum(quantum_results))


@pytest.fixture
def default_zne_protocol():
    scale_factors = [1, 3, 5]
    return ZNE(
        folding_fn=partial(fold_global),
        scale_factors=scale_factors,
        extrapolation_factory=ExpFactory(scale_factors=scale_factors),
    )


@pytest.fixture
def weights_syms():
    return sp.symarray("w", 4)


@pytest.fixture
def sample_circuit(weights_syms):
    ops = [
        qml.AngleEmbedding(weights_syms, wires=range(4), rotation="Y"),
        qml.AngleEmbedding(weights_syms, wires=range(4), rotation="X"),
    ]
    return qml.tape.QuantumScript(ops=ops, measurements=[qml.probs()])


class TestQEMStage:
    """Spec: QEMStage expand applies protocol to body QASMs (fan-out); reduce postprocesses."""

    def test_qem_fanout_and_reduce(self, dummy_pipeline_env):
        class _ScaleFactorProtocol(_DummyQEMProtocol):
            def __init__(self, scale_factors: tuple[float, ...]) -> None:
                self.scale_factors = scale_factors

            def expand(
                self, cirq_circuit: Circuit, observable: Any | None = None
            ) -> tuple[tuple[Circuit, ...], QEMContext]:
                return tuple(cirq_circuit for _ in self.scale_factors), QEMContext()

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
                self, cirq_circuit: Circuit, observable: Any | None = None
            ) -> tuple[tuple[Circuit, ...], QEMContext]:
                return tuple(cirq_circuit for _ in self.scale_factors), QEMContext()

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


class TestPipelineOutputMetaCircuitWithQEM:
    """Spec: Pipeline with ObservableGroupingStage + QEMStage produces MetaCircuits with correct structure."""

    def test_zne_fanout_produces_expected_structure(
        self, sample_circuit, weights_syms, dummy_pipeline_env, default_zne_protocol
    ):
        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
        )
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta_circuit),
                QEMStage(protocol=default_zne_protocol),
                MeasurementStage(),
            ],
        )
        trace = pipeline.run_forward_pass(42, dummy_pipeline_env)
        assert len(trace.final_batch) == 1
        meta = next(iter(trace.final_batch.values()))
        assert hasattr(meta, "circuit_body_qasms") and meta.circuit_body_qasms
        assert len(meta.circuit_body_qasms) >= len(default_zne_protocol.scale_factors)

    def test_no_mitigation_single_body_per_key(
        self, sample_circuit, weights_syms, dummy_pipeline_env
    ):
        meta_circuit = MetaCircuit(
            source_circuit=sample_circuit,
            symbols=weights_syms,
        )
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta_circuit),
                MeasurementStage(),
                QEMStage(protocol=_NoMitigation()),
            ],
        )
        trace = pipeline.run_forward_pass(42, dummy_pipeline_env)
        for key, meta in trace.final_batch.items():
            assert len(meta.circuit_body_qasms) == 1
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
