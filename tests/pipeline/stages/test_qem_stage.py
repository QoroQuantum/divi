# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qem_stage."""

from collections.abc import Sequence
from functools import partial

import pennylane as qml
import pytest
import sympy as sp
from cirq.circuits.circuit import Circuit
from mitiq.zne.inference import ExpFactory
from mitiq.zne.scaling import fold_global

from divi.circuits import MetaCircuit
from divi.circuits.qem import ZNE, QEMProtocol, _NoMitigation
from divi.pipeline import CircuitPipeline
from divi.pipeline.stages import MeasurementStage, QEMStage
from tests.pipeline.helpers import DummySpecStage, ones_execute_fn, two_group_meta


class _DummyQEMProtocol(QEMProtocol):
    """Minimal QEM protocol for tests (used only in this module)."""

    @property
    def name(self) -> str:
        return "dummy-qem"

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        return [cirq_circuit]

    def postprocess_results(self, results: Sequence[float]) -> float:
        return float(sum(results))


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

            def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
                return [cirq_circuit for _ in self.scale_factors]

        protocol = _ScaleFactorProtocol((1.0, 2.0, 3.0))
        meta = two_group_meta()

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                QEMStage(protocol=protocol),
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

    def test_reduce_raises_on_dict_valued_results(self, dummy_pipeline_env):
        """QEM reduce must reject probability dicts — only scalars are valid."""
        protocol = _DummyQEMProtocol()
        meta = two_group_meta()

        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                QEMStage(protocol=protocol),
            ],
        )

        def probs_execute_fn(trace, env):
            """Return probability dicts instead of scalars."""
            from divi.pipeline._compilation import _compile_batch

            _, lineage = _compile_batch(trace.final_batch)
            return {bk: {"00": 0.5, "11": 0.5} for bk in lineage.values()}

        with pytest.raises(TypeError, match="scalar expectation values"):
            pipeline.run(
                initial_spec="ignored",
                env=dummy_pipeline_env,
                execute_fn=probs_execute_fn,
            )

    def test_reduce_handles_multi_obs_expval_dicts(self, dummy_pipeline_env):
        """QEM reduce applies postprocessing per observable when values are {int: float} dicts.

        This mirrors what _counts_to_expvals produces for multi-observable
        measurement groups: each branch key maps to {obs_idx: float} instead
        of a scalar.  Calls reduce() directly to validate the per-observable
        logic in isolation.
        """

        class _ScaleFactorProtocol(_DummyQEMProtocol):
            """Produces 3 body variants (like 3 scale factors) and sums during reduce."""

            def __init__(self) -> None:
                self.scale_factors = (1.0, 2.0, 3.0)

            def modify_circuit(self, cirq_circuit):
                return [cirq_circuit for _ in self.scale_factors]

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

        # _DummyQEMProtocol.postprocess_results sums values across scale factors.
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
                MeasurementStage(),
                QEMStage(protocol=default_zne_protocol),
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
