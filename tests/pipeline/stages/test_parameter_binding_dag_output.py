# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ParameterBindingStage dual-path (QASM-string fast path vs bound DAG slow path)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator, SparsePauliOp

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, PipelineEnv, PipelineTrace
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
)
from tests.pipeline.helpers import DummySpecStage


def _two_qubit_parametric_meta() -> MetaCircuit:
    """Two-qubit parametric MetaCircuit with two independent parameters."""
    params = (Parameter("theta"), Parameter("phi"))
    qc = QuantumCircuit(2)
    qc.rx(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.rz(params[0], 1)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=params,
        observable=SparsePauliOp.from_list([("ZZ", 1.0)]),
    )


def _param_bind_output(trace: PipelineTrace) -> MetaCircuit:
    """Extract the MetaCircuit emitted by ParameterBindingStage from a run_forward_pass trace.

    Avoids peeking at private stage state — tests the observable post-expand
    output without depending on subsequent stages having already mutated it.
    """
    pb_expansion = next(
        exp
        for exp in trace.stage_expansions
        if exp.stage_name == "ParameterBindingStage"
    )
    return next(iter(pb_expansion.batch.values()))


@pytest.fixture
def meta() -> MetaCircuit:
    return _two_qubit_parametric_meta()


@pytest.fixture
def param_sets() -> np.ndarray:
    return np.array([[0.3, 1.7], [2.1, -0.4], [4.0, 0.5]])


class TestPathSelection:
    """Spec: fast path populates ``bound_circuit_bodies`` with QASM strings;
    slow path replaces ``circuit_bodies`` with bound DAGs.  Choice is driven
    by whether any downstream stage declares ``consumes_dag_bodies=True``.
    """

    def test_fast_path_when_param_binding_is_last_stage(
        self, dummy_pipeline_env, meta, param_sets
    ):
        """MeasurementStage placed before ParameterBindingStage leaves ParamBind with no
        downstream consumers — fast path."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        trace = pipeline.run_forward_pass("x", env)

        node = _param_bind_output(trace)
        assert len(node.bound_circuit_bodies) == len(param_sets)
        for _tag, body in node.bound_circuit_bodies:
            assert isinstance(body, str)

    def test_fast_path_when_only_measurement_follows(
        self, dummy_pipeline_env, meta, param_sets
    ):
        """MeasurementStage has ``consumes_dag_bodies=False`` — fast path preserved."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        trace = pipeline.run_forward_pass("x", env)

        node = _param_bind_output(trace)
        assert len(node.bound_circuit_bodies) == len(param_sets)
        for _tag, body in node.bound_circuit_bodies:
            assert isinstance(body, str)

    def test_slow_path_when_pauli_twirl_follows(
        self, dummy_pipeline_env, meta, param_sets
    ):
        """PauliTwirlStage reads ``circuit_bodies`` — ParamBind emits bound DAGs."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=1, seed=0),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        trace = pipeline.run_forward_pass("x", env)

        node = _param_bind_output(trace)
        assert node.bound_circuit_bodies == ()
        assert len(node.circuit_bodies) == len(param_sets)
        for _tag, body in node.circuit_bodies:
            assert isinstance(body, DAGCircuit)
            assert not dag_to_circuit(body).parameters

    def test_slow_path_when_qem_follows(self, dummy_pipeline_env, meta, param_sets):
        """QEMStage reads ``circuit_bodies`` — ParamBind emits bound DAGs."""
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                QEMStage(),  # _NoMitigation protocol — no perf warnings
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        trace = pipeline.run_forward_pass("x", env)

        node = _param_bind_output(trace)
        assert node.bound_circuit_bodies == ()
        assert len(node.circuit_bodies) == len(param_sets)
        for _tag, body in node.circuit_bodies:
            assert isinstance(body, DAGCircuit)
            assert not dag_to_circuit(body).parameters


class TestFastPathZeroCopy:
    """Spec: fast path does not rewrite the symbolic ``circuit_bodies`` DAGs."""

    def test_original_dag_identity_preserved(
        self, dummy_pipeline_env, meta, param_sets
    ):
        original_dag = meta.circuit_bodies[0][1]
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        trace = pipeline.run_forward_pass("x", env)

        node = _param_bind_output(trace)
        assert len(node.circuit_bodies) == 1
        assert node.circuit_bodies[0][1] is original_dag


class TestSlowPathValidation:
    """Spec: slow-path expand validates param-set shape."""

    def test_param_count_mismatch_raises(self, dummy_pipeline_env, meta):
        stage = ParameterBindingStage()
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                stage,
                PauliTwirlStage(n_twirls=1, seed=0),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.0]]),  # meta expects 2 parameters
        )
        with pytest.raises(ValueError, match="expected 2 parameters"):
            stage.expand({(("spec", "circ"),): meta}, env)


class TestFastSlowEquivalence:
    """Spec: both paths produce semantically equivalent bound circuits."""

    def test_operators_match(self, dummy_pipeline_env, meta, param_sets):
        fast_pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                MeasurementStage(),
            ]
        )
        slow_pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                ParameterBindingStage(),
                PauliTwirlStage(n_twirls=1, seed=0),
                MeasurementStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=param_sets)
        fast_node = _param_bind_output(fast_pipeline.run_forward_pass("x", env))
        slow_node = _param_bind_output(slow_pipeline.run_forward_pass("x", env))

        qreg_header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        for ps_idx in range(len(param_sets)):
            fast_body = next(
                body
                for tag, body in fast_node.bound_circuit_bodies
                if ("param_set", ps_idx) in tag
            )
            slow_dag = next(
                dag
                for tag, dag in slow_node.circuit_bodies
                if ("param_set", ps_idx) in tag
            )
            fast_qc = qasm2.loads(qreg_header + fast_body)
            slow_qc = dag_to_circuit(slow_dag)
            assert Operator(fast_qc).equiv(Operator(slow_qc))
