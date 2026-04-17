# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qiskit_spec_stage."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    QiskitSpecStage,
)


def _bell_qiskit():
    """Non-parametric Bell-state Qiskit QuantumCircuit with measurements."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _parametric_qiskit():
    """Parametric Qiskit QuantumCircuit with two parameters."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)
    qc.measure(0, 0)
    return qc


def _no_measure_qiskit():
    """Qiskit QuantumCircuit with no measurement instructions."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


class TestQiskitSpecStageExpand:
    """Expand: QuantumCircuit(s) are converted to MetaCircuit batches."""

    def test_single_non_parametric(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _bell_qiskit()

        batch, _ = stage.expand(qc, dummy_pipeline_env)

        assert len(batch) == 1
        key = next(iter(batch))
        assert key == (("circuit", 0),)
        meta = batch[key]
        assert meta.parameters == ()
        # Bell state measures both qubits.
        assert meta.measured_wires == (0, 1)
        # DAG has the two gates, no measurements.
        _, dag = meta.circuit_bodies[0]
        assert dag.num_qubits() == 2
        op_names = {node.op.name for node in dag.op_nodes()}
        assert op_names == {"h", "cx"}

    def test_single_parametric(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _parametric_qiskit()

        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 2
        # QuantumCircuit.parameters orders alphabetically.
        param_names = {p.name for p in meta.parameters}
        assert param_names == {"theta", "phi"}
        assert meta.measured_wires == (0,)

    def test_no_measurements_warns_and_defaults(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _no_measure_qiskit()

        with pytest.warns(UserWarning, match="no measurement operations"):
            batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert meta.measured_wires == (0, 1)

    def test_sequence(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        circuits = [_bell_qiskit(), _parametric_qiskit()]

        batch, _ = stage.expand(circuits, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", 0),) in batch
        assert (("circuit", 1),) in batch
        assert len(batch[(("circuit", 0),)].parameters) == 0
        assert len(batch[(("circuit", 1),)].parameters) == 2

    def test_mapping(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        circuits = {"bell": _bell_qiskit(), "param": _parametric_qiskit()}

        batch, _ = stage.expand(circuits, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", "bell"),) in batch
        assert (("circuit", "param"),) in batch

    def test_invalid_input_raises(self, dummy_pipeline_env):
        stage = QiskitSpecStage()

        with pytest.raises(TypeError, match="QiskitSpecStage expects"):
            stage.expand(42, dummy_pipeline_env)


class TestQiskitParameterExpressions:
    """ParameterExpressions (e.g. 2*theta) flow through as Qiskit-native."""

    def test_parameter_expression_preserved(self, dummy_pipeline_env):
        """rx(2*theta) — the DAG op carries a ParameterExpression, not a float."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.rx(2 * theta, 0)
        qc.measure(0, 0)

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 1
        assert meta.parameters[0].name == "theta"
        # Exactly one rx gate; its param is a composite ParameterExpression.
        _, dag = meta.circuit_bodies[0]
        rx_nodes = [n for n in dag.op_nodes() if n.op.name == "rx"]
        assert len(rx_nodes) == 1
        (param,) = rx_nodes[0].op.params
        assert isinstance(param, ParameterExpression)
        # Binding theta=0.5 gives 1.0.
        assert float(param.bind({meta.parameters[0]: 0.5})) == pytest.approx(1.0)

    def test_sum_expression_preserved(self, dummy_pipeline_env):
        """rx(theta + phi) — DAG carries a two-parameter expression."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(1, 1)
        qc.rx(theta + phi, 0)
        qc.measure(0, 0)

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert {p.name for p in meta.parameters} == {"theta", "phi"}
        _, dag = meta.circuit_bodies[0]
        (rx_param,) = next(n for n in dag.op_nodes() if n.op.name == "rx").op.params
        assert isinstance(rx_param, ParameterExpression)
        pmap = {p.name: p for p in meta.parameters}
        bound = float(rx_param.bind({pmap["theta"]: 0.3, pmap["phi"]: 0.7}))
        assert bound == pytest.approx(1.0)

    def test_mixed_params_and_constants(self, dummy_pipeline_env):
        """Float constants remain floats on the DAG; parameters stay parametric."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2, 2)
        qc.rx(theta, 0)
        qc.rz(3.14, 0)
        qc.ry(phi, 1)
        qc.measure([0, 1], [0, 1])

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert {p.name for p in meta.parameters} == {"theta", "phi"}
        _, dag = meta.circuit_bodies[0]
        # rz(3.14) stays numeric.
        rz_node = next(n for n in dag.op_nodes() if n.op.name == "rz")
        (rz_param,) = rz_node.op.params
        assert isinstance(rz_param, float)
        assert rz_param == pytest.approx(3.14)


class TestQiskitSpecStageProperties:
    """Inherited properties from CircuitSpecStage."""

    def test_axis_name(self):
        assert QiskitSpecStage().axis_name == "circuit"

    def test_stateful_is_false(self):
        assert QiskitSpecStage().stateful is False


class TestQiskitSpecStagePipeline:
    """Full pipeline: spec → measurement → execute → reduce."""

    def test_bell_probs(self, default_test_simulator):
        """Non-parametric Bell circuit produces ~50/50 probabilities."""
        pipeline = CircuitPipeline(stages=[QiskitSpecStage(), MeasurementStage()])
        env = PipelineEnv(backend=default_test_simulator)
        result = pipeline.run(initial_spec=_bell_qiskit(), env=env)

        probs = result[()]
        assert "00" in probs
        assert "11" in probs
        assert probs["00"] + probs["11"] == pytest.approx(1.0, abs=0.05)

    def test_parametric_with_binding(self, default_test_simulator):
        """Parametric Qiskit circuit with parameter binding produces valid probs."""
        pipeline = CircuitPipeline(
            stages=[
                QiskitSpecStage(),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        # rx(0) rz(0) → |0>, probs should be ~{0: 1.0, 1: 0.0}
        param_sets = np.array([[0.0, 0.0]])
        env = PipelineEnv(backend=default_test_simulator, param_sets=param_sets)
        result = pipeline.run(initial_spec=_parametric_qiskit(), env=env)

        probs = next(iter(result.values()))
        assert "0" in probs
        assert probs["0"] == pytest.approx(1.0, abs=0.05)

    def test_parameter_expression_executes_correctly(self, default_test_simulator):
        """rx(2*theta) with theta=π/2 → rx(π) flips |0⟩ to |1⟩."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.rx(2 * theta, 0)
        qc.measure(0, 0)

        pipeline = CircuitPipeline(
            stages=[
                QiskitSpecStage(),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        param_sets = np.array([[np.pi / 2]])
        env = PipelineEnv(backend=default_test_simulator, param_sets=param_sets)
        result = pipeline.run(initial_spec=qc, env=env)

        probs = next(iter(result.values()))
        assert "1" in probs
        assert probs["1"] == pytest.approx(1.0, abs=0.05)
