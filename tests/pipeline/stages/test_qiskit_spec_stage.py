# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._qiskit_spec_stage."""

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    QiskitSpecStage,
)
from divi.pipeline.stages._qiskit_spec_stage import qiskit_to_pennylane


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


class TestQiskitToPennylane:
    """Unit tests for the shared qiskit_to_pennylane utility."""

    def test_converts_with_probs(self):
        qc = _bell_qiskit()
        qs = qiskit_to_pennylane(qc, lambda wires: qml.probs(wires=wires))

        assert len(qs.measurements) == 1
        assert isinstance(qs.measurements[0], qml.measurements.ProbabilityMP)

    def test_converts_with_expval(self):
        qc = _bell_qiskit()
        qs = qiskit_to_pennylane(qc, lambda wires: qml.expval(qml.Z(wires[0])))

        assert len(qs.measurements) == 1
        assert isinstance(qs.measurements[0], qml.measurements.ExpectationMP)

    def test_no_measurements_warns(self):
        qc = _no_measure_qiskit()
        with pytest.warns(UserWarning, match="no measurement operations"):
            qs = qiskit_to_pennylane(qc, lambda wires: qml.probs(wires=wires))

        assert len(qs.measurements) == 1

    def test_circuit_with_classical_registers(self):
        """Circuits with classical registers (e.g. for c_if) don't crash."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qs = qiskit_to_pennylane(qc, lambda wires: qml.probs(wires=wires))

        assert len(qs.measurements) == 1


class TestQiskitSpecStageExpand:
    """Expand: QuantumCircuit(s) are converted to MetaCircuit batches."""

    def test_single_non_parametric(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _bell_qiskit()

        batch, token = stage.expand(qc, dummy_pipeline_env)

        assert len(batch) == 1
        key = next(iter(batch))
        assert key == (("circuit", 0),)
        meta = batch[key]
        assert len(meta.symbols) == 0
        # The converted circuit should have a probs measurement
        assert isinstance(
            meta.source_circuit.measurements[0],
            qml.measurements.ProbabilityMP,
        )

    def test_single_parametric(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _parametric_qiskit()

        batch, token = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.symbols) == 2
        symbol_names = {str(s) for s in meta.symbols}
        assert symbol_names == {"theta", "phi"}

    def test_no_measurements_warns_and_defaults(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        qc = _no_measure_qiskit()

        with pytest.warns(UserWarning, match="no measurement operations"):
            batch, token = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert isinstance(
            meta.source_circuit.measurements[0],
            qml.measurements.ProbabilityMP,
        )

    def test_sequence(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        circuits = [_bell_qiskit(), _parametric_qiskit()]

        batch, token = stage.expand(circuits, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", 0),) in batch
        assert (("circuit", 1),) in batch
        assert len(batch[(("circuit", 0),)].symbols) == 0
        assert len(batch[(("circuit", 1),)].symbols) == 2

    def test_mapping(self, dummy_pipeline_env):
        stage = QiskitSpecStage()
        circuits = {"bell": _bell_qiskit(), "param": _parametric_qiskit()}

        batch, token = stage.expand(circuits, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", "bell"),) in batch
        assert (("circuit", "param"),) in batch

    def test_invalid_input_raises(self, dummy_pipeline_env):
        stage = QiskitSpecStage()

        with pytest.raises(TypeError, match="QiskitSpecStage expects"):
            stage.expand(42, dummy_pipeline_env)


class TestQiskitParameterExpressions:
    """ParameterExpressions (e.g. 2*theta) are preserved as sympy expressions."""

    def test_parameter_expression_preserved(self, dummy_pipeline_env):
        """rx(2*theta) should produce QASM with '2*theta', not just 'theta'."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1, 1)
        qc.rx(2 * theta, 0)
        qc.measure(0, 0)

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        # Base symbol is just 'theta'
        assert len(meta.symbols) == 1
        assert str(meta.symbols[0]) == "theta"
        # QASM body should contain the expression '2*theta'
        for _tag, body in meta.circuit_body_qasms:
            assert "2*theta" in body

    def test_sum_expression_preserved(self, dummy_pipeline_env):
        """rx(theta + phi) should produce QASM with the sum expression."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(1, 1)
        qc.rx(theta + phi, 0)
        qc.measure(0, 0)

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.symbols) == 2
        symbol_names = {str(s) for s in meta.symbols}
        assert symbol_names == {"theta", "phi"}
        # QASM should contain both symbols in an expression
        for _tag, body in meta.circuit_body_qasms:
            assert "phi" in body
            assert "theta" in body

    def test_mixed_params_and_constants(self, dummy_pipeline_env):
        """Float constants are preserved; only ParameterExpressions become sympy."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2, 2)
        qc.rx(theta, 0)
        qc.rz(3.14, 0)  # fixed constant — must stay in QASM as-is
        qc.ry(phi, 1)
        qc.measure([0, 1], [0, 1])

        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        # Only theta and phi are base symbols (not 3.14)
        assert len(meta.symbols) == 2
        symbol_names = {str(s) for s in meta.symbols}
        assert symbol_names == {"theta", "phi"}
        # QASM should contain both symbols and the constant
        for _tag, body in meta.circuit_body_qasms:
            assert "theta" in body
            assert "phi" in body
            assert "3.14" in body

    def test_bare_parameter_still_works(self, dummy_pipeline_env):
        """Simple Parameter (no expression) continues to work as before."""
        qc = _parametric_qiskit()
        stage = QiskitSpecStage()
        batch, _ = stage.expand(qc, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.symbols) == 2
        symbol_names = {str(s) for s in meta.symbols}
        assert symbol_names == {"theta", "phi"}


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
        # theta=π/2 → rx(π) → |1⟩
        param_sets = np.array([[np.pi / 2]])
        env = PipelineEnv(backend=default_test_simulator, param_sets=param_sets)
        result = pipeline.run(initial_spec=qc, env=env)

        probs = next(iter(result.values()))
        assert "1" in probs
        assert probs["1"] == pytest.approx(1.0, abs=0.05)
