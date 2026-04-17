# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._pennylane_spec_stage."""

import numpy as np
import pennylane as qml
import pytest
import sympy

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PennyLaneSpecStage,
)


def _bell_script():
    """Non-parametric Bell-state QuantumScript."""
    return qml.tape.QuantumScript(
        ops=[qml.Hadamard(0), qml.CNOT(wires=[0, 1])],
        measurements=[qml.probs()],
    )


def _parametric_script():
    """Parametric QuantumScript with two sympy symbols."""
    theta, phi = sympy.symbols("theta phi")
    return qml.tape.QuantumScript(
        ops=[qml.RX(theta, wires=0), qml.RZ(phi, wires=0)],
        measurements=[qml.expval(qml.Z(0))],
    )


class TestPennyLaneSpecStageExpand:
    """Expand: QuantumScript(s) are converted to MetaCircuit batches."""

    def test_single_non_parametric(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()
        qs = _bell_script()

        batch, _ = stage.expand(qs, dummy_pipeline_env)

        assert len(batch) == 1
        key = next(iter(batch))
        assert key == (("circuit", 0),)
        meta = batch[key]
        assert meta.parameters == ()
        # Bell uses qml.probs() over all wires → measured_wires = (0, 1).
        assert meta.measured_wires == (0, 1)
        assert meta.observable is None

    def test_single_parametric(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()
        qs = _parametric_script()

        batch, _ = stage.expand(qs, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 2
        param_names = {p.name for p in meta.parameters}
        assert param_names == {"theta", "phi"}
        # expval(Z(0)) → observable set, no measured_wires.
        assert meta.observable is not None
        assert meta.measured_wires is None

    def test_sequence(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()
        scripts = [_bell_script(), _parametric_script()]

        batch, _ = stage.expand(scripts, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", 0),) in batch
        assert (("circuit", 1),) in batch
        assert len(batch[(("circuit", 0),)].parameters) == 0
        assert len(batch[(("circuit", 1),)].parameters) == 2

    def test_mapping(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()
        scripts = {"bell": _bell_script(), "param": _parametric_script()}

        batch, _ = stage.expand(scripts, dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", "bell"),) in batch
        assert (("circuit", "param"),) in batch

    def test_invalid_input_raises(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()

        with pytest.raises(TypeError, match="PennyLaneSpecStage expects"):
            stage.expand(42, dummy_pipeline_env)


class TestPennyLaneSpecStageQNode:
    """Expand: QNode(s) are converted to MetaCircuit batches."""

    def test_non_parametric_qnode(self, dummy_pipeline_env):
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def bell():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.probs()

        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(bell, dummy_pipeline_env)

        assert len(batch) == 1
        meta = batch[(("circuit", 0),)]
        assert meta.parameters == ()

    def test_parametric_qnode(self, dummy_pipeline_env):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.expval(qml.Z(0))

        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(circuit, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 2
        param_names = {p.name for p in meta.parameters}
        assert param_names == {"p0", "p1"}

    def test_sequence_mixed_qnode_and_script(self, dummy_pipeline_env):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def qnode_circuit():
            qml.Hadamard(0)
            return qml.probs()

        qs = _bell_script()
        stage = PennyLaneSpecStage()

        batch, _ = stage.expand([qs, qnode_circuit], dummy_pipeline_env)

        assert len(batch) == 2
        assert (("circuit", 0),) in batch
        assert (("circuit", 1),) in batch

    def test_single_param_qnode(self, dummy_pipeline_env):
        """QNode with exactly one parameter doesn't break tuple unpacking."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def single_param(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(single_param, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 1
        assert meta.parameters[0].name == "p0"


class TestPennyLaneSpecStageQNodeArray:
    """QNodes with array parameters are auto-detected and converted."""

    def test_qnode_array_param_auto_detected(self, dummy_pipeline_env):
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return qml.expval(qml.Z(0))

        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(circuit, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        assert len(meta.parameters) == 2

    def test_qnode_array_with_fixed_constants(self, dummy_pipeline_env):
        """Only array-derived parameters become symbols; fixed constants stay numeric."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(np.pi / 2, wires=1)  # fixed constant
            return qml.probs()

        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(circuit, dummy_pipeline_env)

        meta = batch[(("circuit", 0),)]
        # Only the array-derived parameter is parametric on the DAG.
        assert len(meta.parameters) == 1

    def test_qnode_multi_array_param_raises(self, dummy_pipeline_env):
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(weights, biases):
            qml.RX(weights[0], wires=0)
            qml.RY(biases[0], wires=1)
            return qml.probs()

        stage = PennyLaneSpecStage()

        with pytest.raises(TypeError, match="array parameters"):
            stage.expand(circuit, dummy_pipeline_env)


class TestPennyLaneSpecStageMeasurementValidation:
    """Measurement type validation: only probs, expval, counts are supported."""

    def test_multi_measurement_raises(self, dummy_pipeline_env):
        qs = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.expval(qml.Z(0)), qml.probs()],
        )
        stage = PennyLaneSpecStage()

        with pytest.raises(ValueError, match="exactly one measurement"):
            stage.expand(qs, dummy_pipeline_env)

    def test_unsupported_measurement_raises(self, dummy_pipeline_env):
        qs = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.sample()],
        )
        stage = PennyLaneSpecStage()

        with pytest.raises(ValueError, match="probs.*expval.*counts"):
            stage.expand(qs, dummy_pipeline_env)

    def test_counts_measurement_accepted(self, dummy_pipeline_env):
        qs = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.counts()],
        )
        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(qs, dummy_pipeline_env)
        assert len(batch) == 1

    def test_expval_measurement_accepted(self, dummy_pipeline_env):
        qs = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.expval(qml.Z(0))],
        )
        stage = PennyLaneSpecStage()

        batch, _ = stage.expand(qs, dummy_pipeline_env)
        assert len(batch) == 1


class TestPennyLaneSpecStageReduce:
    """Reduce is inherited from CircuitSpecStage."""

    def test_strip_circuit_axis(self, dummy_pipeline_env):
        stage = PennyLaneSpecStage()
        results = {(("circuit", 0), ("obs_group", 0)): 1.5}

        reduced = stage.reduce(results, dummy_pipeline_env, token="single")

        assert (("obs_group", 0),) in reduced
        assert reduced[(("obs_group", 0),)] == 1.5


class TestPennyLaneSpecStageProperties:
    """Inherited properties from CircuitSpecStage."""

    def test_axis_name(self):
        assert PennyLaneSpecStage().axis_name == "circuit"

    def test_stateful_is_false(self):
        assert PennyLaneSpecStage().stateful is False


class TestPennyLaneSpecStagePipeline:
    """Full pipeline: spec → measurement → execute → reduce."""

    def test_bell_probs(self, default_test_simulator):
        """Non-parametric Bell state produces ~50/50 probabilities."""
        pipeline = CircuitPipeline(stages=[PennyLaneSpecStage(), MeasurementStage()])
        env = PipelineEnv(backend=default_test_simulator)
        result = pipeline.run(initial_spec=_bell_script(), env=env)

        probs = result[()]
        assert "00" in probs
        assert "11" in probs
        assert probs["00"] + probs["11"] == pytest.approx(1.0, abs=0.05)

    def test_parametric_qnode_with_binding(self, default_test_simulator):
        """QNode with parameter binding produces expectation value."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.expval(qml.Z(0))

        pipeline = CircuitPipeline(
            stages=[
                PennyLaneSpecStage(),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        param_sets = np.array([[0.0, 0.0]])
        env = PipelineEnv(backend=default_test_simulator, param_sets=param_sets)
        result = pipeline.run(initial_spec=circuit, env=env)

        # RX(0) RZ(0) |0> = |0>, so <Z> ≈ 1.0
        expval = next(iter(result.values()))
        assert expval == pytest.approx(1.0, abs=0.05)
