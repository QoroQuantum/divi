# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.pipeline.stages._parameter_binding_stage."""

import numpy as np
import pennylane as qml
import pytest
import sympy

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, ContractViolation, PipelineEnv
from divi.pipeline.stages import MeasurementStage, ParameterBindingStage, QEMStage
from divi.pipeline.stages._parameter_binding_stage import _format_param
from tests.pipeline.helpers import DummySpecStage, two_group_meta


def _parametric_meta(symbol_names: tuple[str, ...] = ("theta", "phi")) -> MetaCircuit:
    """Build a MetaCircuit whose QASM body contains symbolic parameter names."""
    symbols = np.array([sympy.Symbol(s) for s in symbol_names], dtype=object)
    qscript = qml.tape.QuantumScript(
        ops=[
            qml.RX(sympy.Symbol(symbol_names[0]), 0),
            qml.RZ(sympy.Symbol(symbol_names[1]), 0),
        ],
        measurements=[qml.expval(qml.Z(0))],
    )
    return MetaCircuit(source_circuit=qscript, symbols=symbols)


class TestParameterBindingStage:
    """Spec: ParameterBindingStage expand binds env.param_sets into circuit body QASMs; reduce is identity."""

    def test_requires_2d_param_sets(self, dummy_pipeline_env):
        meta = two_group_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        env = PipelineEnv(backend=dummy_pipeline_env.backend, param_sets=[1.0, 2.0])
        with pytest.raises(ValueError, match="param_sets to be 2D"):
            pipeline.run_forward_pass("x", env)

    def test_passthrough_when_no_symbols(self, dummy_pipeline_env):
        meta = two_group_meta()
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[0.0]]),
        )
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        trace = pipeline.run_forward_pass("x", env)
        for node in trace.final_batch.values():
            assert node.circuit_body_qasms

    def test_binds_parameters_into_qasm(self, dummy_pipeline_env):
        """Core spec: symbolic names in QASM are replaced by formatted parameter values."""
        meta = _parametric_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.5, 2.7]]),
        )
        trace = pipeline.run_forward_pass("x", env)

        # All bound bodies should contain the formatted values, not the symbol names
        for node in trace.final_batch.values():
            for _tag, body in node.circuit_body_qasms:
                assert "theta" not in body
                assert "phi" not in body
                assert "1.5" in body
                assert "2.7" in body

    def test_multiple_param_sets_produce_multiple_bodies(self, dummy_pipeline_env):
        """Each param set produces a separate body variant tagged with param_set axis."""
        meta = _parametric_meta()
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=params,
        )
        trace = pipeline.run_forward_pass("x", env)

        # The param_set axis appears on the QASM tags inside each MetaCircuit node
        for node in trace.final_batch.values():
            param_set_indices = set()
            for tag, _body in node.circuit_body_qasms:
                for axis_name, axis_value in tag:
                    if axis_name == "param_set":
                        param_set_indices.add(axis_value)
            assert param_set_indices == {0, 1, 2}

    def test_param_count_mismatch_raises(self, dummy_pipeline_env):
        """Providing wrong number of parameters for a circuit raises ValueError."""
        meta = _parametric_meta()  # expects 2 symbols
        pipeline = CircuitPipeline(
            stages=[
                DummySpecStage(meta=meta),
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        env = PipelineEnv(
            backend=dummy_pipeline_env.backend,
            param_sets=np.array([[1.0]]),  # only 1 value for 2 symbols
        )
        with pytest.raises(ValueError, match="expected 2 parameters"):
            pipeline.run_forward_pass("x", env)

    def test_reduce_is_identity(self):
        """Reduce returns its input unchanged."""
        stage = ParameterBindingStage()
        sentinel = {(("spec", "circ"),): 42.0}
        assert stage.reduce(sentinel, None, None) is sentinel

    def test_axis_name_is_param_set(self):
        assert ParameterBindingStage().axis_name == "param_set"

    def test_stateful_is_true(self):
        assert ParameterBindingStage().stateful is True


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


class TestParameterBindingStageValidate:
    """Spec: ParameterBindingStage must come before QEMStage."""

    def test_param_binding_before_qem_passes(self):
        CircuitPipeline(
            stages=[
                DummySpecStage(meta=two_group_meta()),
                ParameterBindingStage(),
                QEMStage(),
                MeasurementStage(),
            ]
        )

    def test_param_binding_after_qem_raises(self):
        with pytest.raises(
            ContractViolation,
            match="ParameterBindingStage must come before QEMStage",
        ):
            CircuitPipeline(
                stages=[
                    DummySpecStage(meta=two_group_meta()),
                    QEMStage(),
                    ParameterBindingStage(),
                    MeasurementStage(),
                ]
            )
