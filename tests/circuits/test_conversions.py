# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PennyLane-free circuit conversion utilities."""

import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag

from divi.circuits import dag_to_qasm_body
from divi.circuits._conversions import (
    _assert_finite,
    _format_bound_param,
    _format_gate_param,
    _sympy_to_qiskit,
)


class TestFormatBoundParam:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "0"),
            (-0.0, "0"),
            (4.9e-9, "0"),
            (np.pi, "3.14159265"),
            (1234567.891, "1234567.891"),
        ],
    )
    def test_renders_finite_angles(self, value, expected):
        assert _format_bound_param(value, 8) == expected


class TestAssertFinite:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite(self, bad):
        values = np.array([[0.0, bad], [1.0, 2.0]])
        with pytest.raises(ValueError, match="non-finite gate parameters"):
            _assert_finite(values, source="env.param_sets")

    def test_passes_finite_matrix(self):
        _assert_finite(np.array([[0.0, 1.0], [2.0, 3.0]]), source="env.feature_batch")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_format_gate_param_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="non-finite gate parameter"):
        _format_gate_param(bad, 8)


class TestSympyToQiskit:
    def test_bare_symbol_maps_to_parameter(self):
        theta = sp.Symbol("theta")
        parameter = Parameter("theta")
        out = _sympy_to_qiskit(theta, {theta: parameter})
        assert isinstance(out, ParameterExpression)
        assert float(out.bind({parameter: 2.5})) == pytest.approx(2.5)

    def test_numeric_constants_return_float(self):
        assert _sympy_to_qiskit(sp.Float(1.25), {}) == 1.25
        assert _sympy_to_qiskit(sp.Integer(3), {}) == 3.0
        assert _sympy_to_qiskit(sp.pi, {}) == pytest.approx(np.pi)

    def test_plain_python_number_passes_through(self):
        assert _sympy_to_qiskit(2.5, {}) == 2.5
        assert _sympy_to_qiskit(1, {}) == 1.0

    def test_add_composes_via_parameter_arithmetic(self):
        a, b = sp.Symbol("a"), sp.Symbol("b")
        pa, pb = Parameter("a"), Parameter("b")
        out = _sympy_to_qiskit(a + b, {a: pa, b: pb})
        assert isinstance(out, ParameterExpression)
        assert float(out.bind({pa: 1.0, pb: 2.0})) == pytest.approx(3.0)

    def test_mul_with_numeric_coefficient(self):
        theta = sp.Symbol("theta")
        parameter = Parameter("theta")
        out = _sympy_to_qiskit(2 * theta, {theta: parameter})
        assert float(out.bind({parameter: 0.5})) == pytest.approx(1.0)

    def test_pow_composes(self):
        theta = sp.Symbol("theta")
        parameter = Parameter("theta")
        out = _sympy_to_qiskit(theta**2, {theta: parameter})
        assert float(out.bind({parameter: 3.0})) == pytest.approx(9.0)

    def test_sin_maps_to_parameter_method(self):
        theta = sp.Symbol("theta")
        parameter = Parameter("theta")
        out = _sympy_to_qiskit(sp.sin(theta), {theta: parameter})
        assert float(out.bind({parameter: np.pi / 2})) == pytest.approx(1.0)

    def test_unmapped_symbol_raises(self):
        theta = sp.Symbol("theta")
        with pytest.raises(ValueError, match="Unmapped sympy symbol"):
            _sympy_to_qiskit(theta, {})

    def test_unknown_expression_type_raises(self):
        x = sp.Symbol("x")
        with pytest.raises(NotImplementedError):
            _sympy_to_qiskit(sp.factorial(x), {x: Parameter("x")})


class TestDagToQasmBody:
    def test_preamble_is_not_emitted(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)
        body = dag_to_qasm_body(circuit_to_dag(circuit))
        assert "OPENQASM" not in body
        assert "include" not in body
        assert "qreg" not in body
        assert "creg" not in body
        assert "h q[0];" in body

    def test_parametric_gate_emits_identifier(self):
        circuit = QuantumCircuit(1)
        circuit.rx(Parameter("theta"), 0)
        body = dag_to_qasm_body(circuit_to_dag(circuit))
        assert "rx(theta) q[0];" in body

    def test_numeric_gate_uses_precision(self):
        circuit = QuantumCircuit(1)
        circuit.rx(0.123456789, 0)
        dag = circuit_to_dag(circuit)
        assert "rx(0.123) q[0];" in dag_to_qasm_body(dag, precision=3)
        assert "rx(0.12346) q[0];" in dag_to_qasm_body(dag, precision=5)

    def test_cnot_emits_two_qubit_args(self):
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        assert "cx q[0],q[1];" in dag_to_qasm_body(circuit_to_dag(circuit))
