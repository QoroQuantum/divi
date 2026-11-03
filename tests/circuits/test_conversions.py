# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PennyLane-free circuit conversion utilities."""

import subprocess
import sys

import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

from divi.circuits import dag_to_qasm_body, measurement_qasms_from_groups
from divi.circuits._conversions import (
    _QISKIT_TO_QASM2,
    _assert_finite,
    _bind_op_params,
    _format_bound_param,
    _format_gate_param,
    _real_float_array,
    _sparse_pauli_op_to_ham_string,
    _sympy_to_qiskit,
    bind_parameters_in_dag,
    lower_to_basis_gates,
)
from tests._helpers import exact_match


@pytest.mark.parametrize("formatter", [_format_bound_param, _format_gate_param])
@pytest.mark.parametrize(
    "value,precision,expected",
    [
        (0.0, 8, "0"),
        (-0.0, 8, "0"),
        (4.9e-9, 8, "4.9e-09"),
        (2.74e-8, 8, "2.74e-08"),
        (-1.5e-10, 8, "-1.5e-10"),
        (0.0123456789, 8, "0.012345679"),
        (np.pi, 8, "3.14159265"),
        (1234567.891, 8, "1234567.891"),
        (1.5, 10, "1.5"),
        (3.0, 10, "3"),
        (-2.5, 10, "-2.5"),
        (0.001, 10, "0.001"),
        (1e-20, 10, "1e-20"),
        (1.123456789, 4, "1.1235"),
        (0.0, 0, "0"),
        (10.0, 0, "10"),
        (100.0, 0, "100"),
        (2.6, 0, "3"),
        (-0.2, 0, "0"),
    ],
)
def test_numeric_angles_render_in_the_one_qasm_format(
    formatter, value, precision, expected
):
    assert formatter(value, precision) == expected


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


_THETA = sp.Symbol("theta")


class TestSympyToQiskit:
    @pytest.mark.parametrize(
        "expr, bind, expected",
        [
            pytest.param(_THETA, 2.5, 2.5, id="bare_symbol"),
            pytest.param(2 * _THETA, 0.5, 1.0, id="mul_numeric_coefficient"),
            pytest.param(_THETA**2, 3.0, 9.0, id="pow"),
            pytest.param(sp.sin(_THETA), np.pi / 2, 1.0, id="sin"),
        ],
    )
    def test_single_symbol_expression_maps_to_parameter_expression(
        self, expr, bind, expected
    ):
        parameter = Parameter("theta")
        out = _sympy_to_qiskit(expr, {_THETA: parameter})
        assert isinstance(out, ParameterExpression)
        assert float(out.bind({parameter: bind})) == pytest.approx(expected)

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

    def test_unmapped_symbol_raises(self):
        theta = sp.Symbol("theta")
        with pytest.raises(ValueError, match="Unmapped sympy symbol"):
            _sympy_to_qiskit(theta, {})

    def test_unknown_expression_type_raises(self):
        x = sp.Symbol("x")
        with pytest.raises(
            NotImplementedError, match="Cannot convert sympy expression"
        ):
            _sympy_to_qiskit(sp.factorial(x), {x: Parameter("x")})


def test_bind_parameters_in_dag_binds_only_substituted_params_and_keeps_wiring():
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(2, 2)
    qc.rx(a, 0)
    qc.cx(0, 1)
    qc.ry(b, 1)
    qc.measure(1, 1)

    out = dag_to_circuit(bind_parameters_in_dag(circuit_to_dag(qc), {a: 0.5}))

    assert [
        (
            inst.operation.name,
            [str(p) for p in inst.operation.params],
            [out.find_bit(q).index for q in inst.qubits],
            [out.find_bit(c).index for c in inst.clbits],
        )
        for inst in out.data
    ] == [
        ("rx", ["0.5"], [0], []),
        ("cx", [], [0, 1], []),
        ("ry", ["b"], [1], []),
        ("measure", [], [1], [1]),
    ]


@pytest.mark.parametrize(
    "value",
    [np.float32(0.5), np.int64(1), np.array(0.5), np.float64(0.5)],
    ids=["float32", "int64", "zero-dimensional-array", "float64"],
)
def test_bind_parameters_in_dag_accepts_numpy_values(value):
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    out = dag_to_circuit(bind_parameters_in_dag(circuit_to_dag(qc), {theta: value}))

    assert float(out.data[0].operation.params[0]) == float(value)


class TestRealFloatArray:
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[1, 2], [3, 4]], id="int-lists"),
            pytest.param(np.array([[1, 2], [3, 4]], dtype=np.float32), id="float32"),
            pytest.param(np.array([[1 + 0j, 2], [3, 4]]), id="zero-imaginary-complex"),
            pytest.param(np.array([[1 + 0j, 2], [3, 4]], dtype=np.complex64), id="c64"),
            pytest.param([[1 + 0j, 2], [3, 4]], id="complex-lists"),
        ],
    )
    def test_casts_to_float64(self, values):
        out = _real_float_array(values, source="env.param_sets")

        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, [[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.array([[1 + 1j, 2], [3, 4]]), id="complex-array"),
            pytest.param(np.array([[1j, 2], [3, 4]], dtype=np.complex64), id="c64"),
            pytest.param([[1 + 1j, 2], [3, 4]], id="complex-lists"),
        ],
    )
    def test_rejects_a_non_zero_imaginary_part(self, values):
        with pytest.raises(
            ValueError,
            match=exact_match(
                "env.param_sets must be real; got complex values with a non-zero "
                "imaginary part."
            ),
        ):
            _real_float_array(values, source="env.param_sets")


def test_bind_op_params_returns_the_original_op_for_unrelated_bindings():
    op = RZGate(Parameter("t"))
    assert _bind_op_params(op, {Parameter("u"): 1.0}) is op


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

    @pytest.mark.parametrize(
        "add_instruction,expected",
        [
            pytest.param(
                lambda circuit: circuit.barrier(),
                "Instruction 'barrier' not supported by the QASM body emitter. "
                "`barrier` is emitted by QuantumCircuit.measure_all(); use explicit "
                "`measure(i, i)` on a circuit with a classical register instead.",
                id="barrier",
            ),
            pytest.param(
                lambda circuit: circuit.ryy(0.1, 0, 1),
                "Instruction 'ryy' not supported by the QASM body emitter. "
                "Decompose to basis gates before calling dag_to_qasm_body.",
                id="non-basis-gate",
            ),
        ],
    )
    def test_unsupported_instruction_raises_with_hint(self, add_instruction, expected):
        circuit = QuantumCircuit(2)
        add_instruction(circuit)
        with pytest.raises(ValueError, match=exact_match(expected)):
            dag_to_qasm_body(circuit_to_dag(circuit))


def _gates_outside_the_qasm_basis():
    weights = ParameterVector("w", 2)
    qc = QuantumCircuit(3)
    qc.ryy(weights[0], 0, 1)
    qc.ccx(0, 1, 2)
    qc.cswap(2, 0, 1)
    qc.rzz(weights[1], 1, 2)
    qc.sxdg(0)
    qc.ch(0, 2)
    return qc


def _measured_bell_pair():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _trotter_evolution():
    hamiltonian = SparsePauliOp.from_sparse_list(
        [("XX", [0, 1], 1.0), ("YY", [1, 2], 0.5), ("Z", [0], 0.3)], num_qubits=3
    )
    qc = QuantumCircuit(3)
    qc.append(
        PauliEvolutionGate(
            hamiltonian,
            time=Parameter("t"),
            synthesis=SuzukiTrotter(order=2, reps=2, preserve_order=True),
        ),
        range(3),
    )
    return qc


def _control_flow_and_timing():
    theta = Parameter("θ")
    qc = QuantumCircuit(2, 1, global_phase=theta)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.rzz(theta, 0, 1)
    with qc.for_loop(range(2)):
        qc.sxdg(1)
    qc.barrier()
    qc.delay(10, 0, unit="dt")
    qc.reset(1)
    qc.ryy(2 * theta + 0.5, 0, 1)
    return qc


def _custom_and_controlled_gates():
    inner = QuantumCircuit(2, name="entangle")
    inner.h(0)
    inner.cx(0, 1)
    inner.rzz(0.3, 0, 1)
    gate = inner.to_gate()
    qc = QuantumCircuit(3)
    qc.append(gate, [0, 1])
    qc.append(gate.control(1), [2, 0, 1])
    return qc


class TestLowerToBasisGates:
    @pytest.mark.parametrize(
        "make_circuit",
        [
            _gates_outside_the_qasm_basis,
            _trotter_evolution,
            _measured_bell_pair,
            _control_flow_and_timing,
            _custom_and_controlled_gates,
        ],
        ids=[
            "non-basis gates",
            "Trotter evolution",
            "measurements",
            "control flow and timing",
            "custom and controlled gates",
        ],
    )
    @pytest.mark.parametrize(
        "basis_gates", [tuple(_QISKIT_TO_QASM2), ("u", "cx")], ids=["qasm", "u-cx"]
    )
    def test_matches_transpile_at_optimization_level_zero(
        self, make_circuit, basis_gates
    ):
        circuit = make_circuit()
        expected = transpile(
            circuit, basis_gates=list(basis_gates), optimization_level=0
        )
        lowered = lower_to_basis_gates(circuit, basis_gates)
        assert lowered == expected
        assert lowered.global_phase == expected.global_phase

    def test_does_not_load_qiskits_transpiler_plugins(self):
        pytest.importorskip(
            "qiskit_ibm_runtime",
            reason="the plugin loading this guards against needs qiskit-ibm-runtime",
        )
        script = (
            "import sys\n"
            "from qiskit import QuantumCircuit\n"
            "from divi.circuits._conversions import lower_to_basis_gates\n"
            "qc = QuantumCircuit(3)\n"
            "qc.ccx(0, 1, 2)\n"
            "lower_to_basis_gates(qc)\n"
            "assert 'qiskit_ibm_runtime' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "label,rotation",
    [("X", "h q[0];\n"), ("Y", "sdg q[0];\nh q[0];\n"), ("Z", "")],
    ids=["X", "Y", "Z"],
)
def test_measurement_qasm_diagonalises_each_basis(label, rotation):
    assert measurement_qasms_from_groups(((label,),), 1) == [
        rotation + "measure q[0] -> c[0];\n"
    ]


@pytest.mark.parametrize("label", ["IZ", "ZI"])
def test_measurement_qasm_rejects_label_wider_than_register(label):
    with pytest.raises(
        ValueError,
        match=exact_match(
            f"Pauli label {label!r} spans 2 qubits but the register has only 1."
        ),
    ):
        measurement_qasms_from_groups(((label,),), 1)


def test_sparse_pauli_op_to_ham_string_renders_empty_op_as_empty_string():
    no_terms = np.zeros((0, 2), dtype=bool)
    empty = SparsePauliOp(
        PauliList.from_symplectic(no_terms, no_terms), coeffs=np.zeros(0)
    )
    assert _sparse_pauli_op_to_ham_string(empty) == ""
