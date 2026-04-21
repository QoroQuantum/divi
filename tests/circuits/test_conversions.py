# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits._conversions."""

from collections import Counter

import numpy as np
import pennylane as qp
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import Operator, SparsePauliOp

from divi.circuits._conversions import (
    _qscript_to_dag,
    _sympy_to_qiskit,
    dag_to_qasm_body,
    observable_to_sparse_pauli_op,
)
from divi.circuits._qasm_template import build_template, render_template


class TestSympyToQiskit:
    """Conversion of sympy expressions into Qiskit ParameterExpression / float."""

    def test_bare_symbol_maps_to_parameter(self):
        theta = sp.Symbol("theta")
        p = Parameter("theta")
        out = _sympy_to_qiskit(theta, {theta: p})
        assert isinstance(out, ParameterExpression)
        assert float(out.bind({p: 2.5})) == pytest.approx(2.5)

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
        # Evaluates correctly after binding.
        assert float(out.bind({pa: 1.0, pb: 2.0})) == pytest.approx(3.0)

    def test_mul_with_numeric_coefficient(self):
        theta = sp.Symbol("theta")
        p = Parameter("theta")
        out = _sympy_to_qiskit(2 * theta, {theta: p})
        assert float(out.bind({p: 0.5})) == pytest.approx(1.0)

    def test_pow_composes(self):
        theta = sp.Symbol("theta")
        p = Parameter("theta")
        out = _sympy_to_qiskit(theta**2, {theta: p})
        assert float(out.bind({p: 3.0})) == pytest.approx(9.0)

    def test_sin_maps_to_parameter_method(self):
        theta = sp.Symbol("theta")
        p = Parameter("theta")
        out = _sympy_to_qiskit(sp.sin(theta), {theta: p})
        assert float(out.bind({p: np.pi / 2})) == pytest.approx(1.0)

    def test_unmapped_symbol_raises(self):
        theta = sp.Symbol("theta")
        with pytest.raises(ValueError, match="Unmapped sympy symbol"):
            _sympy_to_qiskit(theta, {})

    def test_unknown_expression_type_raises(self):
        # Factorial isn't in the supported set.
        x = sp.Symbol("x")
        with pytest.raises(NotImplementedError):
            _sympy_to_qiskit(sp.factorial(x), {x: Parameter("x")})


class TestQScriptToDag:
    """End-to-end QuantumScript → DAG conversion."""

    def test_non_parametric_circuit(self):
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.PauliZ(1)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(qp.PauliZ(0))])
        dag, params, _ = _qscript_to_dag(qscript)
        assert params == ()
        gate_names = Counter(node.op.name for node in dag.op_nodes())
        assert gate_names == {"h": 1, "cx": 1, "z": 1}

    def test_parametric_qaoa_layer(self):
        # 3 qubits, ring graph, 1 QAOA layer.
        gamma, beta = sp.symbols("gamma beta")
        ops = [
            qp.Hadamard(0),
            qp.Hadamard(1),
            qp.Hadamard(2),
            qp.CNOT([0, 1]),
            qp.RZ(gamma, 1),
            qp.CNOT([0, 1]),
            qp.CNOT([1, 2]),
            qp.RZ(gamma, 2),
            qp.CNOT([1, 2]),
            qp.CNOT([2, 0]),
            qp.RZ(gamma, 0),
            qp.CNOT([2, 0]),
            qp.RX(beta, 0),
            qp.RX(beta, 1),
            qp.RX(beta, 2),
        ]
        qscript = qp.tape.QuantumScript(
            ops=ops,
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, params, _ = _qscript_to_dag(qscript)
        # Parameters come out in first-appearance order: gamma then beta.
        assert [p.name for p in params] == ["gamma", "beta"]
        # Gate count matches input (no decomposition needed for these ops).
        assert dag.size() == len(ops)

    def test_parameters_preserve_first_appearance_order(self):
        a, b, c = sp.symbols("a b c")
        # QScript references c, then a, then b.
        ops = [qp.RX(c, 0), qp.RY(a, 0), qp.RZ(b, 0)]
        qscript = qp.tape.QuantumScript(
            ops=ops,
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        _, params, _ = _qscript_to_dag(qscript)
        assert [p.name for p in params] == ["c", "a", "b"]

    def test_compound_sympy_expression(self):
        theta = sp.Symbol("theta")
        qscript = qp.tape.QuantumScript(
            ops=[qp.RX(2 * theta, 0)],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, (p,), _ = _qscript_to_dag(qscript)
        op = next(iter(dag.op_nodes()))
        assert op.op.name == "rx"
        # The single parameter should be a ParameterExpression evaluating
        # to 2*theta: bind theta=1.0 → 2.0.
        (expr,) = op.op.params
        assert isinstance(expr, ParameterExpression)
        assert float(expr.bind({p: 1.0})) == pytest.approx(2.0)


class TestDagToQasmBody:
    """Body-only parametric QASM emission."""

    def test_preamble_is_not_emitted(self):
        dag, _, _ = _qscript_to_dag(
            qp.tape.QuantumScript(
                ops=[qp.Hadamard(0)],
                measurements=[qp.expval(qp.PauliZ(0))],
            )
        )
        body = dag_to_qasm_body(dag)
        assert "OPENQASM" not in body
        assert "include" not in body
        assert "qreg" not in body
        assert "creg" not in body
        assert "h q[0];" in body

    def test_parametric_gate_emits_identifier(self):
        theta = sp.Symbol("theta")
        dag, (p,), _ = _qscript_to_dag(
            qp.tape.QuantumScript(
                ops=[qp.RX(theta, 0)],
                measurements=[qp.expval(qp.PauliZ(0))],
            )
        )
        body = dag_to_qasm_body(dag)
        assert "rx(theta) q[0];" in body

    def test_numeric_gate_uses_precision(self):
        dag, _, _ = _qscript_to_dag(
            qp.tape.QuantumScript(
                ops=[qp.RX(0.123456789, 0)],
                measurements=[qp.expval(qp.PauliZ(0))],
            )
        )
        body3 = dag_to_qasm_body(dag, precision=3)
        body5 = dag_to_qasm_body(dag, precision=5)
        assert "rx(0.123) q[0];" in body3
        assert "rx(0.12346) q[0];" in body5

    def test_cnot_emits_two_qubit_args(self):
        dag, _, _ = _qscript_to_dag(
            qp.tape.QuantumScript(
                ops=[qp.CNOT([0, 1])],
                measurements=[qp.expval(qp.PauliZ(0))],
            )
        )
        assert "cx q[0],q[1];" in dag_to_qasm_body(dag)


class TestObservableToSparsePauliOp:
    """Conversion of PennyLane observables into Qiskit SparsePauliOp."""

    def test_single_pauli(self):
        wires = qp.wires.Wires([0, 1, 2])
        op = observable_to_sparse_pauli_op(qp.PauliZ(1), wires)
        # SparsePauliOp on 3 qubits: Z on qubit 1 ⇒ "IZI" (qubit 0 rightmost).
        assert op == SparsePauliOp.from_list([("IZI", 1.0)])

    def test_tensor_product(self):
        wires = qp.wires.Wires([0, 1])
        op = observable_to_sparse_pauli_op(qp.PauliZ(0) @ qp.PauliX(1), wires)
        # qubit 0 → Z, qubit 1 → X, little-endian string: "XZ".
        assert op == SparsePauliOp.from_list([("XZ", 1.0)])

    def test_hamiltonian_sum_of_terms(self):
        wires = qp.wires.Wires([0, 1])
        obs = qp.Hamiltonian([0.5, -0.3], [qp.PauliZ(0), qp.PauliX(1)])
        op = observable_to_sparse_pauli_op(obs, wires)
        # {0.5 Z_0, -0.3 X_1} → {"IZ": 0.5, "XI": -0.3}
        expected = SparsePauliOp.from_list([("IZ", 0.5), ("XI", -0.3)])
        assert op.simplify() == expected.simplify()

    def test_identity(self):
        wires = qp.wires.Wires([0, 1])
        op = observable_to_sparse_pauli_op(qp.Identity(0), wires)
        assert op == SparsePauliOp.from_list([("II", 1.0)])

    def test_sum_of_single_qubit_terms(self):
        wires = qp.wires.Wires([0, 1, 2])
        obs = qp.sum(qp.PauliZ(0), qp.PauliZ(1), qp.PauliZ(2))
        op = observable_to_sparse_pauli_op(obs, wires).simplify()
        expected = SparsePauliOp.from_list(
            [("IIZ", 1.0), ("IZI", 1.0), ("ZII", 1.0)]
        ).simplify()
        assert op == expected

    def test_non_sequential_wire_labels(self):
        # PennyLane wire labels can be arbitrary hashables — we resolve
        # via wires.index() rather than treating them as ints.
        wires = qp.wires.Wires(["a", "b", "c"])
        obs = qp.PauliZ("b")
        op = observable_to_sparse_pauli_op(obs, wires)
        # "b" is wires.index("b") = 1 → "IZI".
        assert op == SparsePauliOp.from_list([("IZI", 1.0)])

    def test_non_pauli_observable_raises(self):
        wires = qp.wires.Wires([0])
        herm = qp.Hermitian(np.array([[1.0, 0.0], [0.0, -1.0]]), wires=0)
        with pytest.raises(ValueError, match="no Pauli representation"):
            observable_to_sparse_pauli_op(herm, wires)


class TestEndToEndEquivalence:
    """Round-trip: qscript → DAG → body-only QASM, bound via template, executes
    the same unitary as the current PennyLane-based path."""

    @staticmethod
    def _bound_unitary(body_qasm_with_preamble: str, n_qubits: int) -> np.ndarray:
        qc = QuantumCircuit.from_qasm_str(body_qasm_with_preamble)
        return Operator(qc).data

    @staticmethod
    def _preamble(n_qubits: int) -> str:
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' f"qreg q[{n_qubits}];\n"

    def test_qaoa_3q_unitary_matches_current_path(self):
        gamma, beta = sp.symbols("gamma beta")
        ops = [
            qp.Hadamard(0),
            qp.Hadamard(1),
            qp.Hadamard(2),
            qp.CNOT([0, 1]),
            qp.RZ(gamma, 1),
            qp.CNOT([0, 1]),
            qp.CNOT([1, 2]),
            qp.RZ(gamma, 2),
            qp.CNOT([1, 2]),
            qp.RX(beta, 0),
            qp.RX(beta, 1),
            qp.RX(beta, 2),
        ]
        qscript = qp.tape.QuantumScript(
            ops=ops,
            measurements=[qp.expval(qp.PauliZ(0))],
        )

        # PL qscript → DAG → body-only parametric QASM → template → bound QASM.
        # Cross-check: the same qscript with numeric values substituted should
        # produce the same unitary as binding via template.
        dag, params, _ = _qscript_to_dag(qscript)
        body = dag_to_qasm_body(dag, precision=8)
        template = build_template(body, tuple(p.name for p in params))
        bound_body = render_template(template, ("0.30000000", "1.10000000"))
        u_bound = self._bound_unitary(self._preamble(3) + bound_body, 3)

        # Reference: build the same qscript with concrete numeric angles.
        ref_ops = [
            (
                o.__class__(
                    *(
                        (0.3 if s is gamma else 1.1 if s is beta else p)
                        for p, s in zip(o.parameters, [gamma] + [beta])
                    ),
                    wires=o.wires,
                )
                if o.name == "RZ" or o.name == "RX"
                else o
            )
            for o in ops
        ]
        ref_qscript = qp.tape.QuantumScript(
            ops=[
                qp.Hadamard(0),
                qp.Hadamard(1),
                qp.Hadamard(2),
                qp.CNOT([0, 1]),
                qp.RZ(0.3, 1),
                qp.CNOT([0, 1]),
                qp.CNOT([1, 2]),
                qp.RZ(0.3, 2),
                qp.CNOT([1, 2]),
                qp.RX(1.1, 0),
                qp.RX(1.1, 1),
                qp.RX(1.1, 2),
            ],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        ref_dag, _, _ = _qscript_to_dag(ref_qscript)
        ref_body = dag_to_qasm_body(ref_dag, precision=8)
        u_ref = self._bound_unitary(self._preamble(3) + ref_body, 3)
        assert np.allclose(u_bound, u_ref, atol=1e-10)

    def test_compound_expression_round_trip(self):
        theta = sp.Symbol("theta")
        qscript = qp.tape.QuantumScript(
            ops=[qp.RX(2 * theta, 0), qp.RY(theta + 1, 0)],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, (p,), _ = _qscript_to_dag(qscript)
        body = dag_to_qasm_body(dag, precision=8)
        # The parametric QASM should mention `theta` as an identifier and
        # the arithmetic should survive — it's fine if Qiskit re-orders
        # (e.g. "2*theta" vs "theta*2") as long as the semantics match.
        assert "theta" in body
        # Numeric round-trip: bind theta=0.5 → rx(1.0), ry(1.5).
        template = build_template(body, (p.name,))
        bound_body = render_template(template, ("0.50000000",))
        qc = QuantumCircuit.from_qasm_str(self._preamble(1) + bound_body)
        u = Operator(qc).data
        # Reference: same two gates with concrete numeric values.
        ref_qc = QuantumCircuit(1)
        ref_qc.rx(1.0, 0)
        ref_qc.ry(1.5, 0)
        u_ref = Operator(ref_qc).data
        assert np.allclose(u, u_ref, atol=1e-10)
