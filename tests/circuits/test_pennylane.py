# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Divi's PennyLane circuit adapter."""

from collections import Counter

import numpy as np
import pytest
import sympy
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import Operator, SparsePauliOp

# Precedes the divi import below, which imports PennyLane itself.
qp = pytest.importorskip("pennylane")

from divi.circuits import build_template, dag_to_qasm_body, render_template
from divi.circuits._pennylane import (
    _detect_batch_input_argnames,
    _fresh_symbols,
    _qnode_to_symbolic_qscript,
    _qscript_to_dag,
    _symbolize_trainable_subset,
    _validate_single_measurement,
    qscript_to_meta,
)

CountsMP = qp.measurements.CountsMP
ExpectationMP = qp.measurements.ExpectationMP
ProbabilityMP = qp.measurements.ProbabilityMP


def test_public_pennylane_conversion_exports():
    from divi.circuits import qnode_to_meta, qscript_to_meta

    assert callable(qnode_to_meta)
    assert callable(qscript_to_meta)


class TestQnodeToSymbolicQscript:
    def test_scalar_params_become_sympy_symbols(self):
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(theta, phi):
            qp.RX(theta, wires=0)
            qp.RZ(phi, wires=0)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(circuit)
        assert isinstance(qs, qp.tape.QuantumScript)
        params = qs.get_parameters()
        # Two sympy symbols were created, one per function parameter.
        assert len(params) == 2

    def test_array_param_is_probed(self):
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(params):
            qp.RX(params[0], wires=0)
            qp.RY(params[1], wires=1)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(circuit)
        assert isinstance(qs, qp.tape.QuantumScript)
        assert len(qs.get_parameters()) == 2

    def test_zero_param_qnode(self):
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(circuit)
        assert isinstance(qs, qp.tape.QuantumScript)
        assert len(qs.get_parameters()) == 0

    @pytest.mark.filterwarnings("ignore:Setting shots on device is deprecated")
    def test_device_with_shots_warns(self):
        # divi runs its own backend/shots, so a shot count on the QNode device
        # is ignored — and that should be flagged, not silent.
        dev = qp.device("default.qubit", wires=1, shots=100)

        @qp.qnode(dev)
        def circuit(theta):
            qp.RX(theta, wires=0)
            return qp.expval(qp.Z(0))

        with pytest.warns(UserWarning, match="divi ignores it"):
            _qnode_to_symbolic_qscript(circuit)

    def test_default_valued_param_is_frozen_non_trainable(self):
        # A plain-Python-default argument is non-trainable in PennyLane
        # (requires_grad=False); only the no-default arg should be symbolized
        # and marked trainable. The default value stays baked in.
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(theta, phi=0.5):
            qp.RX(theta, wires=0)
            qp.RZ(phi, wires=0)
            return qp.expval(qp.Z(0))

        # A frozen default angle is surprising, so conversion warns.
        with pytest.warns(UserWarning, match="default-valued QNode parameter"):
            qs = _qnode_to_symbolic_qscript(circuit)
        # Only theta is trainable; phi=0.5 is frozen, matching PennyLane's
        # verdict when the QNode is traced with requires_grad inputs.
        assert qs.trainable_params == [0]
        trainable = qs.get_parameters()
        assert len(trainable) == 1
        full = qs.get_parameters(trainable_only=False)
        assert len(full) == 2
        # The frozen slot is the literal default, not a symbol.
        assert full[1] == pytest.approx(0.5)

    def test_structural_default_hyperparameter_is_respected(self):
        # A structural default like n_layers=2 must keep its int value so the
        # function's control flow (range(n_layers)) works; it is never a gate
        # parameter, so it does not appear in trainable_params at all.
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(weights, n_layers=2):
            for layer in range(n_layers):
                qp.RX(weights[layer], wires=0)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(circuit)
        # 2 layers -> 2 trainable gate params from the array probe.
        assert qs.trainable_params == [0, 1]
        assert len(qs.get_parameters()) == 2

    def test_angle_embedding_template_converts_symbolically(self):
        # A single 1-D-array AngleEmbedding encoder traces symbolically (numpy
        # object array of symbols) and decomposes to one RY symbol per input.
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def circuit(inputs):
            qp.AngleEmbedding(inputs, wires=range(3), rotation="Y")
            return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))

        qs = _qnode_to_symbolic_qscript(circuit)
        assert isinstance(qs, qp.tape.QuantumScript)
        # 3 inputs -> 3 RY gates -> 3 trainable symbols.
        assert len(qs.get_parameters()) == 3

    def test_nonlinear_template_converts_symbolically(self):
        # IQPEmbedding's entangling angle is a product of inputs (x_i * x_j).
        # Symbolic tracing preserves the expression, so it converts — and the
        # product is one of the gate parameters.
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(x):
            qp.IQPEmbedding(x, wires=range(2))
            return qp.expval(qp.Z(0) @ qp.Z(1))

        qs = _qnode_to_symbolic_qscript(circuit)
        param_strs = {str(p) for p in qs.get_parameters()}
        # The nonlinear product survives symbolically.
        assert any("*" in s for s in param_strs), param_strs

    @pytest.mark.parametrize(
        "template",
        [
            # SEL/BEL need a structured (multi-dim) weight shape that can't be
            # inferred from the device wire count alone.
            lambda w: qp.StronglyEntanglingLayers(w, wires=range(3)),
            lambda w: qp.BasicEntanglerLayers(w, wires=range(3)),
        ],
        ids=["StronglyEntanglingLayers", "BasicEntanglerLayers"],
    )
    def test_structured_shape_template_raises_clear_error(self, template):
        # Templates needing a multi-dimensional shape can't be inferred from the
        # wire count; the failure must be a clear shape message, not a leak.
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def circuit(weights):
            template(weights)
            return qp.expval(qp.Z(0))

        with pytest.raises(TypeError, match="couldn't infer the array shape"):
            _qnode_to_symbolic_qscript(circuit)

    def test_arg_shapes_enables_multiarg_structured_conversion(self):
        # With explicit per-arg shapes, a multi-argument template circuit
        # (AngleEmbedding data + StronglyEntanglingLayers weights) converts.
        n = 3
        dev = qp.device("default.qubit", wires=n)

        @qp.qnode(dev)
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            qp.StronglyEntanglingLayers(weights, wires=range(n))
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(
            circuit, arg_shapes={"inputs": (n,), "weights": (1, n, 3)}
        )
        names = [str(p) for p in qs.get_parameters()]
        # 3 data + 9 weight symbols, named by argument; all bare (unwrapped).
        assert sum(s.startswith("inputs__") for s in names) == 3
        assert sum(s.startswith("weights__") for s in names) == 9
        assert all(isinstance(p, sympy.Basic) for p in qs.get_parameters())


class TestDetectBatchInput:
    """Guards the batch_input introspection against the installed PennyLane."""

    def test_detects_single_argnum(self):
        @qp.batch_input(argnum=0)
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(2))
            qp.RY(weights[0], wires=0)
            return qp.expval(qp.Z(0))

        assert _detect_batch_input_argnames(circuit) == ["inputs"]

    def test_detects_multiple_argnums(self):
        @qp.batch_input(argnum=[0, 1])
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(a, b, weights):
            qp.RX(a, wires=0)
            qp.RX(b, wires=1)
            qp.RY(weights, wires=0)
            return qp.expval(qp.Z(0))

        assert _detect_batch_input_argnames(circuit) == ["a", "b"]

    def test_plain_qnode_has_no_batch_input(self):
        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            return qp.expval(qp.Z(0))

        assert _detect_batch_input_argnames(circuit) == []


class TestValidateSingleMeasurement:
    @pytest.fixture
    def expval_script(self):
        return qp.tape.QuantumScript(
            ops=[qp.RX(0.0, wires=0)],
            measurements=[qp.expval(qp.Z(0))],
        )

    @pytest.fixture
    def probs_script(self):
        return qp.tape.QuantumScript(
            ops=[qp.RX(0.0, wires=0)],
            measurements=[qp.probs(wires=0)],
        )

    def test_accepts_allowed_measurement(self, expval_script):
        # Permissive caller — should not raise.
        _validate_single_measurement(
            expval_script,
            allowed=(ProbabilityMP, ExpectationMP, CountsMP),
            caller="PennyLaneSpecStage",
        )

    def test_rejects_disallowed_measurement(self, probs_script):
        # Strict expval-only caller rejects probs — the error names the
        # offending measurement type, not just the caller.
        with pytest.raises(
            ValueError, match=r"CustomVQA requires.*Got:.*ProbabilityMP"
        ):
            _validate_single_measurement(
                probs_script,
                allowed=(ExpectationMP,),
                caller="CustomVQA",
            )

    def test_rejects_no_measurement(self):
        qs = qp.tape.QuantumScript(ops=[qp.RX(0.0, wires=0)], measurements=[])
        with pytest.raises(ValueError, match=r"exactly one measurement.*Got: \[\]"):
            _validate_single_measurement(
                qs, allowed=(ExpectationMP,), caller="CustomVQA"
            )

    def test_rejects_multiple_measurements(self):
        qs = qp.tape.QuantumScript(
            ops=[qp.RX(0.0, wires=0)],
            measurements=[qp.expval(qp.Z(0)), qp.expval(qp.Z(0))],
        )
        with pytest.raises(
            ValueError, match=r"exactly one measurement.*ExpectationMP.*ExpectationMP"
        ):
            _validate_single_measurement(
                qs, allowed=(ExpectationMP,), caller="CustomVQA"
            )

    def test_custom_description_appears_in_error(self, probs_script):
        with pytest.raises(ValueError, match="my-friendly-description"):
            _validate_single_measurement(
                probs_script,
                allowed=(ExpectationMP,),
                caller="X",
                description="my-friendly-description",
            )

    def test_default_description_uses_class_names(self, probs_script):
        with pytest.raises(ValueError, match="ExpectationMP"):
            _validate_single_measurement(
                probs_script, allowed=(ExpectationMP,), caller="X"
            )


class TestSymbolizeTrainableSubset:
    """A proper-subset ``trainable_params`` symbolizes only operation slots."""

    def test_leaves_observable_coefficient_untouched(self):
        """Observable coefficients must never become circuit parameters."""
        ops = [qp.RX(0.1, wires=0), qp.RY(0.2, wires=0)]
        hamiltonian = qp.Hamiltonian([0.7], [qp.Z(0)])
        qs = qp.tape.QuantumScript(ops, [qp.expval(hamiltonian)])
        qs.trainable_params = [2]

        out = _symbolize_trainable_subset(qs)

        assert out.get_parameters(trainable_only=False)[2] == 0.7

    def test_fresh_symbols_avoid_name_collision(self):
        existing = [sp.Symbol("p0") + sp.Symbol("p2")]
        fresh = _fresh_symbols(2, existing)
        names = {symbol.name for symbol in fresh}
        assert names.isdisjoint({"p0", "p2"})
        assert len(names) == 2


class TestQScriptToDag:
    """End-to-end QuantumScript to DAG conversion."""

    def test_non_parametric_circuit(self):
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.PauliZ(1)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(qp.PauliZ(0))])
        dag, params, _ = _qscript_to_dag(qscript)
        assert params == ()
        gate_names = Counter(node.op.name for node in dag.op_nodes())
        assert gate_names == {"h": 1, "cx": 1, "z": 1}

    def test_parametric_qaoa_layer(self):
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
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(qp.PauliZ(0))])
        dag, params, _ = _qscript_to_dag(qscript)
        assert [param.name for param in params] == ["gamma", "beta"]
        assert dag.size() == len(ops)

    def test_parameters_preserve_first_appearance_order(self):
        a, b, c = sp.symbols("a b c")
        qscript = qp.tape.QuantumScript(
            ops=[qp.RX(c, 0), qp.RY(a, 0), qp.RZ(b, 0)],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        _, params, _ = _qscript_to_dag(qscript)
        assert [param.name for param in params] == ["c", "a", "b"]

    def test_compound_sympy_expression(self):
        theta = sp.Symbol("theta")
        qscript = qp.tape.QuantumScript(
            ops=[qp.RX(2 * theta, 0)],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, (param,), _ = _qscript_to_dag(qscript)
        operation = next(iter(dag.op_nodes()))
        assert operation.op.name == "rx"
        (expression,) = operation.op.params
        assert isinstance(expression, ParameterExpression)
        assert float(expression.bind({param: 1.0})) == pytest.approx(2.0)


class TestEndToEndEquivalence:
    """PennyLane conversion and QASM binding preserve circuit semantics."""

    @staticmethod
    def _bound_unitary(body_qasm_with_preamble: str) -> np.ndarray:
        return Operator(QuantumCircuit.from_qasm_str(body_qasm_with_preamble)).data

    @staticmethod
    def _preamble(n_qubits: int) -> str:
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' f"qreg q[{n_qubits}];\n"

    def test_qaoa_3q_unitary_matches_numeric_conversion(self):
        gamma, beta = sp.symbols("gamma beta")
        qscript = qp.tape.QuantumScript(
            ops=[
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
            ],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, params, _ = _qscript_to_dag(qscript)
        body = dag_to_qasm_body(dag, precision=8)
        template = build_template(body, tuple(param.name for param in params))
        bound_body = render_template(template, ("0.30000000", "1.10000000"))
        actual = self._bound_unitary(self._preamble(3) + bound_body)

        reference = qp.tape.QuantumScript(
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
        reference_dag, _, _ = _qscript_to_dag(reference)
        expected = self._bound_unitary(
            self._preamble(3) + dag_to_qasm_body(reference_dag, precision=8)
        )
        assert np.allclose(actual, expected, atol=1e-10)

    def test_compound_expression_round_trip(self):
        theta = sp.Symbol("theta")
        qscript = qp.tape.QuantumScript(
            ops=[qp.RX(2 * theta, 0), qp.RY(theta + 1, 0)],
            measurements=[qp.expval(qp.PauliZ(0))],
        )
        dag, (param,), _ = _qscript_to_dag(qscript)
        body = dag_to_qasm_body(dag, precision=8)
        assert "theta" in body
        template = build_template(body, (param.name,))
        bound_body = render_template(template, ("0.50000000",))
        actual = self._bound_unitary(self._preamble(1) + bound_body)
        reference = QuantumCircuit(1)
        reference.rx(1.0, 0)
        reference.ry(1.5, 0)
        assert np.allclose(actual, Operator(reference).data, atol=1e-10)


class TestQscriptToMetaObservable:
    """``MetaCircuit.observable`` reflects the QuantumScript measurement shape."""

    def test_single_expval_yields_length_one_tuple(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0)], measurements=[qp.expval(qp.PauliZ(0))]
        )
        meta = qscript_to_meta(script)
        assert isinstance(meta.observable, tuple)
        assert len(meta.observable) == 1
        assert isinstance(meta.observable[0], SparsePauliOp)
        assert meta.measured_wires is None

    def test_two_expvals_yields_tuple_of_sparse_pauli_ops(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT([0, 1])],
            measurements=[
                qp.expval(qp.PauliZ(0)),
                qp.expval(qp.PauliZ(0) @ qp.PauliZ(1)),
            ],
        )
        meta = qscript_to_meta(script)
        assert isinstance(meta.observable, tuple)
        assert len(meta.observable) == 2
        assert all(isinstance(obs, SparsePauliOp) for obs in meta.observable)
        assert meta.measured_wires is None

    def test_three_expvals_preserve_order(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT([0, 1])],
            measurements=[
                qp.expval(qp.PauliX(0)),
                qp.expval(qp.PauliY(0)),
                qp.expval(qp.PauliZ(0)),
            ],
        )
        meta = qscript_to_meta(script)
        assert [str(obs.paulis.to_labels()[0]) for obs in meta.observable] == [
            "IX",
            "IY",
            "IZ",
        ]

    def test_mixing_multi_expval_with_probs_raises(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0)],
            measurements=[
                qp.expval(qp.PauliZ(0)),
                qp.expval(qp.PauliX(0)),
                qp.probs(wires=[0]),
            ],
        )
        with pytest.raises(ValueError, match="mixing"):
            qscript_to_meta(script)

    def test_single_probs_yields_measured_wires_no_observable(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0)], measurements=[qp.probs(wires=[0])]
        )
        meta = qscript_to_meta(script)
        assert meta.observable is None
        assert meta.measured_wires == (0,)

    def test_no_measurement_yields_no_observable(self):
        meta = qscript_to_meta(
            qp.tape.QuantumScript(ops=[qp.Hadamard(0)], measurements=[])
        )
        assert meta.observable is None
        assert meta.measured_wires is None
