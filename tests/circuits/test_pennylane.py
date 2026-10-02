# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Divi's PennyLane circuit adapter."""

import sys
from collections import Counter

import numpy as np
import pytest
import sympy
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator

# Precedes the divi import below, which imports PennyLane itself.
qp = pytest.importorskip("pennylane")

import divi.circuits
from divi.circuits import build_template, dag_to_qasm_body, render_template
from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.circuits._pennylane import (
    _PL_TO_QISKIT_GATE,
    _SHAPE_HINT,
    _batch_input_argnums,
    _fresh_symbols,
    _qnode_to_symbolic_qscript,
    _qscript_to_dag,
    _symbol_arg_name,
    _symbolize_trainable_ops,
    _symbolize_trainable_subset,
    _validate_expectation_measurement,
    _validate_single_measurement,
    qnode_to_meta,
    qscript_to_meta,
)
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import CircuitSpecStage, MeasurementStage
from tests._helpers import exact_match

CountsMP = qp.measurements.CountsMP
ExpectationMP = qp.measurements.ExpectationMP
ProbabilityMP = qp.measurements.ProbabilityMP

_DEFAULT_BAKED_IN_WARNING = (
    "A default-valued QNode parameter was baked into a gate as a fixed constant "
    "and will not be trained. Remove the default (pass the value as a required "
    "argument) to make it trainable."
)
_ARRAY_PARAMETER_ERROR = "Failed to convert QNode with array parameter. " + _SHAPE_HINT


def _gate_list(dag):
    return [
        (node.op.name, tuple(dag.find_bit(qubit).index for qubit in node.qargs))
        for node in dag.topological_op_nodes()
    ]


def _expval_script(ops, observable=None):
    return qp.tape.QuantumScript(
        ops, [qp.expval(qp.Z(0) if observable is None else observable)]
    )


def _observable_free_expval_script():
    return qp.tape.QuantumScript(
        [qp.RX(0.1, wires=0)], [ExpectationMP(wires=qp.wires.Wires([0]))]
    )


def _qnode(n_wires, func):
    return qp.qnode(qp.device("default.qubit", wires=n_wires))(func)


def _rx_with_default_rz(theta, phi=0.5):
    qp.RX(theta, wires=0)
    qp.RZ(phi, wires=0)
    return qp.expval(qp.Z(0))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs skip_file_prefixes")
@pytest.mark.parametrize(
    "convert",
    [
        _qnode_to_symbolic_qscript,
        qnode_to_meta,
        lambda qnode: qnode_to_meta(qnode, arg_shapes={"theta": ()}),
    ],
    ids=["symbolic-qscript", "qnode-to-meta", "arg-shapes"],
)
def test_default_valued_warning_points_at_the_caller(convert):
    with pytest.warns(
        UserWarning, match=exact_match(_DEFAULT_BAKED_IN_WARNING)
    ) as record:
        convert(_qnode(1, _rx_with_default_rz))
    assert [w.filename for w in record] == [__file__]


def _flat_array_with_default(weights, phi=0.5):
    qp.RX(weights[0], wires=0)
    qp.RZ(phi, wires=0)
    return qp.expval(qp.Z(0))


def _template_with_default(x, phi=0.5):
    qp.AngleEmbedding(x, wires=range(2))
    qp.RZ(phi, wires=0)
    return qp.expval(qp.Z(0))


def _shaped_inputs_with_default(inputs, phi=0.5):
    qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
    qp.RZ(phi, wires=0)
    return qp.expval(qp.Z(0))


def _scalar_with_constant(theta):
    qp.RX(theta, wires=0)
    qp.RZ(0.5, wires=0)
    return qp.expval(qp.Z(0))


def _shaped_inputs_with_constant(inputs):
    qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
    qp.RZ(0.5, wires=0)
    return qp.expval(qp.Z(0))


def _shaped_inputs_then_rx(inputs, theta):
    qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
    qp.RX(theta, wires=0)
    return qp.expval(qp.Z(0))


def _layers_with_structural_default(weights, n_layers=2):
    for layer in range(n_layers):
        qp.RX(weights[layer], wires=0)
    return qp.expval(qp.Z(0))


def _indexed_weights(weights):
    qp.RX(weights[0], wires=0)
    qp.RY(weights[1], wires=1)
    return qp.expval(qp.Z(0))


def _iterated_weights(weights):
    for weight in weights:
        qp.RX(weight, wires=0)
    return qp.expval(qp.Z(0))


def _single_wire_embedding(x):
    qp.AngleEmbedding(x, wires=[0])
    return qp.expval(qp.Z(0))


def _three_wire_embedding(inputs):
    qp.AngleEmbedding(inputs, wires=range(3), rotation="Y")
    return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))


@pytest.mark.parametrize(
    "name,implementation",
    [("qnode_to_meta", qnode_to_meta), ("qscript_to_meta", qscript_to_meta)],
)
def test_public_pennylane_conversion_exports(name, implementation):
    assert getattr(divi.circuits, name) is implementation


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

    def test_zero_param_qnode(self):
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(circuit)
        assert isinstance(qs, qp.tape.QuantumScript)
        assert len(qs.get_parameters()) == 0

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

        with pytest.raises(
            TypeError, match=exact_match(_ARRAY_PARAMETER_ERROR)
        ) as excinfo:
            _qnode_to_symbolic_qscript(circuit)
        assert excinfo.value.__cause__ is not None

    def test_multiple_array_parameters_raise(self):
        def circuit(a, b):
            qp.RX(a[0], wires=0)
            qp.RY(b[0], wires=0)
            return qp.expval(qp.Z(0))

        with pytest.raises(
            TypeError,
            match=exact_match(
                "Failed to convert QNode — the function appears to use array "
                "parameters or numpy operations on its arguments. QNodes with "
                "multiple array parameters are not supported. Pass a "
                "QuantumScript with explicit sympy symbols instead."
            ),
        ) as excinfo:
            _qnode_to_symbolic_qscript(_qnode(1, circuit))
        assert excinfo.value.__cause__ is not None

    def test_arg_shapes_trace_failure_raises_clear_error(self):
        def circuit(weights):
            qp.StronglyEntanglingLayers(weights, wires=range(2))
            return qp.expval(qp.Z(0))

        with pytest.raises(
            TypeError, match=exact_match("Failed to convert QNode. " + _SHAPE_HINT)
        ) as excinfo:
            _qnode_to_symbolic_qscript(_qnode(2, circuit), arg_shapes={})
        assert excinfo.value.__cause__ is not None

    @pytest.mark.parametrize(
        "func,n_wires,names",
        [
            (_indexed_weights, 2, ["p0", "p1"]),
            (_iterated_weights, 2, ["p0", "p1"]),
            (_single_wire_embedding, 1, ["p0"]),
            (_three_wire_embedding, 3, ["p0", "p1", "p2"]),
        ],
        ids=[
            "indexed-flat-array",
            "iterated-array-sized-by-wires",
            "single-wire-template",
            "template",
        ],
    )
    def test_array_argument_gets_one_symbol_per_slot(self, func, n_wires, names):
        qs = _qnode_to_symbolic_qscript(_qnode(n_wires, func))
        assert [str(p) for p in qs.get_parameters()] == names

    def test_template_trace_without_gate_parameters_raises(self):
        def circuit(x):
            if len(x):
                qp.Hadamard(wires=0)
            return qp.expval(qp.Z(0))

        with pytest.raises(TypeError, match=exact_match(_ARRAY_PARAMETER_ERROR)):
            _qnode_to_symbolic_qscript(_qnode(2, circuit))

    @pytest.mark.parametrize(
        "func,arg_shapes,trainable_params",
        [
            (_rx_with_default_rz, None, [0]),
            (_flat_array_with_default, None, [0]),
            (_template_with_default, None, [0, 1]),
            (_shaped_inputs_with_default, {"inputs": (2,)}, [0, 1]),
        ],
        ids=["scalar", "flat-array", "template", "arg-shapes"],
    )
    def test_baked_in_default_warns_on_every_path(
        self, func, arg_shapes, trainable_params
    ):
        with pytest.warns(UserWarning, match=exact_match(_DEFAULT_BAKED_IN_WARNING)):
            qs = _qnode_to_symbolic_qscript(_qnode(2, func), arg_shapes=arg_shapes)
        assert qs.trainable_params == trainable_params
        assert qs.get_parameters(trainable_only=False)[-1] == pytest.approx(0.5)

    @pytest.mark.filterwarnings("error::UserWarning")
    @pytest.mark.parametrize(
        "func,arg_shapes,trainable_params",
        [
            (_scalar_with_constant, None, [0]),
            (_shaped_inputs_with_constant, {"inputs": (2,)}, [0, 1]),
            (_layers_with_structural_default, None, [0, 1]),
        ],
        ids=["scalar", "arg-shapes", "structural-default"],
    )
    def test_literal_constant_without_default_is_silent(
        self, func, arg_shapes, trainable_params
    ):
        qs = _qnode_to_symbolic_qscript(_qnode(2, func), arg_shapes=arg_shapes)
        assert qs.trainable_params == trainable_params

    def test_arg_shapes_unwraps_mixed_operation_data(self):
        def circuit(w):
            qp.U3(w[..., 0], w[1], 0.3, wires=0)
            return qp.expval(qp.Z(0))

        qs = _qnode_to_symbolic_qscript(_qnode(1, circuit), arg_shapes={"w": (2,)})
        (operation,) = qs.operations
        assert [str(d) for d in operation.data] == ["w__0", "w__1", "0.3"]
        assert not any(isinstance(d, np.ndarray) for d in operation.data)

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
        assert all(op.name in _PL_TO_QISKIT_GATE for op in qs.operations)


def _three_params_on_two_wires(w):
    qp.RX(w[0], wires=0)
    qp.RY(w[1], wires=1)
    qp.RZ(w[2], wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.Z(0))


def _three_params_on_two_wires_with_default(w, c=0.3):
    return _three_params_on_two_wires(w)


@pytest.mark.parametrize(
    "func", [_three_params_on_two_wires, _three_params_on_two_wires_with_default]
)
def test_qnode_to_meta_counts_indexed_params_beyond_wire_count(func):
    assert len(qnode_to_meta(_qnode(2, func)).parameters) == 3


@pytest.mark.parametrize(
    "angle,expected_params",
    [
        pytest.param(np.array([sp.Symbol("x")], dtype=object), ["x"], id="symbolic"),
        pytest.param(np.array([0.3]), [], id="numeric"),
    ],
)
def test_qscript_to_dag_collapses_a_broadcast_of_one(angle, expected_params):
    qs = qp.tape.QuantumScript([qp.RX(angle, wires=0)], [qp.expval(qp.Z(0))])
    dag, params = _qscript_to_dag(qs)
    assert _gate_list(dag) == [("rx", (0,))]
    assert [p.name for p in params] == expected_params


def test_qnode_to_meta_traces_with_arg_shapes_and_precision():
    meta = qnode_to_meta(
        _qnode(2, _shaped_inputs_then_rx), arg_shapes={"inputs": (2,)}, precision=5
    )
    assert [p.name for p in meta.parameters] == ["inputs__0", "inputs__1", "theta__0"]
    assert meta.precision == 5


def test_symbol_arg_name_strips_only_the_trailing_index():
    assert _symbol_arg_name("my__arg__3") == "my__arg"


class TestDetectBatchInput:
    """Guards the batch_input introspection against the installed PennyLane."""

    def test_detects_single_argnum(self):
        @qp.batch_input(argnum=0)
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(inputs, weights):
            qp.AngleEmbedding(inputs, wires=range(2))
            qp.RY(weights[0], wires=0)
            return qp.expval(qp.Z(0))

        assert _batch_input_argnums(circuit) == [0]

    def test_detects_multiple_argnums(self):
        @qp.batch_input(argnum=[0, 1])
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(a, b, weights):
            qp.RX(a, wires=0)
            qp.RX(b, wires=1)
            qp.RY(weights, wires=0)
            return qp.expval(qp.Z(0))

        assert _batch_input_argnums(circuit) == [0, 1]

    def test_plain_qnode_has_no_batch_input(self):
        @qp.qnode(qp.device("default.qubit", wires=1))
        def circuit(theta):
            qp.RX(theta, wires=0)
            return qp.expval(qp.Z(0))

        assert _batch_input_argnums(circuit) == []

    def test_detects_batch_input_after_another_transform(self):
        @qp.batch_input(argnum=0)
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(inputs, weights):
            qp.RX(inputs, wires=0)
            qp.RY(weights, wires=0)
            return qp.expval(qp.Z(0))

        assert _batch_input_argnums(circuit) == [0]

    def test_detects_positional_argnum(self):
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(inputs, weights):
            qp.RX(inputs, wires=0)
            qp.RY(weights, wires=0)
            return qp.expval(qp.Z(0))

        assert _batch_input_argnums(qp.batch_input(circuit, 1)) == [1]


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

    @pytest.mark.parametrize(
        "measurements,allowed,expected",
        [
            (
                [qp.probs(wires=0)],
                (ExpectationMP,),
                "ExpectationMP. Got: ['ProbabilityMP']",
            ),
            ([], (ExpectationMP,), "ExpectationMP. Got: []"),
            (
                [qp.expval(qp.Z(0)), qp.expval(qp.Z(0))],
                (ExpectationMP,),
                "ExpectationMP. Got: ['ExpectationMP', 'ExpectationMP']",
            ),
            (
                [qp.counts(wires=0)],
                (ProbabilityMP, ExpectationMP),
                "ProbabilityMP, ExpectationMP. Got: ['CountsMP']",
            ),
        ],
        ids=["disallowed", "none", "multiple", "default-description-lists-allowed"],
    )
    def test_rejects_anything_but_one_allowed_measurement(
        self, measurements, allowed, expected
    ):
        qs = qp.tape.QuantumScript(ops=[qp.RX(0.0, wires=0)], measurements=measurements)
        with pytest.raises(
            ValueError,
            match=exact_match(
                f"CustomVQA requires exactly one measurement of type {expected}"
            ),
        ):
            _validate_single_measurement(qs, allowed=allowed, caller="CustomVQA")

    def test_custom_description_appears_in_error(self, probs_script):
        with pytest.raises(ValueError, match="my-friendly-description"):
            _validate_single_measurement(
                probs_script,
                allowed=(ExpectationMP,),
                caller="X",
                description="my-friendly-description",
            )


@pytest.mark.parametrize(
    "script,message",
    [
        (
            qp.tape.QuantumScript([qp.RX(0.1, wires=0)], [qp.probs(wires=0)]),
            "CustomVQA requires exactly one measurement of type "
            "expectation-value (expval()). Got: ['ProbabilityMP']",
        ),
        (
            _observable_free_expval_script(),
            "CustomVQA requires the QuantumScript's expectation-value measurement "
            "to declare an observable; got expval() with obs=None.",
        ),
    ],
    ids=["non-expval", "no-observable"],
)
def test_validate_expectation_measurement_rejects(script, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        _validate_expectation_measurement(script, caller="CustomVQA")


def test_symbolize_trainable_ops_binds_only_the_trainable_subset():
    qs = _expval_script([qp.RX(0.1, wires=0), qp.RY(0.2, wires=0)])
    qs.trainable_params = [1]

    out = _symbolize_trainable_ops(qs)

    assert [str(p) for p in out.get_parameters(trainable_only=False)] == [
        "0.1",
        "p0",
    ]
    assert out.trainable_params == [1]


def test_symbolize_trainable_ops_rejects_observable_only_indices():
    qs = _expval_script([qp.RX(0.1, wires=0)], qp.Hamiltonian([0.7], [qp.Z(0)]))
    qs.trainable_params = [1]

    with pytest.raises(
        ValueError,
        match=exact_match(
            "QuantumScript's trainable_params point only at observable "
            "coefficients; CustomVQA only trains operation parameters. "
            "Remove observable-coefficient indices from qs.trainable_params."
        ),
    ):
        _symbolize_trainable_ops(qs)


class TestSymbolizeTrainableSubset:
    """A proper-subset ``trainable_params`` symbolises only operation slots."""

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

    @pytest.mark.parametrize(
        "n_symbols,existing,expected",
        [
            (2, [], ["p0", "p1"]),
            (
                3,
                [sp.Symbol("p1"), sp.Symbol("p2"), Parameter("p4"), Parameter("p5")],
                ["p0", "p3", "p6"],
            ),
        ],
        ids=["nothing-taken", "sympy-and-qiskit-taken"],
    )
    def test_fresh_symbols_take_lowest_free_names(self, n_symbols, existing, expected):
        assert [s.name for s in _fresh_symbols(n_symbols, existing)] == expected


class TestQScriptToDag:
    """End-to-end QuantumScript to DAG conversion."""

    def test_non_parametric_circuit(self):
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.PauliZ(1)]
        qscript = qp.tape.QuantumScript(ops=ops, measurements=[qp.expval(qp.PauliZ(0))])
        dag, params = _qscript_to_dag(qscript)
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
        dag, params = _qscript_to_dag(qscript)
        assert [param.name for param in params] == ["gamma", "beta"]
        assert dag.size() == len(ops)

    def test_qiskit_parameters_are_deduplicated_in_first_appearance_order(self):
        a, b = Parameter("a"), Parameter("b")
        dag, params = _qscript_to_dag(
            _expval_script([qp.RX(b, 0), qp.RY(a, 0), qp.RZ(a + b, 0)])
        )
        assert params == (b, a)
        assert [name for name, _ in _gate_list(dag)] == ["rx", "ry", "rz"]

    @pytest.mark.parametrize(
        "ops,observable,expected",
        [
            (
                [qp.Hadamard("a"), qp.CNOT(["a", "b"])],
                qp.Z("b"),
                [("h", (0,)), ("cx", (0, 1))],
            ),
            (
                [qp.Hadamard(0), qp.CNOT([0, 2])],
                qp.Z(2),
                [("h", (0,)), ("cx", (0, 1))],
            ),
            (
                [qp.Hadamard(1), qp.CNOT([1, 0])],
                qp.Z(0),
                [("h", (1,)), ("cx", (1, 0))],
            ),
        ],
        ids=["string-labels", "non-contiguous", "contiguous-out-of-order"],
    )
    def test_wires_map_onto_a_compact_register(self, ops, observable, expected):
        dag, _ = _qscript_to_dag(_expval_script(ops, observable))
        assert dag.num_qubits() == 2
        assert _gate_list(dag) == expected

    def test_unsupported_operations_are_decomposed(self):
        dag, _ = _qscript_to_dag(
            _expval_script(
                [qp.Rot(0.1, 0.2, 0.3, wires=0), qp.IsingZZ(0.5, wires=[0, 1])]
            )
        )
        assert _gate_list(dag) == [
            ("rz", (0,)),
            ("ry", (0,)),
            ("rz", (0,)),
            ("cx", (0, 1)),
            ("rz", (1,)),
            ("cx", (0, 1)),
        ]

    @pytest.mark.parametrize(
        "operation",
        [
            qp.QubitUnitary(np.array([[0, 1], [1, 0]]), wires=0),
            qp.QubitUnitary(np.eye(4)[[0, 1, 3, 2]], wires=[0, 1]),
            qp.StatePrep(np.array([0, 1, 0, 0]), wires=[0, 1]),
        ],
        ids=["one-qubit-unitary", "two-qubit-unitary", "state-prep"],
    )
    def test_matrix_operations_lower_to_the_qasm2_basis(self, operation):
        dag, _ = _qscript_to_dag(_expval_script([operation]))
        names = [name for name, _ in _gate_list(dag)]
        assert names
        assert set(names) <= set(_QISKIT_TO_QASM2)

    @pytest.mark.parametrize(
        "matrix,wires",
        [
            (np.array([[0, 1], [1, 0]]), [0]),
            (np.eye(4)[[0, 1, 3, 2]], [0, 1]),
        ],
        ids=["one-qubit", "two-qubit"],
    )
    def test_matrix_unitaries_keep_their_action(self, matrix, wires):
        dag, _ = _qscript_to_dag(_expval_script([qp.QubitUnitary(matrix, wires=wires)]))
        # PennyLane orders wire 0 as the most significant bit; Qiskit as the least.
        expected = Operator(matrix).reverse_qargs()
        assert Operator(dag_to_circuit(dag)).equiv(expected)

    def test_adjacent_inverse_gates_are_kept(self):
        dag, _ = _qscript_to_dag(_expval_script([qp.Hadamard(0), qp.Hadamard(0)]))
        assert _gate_list(dag) == [("h", (0,)), ("h", (0,))]


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
        dag, params = _qscript_to_dag(qscript)
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
        reference_dag, _ = _qscript_to_dag(reference)
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
        dag, (param,) = _qscript_to_dag(qscript)
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

    @pytest.mark.parametrize(
        "observables,labels",
        [
            ([qp.PauliZ(0)], ["IZ"]),
            ([qp.PauliZ(0), qp.PauliZ(0) @ qp.PauliZ(1)], ["IZ", "ZZ"]),
            ([qp.PauliX(0), qp.PauliY(0), qp.PauliZ(0)], ["IX", "IY", "IZ"]),
        ],
        ids=["one", "two", "three-in-order"],
    )
    def test_expvals_become_a_tuple_of_observables_in_order(self, observables, labels):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT([0, 1])],
            measurements=[qp.expval(obs) for obs in observables],
        )
        meta = qscript_to_meta(script)
        assert isinstance(meta.observable, tuple)
        assert [obs.paulis.to_labels() for obs in meta.observable] == [
            [label] for label in labels
        ]
        assert meta.measured_wires is None

    def test_mixing_multi_expval_with_probs_raises(self):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0)],
            measurements=[
                qp.expval(qp.PauliZ(0)),
                qp.expval(qp.PauliX(0)),
                qp.probs(wires=[0]),
            ],
        )
        with pytest.raises(
            ValueError,
            match=exact_match(
                "qscript_to_meta: mixing `expval` with `probs`/`counts` "
                "measurements in a single QuantumScript is not supported."
            ),
        ):
            qscript_to_meta(script)

    def test_expval_without_observable_raises(self):
        with pytest.raises(
            ValueError,
            match=exact_match("ExpectationMP without an observable is not supported."),
        ):
            qscript_to_meta(_observable_free_expval_script())

    @pytest.mark.parametrize(
        "measurement,measured_wires",
        [
            (qp.probs(wires=[0]), (0,)),
            (qp.probs(wires=[1]), (1,)),
            (qp.probs(), (0, 1)),
        ],
        ids=["first-wire", "explicit-subset", "all-wires"],
    )
    def test_probs_measured_wires(self, measurement, measured_wires):
        script = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), qp.CNOT([0, 1])], measurements=[measurement]
        )
        meta = qscript_to_meta(script)
        assert meta.observable is None
        assert meta.measured_wires == measured_wires

    @pytest.mark.parametrize(
        "was_multi_obs,expected",
        [(None, 0.0), (True, [0.0])],
        ids=["inferred", "explicit"],
    )
    def test_single_expval_result_shape_follows_multi_obs_flag(
        self, default_test_simulator, was_multi_obs, expected
    ):
        meta = qscript_to_meta(
            _expval_script([qp.Hadamard(0)]), was_multi_obs=was_multi_obs
        )
        result = CircuitPipeline(stages=[CircuitSpecStage(), MeasurementStage()]).run(
            meta, PipelineEnv(backend=default_test_simulator)
        )
        assert result.value == pytest.approx(expected, abs=1e-9)
        assert isinstance(result.value, type(expected))

    def test_explicit_parameter_order_is_kept(self):
        a, b = sp.symbols("a b")
        order = (Parameter("b"), Parameter("a"))
        meta = qscript_to_meta(
            _expval_script([qp.RX(a, 0), qp.RY(b, 0)]), parameter_order=order
        )
        assert meta.parameters == order

    def test_precision_is_forwarded(self):
        assert (
            qscript_to_meta(_expval_script([qp.Hadamard(0)]), precision=4).precision
            == 4
        )

    def test_no_measurement_yields_no_observable(self):
        meta = qscript_to_meta(
            qp.tape.QuantumScript(ops=[qp.Hadamard(0)], measurements=[])
        )
        assert meta.observable is None
        assert meta.measured_wires is None
