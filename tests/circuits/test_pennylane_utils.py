# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.circuits._pennylane_utils."""

import pennylane as qp
import pytest
import sympy
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP

from divi.circuits._pennylane_utils import (
    _detect_batch_input_argnames,
    _qnode_to_symbolic_qscript,
    _validate_single_measurement,
)


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
