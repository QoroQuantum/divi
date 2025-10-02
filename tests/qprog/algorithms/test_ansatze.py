# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest
import sympy

from divi.qprog import (
    GenericLayerAnsatz,
    HardwareEfficientAnsatz,
    HartreeFockAnsatz,
    QAOAAnsatz,
    UCCSDAnsatz,
)


# --- Helper Function ---
def get_circuit_operations(ansatz, test_params, n_qubits, n_layers, **kwargs):
    """Helper to build an ansatz and return its operations list."""

    def circuit(params):
        ansatz.build(params, n_qubits, n_layers, **kwargs)
        return qml.expval(qml.PauliZ(0))

    qscript = qml.tape.make_qscript(circuit)(test_params)
    return qscript.operations


# --- Test GenericLayerAnsatz ---
class TestGenericLayerAnsatz:
    """Tests for the GenericLayerAnsatz class."""

    @pytest.mark.parametrize(
        "gate_sequence, entangler, layout",
        [
            ([qml.RX], qml.CNOT, "linear"),
            ([qml.RY, qml.RZ], qml.CZ, "circular"),
            ([qml.Rot], None, "all_to_all"),
            ([qml.RX], qml.CNOT, [(0, 2), (1, 3)]),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initialization_valid(self, gate_sequence, entangler, layout):
        """Tests that the ansatz can be initialized with valid parameters."""
        try:
            GenericLayerAnsatz(
                gate_sequence=gate_sequence,
                entangler=entangler,
                entangling_layout=layout,
            )
        except (ValueError, TypeError):
            pytest.fail("GenericLayerAnsatz initialization failed with valid inputs.")

    def test_initialization_invalid_gate_sequence_type_error(self):
        """Tests that a TypeError is raised for non-class items in the sequence."""
        bad_sequence = [qml.RX, "not-a-gate"]
        with pytest.raises(TypeError, match="issubclass\(\) arg 1 must be a class"):
            GenericLayerAnsatz(gate_sequence=bad_sequence)

    def test_initialization_invalid_gate_sequence_value_error(self):
        """Tests that a ValueError is raised for multi-qubit gates."""
        bad_sequence = [qml.CNOT]
        with pytest.raises(ValueError, match="must be PennyLane one-qubit gate"):
            GenericLayerAnsatz(gate_sequence=bad_sequence)

    def test_initialization_invalid_entangler(self):
        """Tests that initialization fails with an unsupported entangler."""
        with pytest.raises(ValueError, match="Only qml.CNOT and qml.CZ are supported"):
            GenericLayerAnsatz(
                gate_sequence=[qml.RX], entangler=qml.CRX, entangling_layout="linear"
            )

    def test_initialization_invalid_layout_string(self):
        """Tests that initialization fails with an invalid layout string."""
        with pytest.raises(
            ValueError, match="must be 'linear', 'circular', 'all_to_all'"
        ):
            GenericLayerAnsatz(
                gate_sequence=[qml.RX],
                entangler=qml.CNOT,
                entangling_layout="invalid_layout",
            )

    def test_initialization_warns_on_layout_without_entangler(self):
        """Tests that a warning is issued if a layout is given but entangler is None."""
        with pytest.warns(UserWarning, match="`entangler` is None"):
            GenericLayerAnsatz(
                gate_sequence=[qml.RX], entangler=None, entangling_layout="linear"
            )

    @pytest.mark.parametrize(
        "gate_sequence, n_qubits, expected_params",
        [
            ([qml.RX], 4, 4),
            ([qml.RX, qml.RZ], 4, 8),
            ([qml.Rot], 3, 9),
            ([qml.RY, qml.Rot], 2, 8),  # 1 + 3 params per qubit
        ],
    )
    def test_n_params_per_layer(self, gate_sequence, n_qubits, expected_params):
        """Tests the parameter calculation."""
        ansatz = GenericLayerAnsatz(gate_sequence=gate_sequence)
        assert ansatz.n_params_per_layer(n_qubits) == expected_params

    def test_build_no_entangler(self):
        """Tests building a circuit with only rotation gates."""
        n_qubits, n_layers = 2, 2
        ansatz = GenericLayerAnsatz(gate_sequence=[qml.RX, qml.RY], entangler=None)
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = sympy.symarray("p", n_params)

        ops = get_circuit_operations(ansatz, params, n_qubits, n_layers)

        assert len(ops) == n_qubits * n_layers * 2  # 2 qubits * 2 layers * 2 gates
        assert all(op.name in ["RX", "RY"] for op in ops)
        # Check if params are correctly assigned to the first qubit, first layer
        assert ops[0].parameters[0] == params[0]
        assert ops[1].parameters[0] == params[1]

    def test_build_with_entangler(self):
        """Tests building a circuit with rotation and CNOT gates."""
        n_qubits, n_layers = 3, 1
        ansatz = GenericLayerAnsatz(
            gate_sequence=[qml.RX], entangler=qml.CNOT, entangling_layout="linear"
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = sympy.symarray("p", n_params)

        ops = get_circuit_operations(ansatz, params, n_qubits, n_layers)

        # Expected ops: 3 RX + 2 CNOTs for linear layout on 3 qubits
        assert len(ops) == 5
        assert [op.name for op in ops] == ["RX", "RX", "RX", "CNOT", "CNOT"]
        assert ops[3].wires.tolist() == [0, 1]
        assert ops[4].wires.tolist() == [1, 2]


# --- Test QAOAAnsatz ---
class TestQAOAAnsatz:
    """Tests for the QAOAAnsatz class."""

    def test_n_params_per_layer(self):
        """Tests that n_params_per_layer returns 8."""
        assert QAOAAnsatz.n_params_per_layer(n_qubits=4) == 8

    def test_build(self):
        """Tests that the build method creates a QAOAEmbedding."""
        n_qubits, n_layers = 4, 3
        ansatz = QAOAAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = sympy.symarray("p", n_params)

        ops = get_circuit_operations(ansatz, params, n_qubits, n_layers)

        assert len(ops) == 1
        assert isinstance(ops[0], qml.QAOAEmbedding)


# --- Test HardwareEfficientAnsatz ---
class TestHardwareEfficientAnsatz:
    """Tests for the HardwareEfficientAnsatz placeholder class."""

    def test_n_params_per_layer_raises_error(self):
        """Tests that n_params_per_layer raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            HardwareEfficientAnsatz.n_params_per_layer(n_qubits=4)

    def test_build_raises_error(self):
        """Tests that build raises NotImplementedError."""
        ansatz = HardwareEfficientAnsatz()
        with pytest.raises(NotImplementedError):
            ansatz.build(params=None, n_qubits=4, n_layers=1)


# --- Test Chemistry Ansaetze ---
class TestUCCSDAnsatz:
    """Tests for the UCCSDAnsatz class."""

    def test_n_params_per_layer(self):
        """Tests parameter count for UCCSD."""

        assert UCCSDAnsatz.n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build(self):
        """Tests that the build method constructs a UCCSD operation."""
        n_electrons, n_qubits, n_layers = 2, 4, 2

        # Mock values
        mock_hf = np.array([1, 1, 0, 0])

        ansatz = UCCSDAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits, n_electrons)
        params = sympy.symarray("p", n_params)

        # n_layers is handled internally by qml.UCCSD via n_repeats
        ops = get_circuit_operations(
            ansatz, params, n_qubits, n_layers=n_layers, n_electrons=n_electrons
        )

        assert len(ops) == 1
        uccsd_op = ops[0]
        assert isinstance(uccsd_op, qml.UCCSD)
        assert uccsd_op.hyperparameters["s_wires"] == ((0, 1, 2), (1, 2, 3))
        assert uccsd_op.hyperparameters["d_wires"] == (((0, 1), (2, 3)),)
        assert uccsd_op.hyperparameters["n_repeats"] == n_layers
        assert np.all(uccsd_op.hyperparameters["init_state"] == mock_hf)


class TestHartreeFockAnsatz:
    """Tests for the HartreeFockAnsatz class."""

    def test_n_params_per_layer(self):
        """Tests parameter count for HartreeFockAnsatz."""
        assert HartreeFockAnsatz.n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build(self):
        """Tests the special hf_state behavior for multi-layer circuits."""
        n_electrons, n_qubits, n_layers = 2, 4, 2
        mock_hf = np.array([1, 1, 0, 0])

        ansatz = HartreeFockAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits, n_electrons)
        params = sympy.symarray("p", n_params)

        ops = get_circuit_operations(
            ansatz, params, n_qubits, n_layers, n_electrons=n_electrons
        )

        # Expected: 2 AllSinglesDoubles
        assert len(ops) == 2

        # First AllSinglesDoubles should have hf_state (it gets auto-added by qml.layer)
        assert isinstance(ops[0], qml.AllSinglesDoubles)
        assert "hf_state" in ops[0].hyperparameters
        assert np.all(ops[0].hyperparameters["hf_state"] == mock_hf)

        # Second AllSinglesDoubles should have hf_state reset to 0
        assert isinstance(ops[1], qml.AllSinglesDoubles)
        assert ops[1].hyperparameters["hf_state"] == 0
