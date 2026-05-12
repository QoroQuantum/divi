# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SwapGate,
    UGate,
)

from divi.qprog import (
    GenericLayerAnsatz,
    HartreeFockAnsatz,
    QAOAAnsatz,
    QCCAnsatz,
    UCCSDAnsatz,
)


def _build_circuit(ansatz, params, n_qubits, n_layers, **kwargs) -> QuantumCircuit:
    return ansatz.build(params, n_qubits, n_layers, **kwargs)


def _gate_names(qc: QuantumCircuit) -> list[str]:
    return [instr.operation.name for instr in qc.data]


def _gate_qubits(qc: QuantumCircuit) -> list[list[int]]:
    return [[qc.find_bit(q).index for q in instr.qubits] for instr in qc.data]


# --- Test GenericLayerAnsatz ---
class TestGenericLayerAnsatz:
    """Tests for the GenericLayerAnsatz class."""

    @pytest.mark.parametrize(
        "gate_sequence, entangler, layout",
        [
            ([RXGate], CXGate, "linear"),
            ([RYGate, RZGate], CZGate, "circular"),
            ([UGate], None, "all-to-all"),
            ([RXGate], CXGate, [(0, 2), (1, 3)]),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initialization_valid(self, gate_sequence, entangler, layout):
        try:
            GenericLayerAnsatz(
                gate_sequence=gate_sequence,
                entangler=entangler,
                entangling_layout=layout,
            )
        except (ValueError, TypeError):
            pytest.fail("GenericLayerAnsatz initialization failed with valid inputs.")

    def test_initialization_rejects_string_gate(self):
        with pytest.raises(TypeError, match="Expected a Qiskit Gate subclass"):
            GenericLayerAnsatz(gate_sequence=[RXGate, "rx"])

    def test_initialization_rejects_gate_instance(self):
        with pytest.raises(TypeError, match="Expected a Qiskit Gate subclass"):
            GenericLayerAnsatz(gate_sequence=[RXGate(0.0)])

    def test_initialization_invalid_layout_string(self):
        with pytest.raises(ValueError, match="Unknown entangling_layout:"):
            GenericLayerAnsatz(
                gate_sequence=[RXGate],
                entangler=CXGate,
                entangling_layout="invalid_layout",
            )

    def test_initialization_warns_on_layout_without_entangler(self):
        with pytest.warns(UserWarning, match="`entangler` is None"):
            GenericLayerAnsatz(
                gate_sequence=[RXGate], entangler=None, entangling_layout="linear"
            )

    @pytest.mark.parametrize(
        "gate_sequence, n_qubits, expected_params",
        [
            ([RXGate], 4, 4),
            ([RXGate, RZGate], 4, 8),
            ([UGate], 3, 9),
            ([RYGate, UGate], 2, 8),  # 1 + 3 params per qubit
        ],
    )
    def test_n_params_per_layer(self, gate_sequence, n_qubits, expected_params):
        ansatz = GenericLayerAnsatz(gate_sequence=gate_sequence)
        assert ansatz.n_params_per_layer(n_qubits) == expected_params

    def test_n_params_per_layer_rejects_parameter_free_ansatz(self):
        ansatz = GenericLayerAnsatz(gate_sequence=[HGate])
        with pytest.raises(ValueError, match="must define at least one trainable"):
            ansatz.n_params_per_layer(n_qubits=2)

    def test_build_no_entangler(self):
        n_qubits, n_layers = 2, 2
        ansatz = GenericLayerAnsatz(gate_sequence=[RXGate, RYGate], entangler=None)
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)

        names = _gate_names(qc)
        # 2 qubits * 2 layers * 2 gates = 8 gates
        assert len(names) == 8
        assert all(name in ("rx", "ry") for name in names)
        # First two gates are on qubit 0 then qubit 1 with the first two params.
        assert qc.data[0].operation.params[0] == params[0]
        assert qc.data[2].operation.params[0] == params[2]

    def test_build_with_entangler(self):
        n_qubits, n_layers = 3, 1
        ansatz = GenericLayerAnsatz(
            gate_sequence=[RXGate], entangler=CXGate, entangling_layout="linear"
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)

        # 3 RX + 2 CNOTs for linear layout on 3 qubits
        assert _gate_names(qc) == ["rx", "rx", "rx", "cx", "cx"]
        assert _gate_qubits(qc)[3:] == [[0, 1], [1, 2]]

    def test_build_with_swap_entangler(self):
        """Non-CX/CZ 2-qubit entanglers work — SwapGate isn't in any whitelist."""
        n_qubits, n_layers = 3, 1
        ansatz = GenericLayerAnsatz(
            gate_sequence=[RXGate], entangler=SwapGate, entangling_layout="linear"
        )
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        qc = _build_circuit(
            ansatz, list(ParameterVector("p", n_params)), n_qubits, n_layers
        )
        assert _gate_names(qc) == ["rx", "rx", "rx", "swap", "swap"]


# --- Test QAOAAnsatz ---
class TestQAOAAnsatz:
    """Tests for the QAOAAnsatz class."""

    def test_n_params_per_layer(self):
        # 2 * n_qubits per layer.
        assert QAOAAnsatz().n_params_per_layer(n_qubits=4) == 8

    def test_build_structure(self):
        n_qubits, n_layers = 4, 3
        ansatz = QAOAAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(ansatz, list(params), n_qubits, n_layers)

        names = _gate_names(qc)
        # Initial Hadamards.
        assert names[:n_qubits] == ["h"] * n_qubits
        # RZZ decomposes to CX-RZ-CX (3 basis gates), so the cost layer emits
        # 3 * (n_qubits - 1) + n_qubits gates; mixer adds n_qubits RY.
        assert "cx" in names
        assert "ry" in names
        assert "rz" in names


# --- Test Chemistry Ansaetze ---
class TestUCCSDAnsatz:
    """Tests for the UCCSDAnsatz class."""

    def test_n_params_per_layer(self):
        assert UCCSDAnsatz().n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build_emits_qiskit_circuit(self):
        n_electrons, n_qubits, n_layers = 2, 4, 1
        ansatz = UCCSDAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(
            n_qubits, n_electrons=n_electrons
        )
        params = ParameterVector("p", n_params)
        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )
        # Should produce a non-empty Qiskit circuit on the requested qubit count.
        assert qc.num_qubits == n_qubits
        assert len(qc.data) > 0


class TestHartreeFockAnsatz:
    """Tests for the HartreeFockAnsatz class."""

    def test_n_params_per_layer(self):
        assert HartreeFockAnsatz().n_params_per_layer(n_qubits=4, n_electrons=2) == 3

    def test_build_emits_qiskit_circuit(self):
        n_electrons, n_qubits, n_layers = 2, 4, 2
        ansatz = HartreeFockAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(
            n_qubits, n_electrons=n_electrons
        )
        params = ParameterVector("p", n_params)
        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )
        assert qc.num_qubits == n_qubits
        assert len(qc.data) > 0


# --- Test QCCAnsatz ---
class TestQCCAnsatz:
    """Tests for the QCCAnsatz class."""

    def test_n_params_per_layer(self):
        # n_qubits RY + 3*(n_qubits-1) entanglers.
        assert QCCAnsatz().n_params_per_layer(n_qubits=4) == 13
        assert QCCAnsatz().n_params_per_layer(n_qubits=2) == 5

    def test_build_structure(self):
        n_electrons, n_qubits, n_layers = 2, 4, 1
        ansatz = QCCAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )

        names = _gate_names(qc)
        # Hartree-Fock prep: 2 X gates (n_electrons=2 bits set in hf_state).
        # Then 4 RY rotations, then 9 two-qubit Pauli rotations
        # (XX/YY/ZZ for each adjacent pair) each emitted as basis gates.
        assert names[:2] == ["x", "x"]
        assert names[2:6] == ["ry", "ry", "ry", "ry"]
        # Each XX block: 2 H + cx + rz + cx + 2 H = 7 gates.
        # Each YY block: 2 RX + cx + rz + cx + 2 RX = 7 gates.
        # Each ZZ block: cx + rz + cx = 3 gates.
        # 3 adjacent pairs × (7 + 7 + 3) = 51 entangler gates.
        # Total: 2 + 4 + 51 = 57.
        assert len(names) == 57
        assert names.count("cx") == 6 * 3  # CX appears in all three rotations
        assert names.count("rz") == 9  # one RZ per two-qubit Pauli rotation
        assert names.count("h") == 12  # XX needs 4 H per pair × 3 pairs
        assert names.count("rx") == 12  # YY needs 4 RX per pair × 3 pairs

    def test_build_multi_layer(self):
        n_electrons, n_qubits, n_layers = 2, 4, 2
        ansatz = QCCAnsatz()
        n_params = n_layers * ansatz.n_params_per_layer(n_qubits)
        params = ParameterVector("p", n_params)

        qc = _build_circuit(
            ansatz, list(params), n_qubits, n_layers, n_electrons=n_electrons
        )

        names = _gate_names(qc)
        # 2 X (HF prep, once total) + 2 layers × (4 RY + 51 entangler basis gates).
        assert len(names) == 2 + 2 * (4 + 51)
        assert names.count("ry") == 8
