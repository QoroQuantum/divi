# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared initial-state utility and its integration into algorithms."""

import networkx as nx
import pennylane as qp
import pytest
import scipy.linalg
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from divi.hamiltonians import xy_mixer
from divi.qprog import QAOA, VQE, TimeEvolution
from divi.qprog.algorithms import (
    CustomPerQubitState,
    OnesState,
    QAOAAnsatz,
    SuperpositionState,
    WState,
    ZerosState,
)
from divi.qprog.algorithms._initial_state import build_block_xy_mixer_graph
from divi.qprog.problems import MaxCutProblem


def _gate_names(qc: QuantumCircuit) -> list[str]:
    return [instr.operation.name for instr in qc.data]


def _gate_qubits(qc: QuantumCircuit) -> list[list[int]]:
    return [[qc.find_bit(q).index for q in instr.qubits] for instr in qc.data]


class TestZerosState:
    def test_build_emits_nothing(self):
        qc = ZerosState().build([0, 1, 2])
        assert _gate_names(qc) == []

    def test_name(self):
        assert ZerosState().name == "ZerosState"


class TestOnesState:
    def test_build_emits_x_per_qubit(self):
        qc = OnesState().build([0, 1])
        assert _gate_names(qc) == ["x", "x"]
        assert _gate_qubits(qc) == [[0], [1]]


class TestSuperpositionState:
    def test_build_emits_hadamard_per_qubit(self):
        qc = SuperpositionState().build([0, 1, 2])
        assert _gate_names(qc) == ["h", "h", "h"]
        assert _gate_qubits(qc) == [[0], [1], [2]]


class TestCustomPerQubitState:
    def test_zeros_string_is_no_op(self):
        qc = CustomPerQubitState("00").build([0, 1])
        assert _gate_names(qc) == []

    def test_ones_string_is_x(self):
        qc = CustomPerQubitState("11").build([0, 1])
        assert _gate_names(qc) == ["x", "x"]

    def test_plus_is_hadamard(self):
        qc = CustomPerQubitState("++").build([0, 1])
        assert _gate_names(qc) == ["h", "h"]

    def test_minus_is_x_then_h(self):
        qc = CustomPerQubitState("-").build([0])
        assert _gate_names(qc) == ["x", "h"]
        assert _gate_qubits(qc) == [[0], [0]]

    def test_mixed_string(self):
        qc = CustomPerQubitState("01+-").build([0, 1, 2, 3])
        assert _gate_names(qc) == ["x", "h", "x", "h"]
        assert _gate_qubits(qc) == [[1], [2], [3], [3]]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="wire count"):
            CustomPerQubitState("01").build([0, 1, 2])

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="state_string"):
            CustomPerQubitState("xy")


class TestWState:
    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            WState(0, 1)

    def test_wrong_wire_count_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            WState(3, 2).build([0, 1, 2])


class TestBlockXYMixer:
    def test_graph_structure(self):
        g = build_block_xy_mixer_graph(3, 2, range(6))
        # All-to-all within each block (complete graph per block)
        assert set(g.edges()) == {
            (0, 1),
            (0, 2),
            (1, 2),  # block 0
            (3, 4),
            (3, 5),
            (4, 5),  # block 1
        }

    def test_graph_structure_path(self):
        g = build_block_xy_mixer_graph(3, 2, range(6), connectivity="path")
        # Nearest-neighbour within each block (path graph per block)
        assert set(g.edges()) == {(0, 1), (1, 2), (3, 4), (4, 5)}

    def test_wrong_wire_count_raises(self):
        with pytest.raises(ValueError, match="Expected 6 wires"):
            build_block_xy_mixer_graph(3, 2, [0, 1, 2])

    def test_preserves_one_hot_subspace(self):
        """XY mixer applied to a W-state should produce only one-hot outputs."""
        n = 3
        wires = list(range(n))

        xy_graph = build_block_xy_mixer_graph(n, 1, wires)
        mixer = xy_mixer(xy_graph)

        qc = WState(n, 1).build(wires)
        U = Operator(scipy.linalg.expm(-1j * 1.5 * mixer.to_matrix()))
        probs = Statevector(qc).evolve(U).probabilities()

        one_hot_indices = [1 << i for i in range(n)]  # 001, 010, 100
        total_one_hot = sum(probs[i] for i in one_hot_indices)
        assert total_one_hot == pytest.approx(1.0, abs=1e-10)

    def test_preserves_one_hot_multi_block(self):
        block_size, n_blocks = 3, 2
        wires = list(range(block_size * n_blocks))
        n = len(wires)

        xy_graph = build_block_xy_mixer_graph(block_size, n_blocks, wires)
        mixer = xy_mixer(xy_graph)

        qc = WState(block_size, n_blocks).build(wires)
        U = Operator(scipy.linalg.expm(-1j * 2.0 * mixer.to_matrix()))
        probs = Statevector(qc).evolve(U).probabilities()

        total_valid = 0.0
        for idx in range(2**n):
            bits = format(idx, f"0{n}b")
            valid = all(
                sum(int(bits[b * block_size + j]) for j in range(block_size)) == 1
                for b in range(n_blocks)
            )
            if valid:
                total_valid += probs[idx]
        assert total_valid == pytest.approx(1.0, abs=1e-10)


class TestTimeEvolutionCustomInitialState:
    """Test that TimeEvolution accepts and uses custom string initial states."""

    def test_custom_string_10_behaves_like_swapped_ones(self, default_test_simulator):
        """H=Z₀+Z₁ with initial_state='10' is an eigenstate, so P(10)≈1."""
        te = TimeEvolution(
            hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
            time=0.5,
            initial_state=CustomPerQubitState("10"),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        probs = te.results
        assert probs.get("10", 0.0) >= 0.95

    def test_custom_string_plus_minus(self, default_test_simulator):
        """Smoke test: '+-' creates |+⟩|−⟩ and runs successfully."""
        te = TimeEvolution(
            hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
            time=0.1,
            initial_state=CustomPerQubitState("+-"),
            backend=default_test_simulator,
        )
        te.run()
        assert te.total_circuit_count >= 1
        assert te.results is not None

    def test_invalid_custom_string_raises(self, default_test_simulator):
        with pytest.raises(TypeError):
            TimeEvolution(
                hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
                initial_state="xy",
                backend=default_test_simulator,
            )


class TestQAOACustomInitialState:
    """Test that QAOA accepts custom string initial states."""

    def test_custom_string_accepted(self, default_test_simulator):
        """QAOA with initial_state=CustomPerQubitState('00') should behave like ZerosState."""
        graph = nx.Graph([(0, 1)])
        qaoa = QAOA(
            MaxCutProblem(graph),
            initial_state=CustomPerQubitState("00"),
            backend=default_test_simulator,
            max_iterations=1,
        )
        assert isinstance(qaoa.initial_state, CustomPerQubitState)


class TestVQEInitialState:
    """Test that VQE accepts and uses the initial_state parameter."""

    def test_initial_state_default_is_zeros(self, default_test_simulator):
        vqe = VQE(
            hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
            ansatz=QAOAAnsatz(),
            backend=default_test_simulator,
        )
        assert isinstance(vqe.initial_state, ZerosState)

    def test_initial_state_superposition_runs(self, default_test_simulator):
        vqe = VQE(
            hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
            ansatz=QAOAAnsatz(),
            initial_state=SuperpositionState(),
            backend=default_test_simulator,
            max_iterations=1,
        )
        vqe.run()
        assert vqe.total_circuit_count >= 1

    def test_chemistry_ansatz_warns(self, default_test_simulator):
        with pytest.warns(UserWarning, match="chemistry ansatz"):
            VQE(
                hamiltonian=qp.PauliZ(0) + qp.PauliZ(1),
                initial_state=SuperpositionState(),
                n_electrons=1,
                backend=default_test_simulator,
            )
