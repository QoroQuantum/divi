# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared initial-state utility and its integration into algorithms."""

import networkx as nx
import pennylane as qml
import pennylane.qaoa as pqaoa
import pytest

from divi.qprog import QAOA, VQE, TimeEvolution
from divi.qprog.algorithms._ansatze import QAOAAnsatz
from divi.qprog.algorithms._initial_state import (
    CustomPerQubitState,
    OnesState,
    SuperpositionState,
    WState,
    ZerosState,
    build_block_xy_mixer_graph,
)
from divi.qprog.problems import MaxCutProblem

# ---------------------------------------------------------------------------
# InitialState classes
# ---------------------------------------------------------------------------


class TestZerosState:
    def test_build_returns_empty(self):
        assert ZerosState().build([0, 1, 2]) == []

    def test_name(self):
        assert ZerosState().name == "ZerosState"


class TestOnesState:
    def test_build_returns_paulix(self):
        ops = OnesState().build([0, 1])
        assert len(ops) == 2
        assert all(isinstance(op, qml.PauliX) for op in ops)
        assert [op.wires.tolist() for op in ops] == [[0], [1]]


class TestSuperpositionState:
    def test_build_returns_hadamard(self):
        ops = SuperpositionState().build([0, 1, 2])
        assert len(ops) == 3
        assert all(isinstance(op, qml.Hadamard) for op in ops)


class TestCustomPerQubitState:
    def test_zeros_string_is_no_op(self):
        assert CustomPerQubitState("00").build([0, 1]) == []

    def test_ones_string_is_paulix(self):
        ops = CustomPerQubitState("11").build([0, 1])
        assert len(ops) == 2
        assert all(isinstance(op, qml.PauliX) for op in ops)

    def test_plus_is_hadamard(self):
        ops = CustomPerQubitState("++").build([0, 1])
        assert all(isinstance(op, qml.Hadamard) for op in ops)

    def test_minus_is_paulix_then_hadamard(self):
        ops = CustomPerQubitState("-").build([0])
        assert len(ops) == 2
        assert isinstance(ops[0], qml.PauliX)
        assert isinstance(ops[1], qml.Hadamard)

    def test_mixed_string(self):
        ops = CustomPerQubitState("01+-").build([0, 1, 2, 3])
        assert len(ops) == 4
        assert isinstance(ops[0], qml.PauliX) and ops[0].wires.tolist() == [1]
        assert isinstance(ops[1], qml.Hadamard) and ops[1].wires.tolist() == [2]
        assert isinstance(ops[2], qml.PauliX) and ops[2].wires.tolist() == [3]
        assert isinstance(ops[3], qml.Hadamard) and ops[3].wires.tolist() == [3]

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
        init_ops = WState(n, 1).build(wires)
        xy_graph = build_block_xy_mixer_graph(n, 1, wires)
        mixer_op = pqaoa.mixer_layer(1.5, pqaoa.xy_mixer(xy_graph))

        tape = qml.tape.QuantumScript(
            ops=init_ops + [mixer_op], measurements=[qml.probs()]
        )
        dev = qml.device("default.qubit", wires=n)
        result = qml.execute([tape], dev)[0]

        total_one_hot = sum(result[i] for i in [1, 2, 4])
        assert total_one_hot == pytest.approx(1.0, abs=1e-10)

    def test_preserves_one_hot_multi_block(self):
        block_size, n_blocks = 3, 2
        wires = list(range(block_size * n_blocks))
        init_ops = WState(block_size, n_blocks).build(wires)
        xy_graph = build_block_xy_mixer_graph(block_size, n_blocks, wires)
        mixer_op = pqaoa.mixer_layer(2.0, pqaoa.xy_mixer(xy_graph))

        tape = qml.tape.QuantumScript(
            ops=init_ops + [mixer_op], measurements=[qml.probs()]
        )
        dev = qml.device("default.qubit", wires=len(wires))
        result = qml.execute([tape], dev)[0]

        total_valid = 0.0
        for idx in range(2 ** len(wires)):
            bits = format(idx, f"0{len(wires)}b")
            valid = all(
                sum(int(bits[b * block_size + j]) for j in range(block_size)) == 1
                for b in range(n_blocks)
            )
            if valid:
                total_valid += result[idx]
        assert total_valid == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Integration: TimeEvolution with custom initial states
# ---------------------------------------------------------------------------


class TestTimeEvolutionCustomInitialState:
    """Test that TimeEvolution accepts and uses custom string initial states."""

    def test_custom_string_10_behaves_like_swapped_ones(self, default_test_simulator):
        """H=Z₀+Z₁ with initial_state='10' is an eigenstate, so P(10)≈1."""
        te = TimeEvolution(
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
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
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
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
                hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
                initial_state="xy",
                backend=default_test_simulator,
            )


# ---------------------------------------------------------------------------
# Integration: QAOA with custom initial states
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Integration: VQE with initial_state
# ---------------------------------------------------------------------------


class TestVQEInitialState:
    """Test that VQE accepts and uses the initial_state parameter."""

    def test_initial_state_default_is_zeros(self, default_test_simulator):
        vqe = VQE(
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
            ansatz=QAOAAnsatz(),
            backend=default_test_simulator,
        )
        assert isinstance(vqe.initial_state, ZerosState)

    def test_initial_state_superposition_runs(self, default_test_simulator):
        vqe = VQE(
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
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
                hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
                initial_state=SuperpositionState(),
                n_electrons=1,
                backend=default_test_simulator,
            )
