# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared initial-state utility and its integration into algorithms."""

import networkx as nx
import pennylane as qml
import pytest

from divi.qprog import QAOA, VQE, TimeEvolution
from divi.qprog.algorithms._ansatze import QAOAAnsatz
from divi.qprog.algorithms._initial_state import (
    build_initial_state_ops,
    validate_initial_state,
)
from divi.qprog.algorithms._qaoa import GraphProblem

# ---------------------------------------------------------------------------
# validate_initial_state
# ---------------------------------------------------------------------------


class TestValidateInitialState:
    @pytest.mark.parametrize("state", ["Zeros", "Ones", "Superposition"])
    def test_literals_pass(self, state):
        validate_initial_state(state, n_qubits=4)  # no error

    @pytest.mark.parametrize("custom", ["0000", "1111", "01+-", "+-+-"])
    def test_valid_custom_strings_pass(self, custom):
        validate_initial_state(custom, n_qubits=len(custom))

    def test_custom_string_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            validate_initial_state("01", n_qubits=3)

    @pytest.mark.parametrize("bad", ["Invalid", "Bell", "2", "abc", "01x"])
    def test_invalid_strings_raise(self, bad):
        with pytest.raises(ValueError, match="initial_state"):
            validate_initial_state(bad, n_qubits=3)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="initial_state"):
            validate_initial_state("", n_qubits=1)


# ---------------------------------------------------------------------------
# build_initial_state_ops
# ---------------------------------------------------------------------------


class TestBuildInitialStateOps:
    def test_zeros_returns_empty(self):
        assert build_initial_state_ops("Zeros", [0, 1, 2]) == []

    def test_ones_returns_paulix_all_wires(self):
        ops = build_initial_state_ops("Ones", [0, 1])
        assert len(ops) == 2
        assert all(isinstance(op, qml.PauliX) for op in ops)
        assert [op.wires.tolist() for op in ops] == [[0], [1]]

    def test_superposition_returns_hadamard_all_wires(self):
        ops = build_initial_state_ops("Superposition", [0, 1, 2])
        assert len(ops) == 3
        assert all(isinstance(op, qml.Hadamard) for op in ops)

    def test_custom_0_is_no_op(self):
        ops = build_initial_state_ops("00", [0, 1])
        assert ops == []

    def test_custom_1_is_paulix(self):
        ops = build_initial_state_ops("11", [0, 1])
        assert len(ops) == 2
        assert all(isinstance(op, qml.PauliX) for op in ops)

    def test_custom_plus_is_hadamard(self):
        ops = build_initial_state_ops("++", [0, 1])
        assert len(ops) == 2
        assert all(isinstance(op, qml.Hadamard) for op in ops)

    def test_custom_minus_is_paulix_then_hadamard(self):
        ops = build_initial_state_ops("-", [0])
        assert len(ops) == 2
        assert isinstance(ops[0], qml.PauliX)
        assert isinstance(ops[1], qml.Hadamard)

    def test_custom_mixed_string(self):
        ops = build_initial_state_ops("01+-", [0, 1, 2, 3])
        # '0' → nothing, '1' → PauliX, '+' → Hadamard, '-' → PauliX+Hadamard
        assert len(ops) == 4
        assert isinstance(ops[0], qml.PauliX) and ops[0].wires.tolist() == [1]
        assert isinstance(ops[1], qml.Hadamard) and ops[1].wires.tolist() == [2]
        assert isinstance(ops[2], qml.PauliX) and ops[2].wires.tolist() == [3]
        assert isinstance(ops[3], qml.Hadamard) and ops[3].wires.tolist() == [3]


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
            initial_state="10",
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1
        probs = te.results
        assert probs.get("10", 0.0) >= 0.95

    def test_custom_string_plus_minus(self, default_test_simulator):
        """Smoke test: '+-' creates |+⟩|−⟩ and runs successfully."""
        te = TimeEvolution(
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
            time=0.1,
            initial_state="+-",
            backend=default_test_simulator,
        )
        count, _ = te.run()
        assert count >= 1
        assert te.results is not None

    def test_invalid_custom_string_raises(self, default_test_simulator):
        with pytest.raises(ValueError, match="initial_state"):
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
        """QAOA with initial_state='00' should behave like 'Zeros'."""
        graph = nx.Graph([(0, 1)])
        qaoa = QAOA(
            problem=graph,
            graph_problem=GraphProblem.MAXCUT,
            initial_state="00",
            backend=default_test_simulator,
            max_iterations=1,
        )
        assert qaoa.initial_state == "00"


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
        assert vqe.initial_state == "Zeros"

    def test_initial_state_superposition_runs(self, default_test_simulator):
        vqe = VQE(
            hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
            ansatz=QAOAAnsatz(),
            initial_state="Superposition",
            backend=default_test_simulator,
            max_iterations=1,
        )
        count, _ = vqe.run()
        assert count >= 1

    def test_chemistry_ansatz_warns(self, default_test_simulator):
        with pytest.warns(UserWarning, match="chemistry ansatz"):
            VQE(
                hamiltonian=qml.PauliZ(0) + qml.PauliZ(1),
                initial_state="Superposition",
                n_electrons=1,
                backend=default_test_simulator,
            )
