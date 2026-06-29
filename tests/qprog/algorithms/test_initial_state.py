# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared initial-state utility and its integration into algorithms."""

import networkx as nx
import pennylane as qp
import pytest
import scipy.linalg
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
from tests.qprog.algorithms._helpers import gate_names, gate_qubits


@pytest.fixture
def two_z_hamiltonian():
    """``Z₀ + Z₁`` — the shared two-qubit Hamiltonian for the integration tests."""
    return qp.PauliZ(0) + qp.PauliZ(1)


@pytest.fixture
def make_time_evolution(two_z_hamiltonian, default_test_simulator):
    def _make(**overrides):
        return TimeEvolution(
            **{
                "hamiltonian": two_z_hamiltonian,
                "backend": default_test_simulator,
                **overrides,
            }
        )

    return _make


@pytest.fixture
def make_vqe(two_z_hamiltonian, default_test_simulator, default_optimizer):
    def _make(**overrides):
        return VQE(
            **{
                "hamiltonian": two_z_hamiltonian,
                "ansatz": QAOAAnsatz(),
                "optimizer": default_optimizer,
                "backend": default_test_simulator,
                **overrides,
            }
        )

    return _make


class TestZerosState:
    def test_build_emits_nothing(self):
        qc = ZerosState().build([0, 1, 2])
        assert gate_names(qc) == []

    def test_name(self):
        assert ZerosState().name == "ZerosState"


def test_build_emits_x_per_qubit():
    qc = OnesState().build([0, 1])
    assert gate_names(qc) == ["x", "x"]
    assert gate_qubits(qc) == [[0], [1]]


def test_build_emits_hadamard_per_qubit():
    qc = SuperpositionState().build([0, 1, 2])
    assert gate_names(qc) == ["h", "h", "h"]
    assert gate_qubits(qc) == [[0], [1], [2]]


class TestCustomPerQubitState:
    def test_zeros_string_is_no_op(self):
        qc = CustomPerQubitState("00").build([0, 1])
        assert gate_names(qc) == []

    def test_ones_string_is_x(self):
        qc = CustomPerQubitState("11").build([0, 1])
        assert gate_names(qc) == ["x", "x"]

    def test_plus_is_hadamard(self):
        qc = CustomPerQubitState("++").build([0, 1])
        assert gate_names(qc) == ["h", "h"]

    def test_minus_is_x_then_h(self):
        qc = CustomPerQubitState("-").build([0])
        assert gate_names(qc) == ["x", "h"]
        assert gate_qubits(qc) == [[0], [0]]

    def test_mixed_string(self):
        qc = CustomPerQubitState("01+-").build([0, 1, 2, 3])
        assert gate_names(qc) == ["x", "h", "x", "h"]
        assert gate_qubits(qc) == [[1], [2], [3], [3]]

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

    @pytest.mark.parametrize("block_size, n_blocks, t", [(3, 1, 1.5), (3, 2, 2.0)])
    def test_preserves_one_hot_subspace(self, block_size, n_blocks, t):
        """The XY mixer evolving a W-state keeps all amplitude in the
        per-block one-hot subspace (each block has exactly one excitation).
        With ``n_blocks=1`` this is the plain single-block one-hot case."""
        wires = list(range(block_size * n_blocks))
        n = len(wires)

        xy_graph = build_block_xy_mixer_graph(block_size, n_blocks, wires)
        mixer = xy_mixer(xy_graph)

        qc = WState(block_size, n_blocks).build(wires)
        U = Operator(scipy.linalg.expm(-1j * t * mixer.to_matrix()))
        probs = Statevector(qc).evolve(U).probabilities()

        total_valid = sum(
            probs[idx]
            for idx in range(2**n)
            if all(
                sum(
                    int(bit)
                    for bit in format(idx, f"0{n}b")[b * block_size :][:block_size]
                )
                == 1
                for b in range(n_blocks)
            )
        )
        assert total_valid == pytest.approx(1.0, abs=1e-10)


class TestTimeEvolutionCustomInitialState:
    """Test that TimeEvolution accepts and uses custom string initial states."""

    def test_custom_string_10_behaves_like_swapped_ones(self, make_time_evolution):
        """H=Z₀+Z₁ with initial_state='10' is an eigenstate, so P(10)≈1."""
        te = make_time_evolution(time=0.5, initial_state=CustomPerQubitState("10"))
        te.run()
        assert te.results.get("10", 0.0) >= 0.95

    def test_custom_string_plus_minus(self, make_time_evolution):
        """'+-' prepares |+⟩|−⟩; both qubits are X-eigenstates, so the Z-basis
        readout is uniform over all four bitstrings (the Z-diagonal evolution
        leaves those probabilities unchanged)."""
        te = make_time_evolution(time=0.1, initial_state=CustomPerQubitState("+-"))
        te.run()
        for bitstring in ("00", "01", "10", "11"):
            assert te.results.get(bitstring, 0.0) == pytest.approx(0.25, abs=0.05)

    def test_invalid_custom_string_raises(self, make_time_evolution):
        with pytest.raises(TypeError):
            make_time_evolution(initial_state="xy")


def test_qaoa_accepts_custom_string_initial_state(
    default_test_simulator, default_optimizer
):
    """QAOA accepts a ``CustomPerQubitState`` and stores it as its initial state."""
    qaoa = QAOA(
        MaxCutProblem(nx.Graph([(0, 1)])),
        initial_state=CustomPerQubitState("00"),
        optimizer=default_optimizer,
        backend=default_test_simulator,
    )
    assert isinstance(qaoa.initial_state, CustomPerQubitState)


class TestVQEInitialState:
    """Test that VQE accepts and uses the initial_state parameter."""

    def test_initial_state_default_is_zeros(self, make_vqe):
        assert isinstance(make_vqe().initial_state, ZerosState)

    def test_initial_state_superposition_runs(self, make_vqe):
        vqe = make_vqe(initial_state=SuperpositionState(), max_iterations=1)
        vqe.run()
        assert len(vqe.losses_history) == 1

    def test_chemistry_ansatz_warns(self, make_vqe):
        with pytest.warns(UserWarning, match="chemistry ansatz"):
            # ansatz=None selects the default chemistry ansatz, which warns when
            # paired with an explicit initial_state.
            make_vqe(ansatz=None, initial_state=SuperpositionState(), n_electrons=1)
