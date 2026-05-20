# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QAOAProblem base protocol."""

from qiskit.quantum_info import SparsePauliOp

from divi.qprog.problems import QAOAProblem


class MinimalProblem(QAOAProblem):
    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        return SparsePauliOp.from_list([("ZI", 1.0), ("IZ", -1.0)])

    @property
    def loss_constant(self) -> float:
        return 0.0

    @property
    def decode_fn(self):
        return lambda bitstring: bitstring


def test_default_mixer_hamiltonian_is_x_mixer():
    mixer = MinimalProblem().mixer_hamiltonian

    assert mixer.num_qubits == 2
    assert set(mixer.paulis.to_labels()) == {"IX", "XI"}
