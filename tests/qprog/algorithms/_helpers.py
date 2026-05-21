# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for algorithm tests (explicit import only)."""

from qiskit.circuit import QuantumCircuit


def gate_names(qc: QuantumCircuit) -> list[str]:
    return [instr.operation.name for instr in qc.data]


def gate_qubits(qc: QuantumCircuit) -> list[list[int]]:
    return [[qc.find_bit(q).index for q in instr.qubits] for instr in qc.data]
