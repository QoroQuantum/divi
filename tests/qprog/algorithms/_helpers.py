# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for algorithm tests (explicit import only)."""

import importlib.util

import pytest
from qiskit.circuit import QuantumCircuit

needs_qiskit_nature = pytest.mark.skipif(
    importlib.util.find_spec("qiskit_nature") is None,
    reason="requires the 'chem' extra",
)


def gate_names(qc: QuantumCircuit) -> list[str]:
    return [instr.operation.name for instr in qc.data]


def gate_qubits(qc: QuantumCircuit) -> list[list[int]]:
    return [[qc.find_bit(q).index for q in instr.qubits] for instr in qc.data]


def seed_best_probs(program, probs: dict[str, float], key=0) -> None:
    """Give ``program`` a sampled distribution, as a finished run leaves it."""
    program._results["best_probs"] = {key: probs}
    program._losses_history = [{0: -1.0}]
