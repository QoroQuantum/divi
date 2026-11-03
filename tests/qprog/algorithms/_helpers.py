# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for algorithm tests (explicit import only)."""

import importlib.util

import numpy as np
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


def amplitudes(t1_alpha, t1_beta, t2_opposite):
    """UCCSD amplitudes with the given singles and opposite-spin doubles, and no
    same-spin doubles."""
    t2_opposite = np.asarray(t2_opposite)
    return (
        (np.asarray(t1_alpha), np.asarray(t1_beta)),
        (np.zeros_like(t2_opposite), t2_opposite, np.zeros_like(t2_opposite)),
    )


def one_pair_amplitudes(double, single=0.0):
    """UCCSD amplitudes for one occupied and one virtual orbital per spin."""
    return amplitudes(
        np.full((1, 1), single),
        np.full((1, 1), -single),
        np.full((1, 1, 1, 1), double),
    )


def seed_best_probs(program, probs: dict[str, float], key=0) -> None:
    """Give ``program`` a sampled distribution, as a finished run leaves it."""
    program._results["best_probs"] = {key: probs}
    program._losses_history = [{0: -1.0}]
