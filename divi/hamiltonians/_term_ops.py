# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level manipulation primitives for ``SparsePauliOp`` Hamiltonians."""

import math
import numbers
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pennylane as qp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

from divi.circuits._conversions import _observable_to_sparse_pauli_op
from divi.circuits._core import _assert_hermitian_spo


def _require_qiskit_num_qubits(num_qubits: int | None) -> int:
    """Return Qiskit's nullable ``num_qubits`` value as Divi's required int."""
    if num_qubits is None:
        raise RuntimeError("SparsePauliOp must have a concrete qubit count.")
    return num_qubits


def generate_empty_spo(num_qubits: int) -> SparsePauliOp:
    """Zero-row ``SparsePauliOp`` over ``num_qubits`` qubits (``size == 0``)."""
    return SparsePauliOp(["I" * num_qubits], coeffs=np.zeros(1, dtype=complex))[
        np.zeros(0, dtype=int)
    ]


def to_spo(op: qp.operation.Operator | SparsePauliOp) -> SparsePauliOp:
    """Convert a PennyLane operator or ``SparsePauliOp`` to ``SparsePauliOp``,
    validating Hermiticity in both cases.

    The PennyLane branch builds a new ``SparsePauliOp`` by walking the
    operator tree. For repeated use on the same observable, convert once
    at setup and reuse the returned ``SparsePauliOp``.
    """
    if isinstance(op, SparsePauliOp):
        _assert_hermitian_spo(op)
        return op
    spo = _observable_to_sparse_pauli_op(op, op.wires)
    _assert_hermitian_spo(spo)
    return spo


def _spo_wires(op: qp.operation.Operator | SparsePauliOp) -> tuple:
    """Wire mapping aligned with :func:`to_spo` (qubit ``i`` ↔ ``wires[i]``)."""
    if isinstance(op, SparsePauliOp):
        return tuple(range(_require_qiskit_num_qubits(op.num_qubits)))
    return tuple(op.wires)


def _clean_hamiltonian_spo(spo: SparsePauliOp) -> tuple[SparsePauliOp, float]:
    """Partition identity-only rows from the rest. Returns ``(non-identity SPO, constant)``.

    The returned SPO has ``size == 0`` when the input contains only identity
    terms; callers must use ``size`` (not ``simplify()``) to detect emptiness.
    """
    if spo.size == 0:
        return spo, 0.0
    non_id_mask = np.any(spo.paulis.x | spo.paulis.z, axis=1)
    # ``math.fsum`` is exact-rounding; protects against alternating-sign
    # cancellation that ``ndarray.sum`` accumulates left-to-right.
    constant = math.fsum(spo.coeffs[~non_id_mask].real)
    cleaned = SparsePauliOp(spo.paulis[non_id_mask], spo.coeffs[non_id_mask])
    if non_id_mask.any():
        cleaned = cleaned.simplify()
    return cleaned, constant


def _sort_hamiltonian_terms_spo(
    spo: SparsePauliOp,
    order: Literal["absolute", "magnitude"] = "absolute",
) -> SparsePauliOp:
    """Sort terms ascending by signed coefficient (``"absolute"``) or |coefficient| (``"magnitude"``)."""
    if spo.size <= 1:
        return spo
    coeffs = np.asarray(spo.coeffs.real)
    keys = np.abs(coeffs) if order == "magnitude" else coeffs
    idx = np.argsort(keys, kind="stable")
    return SparsePauliOp(spo.paulis[idx], spo.coeffs[idx])


_HALF_PI = float(np.pi / 2)


def _spo_to_basis_gate_ops(
    spo: SparsePauliOp,
    time,
    wires: Sequence,
) -> list[qp.operation.Operator]:
    """First-order Trotter decomposition of ``exp(-i * time * H)`` as PennyLane basis gates.

    Output matches ``qp.PauliRot`` → ``MultiRZ`` → CNOT/RZ decomposition.
    """
    wire_list = list(wires)
    ops: list[qp.operation.Operator] = []
    for pauli_str, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
        c = float(np.real(coeff))
        # Qiskit labels are big-endian: pauli_str[-(i+1)] is qubit i.
        active = [
            (wire_list[i], ch) for i, ch in enumerate(reversed(pauli_str)) if ch != "I"
        ]
        if not active:
            continue
        theta = 2 * time * c

        if len(active) == 1:
            w, ch = active[0]
            if ch == "Z":
                ops.append(qp.RZ(theta, wires=w))
            elif ch == "X":
                ops.append(qp.RX(theta, wires=w))
            else:
                ops.append(qp.RY(theta, wires=w))
            continue

        for w, ch in active:
            if ch == "X":
                ops.append(qp.Hadamard(wires=w))
            elif ch == "Y":
                ops.append(qp.RX(_HALF_PI, wires=w))

        active_wires = [w for w, _ in active]
        n = len(active_wires)
        for i in range(n - 1, 0, -1):
            ops.append(qp.CNOT(wires=(active_wires[i], active_wires[i - 1])))
        ops.append(qp.RZ(theta, wires=active_wires[0]))
        for i in range(1, n):
            ops.append(qp.CNOT(wires=(active_wires[i], active_wires[i - 1])))

        for w, ch in active:
            if ch == "X":
                ops.append(qp.Hadamard(wires=w))
            elif ch == "Y":
                ops.append(qp.RX(-_HALF_PI, wires=w))
    return ops


def _spo_to_qiskit_basis_gates(
    qc, spo: SparsePauliOp, time, qubits: Sequence[int]
) -> None:
    """Append first-order-Trotter ``exp(-i * time * spo)`` onto ``qc``.

    Dispatches on ``time`` — numeric → :func:`_spo_to_qiskit_basis_gates_numeric`;
    symbolic (``Parameter`` / ``ParameterExpression``) →
    :func:`_spo_to_qiskit_basis_gates_symbolic`. Exact when the SPO terms
    pairwise commute (QAOA cost/mixer layers).
    """
    if isinstance(time, numbers.Real):
        _spo_to_qiskit_basis_gates_numeric(qc, spo, float(time), qubits)
    else:
        _spo_to_qiskit_basis_gates_symbolic(qc, spo, time, qubits)


def _active_pauli_chars(x_row, z_row) -> tuple[np.ndarray, list[str]]:
    """Return ``(active_qubit_indices, pauli_chars)`` for one SPO row.

    Symplectic ``(x, z)`` decoding: ``(0,0) → I`` (skipped), ``(1,0) → X``,
    ``(0,1) → Z``, ``(1,1) → Y``. Active indices are in ascending order.
    """
    active = np.flatnonzero(x_row | z_row)
    chars: list[str] = []
    for q in active:
        xq = x_row[q]
        zq = z_row[q]
        if xq and zq:
            chars.append("Y")
        elif xq:
            chars.append("X")
        else:
            chars.append("Z")
    return active, chars


def _spo_to_qiskit_basis_gates_numeric(
    qc, spo: SparsePauliOp, time: float, qubits: Sequence[int]
) -> None:
    """Numeric-angle fast path; see :func:`_spo_to_qiskit_basis_gates`."""
    _assert_hermitian_spo(spo)
    qubit_list = list(qubits)

    if len(qubit_list) == 0 or spo.size == 0:
        return

    sub = LieTrotter().synthesize(PauliEvolutionGate(spo, time=time))
    # ``PauliEvolutionGate`` may synthesize to ``R{XX,YY,ZZ}Gate`` for
    # two-qubit Pauli rotations — outside our QASM2 basis. ``decompose`` is
    # a no-op on instructions whose names aren't listed.
    sub = sub.decompose(["rxx", "ryy", "rzz"])
    qc.compose(sub, qubits=qubit_list, inplace=True)


def _spo_to_qiskit_basis_gates_symbolic(
    qc, spo: SparsePauliOp, time, qubits: Sequence[int]
) -> None:
    """CX-RZ-CX ladder emitter for symbolic ``time``.

    Qiskit's ``LieTrotter`` synthesizer only accepts numeric angles, so any
    ``Parameter`` / ``ParameterExpression`` ``time`` comes through here.
    """
    _assert_hermitian_spo(spo)
    qubit_list = list(qubits)
    x_arr = spo.paulis.x  # bool[N_terms, n_qubits]
    z_arr = spo.paulis.z
    coeffs_real = spo.coeffs.real  # zero-copy view

    for i in range(x_arr.shape[0]):
        active_idx, chars = _active_pauli_chars(x_arr[i], z_arr[i])
        if active_idx.size == 0:
            continue

        active: list[tuple[int, str]] = [
            (qubit_list[q], ch) for q, ch in zip(active_idx, chars)
        ]
        theta = 2 * time * float(coeffs_real[i])

        if len(active) == 1:
            q, ch = active[0]
            if ch == "Z":
                qc.rz(theta, q)
            elif ch == "X":
                qc.rx(theta, q)
            else:
                qc.ry(theta, q)
            continue

        for q, ch in active:
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.rx(_HALF_PI, q)

        active_qubits = [q for q, _ in active]
        n = len(active_qubits)
        for j in range(n - 1, 0, -1):
            qc.cx(active_qubits[j], active_qubits[j - 1])
        qc.rz(theta, active_qubits[0])
        for j in range(1, n):
            qc.cx(active_qubits[j], active_qubits[j - 1])

        for q, ch in active:
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.rx(-_HALF_PI, q)
