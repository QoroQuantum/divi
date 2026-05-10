# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level manipulation primitives for Hamiltonian operators.

Defines a few PennyLane-typed helpers (``_get_terms_iterable``,
``_is_empty_hamiltonian``, ``_is_multi_term_sum``) and their
``SparsePauliOp``-native equivalents (``_to_spo``, ``_from_spo``,
``_clean_hamiltonian_spo``, ``_sort_hamiltonian_terms_spo``,
``_spo_to_basis_gate_ops``). The SPO siblings are the hot path; the
PL helpers exist for boundary use.
"""

import math
import threading
import weakref
from collections.abc import Sequence
from typing import Literal, TypeGuard, cast

import numpy as np
import pennylane as qp
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import sparse_pauli_op_to_pl_observable
from divi.circuits._conversions import observable_to_sparse_pauli_op

# Cache keyed on PL-op identity: {pl_op: (spo, canonical_wires)}.
# Populated by ``_from_spo`` and read by ``_to_spo`` / ``_spo_wires``.
# All access goes through ``_PL_TO_SPO_CACHE_LOCK``.
_PL_TO_SPO_CACHE: (
    "weakref.WeakKeyDictionary[qp.operation.Operator, tuple[SparsePauliOp, tuple]]"
) = weakref.WeakKeyDictionary()
_PL_TO_SPO_CACHE_LOCK = threading.Lock()


def _is_multi_term_sum(
    op: qp.operation.Operator,
) -> TypeGuard[qp.Hamiltonian | qp.ops.Sum]:
    """True if op is a multi-term Sum or Hamiltonian (has operands and len)."""
    return isinstance(op, (qp.Hamiltonian, qp.ops.Sum))


def _get_terms_iterable(
    op: qp.operation.Operator,
) -> Sequence[qp.operation.Operator]:
    """Return terms as a sequence for iteration. Works for Sum/Hamiltonian and single-term."""
    return op.operands if _is_multi_term_sum(op) else [op]


def _is_empty_hamiltonian(op: qp.operation.Operator) -> bool:
    """True if op is an empty Sum/Hamiltonian (only constant terms)."""
    return _is_multi_term_sum(op) and len(op) == 0


# --------------------------------------------------------------------------- #
# SparsePauliOp-native siblings of the PL-typed helpers above.
# --------------------------------------------------------------------------- #


def _num_qubits(spo: SparsePauliOp) -> int:
    """SPO ``num_qubits`` narrowed to ``int`` (qiskit types it as ``int | None``)."""
    return cast(int, spo.num_qubits)


def _empty_spo(num_qubits: int) -> SparsePauliOp:
    """Zero-row ``SparsePauliOp`` over ``num_qubits`` qubits (``size == 0``)."""
    return SparsePauliOp(["I" * num_qubits], coeffs=np.zeros(1, dtype=complex))[
        np.zeros(0, dtype=int)
    ]


def _to_spo(op: qp.operation.Operator | SparsePauliOp) -> SparsePauliOp:
    """PennyLane operator → ``SparsePauliOp`` (passthrough if already SPO).

    Reuses the SPO cached by :func:`_from_spo` when ``op`` is its result.
    """
    if isinstance(op, SparsePauliOp):
        return op
    with _PL_TO_SPO_CACHE_LOCK:
        cached = _PL_TO_SPO_CACHE.get(op)
    if cached is not None:
        return cached[0]
    return observable_to_sparse_pauli_op(op, op.wires)


def _spo_wires(op: qp.operation.Operator | SparsePauliOp) -> tuple:
    """Wire mapping aligned with :func:`_to_spo` (qubit ``i`` ↔ ``wires[i]``).

    Reads the cached canonical wires recorded by :func:`_from_spo` —
    necessary because ``op.wires`` on a PL operator can drop or reorder
    entries after ``simplify()``.
    """
    if isinstance(op, SparsePauliOp):
        return tuple(range(_num_qubits(op)))
    with _PL_TO_SPO_CACHE_LOCK:
        cached = _PL_TO_SPO_CACHE.get(op)
    if cached is not None:
        return cached[1]
    return tuple(op.wires)


def _from_spo(
    spo: SparsePauliOp,
    wires: Sequence,
    *,
    simplify: bool = True,
) -> qp.operation.Operator:
    """``SparsePauliOp`` → PennyLane operator.

    Empty SPO maps to ``qp.Hamiltonian([], [])``. With ``simplify=True``
    (default) a final ``.simplify()`` collapses trivial wrappers like
    ``SProd(1.0, X)`` to their bare operator. The source SPO and the
    wire mapping (qubit ``i`` ↔ ``wires[i]``) are recorded in
    :data:`_PL_TO_SPO_CACHE` so :func:`_to_spo` / :func:`_spo_wires`
    recover them on the result.
    """
    if spo.size == 0:
        return qp.Hamiltonian([], [])
    pl = sparse_pauli_op_to_pl_observable(spo, wires)
    if simplify:
        pl = pl.simplify()
    try:
        with _PL_TO_SPO_CACHE_LOCK:
            _PL_TO_SPO_CACHE[pl] = (spo, tuple(wires))
    except TypeError:
        pass
    return pl


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


def _clean_hamiltonian_via_spo(
    hamiltonian: qp.operation.Operator,
) -> tuple[qp.operation.Operator, float]:
    """Partition constant (identity) terms from a PennyLane Hamiltonian.

    Returns ``(non-identity Hamiltonian, summed constant)``. Empty input
    maps to ``(qp.Hamiltonian([], []), constant)``. Routes through
    ``SparsePauliOp`` so the non-identity result carries a canonical
    qubit→wire mapping (qubit ``i`` ↔ ``wires[i]``).
    """
    spo = _to_spo(hamiltonian)
    non_id, constant = _clean_hamiltonian_spo(spo)
    if non_id.size == 0:
        return qp.Hamiltonian([], []), constant
    return _from_spo(non_id, _spo_wires(hamiltonian)), constant


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
    """First-order Trotter decomposition of ``exp(-i * time * H)`` as basis gates.

    Output matches ``qp.PauliRot`` → ``MultiRZ`` → CNOT/RZ decomposition.
    Returns PennyLane ops rather than calling ``qiskit.synthesis.LieTrotter``
    so the result composes with the surrounding PL-typed circuit pipeline.
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
    """Append exp(-i * time * spo) gates to ``qc`` as qiskit basis gates.

    Gate-for-gate equivalent of :func:`_spo_to_basis_gate_ops` but emits qiskit
    instructions directly, skipping the PL→qiskit conversion roundtrip.
    """
    qubit_list = list(qubits)
    for pauli_str, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
        c = float(np.real(coeff))
        # Qiskit labels are big-endian: pauli_str[-(i+1)] is qubit i.
        active = [
            (qubit_list[i], ch) for i, ch in enumerate(reversed(pauli_str)) if ch != "I"
        ]
        if not active:
            continue
        theta = 2 * time * c

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
        for i in range(n - 1, 0, -1):
            qc.cx(active_qubits[i], active_qubits[i - 1])
        qc.rz(theta, active_qubits[0])
        for i in range(1, n):
            qc.cx(active_qubits[i], active_qubits[i - 1])

        for q, ch in active:
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.rx(-_HALF_PI, q)
