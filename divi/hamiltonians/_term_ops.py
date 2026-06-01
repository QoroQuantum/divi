# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level manipulation primitives for ``SparsePauliOp`` Hamiltonians."""

import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pennylane as qp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter


def _assert_hermitian_spo(spo: SparsePauliOp, atol: float = 1e-10) -> None:
    """Validate that a Pauli-basis observable has real coefficients.

    Checks each coefficient's imaginary part directly. Pathological cases
    where individually non-Hermitian Pauli terms cancel after summation
    (e.g. ``+i X`` and ``-i X``) are not caught — callers that may produce
    such inputs should pass an already-simplified ``SparsePauliOp``.
    """
    if spo.size == 0:
        return
    if np.any(np.abs(np.imag(spo.coeffs)) > atol):
        raise ValueError(
            "SparsePauliOp observables must be Hermitian; Pauli coefficients "
            "must be real."
        )


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


_PAULI_CHARS = frozenset("IXYZ")


def _spo_from_pauli_dict(terms: dict[str, float]) -> SparsePauliOp:
    """Build a ``SparsePauliOp`` from a ``{pauli_string: coefficient}`` mapping.

    Pauli strings are read in divi convention: the **leftmost** character is
    qubit 0. ``{"XXIY": 1.0}`` therefore means ``X(0) X(1) I(2) Y(3)``.
    Internally each key is reversed before handing to
    :meth:`qiskit.quantum_info.SparsePauliOp.from_list`, which expects qubit 0
    on the right — so the produced SPO's ``.to_labels()`` output is the
    *reverse* of the input keys, but the symplectic representation
    (``spo.paulis.x[:, qubit]``, ``spo.paulis.z[:, qubit]``) lines up with
    the qubit indices the user specified.

    Validates that every key is a non-empty string of equal length composed
    only of ``I``, ``X``, ``Y``, ``Z`` and that every coefficient is real.
    Dict keys are unique by construction, so the cancellation gap noted on
    :func:`~divi.circuits._core._assert_hermitian_spo` (where ``+i X`` and
    ``-i X`` could cancel to a hermitian sum despite individually being
    non-hermitian) cannot arise on this path.
    """
    if not terms:
        raise ValueError(
            "to_spo: cannot build a SparsePauliOp from an empty dict — "
            "qubit count is undefined."
        )

    first_len: int | None = None
    items: list[tuple[str, float]] = []
    for key, coeff in terms.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"to_spo: Pauli-string keys must be non-empty strings, got {key!r}."
            )
        if not set(key).issubset(_PAULI_CHARS):
            raise ValueError(
                f"to_spo: key {key!r} contains characters outside {{I, X, Y, Z}}."
            )
        if first_len is None:
            first_len = len(key)
        elif len(key) != first_len:
            raise ValueError(
                f"to_spo: all Pauli-string keys must share a length; "
                f"got {first_len} and {len(key)}."
            )
        if isinstance(coeff, complex) and coeff.imag != 0:
            raise ValueError(
                f"to_spo: coefficient for {key!r} must be real, got {coeff!r}."
            )
        # Reverse: divi convention puts qubit 0 leftmost; Qiskit's parser
        # puts qubit 0 rightmost.
        items.append((key[::-1], float(np.real(coeff))))

    return SparsePauliOp.from_list(items)


def _observable_to_sparse_pauli_op(
    obs: qp.operation.Operator,
    wires,
) -> SparsePauliOp:
    """Convert a PennyLane observable to a Qiskit :class:`~qiskit.quantum_info.SparsePauliOp`.

    Handles arbitrary wire labels (strings, tuples, non-contiguous ints)
    by resolving through the provided *wires* register.

    Coefficients are stored as real floats.  A warning is emitted if any
    coefficient has a non-negligible imaginary part (>1e-10), which would
    indicate a non-Hermitian observable.
    """
    pauli_rep = obs.pauli_rep
    if pauli_rep is None:
        raise ValueError(
            f"Observable {obs!r} has no Pauli representation; cannot "
            f"convert to SparsePauliOp."
        )
    wire_list = list(wires)
    num_qubits = len(wire_list)

    sparse: list[tuple[str, list[int], float]] = []
    for pauli_word, coeff in pauli_rep.items():
        c = complex(coeff)
        if abs(c.imag) > 1e-10:
            warnings.warn(
                f"Observable coefficient {c} has non-negligible imaginary "
                f"part ({c.imag:.2e}); dropping it. This may indicate a "
                f"non-Hermitian observable.",
                stacklevel=2,
            )
        pw_items = list(dict(pauli_word).items())
        if not pw_items:
            sparse.append(("", [], c.real))
        else:
            pauli_chars = "".join(ch for _, ch in pw_items)
            qubit_indices = [wire_list.index(w) for w, _ in pw_items]
            sparse.append((pauli_chars, qubit_indices, c.real))

    return SparsePauliOp.from_sparse_list(sparse, num_qubits=num_qubits)


def to_spo(
    op: qp.operation.Operator | SparsePauliOp | dict[str, float],
    *,
    wires=None,
) -> SparsePauliOp:
    """Convert a PennyLane operator, ``SparsePauliOp``, or Pauli-string
    dict to ``SparsePauliOp``, validating Hermiticity in every case.

    The PennyLane branch builds a new ``SparsePauliOp`` by walking the
    operator tree. The dict branch accepts ``{pauli_string: coefficient}``
    mappings such as ``{"XXIY": 1.0, "ZIII": -0.5}`` — every key must be a
    non-empty string over ``{I, X, Y, Z}``, all keys must share a length,
    and coefficients must be real.

    .. note::

        Pauli strings are read in **divi convention**: the leftmost
        character is qubit 0 (so ``"XXIY"`` means ``X(0) X(1) I(2) Y(3)``).
        Qiskit's native :meth:`~qiskit.quantum_info.SparsePauliOp.from_list`
        and the ``.to_labels()`` output of the returned SPO use the
        opposite convention (qubit 0 rightmost), so dict keys you type in
        and the labels you read back will look reversed — the symplectic
        representation is what stays consistent across both forms.

    Args:
        op: Operator to convert.
        wires: Optional wire register to resolve a PennyLane operator
            against. When ``None`` (default), falls back to ``op.wires``,
            which yields an SPO whose qubit count equals the operator's
            own wire count. Pass an explicit wires register when the
            surrounding circuit is wider than the operator's own support
            (e.g. a single-qubit observable inside an n-qubit script).
            Ignored for ``SparsePauliOp`` and dict inputs.

    For repeated use on the same observable, convert once at setup and
    reuse the returned ``SparsePauliOp``.
    """
    if isinstance(op, SparsePauliOp):
        _assert_hermitian_spo(op)
        return op
    if isinstance(op, dict):
        spo = _spo_from_pauli_dict(op)
        _assert_hermitian_spo(spo)
        return spo
    spo = _observable_to_sparse_pauli_op(op, wires if wires is not None else op.wires)
    _assert_hermitian_spo(spo)
    return spo


def _spo_wires(op: qp.operation.Operator | SparsePauliOp) -> tuple:
    """Wire mapping aligned with :func:`to_spo` (qubit ``i`` ↔ ``wires[i]``)."""
    if isinstance(op, SparsePauliOp):
        return tuple(range(_require_qiskit_num_qubits(op.num_qubits)))
    return tuple(op.wires)


def _clean_hamiltonian_spo(
    spo: SparsePauliOp, *, raise_on_constant: bool = False
) -> tuple[SparsePauliOp, float]:
    """Partition identity-only rows from the rest. Returns ``(non-identity SPO, constant)``.

    The returned SPO has ``size == 0`` when the input contains only identity
    terms; callers must use ``size`` (not ``simplify()``) to detect emptiness.

    Set ``raise_on_constant=True`` to reject a constant-only operator instead —
    variational programs (VQE, QAOA, CustomVQA, QNN) have no objective to
    minimize when nothing but identity terms remain. Callers that legitimately
    tolerate constants (e.g. time evolution) keep the default.
    """
    if spo.size == 0:
        if raise_on_constant:
            raise ValueError("Hamiltonian contains only constant terms.")
        return spo, 0.0
    non_id_mask = np.any(spo.paulis.x | spo.paulis.z, axis=1)
    # ``math.fsum`` is exact-rounding; protects against alternating-sign
    # cancellation that ``ndarray.sum`` accumulates left-to-right.
    constant = math.fsum(spo.coeffs[~non_id_mask].real)
    cleaned = SparsePauliOp(spo.paulis[non_id_mask], spo.coeffs[non_id_mask])
    if non_id_mask.any():
        cleaned = cleaned.simplify()
    if raise_on_constant and cleaned.size == 0:
        raise ValueError("Hamiltonian contains only constant terms.")
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
