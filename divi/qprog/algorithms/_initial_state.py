# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared initial-state preparation utilities.

Provides validation and gate-construction helpers consumed by
TimeEvolution, QAOA, VQE, and any future algorithm that prepends
an initial-state preparation layer to its circuit.
"""

from __future__ import annotations

from typing import Literal, Sequence, get_args

import pennylane as qml

_INITIAL_STATE_LITERAL = Literal["Zeros", "Ones", "Superposition"]
_CUSTOM_CHARS = frozenset("01+-")


def validate_initial_state(
    initial_state: str,
    n_qubits: int,
) -> None:
    """Validate an initial-state specification.

    Accepted values:
    * One of the literal keywords: ``"Zeros"``, ``"Ones"``, ``"Superposition"``.
    * A custom per-qubit string composed of the characters ``0``, ``1``,
      ``+`` and ``-`` whose length must equal *n_qubits*.

    Args:
        initial_state: The initial-state specification to validate.
        n_qubits: Expected number of qubits.

    Raises:
        ValueError: If *initial_state* is not a recognised literal and is not a
            valid custom string, or if the custom string length does not match
            *n_qubits*.
    """
    if initial_state in get_args(_INITIAL_STATE_LITERAL):
        return

    is_valid_custom = bool(initial_state) and all(
        c in _CUSTOM_CHARS for c in initial_state
    )
    if is_valid_custom:
        if len(initial_state) != n_qubits:
            raise ValueError(
                f"initial_state string length ({len(initial_state)}) "
                f"must match number of qubits ({n_qubits})."
            )
        return

    raise ValueError(
        f"initial_state must be one of {get_args(_INITIAL_STATE_LITERAL)} "
        f"or a string of '0', '1', '+', '-', got {initial_state!r}"
    )


def build_initial_state_ops(
    initial_state: str,
    wires: Sequence[int],
) -> list[qml.operation.Operator]:
    """Return gate operations that prepare *initial_state* on *wires*.

    The returned list is meant to be **prepended** to any circuit's
    operation list.

    Mapping:
    * ``"Zeros"`` → empty list (computational basis default).
    * ``"Ones"``  → ``PauliX`` on every wire.
    * ``"Superposition"`` → ``Hadamard`` on every wire.
    * Custom string (per-qubit): ``'0'`` → nothing, ``'1'`` → ``PauliX``,
      ``'+'`` → ``Hadamard``, ``'-'`` → ``PauliX`` then ``Hadamard``.

    Args:
        initial_state: A validated initial-state specification.
        wires: Ordered sequence of wire labels.

    Returns:
        List of PennyLane operations.
    """
    if initial_state == "Zeros":
        return []

    ops: list[qml.operation.Operator] = []

    if initial_state == "Ones":
        for w in wires:
            ops.append(qml.PauliX(wires=w))
    elif initial_state == "Superposition":
        for w in wires:
            ops.append(qml.Hadamard(wires=w))
    else:
        # Custom per-qubit string
        for w, char in zip(wires, initial_state):
            if char == "1":
                ops.append(qml.PauliX(wires=w))
            elif char == "+":
                ops.append(qml.Hadamard(wires=w))
            elif char == "-":
                ops.append(qml.PauliX(wires=w))
                ops.append(qml.Hadamard(wires=w))
            # '0' → nothing

    return ops
