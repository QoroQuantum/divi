# Copyright 2018 The Cirq Developers

from __future__ import annotations

from typing import TYPE_CHECKING

from ._parser import QasmParser

if TYPE_CHECKING:
    import cirq


def cirq_circuit_from_qasm(qasm: str) -> cirq.Circuit:
    """Parses an OpenQASM string to `cirq.Circuit`.

    Args:
        qasm: The OpenQASM string

    Returns:
        The parsed circuit
    """

    return QasmParser().parse(qasm).circuit
