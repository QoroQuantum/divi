# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from cirq import ops
from cirq.protocols.qasm import QasmArgs


def patched_format_field(self, value: Any, spec: str) -> str:
    """Method of string.Formatter that specifies the output of format()."""
    if isinstance(value, (float, int)):
        if isinstance(value, float):
            value = round(value, self.precision)
        if spec == "half_turns":
            value = f"pi*{value}" if value != 0 else "0"
            spec = ""

    elif isinstance(value, ops.Qid):
        value = self.qubit_id_map[value]

    elif isinstance(value, str) and spec == "meas":
        value = self.meas_key_id_map[value]
        spec = ""

    from sympy import Expr, pi

    if isinstance(value, Expr):
        if spec == "half_turns":
            value *= pi
        return str(value)

    return super(QasmArgs, self).format_field(value, spec)


QasmArgs.format_field = patched_format_field
