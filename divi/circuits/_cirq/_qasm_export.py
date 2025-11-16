# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Patch for Cirq's QasmArgs.format_field to support symbolic parameters in QASM export.

This module patches Cirq's QasmArgs.format_field method to properly handle:
1. Symbolic parameters (Sympy expressions) with "half_turns" spec for rotation gates
2. Float precision rounding for numeric parameters
3. Qubit ID mapping for Qid objects

Note: Measurement keys are handled directly by Cirq's qasm() function and do not
require special handling in format_field.

The patch is essential for QEM (Quantum Error Mitigation) protocols that modify circuits
with symbolic parameters. When Cirq exports QASM code using cirq.qasm(), it formats rotation
gate angles using the "half_turns" spec. Without this patch, Sympy expressions would not be
properly converted to QASM format (e.g., "theta*pi" instead of just "theta").

Example:
    When a circuit contains a rotation gate with a Sympy symbol:
        rz(theta * pi) q[0]

    Cirq's QASM export calls format_field with the Sympy expression and spec="half_turns".
    This patch ensures the expression is properly formatted for QASM output.
"""

from typing import Any

from cirq import ops
from cirq.protocols.qasm import QasmArgs
from sympy import Expr, pi


def patched_format_field(self, value: Any, spec: str) -> str:
    """Patched version of QasmArgs.format_field for symbolic parameter support.

    This method extends Cirq's QasmArgs.format_field to handle:
    - Sympy expressions with "half_turns" spec (for rotation gate angles)
    - Float precision rounding
    - Qubit ID mapping

    Args:
        self: QasmArgs instance
        value: The value to format (can be float, int, Sympy Expr, or Qid)
        spec: Format specifier (e.g., "half_turns" or empty string)

    Returns:
        Formatted string representation of the value suitable for QASM output
    """
    # Handle numeric values (floats and integers)
    if isinstance(value, (float, int)):
        if isinstance(value, float):
            value = round(value, self.precision)
        if spec == "half_turns":
            # Format as "pi*value" for QASM rotation gates
            value = f"pi*{value}" if value != 0 else "0"
            spec = ""

    # Handle Cirq Qid objects (qubits)
    elif isinstance(value, ops.Qid):
        value = self.qubit_id_map[value]

    # Handle Sympy expressions (symbolic parameters)
    if isinstance(value, Expr):
        if spec == "half_turns":
            # Multiply by pi for rotation gates (Cirq uses half-turns internally)
            value *= pi
        return str(value)

    # Fall back to parent implementation for other cases
    return super(QasmArgs, self).format_field(value, spec)


# Apply the patch to QasmArgs.format_field
# This is done at module import time so all QASM exports use the patched version
QasmArgs.format_field = patched_format_field
