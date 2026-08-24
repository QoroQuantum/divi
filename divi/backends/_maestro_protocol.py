# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The conventions Maestro itself imposes, separate from how it is driven.

Maestro's QASM dialect, its MPS sizing rule and its result bit order are
facts about the simulator, not about any one way of reaching it. Keeping
them here leaves :mod:`~divi.backends.runners._maestro` to deal only with
execution.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Bond dimension applied when auto-MPS selects MPS for a batch.
MPS_AUTO_BOND_DIMENSION = 64

#: Qubit count above which a batch switches from statevector to MPS.
MPS_QUBIT_THRESHOLD = 22

_QREG_RE = re.compile(r"qreg\s+q\[(\d+)\]")


def strip_id_gates(qasm: str) -> str:
    """Remove ``id`` (identity) gates from QASM.

    Maestro's QASM parser does not recognise the ``id`` gate. Since identity
    gates are no-ops, stripping them is safe.
    """
    return re.sub(r"id\s+q\[\d+\]\s*;\n?", "", qasm)


def qasm_n_qubits(qasm: str, label: str) -> int:
    """Read the register width Maestro should allocate for one circuit."""
    match = _QREG_RE.search(qasm)
    if match is None:
        raise ValueError(f"Circuit '{label}' declares no 'qreg q[N]'.")
    return int(match.group(1))


def counts_to_little_endian(counts: Mapping[str, int]) -> dict[str, int]:
    """Flip one circuit's count keys from Maestro's bit order to Qiskit's.

    Maestro puts ``q[0]`` leftmost; Qiskit puts it rightmost.
    """
    return {bits[::-1]: n for bits, n in counts.items()}


def expvals_from_result(
    raw: Mapping[str, Any], terms: Sequence[str]
) -> dict[str, float]:
    """Pair a Maestro estimate result with the Pauli terms it was asked for."""
    return dict(zip(terms, raw["expectation_values"]))
