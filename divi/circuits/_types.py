# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from enum import Enum

AxisLabel = tuple[
    str, Hashable
]  # A single (axis_name, value) pair used in batch and branch keys.

QASMTag = tuple[AxisLabel, ...]  # Sequence of AxisLabels labelling a QASM body variant.


class ResultFormat(Enum):
    """Canonical format that raw backend results should be converted into.

    Set by a measurement stage during ``expand``; read by ``pipeline.run()``
    to apply the correct conversion between execute and reduce.
    """

    COUNTS = "counts"
    """Raw shot counts — no conversion. Used by PCE (nonlinear reduce)."""

    PROBS = "probs"
    """Probability distributions (``{bitstring: probability}``)."""

    EXPVALS = "expvals"
    """Expectation values (``{observable_key: float}`` mapping per branch key)."""
