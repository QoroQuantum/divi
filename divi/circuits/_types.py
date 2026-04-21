# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable

AxisLabel = tuple[
    str, Hashable
]  # A single (axis_name, value) pair used in batch and branch keys.

QASMTag = tuple[AxisLabel, ...]  # Sequence of AxisLabels labelling a QASM body variant.
