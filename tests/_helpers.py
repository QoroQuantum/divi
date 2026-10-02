# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across the whole test suite."""

import re


def exact_match(message: str) -> str:
    """``pytest.raises(match=...)`` pattern matching ``message`` and nothing else."""
    return f"^{re.escape(message)}$"
