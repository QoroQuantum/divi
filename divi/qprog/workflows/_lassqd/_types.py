# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the LASSQD configuration dataclasses."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import ConfigDict


def exact_ints(value: Any) -> Any:
    """``value`` with every numpy integer in it, including inside lists, tuples
    and mappings, made a Python ``int``.

    pydantic converts numpy integers through a float, losing digits past 2**53.
    """
    if isinstance(value, Mapping):
        return {exact_ints(key): exact_ints(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(exact_ints(item) for item in value)
    return int(value) if isinstance(value, np.integer) else value


#: Lets fields hold divi objects such as optimizers, checked by ``isinstance``.
CONFIG = ConfigDict(arbitrary_types_allowed=True)
