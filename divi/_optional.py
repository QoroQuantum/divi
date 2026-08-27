# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Precise loading for dependencies provided through optional extras."""

import importlib
from types import ModuleType


def _missing_requested_package(exc: ModuleNotFoundError, module_name: str) -> bool:
    """Whether *exc* says the requested top-level package is absent."""
    root = module_name.partition(".")[0]
    return exc.name == root


def optional_module(module_name: str) -> ModuleType | None:
    """Import *module_name*, returning ``None`` only when its package is absent."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if _missing_requested_package(exc, module_name):
            return None
        raise


def import_optional(module_name: str, *, extra: str, capability: str) -> ModuleType:
    """Import an optional module or raise an error naming its Divi extra."""
    module = optional_module(module_name)
    if module is None:
        raise ImportError(
            f"{capability} requires the '{extra}' extra; install it with "
            f"`pip install qoro-divi[{extra}]`."
        )
    return module
