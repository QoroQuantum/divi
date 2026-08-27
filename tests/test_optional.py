# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import ModuleType

import pytest

from divi._optional import import_optional, optional_module


def test_optional_module_returns_imported_module(mocker):
    expected = ModuleType("available")
    mocker.patch.object(importlib, "import_module", return_value=expected)

    assert optional_module("available") is expected


def test_optional_module_returns_none_when_requested_package_is_absent(mocker):
    missing = ModuleNotFoundError("missing", name="missing")
    mocker.patch.object(importlib, "import_module", side_effect=missing)

    assert optional_module("missing.submodule") is None


def test_optional_module_preserves_transitive_import_failure(mocker):
    transitive = ModuleNotFoundError("broken dependency", name="dependency")
    mocker.patch.object(importlib, "import_module", side_effect=transitive)

    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        optional_module("available")


def test_import_optional_names_capability_and_extra(mocker):
    missing = ModuleNotFoundError("missing", name="pennylane")
    mocker.patch.object(importlib, "import_module", side_effect=missing)

    with pytest.raises(
        ImportError,
        match=(
            r"QNode conversion requires the 'pennylane' extra.*"
            r"qoro-divi\[pennylane\]"
        ),
    ):
        import_optional("pennylane", extra="pennylane", capability="QNode conversion")
