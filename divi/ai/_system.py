# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""System hardware detection for divi-ai.

Provides lightweight helpers that identify the host CPU architecture and
available RAM so the model-selection UI can highlight the best option.
"""

import platform

import psutil


def detect_arch() -> str:
    """Return a normalized CPU architecture string.

    Returns
    -------
    str
        ``"apple_silicon"``, ``"x86_64"``, ``"arm64"``, or the raw value
        from :func:`platform.machine` if none of these match.
    """
    raw = platform.machine().lower()
    if raw in ("arm64", "aarch64"):
        if platform.system() == "Darwin":
            return "apple_silicon"
        return "arm64"
    if raw in ("x86_64", "amd64"):
        return "x86_64"
    return raw


def detect_ram_gb() -> float | None:
    """Return total system RAM in gigabytes, or ``None`` on failure.

    Uses :mod:`psutil` for cross-platform support (Linux, macOS, Windows).
    """
    try:
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return None
