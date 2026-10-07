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


def detect_cpu_threads() -> int:
    """Return a portable thread count for local model inference.

    Physical cores avoid simultaneous-multithreading contention on CPU-bound
    llama.cpp workloads. The result is capped by the process CPU affinity when
    the platform exposes it, so containers and restricted processes do not use
    unavailable CPUs. Logical CPUs are used only when physical-core detection
    is unavailable.
    """
    try:
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        return 1

    count = physical or logical or 1
    try:
        affinity = len(psutil.Process().cpu_affinity())
    except Exception:
        affinity = 0
    if affinity:
        count = min(count, affinity)
    return max(1, count)
