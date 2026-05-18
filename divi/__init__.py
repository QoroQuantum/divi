# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# Import maestro early (if available) to prevent C++ library initialization
# order conflicts with Qiskit / PennyLane that trigger segfaults in
# maestro.simple_estimate.  See internal docs for details.
try:
    # pyrefly: ignore[missing-import]  # ``maestro`` ships separately
    import maestro as _maestro  # noqa: F401
except ImportError:
    pass

from rich.traceback import install as _install_rich_tracebacks

from .reporting import enable_logging

enable_logging()

# Replace Python's default excepthook with Rich's pretty traceback renderer.
# Applies process-wide on first ``import divi``. Cancellation paths still
# raise their own exceptions; Rich just formats whatever reaches stderr.
_install_rich_tracebacks(show_locals=False)
