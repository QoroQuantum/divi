# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# Import maestro first to prevent C++ library initialisation order conflicts
# with Qiskit / PennyLane that trigger segfaults in maestro.simple_estimate.
import maestro as _maestro  # noqa: F401

import logging as _logging

from rich.traceback import install as _install_rich_tracebacks

_logger = _logging.getLogger(__name__)
if not _logger.handlers:
    _logger.addHandler(_logging.NullHandler())

# Replace Python's default excepthook with Rich's pretty traceback renderer.
# Applies process-wide on first ``import divi``. Cancellation paths still
# raise their own exceptions; Rich just formats whatever reaches stderr.
_install_rich_tracebacks(show_locals=False)
