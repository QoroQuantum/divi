# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# Import maestro early (if available) to prevent C++ library initialisation
# order conflicts with Qiskit / PennyLane that trigger segfaults in
# maestro.simple_estimate.  See internal docs for details.
try:
    import maestro as _maestro  # noqa: F401
except ImportError:
    pass

from .reporting import enable_logging

enable_logging()
