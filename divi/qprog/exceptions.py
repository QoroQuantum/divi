# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


class _CancelledError(Exception):
    """Internal exception to signal a task to stop due to cancellation."""

    pass
