# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


class ExecutionCancelledError(Exception):
    """Signal that a running program, job, or batch was cooperatively cancelled."""
