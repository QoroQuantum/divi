# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contractual verification helpers for CircuitRunner depth tracking.

Every CircuitRunner subclass must satisfy these contracts.  Each function
takes a *ready-to-use* runner and the QASM circuits needed for the check;
the caller is responsible for mocking whatever the concrete backend
needs (API calls, simulators, transpilation, …).
"""

import pytest

from divi.backends import CircuitRunner

# ---------------------------------------------------------------------------
# Re-usable QASM snippets – callers may also supply their own.
# ---------------------------------------------------------------------------

# h + measure → depth 2
QASM_DEPTH_2 = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)

# h + cx + measure → depth 3
QASM_DEPTH_3 = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


# ---------------------------------------------------------------------------
# Contract functions
# ---------------------------------------------------------------------------


def verify_depth_tracking_disabled(runner: CircuitRunner, circuits: dict[str, str]):
    """When ``track_depth`` is False, no depth data is recorded."""
    assert runner.track_depth is False

    runner.submit_circuits(circuits)

    assert runner.depth_history == []
    assert runner.average_depth() == 0.0
    assert runner.std_depth() == 0.0


def verify_depth_tracking_records(
    runner: CircuitRunner,
    circuits: dict[str, str],
    expected_depths_sorted: list[int],
):
    """When ``track_depth`` is True, depths are correctly recorded for a single batch."""
    assert runner.track_depth is True

    runner.submit_circuits(circuits)

    assert len(runner.depth_history) == 1
    assert sorted(runner.depth_history[0]) == expected_depths_sorted
    assert runner.average_depth() == pytest.approx(
        sum(expected_depths_sorted) / len(expected_depths_sorted)
    )


def verify_depth_history_accumulates(
    runner: CircuitRunner,
    circuits_batch_1: dict[str, str],
    circuits_batch_2: dict[str, str],
):
    """Multiple ``submit_circuits`` calls each append a new batch entry."""
    assert runner.track_depth is True

    runner.submit_circuits(circuits_batch_1)
    runner.submit_circuits(circuits_batch_2)

    assert len(runner.depth_history) == 2
    assert len(runner.depth_history[0]) == len(circuits_batch_1)
    assert len(runner.depth_history[1]) == len(circuits_batch_2)


def verify_clear_depth_history(runner: CircuitRunner, circuits: dict[str, str]):
    """``clear_depth_history`` resets all depth state."""
    assert runner.track_depth is True

    runner.submit_circuits(circuits)
    assert len(runner.depth_history) == 1

    runner.clear_depth_history()

    assert runner.depth_history == []
    assert runner.average_depth() == 0.0
    assert runner.std_depth() == 0.0


def verify_depth_history_returns_copy(runner: CircuitRunner, circuits: dict[str, str]):
    """``depth_history`` returns a copy; mutating it does not affect internal state."""
    assert runner.track_depth is True

    runner.submit_circuits(circuits)
    history = runner.depth_history
    history.clear()

    assert len(runner.depth_history) == 1
    assert len(history) == 0


def verify_std_depth_zero_for_single_value(
    runner: CircuitRunner, circuits: dict[str, str]
):
    """``std_depth`` returns 0.0 when only one depth value exists."""
    assert runner.track_depth is True

    runner.submit_circuits(circuits)

    assert runner.std_depth() == 0.0
