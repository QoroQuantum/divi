# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Contractual verification helpers for :class:`~divi.backends.CircuitRunner`.

Each ``verify_*`` takes a ready-to-use runner; the backend ``TestContracts``
class mocks dependencies and injects the instance.

Lists are grouped by runner fixture (``track_depth`` and sync/async shape):

* ``DEPTH_CONTRACTS_DISABLED`` / ``DEPTH_CONTRACTS_ENABLED``
* ``SYNC_RUNNER_CONTRACT_CASES`` / ``ASYNC_RUNNER_CONTRACT_CASES`` — ready-made
  ``(verify, fixture_name)`` tuples for parametrization
"""

from collections.abc import Callable
from threading import Event

import pytest

from divi.backends import CircuitRunner
from divi.exceptions import ExecutionCancelledError

ContractCase = tuple[Callable[[CircuitRunner], None], str]

CONTRACT_TEST_SHOTS = 10

QASM_MINIMAL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
    "h q[0];\nmeasure q[0] -> c[0];\n"
)

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

_CIRCUITS_ONE = {"c1": QASM_DEPTH_2}
_CIRCUITS_TWO = {"c1": QASM_DEPTH_2, "c2": QASM_DEPTH_3}
_BATCH_ONE = {"c1": QASM_DEPTH_2}
_BATCH_TWO = {"c2": QASM_DEPTH_3, "c3": QASM_DEPTH_3}


def flatten_contract_cases(
    *groups: tuple[list[Callable[[CircuitRunner], None]], str],
) -> list[ContractCase]:
    """Expand ``(contract_list, fixture_name)`` groups for parametrization."""
    return [(verify, fixture) for contracts, fixture in groups for verify in contracts]


def contract_case_id(case: ContractCase) -> str:
    verify, _fixture = case
    return verify.__name__.removeprefix("verify_")


def verify_depth_tracking_disabled(runner: CircuitRunner) -> None:
    """When ``track_depth`` is False, no depth data is recorded."""
    assert runner.track_depth is False

    runner.submit_circuits(_CIRCUITS_ONE)

    assert runner.depth_history == []
    assert runner.average_depth() == 0.0
    assert runner.std_depth() == 0.0


def verify_depth_tracking_records(runner: CircuitRunner) -> None:
    """When ``track_depth`` is True, depths are correctly recorded for a single batch."""
    assert runner.track_depth is True

    runner.submit_circuits(_CIRCUITS_TWO)

    assert len(runner.depth_history) == 1
    assert sorted(runner.depth_history[0]) == [2, 3]
    assert runner.average_depth() == pytest.approx(2.5)


def verify_depth_history_accumulates(runner: CircuitRunner) -> None:
    """Multiple ``submit_circuits`` calls each append a new batch entry."""
    assert runner.track_depth is True

    runner.submit_circuits(_BATCH_ONE)
    runner.submit_circuits(_BATCH_TWO)

    assert len(runner.depth_history) == 2
    assert len(runner.depth_history[0]) == len(_BATCH_ONE)
    assert len(runner.depth_history[1]) == len(_BATCH_TWO)


def verify_clear_depth_history(runner: CircuitRunner) -> None:
    """``clear_depth_history`` resets all depth state."""
    assert runner.track_depth is True

    runner.submit_circuits(_CIRCUITS_ONE)
    assert len(runner.depth_history) == 1

    runner.clear_depth_history()

    assert runner.depth_history == []
    assert runner.average_depth() == 0.0
    assert runner.std_depth() == 0.0


def verify_depth_history_returns_copy(runner: CircuitRunner) -> None:
    """``depth_history`` returns a copy; mutating it does not affect internal state."""
    assert runner.track_depth is True

    runner.submit_circuits(_CIRCUITS_ONE)
    history = runner.depth_history
    history.clear()

    assert len(runner.depth_history) == 1
    assert len(history) == 0


def verify_std_depth_zero_for_single_value(runner: CircuitRunner) -> None:
    """``std_depth`` returns 0.0 when only one depth value exists."""
    assert runner.track_depth is True

    runner.submit_circuits(_CIRCUITS_ONE)

    assert runner.std_depth() == 0.0


def verify_cancellation_before_dispatch(runner: CircuitRunner) -> None:
    """A pre-set ``cancellation_event`` aborts before backend dispatch."""
    event = Event()
    event.set()

    with pytest.raises(ExecutionCancelledError):
        runner.submit_circuits({"c1": QASM_MINIMAL}, cancellation_event=event)


DEPTH_CONTRACTS_DISABLED = [
    verify_depth_tracking_disabled,
]

DEPTH_CONTRACTS_ENABLED = [
    verify_depth_tracking_records,
    verify_depth_history_accumulates,
    verify_clear_depth_history,
    verify_depth_history_returns_copy,
    verify_std_depth_zero_for_single_value,
]

SYNC_CANCELLATION_CONTRACTS = [
    verify_cancellation_before_dispatch,
]

SYNC_RUNNER_CONTRACT_CASES = flatten_contract_cases(
    (DEPTH_CONTRACTS_DISABLED, "contract_runner_disabled"),
    (DEPTH_CONTRACTS_ENABLED, "contract_runner_enabled"),
    (SYNC_CANCELLATION_CONTRACTS, "contract_runner_default"),
)

# Async backends (e.g. QoroService) intentionally omit
# ``SYNC_CANCELLATION_CONTRACTS``: ``submit_circuits`` only initiates the job
# and accepts ``cancellation_event`` for interface parity, so a pre-dispatch
# abort isn't meaningful. Async cancellation is exercised through
# ``poll_job_status`` / ``cancel_job`` in the backend's own test file.
ASYNC_RUNNER_CONTRACT_CASES = flatten_contract_cases(
    (DEPTH_CONTRACTS_DISABLED, "contract_runner_disabled"),
    (DEPTH_CONTRACTS_ENABLED, "contract_runner_enabled"),
)


class SyncRunnerContractsBase:
    """Base class wiring ``SYNC_RUNNER_CONTRACT_CASES`` to a backend test class.

    Subclasses must provide ``contract_runner_disabled``,
    ``contract_runner_enabled``, and ``contract_runner_default`` fixtures.
    """

    @pytest.mark.parametrize("case", SYNC_RUNNER_CONTRACT_CASES, ids=contract_case_id)
    def test_contract(self, case, request):
        verify, fixture_name = case
        verify(request.getfixturevalue(fixture_name))


class AsyncRunnerContractsBase:
    """Base class wiring ``ASYNC_RUNNER_CONTRACT_CASES`` to a backend test class.

    Subclasses must provide ``contract_runner_disabled`` and
    ``contract_runner_enabled`` fixtures.
    """

    @pytest.mark.parametrize("case", ASYNC_RUNNER_CONTRACT_CASES, ids=contract_case_id)
    def test_contract(self, case, request):
        verify, fixture_name = case
        verify(request.getfixturevalue(fixture_name))
