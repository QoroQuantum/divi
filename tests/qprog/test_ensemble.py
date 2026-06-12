# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Event
from threading import Event as ThreadingEvent
from threading import Thread

import networkx as nx
import numpy as np
import pytest
from rich.panel import Panel
from rich.traceback import Traceback

import divi.qprog.ensemble as ensemble_module
from divi.backends import AsyncJobBackend, ExecutionResult
from divi.exceptions import ExecutionCancelledError
from divi.qprog._batch_coordinator import _BatchCoordinator, _ProxyBackend
from divi.qprog.ensemble import (
    BatchConfig,
    BatchMode,
    ProgramEnsemble,
    _beam_search_aggregate_top_n,
    _hierarchical_aggregate_top_n,
)
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import GraphPartitioningConfig, MaxCutProblem
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.variational_quantum_algorithm import SolutionEntry
from divi.qprog.workflows import PartitioningProgramEnsemble
from divi.reporting import TerminalStatus
from tests.qprog._program_contracts import verify_basic_program_ensemble_behaviour


class _FakeRunResult:
    """Mimics a program instance returned by run() for mocked futures."""

    def __init__(self, circuit_count: int, run_time: float):
        self._total_circuit_count = circuit_count
        self._total_run_time = run_time


class _StubProgram(QuantumProgram):
    """Base for ProgramEnsemble test programs.

    Stubs the abstract methods and tracks a ``_ran`` flag; subclasses
    implement only ``run()`` (and any extra constructor args they need).
    """

    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self._ran = False

    def _build_pipelines(self) -> None:
        pass

    def has_results(self) -> bool:
        return self._ran

    def _generate_circuits(self, **kwargs):
        return []

    def _post_process_results(self, results):
        pass


class SimpleTestProgram(_StubProgram):
    """A simple mock program whose ``run()`` assigns preset counter values."""

    def __init__(self, circ_count: int, run_time: float, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.circ_count = circ_count
        self.run_time = run_time

    def run(self):
        self._total_circuit_count = self.circ_count
        self._total_run_time = self.run_time
        self._ran = True
        return self


class SampleProgramEnsemble(ProgramEnsemble):
    """A mock ProgramEnsemble for testing."""

    def __init__(self, backend):
        super().__init__(backend)
        self.max_iterations = 5

    def create_programs(self):
        """Creates a set of mock programs."""
        super().create_programs()
        self.programs = {
            "prog1": SimpleTestProgram(
                10, 5.5, backend=self.backend, program_id="prog1"
            ),
            "prog2": SimpleTestProgram(
                5, 10.0, backend=self.backend, program_id="prog2"
            ),
        }

    def aggregate_results(self):
        """A mock aggregation function."""
        # The super() call is important to trigger checks and the join()
        super().aggregate_results()
        return sum(p.circ_count for p in self.programs.values())


@pytest.fixture
def program_ensemble(dummy_simulator):
    batch = SampleProgramEnsemble(backend=dummy_simulator)
    yield batch
    try:
        batch.reset()
    except Exception:
        pass  # Don't break test teardown due to a race condition


@pytest.fixture(autouse=True)
def stop_live_display(program_ensemble):
    """Fixture to automatically stop any active Rich progress bars after a test."""
    yield
    if (
        hasattr(program_ensemble, "_progress_bar")
        and program_ensemble._progress_bar is not None
        and not program_ensemble._progress_bar.finished
    ):
        program_ensemble._progress_bar.stop()


class TestProgramEnsemble:
    def test_correct_initialization(self, program_ensemble):
        assert program_ensemble._executor is None
        assert len(program_ensemble.programs) == 0
        assert program_ensemble.total_circuit_count == 0
        assert program_ensemble.total_run_time == 0.0

    def test_basic_program_ensemble_behaviour(self, program_ensemble, mocker):
        """Uses the contract to verify basic error handling and state checks."""
        verify_basic_program_ensemble_behaviour(program_ensemble, mocker)

    def test_programs_dict_is_correct(self, program_ensemble):
        program_ensemble.create_programs()
        assert len(program_ensemble.programs) == 2
        assert "prog1" in program_ensemble.programs
        assert "prog2" in program_ensemble.programs
        assert hasattr(program_ensemble._queue, "get")  # Check if it's queue-like
        # create_programs() only sets up the progress queue (sub-programs bind
        # to it at construction). _done_event is created per-run by run().
        assert not hasattr(program_ensemble, "_done_event")

    def test_reset_cleans_up_all_resources(self, program_ensemble, mocker):
        """Tests that reset() correctly shuts down and cleans up all resources."""
        # First, create programs to initialize the manager, queue, etc.
        program_ensemble.create_programs()
        # Now, simulate a running state by creating an executor and futures.
        # run() creates _done_event per-run, so set it up here to mirror that.
        program_ensemble._done_event = ThreadingEvent()
        program_ensemble._executor = mocker.MagicMock(spec=ThreadPoolExecutor)
        program_ensemble._listener_thread = mocker.MagicMock(spec=Thread)
        program_ensemble._progress_bar = mocker.MagicMock()
        program_ensemble._live_display = mocker.MagicMock()
        program_ensemble.futures = [mocker.MagicMock()]
        program_ensemble._pb_task_map = {}

        # Configure the mock to behave like a stopped thread to prevent warnings
        program_ensemble._listener_thread.is_alive.return_value = False

        # Spy on the original objects before they are cleared
        mock_executor = program_ensemble._executor
        done_event_set_spy = mocker.spy(program_ensemble._done_event, "set")
        mock_listener_thread = program_ensemble._listener_thread
        mock_live_display = program_ensemble._live_display

        # Call the method under test
        program_ensemble.reset()

        # Assert that cleanup methods were called on the spied objects
        mock_executor.shutdown.assert_called_once_with(wait=False)
        done_event_set_spy.assert_called_once()
        mock_listener_thread.join.assert_called_once()
        mock_live_display.stop.assert_called_once()

        # Assert that state attributes are cleared
        assert program_ensemble._executor is None
        assert program_ensemble.futures == []

    def test_total_circuit_count_setter(self, program_ensemble):
        with pytest.raises(
            AttributeError,
            match="property 'total_circuit_count' of 'SampleProgramEnsemble'",
        ):
            program_ensemble.total_circuit_count = 100

    def test_total_run_time_setter(self, program_ensemble):
        with pytest.raises(
            AttributeError,
            match="property 'total_run_time' of 'SampleProgramEnsemble'",
        ):
            program_ensemble.total_run_time = 100

    def test_run_returns_expected_number_of_futures(self, program_ensemble):
        program_ensemble.create_programs()
        program_ensemble.run()
        assert len(program_ensemble.futures) == 2

    def test_run_fails_if_already_running(self, mocker, program_ensemble):
        program_ensemble.create_programs()

        # Mock ThreadPoolExecutor to simulate a long-running batch
        mock_executor = mocker.patch("divi.qprog.ensemble.ThreadPoolExecutor")
        mock_instance = mock_executor.return_value

        # Create futures that simulate a long-running process
        future1 = Future()
        future2 = Future()
        mock_instance.submit.side_effect = [future1, future2]

        # First run should work
        program_ensemble.run()

        # Subsequent run should raise an exception
        with pytest.raises(RuntimeError, match="An ensemble is already being run."):
            program_ensemble.run()

    def test_run_submits_correct_tasks(self, program_ensemble, mocker):
        """Tests that run() submits the correct number of tasks to the executor."""
        program_ensemble.create_programs()
        mock_executor_class = mocker.patch("divi.qprog.ensemble.ThreadPoolExecutor")
        mock_executor = mock_executor_class.return_value

        # The executor's submit method returns Future objects.
        mock_future_1 = mocker.MagicMock(spec=Future)
        mock_future_2 = mocker.MagicMock(spec=Future)
        mock_executor.submit.side_effect = [mock_future_1, mock_future_2]

        # Mock `as_completed` so that a later call to join() doesn't hang.
        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            return_value=[mock_future_1, mock_future_2],
        )

        # Run non-blocking to inspect the state before it's cleaned up
        program_ensemble.run(blocking=False)

        assert mock_executor.submit.call_count == len(program_ensemble.programs)

        # Each submit call should receive a callable and the correct program
        programs = list(program_ensemble.programs.values())
        for i, call_args in enumerate(mock_executor.submit.call_args_list):
            submitted_fn, submitted_program = call_args[0]
            assert callable(submitted_fn)
            assert submitted_program is programs[i]

        # Clean up the non-blocking run. This should now terminate correctly.
        program_ensemble.join()

    def test_blocking_run_executes_and_joins_correctly(self, program_ensemble):
        """Integration test for a standard blocking run."""
        program_ensemble.create_programs()
        # run(blocking=True) is the default, which should execute and then join.
        result = program_ensemble.run(blocking=True)

        assert result is program_ensemble
        assert program_ensemble.total_circuit_count == 15
        assert program_ensemble.total_run_time == 15.5
        # After a blocking run, the executor should be gone.
        assert program_ensemble._executor is None
        assert len(program_ensemble.futures) == 0

    def test_non_blocking_run_registers_atexit_hook(self, program_ensemble, mocker):
        """Tests that a non-blocking run correctly registers a cleanup hook."""
        mock_atexit_register = mocker.patch("atexit.register")
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        # Check that the executor is active and hook is registered
        assert program_ensemble._executor is not None
        mock_atexit_register.assert_called_once_with(
            program_ensemble._atexit_cleanup_hook
        )

        # Manually clean up to avoid side effects
        program_ensemble.join()

    def test_join_unregisters_atexit_hook(self, program_ensemble, mocker):
        """Tests that join() unregisters the cleanup hook after a non-blocking run."""
        mock_atexit_register = mocker.patch("atexit.register")
        mock_atexit_unregister = mocker.patch("atexit.unregister")

        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)  # This registers the hook

        mock_atexit_register.assert_called_once()

        program_ensemble.join()  # This should unregister it

        mock_atexit_unregister.assert_called_once_with(
            program_ensemble._atexit_cleanup_hook
        )

    def test_check_all_done_true_when_all_futures_ready(self, program_ensemble):
        future_1 = Future()
        future_2 = Future()
        program_ensemble.futures = [future_1, future_2]

        # Test when no futures are done
        assert not program_ensemble.check_all_done()

        # Complete one future
        future_1.set_result(None)
        assert not program_ensemble.check_all_done()

        # Complete second future
        future_2.set_result(None)
        assert program_ensemble.check_all_done()

    def test_join_handles_task_exceptions(self, program_ensemble, mocker):
        """Ensures join() catches exceptions from futures, collects partial results, and cleans up."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        failing_future = Future()
        failing_future.set_exception(ValueError("Task failed"))
        successful_future = Future()
        successful_future.set_result(_FakeRunResult(5, 5.0))
        program_ensemble.futures = [failing_future, successful_future]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            failing_future: progs[0],
            successful_future: progs[1],
        }

        # as_completed is called twice: once in the join() loop (yields the
        # failing future), once in _stop_remaining_programs (no unstoppable
        # futures because both are already done).
        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[
                iter([failing_future, successful_future]),
                iter([]),
            ],
        )
        mock_shutdown = mocker.spy(program_ensemble._executor, "shutdown")

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            program_ensemble.join()

        mock_shutdown.assert_called_once_with(wait=True)
        assert program_ensemble._executor is None
        # The successful future's results should still be collected in the finally block
        assert program_ensemble.total_circuit_count == 5
        assert program_ensemble.total_run_time == 5.0

    def test_aggregate_results_calls_join_and_aggregates(self, program_ensemble):
        """
        Tests that aggregate_results works correctly after a successful run,
        verifying the end-to-end data flow.
        """
        program_ensemble.create_programs()
        program_ensemble.run(blocking=True)
        result = program_ensemble.aggregate_results()

        assert result == 15  # 10 + 5

    def test_reset_listener_thread_timeout(self, program_ensemble, mocker):
        """Test reset handles listener thread timeout warning."""
        program_ensemble.create_programs()
        program_ensemble._done_event = Event()
        mock_thread = mocker.MagicMock(spec=Thread)
        mock_thread.is_alive.return_value = True  # Simulate timeout
        program_ensemble._listener_thread = mock_thread

        with pytest.warns(UserWarning, match="Listener thread did not terminate"):
            program_ensemble.reset()
        mock_thread.join.assert_called_once_with(timeout=1)

    def test_reset_progress_bar_exception(self, program_ensemble, mocker):
        """Test reset handles live display stop exception."""
        program_ensemble.create_programs()
        mock_live_display = mocker.MagicMock()
        mock_live_display.stop.side_effect = Exception("Stop failed")
        program_ensemble._live_display = mock_live_display
        program_ensemble._progress_bar = mocker.MagicMock()
        program_ensemble._pb_task_map = {}

        # Should not raise exception, just pass silently
        program_ensemble.reset()
        mock_live_display.stop.assert_called_once()

    def test_atexit_cleanup_warning(self, program_ensemble, mocker):
        """Test atexit cleanup hook issues warning."""
        program_ensemble.create_programs()
        mock_executor = mocker.MagicMock()
        program_ensemble._executor = mock_executor

        with pytest.warns(
            UserWarning,
            match="non-blocking ProgramEnsemble run was not explicitly closed",
        ):
            program_ensemble._atexit_cleanup_hook()

    def test_emit_progress_message_puts_on_queue(self, program_ensemble, mocker):
        """Synthetic terminal-status messages flow through the queue —
        not direct ``progress_bar.update`` calls — so the listener stays
        the single writer of the display."""
        program_ensemble.create_programs()
        program_ensemble._progress_bar = mocker.MagicMock()
        program_ensemble._emit_progress_message(
            "prog1", final_status=TerminalStatus.FAILED, message="Job failed"
        )
        msg = program_ensemble._queue.get_nowait()
        assert msg == {
            "job_id": "prog1",
            "progress": 0,
            "final_status": TerminalStatus.FAILED,
            "message": "Job failed",
        }

    def test_emit_progress_message_no_op_without_program_id(
        self, program_ensemble, mocker
    ):
        program_ensemble.create_programs()
        program_ensemble._progress_bar = mocker.MagicMock()
        program_ensemble._emit_progress_message(
            None, final_status=TerminalStatus.FAILED
        )
        assert program_ensemble._queue.empty()

    def test_emit_progress_message_no_op_without_progress_bar(self, program_ensemble):
        """When no progress display is active, emitting a message is a
        no-op — without a listener nothing would consume it."""
        program_ensemble.create_programs()
        assert program_ensemble._progress_bar is None
        program_ensemble._emit_progress_message(
            "prog1", final_status=TerminalStatus.FAILED
        )
        assert program_ensemble._queue.empty()

    def test_wait_for_listener_drain_bails_on_dead_listener(
        self, program_ensemble, mocker
    ):
        """If the listener thread has died, ``_wait_for_listener_drain``
        warns and returns rather than hanging on ``queue.join()``."""
        program_ensemble.create_programs()
        program_ensemble._listener_thread = mocker.MagicMock()
        program_ensemble._listener_thread.is_alive.return_value = False
        # Put a message on the queue with no listener to drain it.
        program_ensemble._queue.put({"job_id": "prog1", "progress": 0})

        with pytest.warns(RuntimeWarning, match="listener thread died"):
            program_ensemble._wait_for_listener_drain()

    def test_wait_for_listener_drain_bails_on_timeout(self, program_ensemble, mocker):
        """A live-but-stuck listener must not hang ``join()``: after the
        configured timeout the watchdog warns and returns."""
        program_ensemble.create_programs()
        program_ensemble._listener_thread = mocker.MagicMock()
        program_ensemble._listener_thread.is_alive.return_value = (
            True  # alive but stuck
        )
        program_ensemble._queue.put({"job_id": "prog1", "progress": 0})

        with pytest.warns(RuntimeWarning, match="did not drain within"):
            program_ensemble._wait_for_listener_drain(timeout=0.2)

        # Drop the live mock before fixture teardown — otherwise reset()
        # would treat ``is_alive()`` as still True and emit its own
        # "Listener thread did not terminate" warning, polluting the
        # test's warning capture for downstream runs.
        program_ensemble._listener_thread = None

    def test_handle_cancellation_phases(self, program_ensemble, mocker):
        """Test all three phases of cancellation handling.

        Each phase emits a synthetic progress message via
        ``_emit_progress_message``; we spy on the helper to inspect the
        messages.
        """
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}
        program_ensemble._cancellation_event = mocker.MagicMock()

        future1, future2, future3 = Future(), Future(), Future()
        future3.set_result(_FakeRunResult(5, 2.0))  # already done
        mocker.patch.object(
            future1, "cancel", return_value=True
        )  # pending, can be cancelled
        mocker.patch.object(
            future2, "cancel", return_value=False
        )  # running, cannot be cancelled

        program_ensemble.futures = [future1, future2, future3]
        program_ensemble._future_to_program = {
            future1: program_ensemble.programs["prog1"],
            future2: program_ensemble.programs["prog2"],
            future3: mocker.Mock(),  # Dummy program for completed future
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[future2])

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program_ensemble._handle_cancellation()

        program_ensemble._cancellation_event.set.assert_called_once()

        messages = [call.kwargs.get("message", "") for call in spy.call_args_list]
        assert "Cancelled by user" in messages
        assert "Finishing... ⏳" in messages
        assert "Stopped after current iteration" in messages

    def test_failed_future_with_execution_cancelled_error_emits_cancelled(
        self, program_ensemble, mocker
    ):
        """A future that finished with ``ExecutionCancelledError`` is the
        cooperative result of a user cancel propagating from the worker,
        not a real failure — it must show ``CANCELLED``, not ``FAILED``."""
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()

        failed_future = Future()
        failed_future.set_exception(ExecutionCancelledError("Cancelled by user"))

        program_ensemble.futures = [failed_future]
        program_ensemble._future_to_program = {
            failed_future: program_ensemble.programs["prog1"]
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[])

        program_ensemble._handle_cancellation()

        cancelled_emits = [
            call
            for call in spy.call_args_list
            if call.kwargs.get("final_status") is TerminalStatus.CANCELLED
        ]
        assert cancelled_emits, (
            "expected at least one CANCELLED emit for an ExecutionCancelledError "
            f"future, got {[c.kwargs for c in spy.call_args_list]}"
        )

    def test_failed_future_with_runtime_error_still_emits_failed(
        self, program_ensemble, mocker
    ):
        """A future that crashed for a non-cancellation reason must still
        show ``FAILED`` even when reaped during cancellation cleanup —
        masking it as CANCELLED would hide real bugs from the user."""
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()

        failed_future = Future()
        failed_future.set_exception(RuntimeError("boom"))

        program_ensemble.futures = [failed_future]
        program_ensemble._future_to_program = {
            failed_future: program_ensemble.programs["prog1"]
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[])

        program_ensemble._handle_cancellation()

        statuses = [
            call.kwargs.get("final_status")
            for call in spy.call_args_list
            if call.kwargs.get("final_status") is not None
        ]
        assert TerminalStatus.FAILED in statuses
        assert TerminalStatus.CANCELLED not in statuses

    def test_failed_future_panel_printed_during_cancellation(
        self, program_ensemble, mocker
    ):
        """When cancellation reaps a future that crashed for a non-
        cancellation reason, the exception detail must still surface — a
        red Rich panel with a traceback, mirroring the no-cancel failure
        path. Otherwise the user only sees a red progress row and never
        learns what went wrong."""
        program_ensemble.create_programs()
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}
        program_ensemble._cancellation_event = mocker.MagicMock()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar

        failed_future = Future()
        failed_future.set_exception(RuntimeError("boom"))
        cancelled_future = Future()
        cancelled_future.set_exception(ExecutionCancelledError("Cancelled by user"))

        program_ensemble.futures = [failed_future, cancelled_future]
        program_ensemble._future_to_program = {
            failed_future: program_ensemble.programs["prog1"],
            cancelled_future: program_ensemble.programs["prog2"],
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[])

        program_ensemble._handle_cancellation()

        printed = [
            call.args[0] for call in mock_progress_bar.console.print.call_args_list
        ]
        panels = [p for p in printed if isinstance(p, Panel)]
        tracebacks = [p for p in printed if isinstance(p, Traceback)]
        # One failure → exactly one summary panel and one traceback render.
        assert len(panels) == 1
        assert len(tracebacks) == 1

        panel_text = str(panels[0].renderable)
        assert "RuntimeError" in panel_text
        assert "boom" in panel_text
        # The cancelled program is not rendered as a failure.
        assert "ExecutionCancelledError" not in panel_text
        # Failed program is identified by id.
        prog1_id = program_ensemble.programs["prog1"].program_id
        assert prog1_id in str(panels[0].title)

    def test_cancellation_without_failures_prints_no_failure_panels(
        self, program_ensemble, mocker
    ):
        """When every program either ran cleanly or cancelled cooperatively,
        no Rich failure panels should be printed — only the existing
        progress-row status updates."""
        program_ensemble.create_programs()
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar

        cancelled_future = Future()
        cancelled_future.set_exception(ExecutionCancelledError("Cancelled by user"))

        program_ensemble.futures = [cancelled_future]
        program_ensemble._future_to_program = {
            cancelled_future: program_ensemble.programs["prog1"]
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[])

        program_ensemble._handle_cancellation()

        # No Panel/Traceback emission should have happened on the console.
        assert mock_progress_bar.console.print.call_count == 0

    def test_handle_cancellation_unstoppable_futures(self, program_ensemble, mocker):
        """Test cancellation handling with unstoppable futures."""
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_ensemble.futures = [future]
        program_ensemble._future_to_program = {
            future: program_ensemble.programs["prog1"]
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[future])

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program_ensemble._handle_cancellation()

        finishing_calls = [
            call
            for call in spy.call_args_list
            if call.kwargs.get("message") == "Finishing... ⏳"
        ]
        assert len(finishing_calls) > 0

    def test_handle_cancellation_calls_cancel_unfinished_job(
        self, program_ensemble, mocker
    ):
        """Test that _handle_cancellation calls cancel_unfinished_job for unstoppable futures."""
        program_ensemble.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_ensemble.futures = [future]
        program = program_ensemble.programs["prog1"]
        program_ensemble._future_to_program = {future: program}

        # Mock cancel_unfinished_job - this prevents the warning since the actual method isn't called
        mock_cancel = mocker.patch.object(program, "cancel_unfinished_job")
        # Mock as_completed to return the future so Phase 3 doesn't hang
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[future])

        program_ensemble._handle_cancellation()

        # Verify cancel_unfinished_job was called
        mock_cancel.assert_called_once()

    def test_handle_cancellation_delegates_to_backend_cancel_job(
        self, program_ensemble, mocker
    ):
        """Test that _handle_cancellation delegates to backend.cancel_job via cancel_unfinished_job."""
        program_ensemble.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
        program_ensemble._pb_task_map = {"prog1": 1}
        program_ensemble._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_ensemble.futures = [future]
        program = program_ensemble.programs["prog1"]
        program_ensemble._future_to_program = {future: program}

        # Set up program with execution result
        execution_result = ExecutionResult(job_id="test_job_123")
        program._current_execution_result = execution_result

        # Mock backend cancel_job (spec'd to satisfy AsyncJobBackend protocol)
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_backend.cancel_job = mocker.Mock()
        program.backend = mock_backend

        # Mock as_completed to return the future so Phase 3 doesn't hang
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[future])

        program_ensemble._handle_cancellation()

        # Verify cancel_job was called on the backend
        mock_backend.cancel_job.assert_called_once_with(execution_result)

    def test_handle_failure_sets_cancellation_event(self, program_ensemble, mocker):
        """Failure path should set _cancellation_event to stop VQA loops."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        f_bad = Future()
        f_bad.set_exception(RuntimeError("Job xyz has failed."))
        f_good = Future()
        f_good.set_result(_FakeRunResult(10, 5.0))
        program_ensemble.futures = [f_bad, f_good]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            f_bad: progs[0],
            f_good: progs[1],
        }

        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[iter([f_bad]), iter([])],
        )

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            program_ensemble.join()

        assert program_ensemble._cancellation_event.is_set()

    def test_handle_failure_updates_progress_bars(self, program_ensemble, mocker):
        """Failure path should emit a Failed terminal-status message."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}

        f_bad = Future()
        f_bad.set_exception(RuntimeError("Job xyz has failed."))
        f_good = Future()
        f_good.set_result(_FakeRunResult(10, 5.0))
        program_ensemble.futures = [f_bad, f_good]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            f_bad: progs[0],
            f_good: progs[1],
        }

        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[iter([f_bad]), iter([])],
        )

        with pytest.raises(RuntimeError):
            program_ensemble.join()

        # Identity check on the enum member (not just the string value):
        # `TerminalStatus.FAILED == "Failed"` thanks to the str-mixin,
        # so an `in "Failed"` assertion would silently accept a raw
        # string regression.
        failed_calls = [
            call
            for call in spy.call_args_list
            if call.kwargs.get("final_status") is TerminalStatus.FAILED
        ]
        assert any(
            call.args[0] == progs[0].program_id for call in failed_calls
        ), "the failed program's row was not emitted with final_status=Failed"

    def test_handle_failure_non_batched_cancels_jobs(self, program_ensemble, mocker):
        """Without coordinator, failure should call cancel_unfinished_job on running programs."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        program_ensemble._coordinator = None

        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}

        f_bad = Future()
        f_bad.set_exception(RuntimeError("Job xyz has failed."))
        # Simulate a still-running future that cannot be cancelled.
        # done() returns False during _stop_remaining_programs (so the
        # cancel path is taken) and True afterwards (for result collection).
        f_running = Future()
        mocker.patch.object(f_running, "cancel", return_value=False)
        done_returns = iter([False, False, True, True, True])
        mocker.patch.object(f_running, "done", side_effect=lambda: next(done_returns))

        program_ensemble.futures = [f_bad, f_running]
        prog1, prog2 = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            f_bad: prog1,
            f_running: prog2,
        }

        mock_cancel = mocker.patch.object(prog2, "cancel_unfinished_job")

        # Resolve f_running so _collect_completed_results can pick it up
        # once done() starts returning True.
        f_running.set_result(_FakeRunResult(3, 1.0))

        # First call: join() loop yields the failing future.
        # Second call: _stop_remaining_programs waits for unstoppable futures.
        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[iter([f_bad]), iter([f_running])],
        )

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            program_ensemble.join()

        mock_cancel.assert_called_once()
        # Verify that the running program's results were still collected
        assert program_ensemble.total_circuit_count == 3
        assert program_ensemble.total_run_time == 1.0

    def test_stop_remaining_programs_called_on_failure(self, program_ensemble, mocker):
        """_stop_remaining_programs should be called from the failure path."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        f_bad = Future()
        f_bad.set_exception(RuntimeError("boom"))
        program_ensemble.futures = [f_bad]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {f_bad: progs[0]}

        spy = mocker.spy(program_ensemble, "_stop_remaining_programs")
        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[iter([f_bad]), iter([])],
        )

        with pytest.raises(RuntimeError):
            program_ensemble.join()

        spy.assert_called_once()
        call_kwargs = spy.call_args[1]
        assert call_kwargs["pending_status"] == "Cancelled"
        assert "failure" in call_kwargs["pending_message"].lower()

    def test_handle_failure_all_futures_failed(self, program_ensemble, mocker):
        """When batch coordinator fails all futures, every program should
        emit a Failed terminal-status message via the queue."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}

        # Simulate _fail_futures setting the same exception on all futures
        shared_exc = RuntimeError("Merged batch job xyz has failed.")
        f1 = Future()
        f1.set_exception(shared_exc)
        f2 = Future()
        f2.set_exception(shared_exc)

        program_ensemble.futures = [f1, f2]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            f1: progs[0],
            f2: progs[1],
        }

        # First call: join() loop yields f1 (raises). Second call:
        # _stop_remaining_programs finds no unstoppable futures (all done).
        mocker.patch(
            "divi.qprog.ensemble.as_completed",
            side_effect=[iter([f1]), iter([])],
        )

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            program_ensemble.join()

        # Both programs' rows should have been emitted with the FAILED
        # enum member specifically.
        failed_prog_ids = {
            call.args[0]
            for call in spy.call_args_list
            if call.kwargs.get("final_status") is TerminalStatus.FAILED
        }
        assert failed_prog_ids == {"prog1", "prog2"}

    def test_join_early_return_no_executor(self, program_ensemble):
        """Test join returns early when no executor."""
        program_ensemble._executor = None
        result = program_ensemble.join()
        assert result is None

    def test_join_keyboard_interrupt(self, program_ensemble, mocker):
        """Test join handles KeyboardInterrupt."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        mocker.patch(
            "divi.qprog.ensemble.as_completed", side_effect=KeyboardInterrupt()
        )
        mock_handle_cancellation = mocker.patch.object(
            program_ensemble, "_handle_cancellation"
        )

        result = program_ensemble.join()

        assert result is False
        mock_handle_cancellation.assert_called_once()

    def test_join_keyboard_interrupt_no_double_count(self, program_ensemble, mocker):
        """Results collected before KeyboardInterrupt are not double-counted."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        # Create two pre-resolved futures with known results
        f1 = Future()
        f1.set_result(_FakeRunResult(10, 5.0))
        f2 = Future()
        f2.set_result(_FakeRunResult(7, 3.0))
        program_ensemble.futures = [f1, f2]

        # as_completed yields f1 then raises, simulating an interrupt
        # after one future was already collected by the loop.
        def _partial_as_completed(futures):
            yield f1
            raise KeyboardInterrupt()

        mocker.patch(
            "divi.qprog.ensemble.as_completed", side_effect=_partial_as_completed
        )
        mocker.patch.object(program_ensemble, "_handle_cancellation")

        program_ensemble.join()

        # Both futures completed (10 + 7 = 17), each counted exactly once
        assert program_ensemble.total_circuit_count == 17
        assert program_ensemble.total_run_time == 8.0

    def test_join_exception_no_double_count(self, program_ensemble, mocker):
        """All completed futures are counted exactly once after a task exception."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        f1 = Future()
        f1.set_result(_FakeRunResult(10, 5.0))
        f2 = Future()
        f2.set_result(_FakeRunResult(7, 3.0))
        f_bad = Future()
        f_bad.set_exception(ValueError("boom"))
        program_ensemble.futures = [f1, f2, f_bad]
        progs = list(program_ensemble.programs.values())
        program_ensemble._future_to_program = {
            f1: progs[0],
            f2: progs[1],
            f_bad: mocker.Mock(program_id="prog_bad"),
        }

        call_count = [0]

        # as_completed is called twice: once in the join() loop (yields f1
        # then f_bad which raises), once in _stop_remaining_programs (no
        # unstoppable futures since all are already done).
        def _mock_as_completed(futures):
            call_count[0] += 1
            if call_count[0] == 1:
                yield f1
                yield f_bad
            else:
                yield from futures

        mocker.patch("divi.qprog.ensemble.as_completed", side_effect=_mock_as_completed)

        with pytest.raises(RuntimeError, match="Ensemble execution failed"):
            program_ensemble.join()

        # f1 (10) and f2 (7) both completed, each counted exactly once
        assert program_ensemble.total_circuit_count == 17
        assert program_ensemble.total_run_time == 8.0

    def test_run_rejects_duplicate_program_instances(self, program_ensemble):
        """run() raises when the same program instance is assigned to multiple keys."""
        program_ensemble.create_programs()
        shared = program_ensemble.programs["prog1"]
        program_ensemble.programs = {"a": shared, "b": shared}

        with pytest.raises(RuntimeError, match="Duplicate program instances"):
            program_ensemble.run()

    def test_atexit_unregister_failure(self, program_ensemble, mocker):
        """Test atexit unregister handles TypeError."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        mock_unregister = mocker.patch(
            "atexit.unregister", side_effect=TypeError("Not registered")
        )
        program_ensemble.join()
        mock_unregister.assert_called_once()

    def test_aggregate_results_with_running_executor(self, program_ensemble, mocker):
        """
        Tests that aggregate_results calls join() if the executor is still
        running and then correctly aggregates the results.
        """
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        mock_join = mocker.spy(program_ensemble, "join")
        result = program_ensemble.aggregate_results()
        mock_join.assert_called_once()
        assert result == 15

    def test_run_passes_batch_config_to_coordinator(self, program_ensemble):
        """BatchConfig is forwarded to the coordinator."""
        program_ensemble.create_programs()
        config = BatchConfig(max_batch_size=50)
        program_ensemble.run(blocking=True, batch_config=config)
        assert program_ensemble.total_circuit_count == 15

    def test_run_with_batching_off(self, program_ensemble):
        """BatchMode.OFF disables the coordinator entirely."""
        program_ensemble.create_programs()
        program_ensemble.run(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )
        assert program_ensemble.total_circuit_count == 15
        assert program_ensemble._coordinator is None


class TestRegistrationFailureCleanup:
    """Regression tests for partial-registration failure cleanup in ``run()``."""

    @staticmethod
    def _flush_threads_alive() -> int:
        # Thread name prefixes are not part of the public contract; this
        # check is a fragile heuristic and may need to be updated if the
        # coordinator's daemon-thread naming changes.
        return sum(
            1
            for t in threading.enumerate()
            if t.is_alive()
            and (t.name.startswith("flush") or "BatchCoordinator" in t.name)
        )

    def test_register_program_failure_clears_state(self, dummy_simulator, mocker):
        """``register_program`` raising mid-loop must leave the coordinator
        with empty ``_active_programs`` (and the ensemble with no live
        executor or coordinator handle)."""
        ensemble = SampleProgramEnsemble(backend=dummy_simulator)
        ensemble.create_programs()
        ensemble.programs = {
            f"prog{i}": SimpleTestProgram(
                1, 0.1, backend=dummy_simulator, program_id=f"prog{i}"
            )
            for i in range(1, 5)
        }

        # Capture the coordinator instance via a side-effect on
        # construction so we can inspect its state *after* run() has
        # cleared the ensemble's reference to it.
        captured: dict = {}
        original_init = _BatchCoordinator.__init__

        def _capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured["coord"] = self

        mocker.patch.object(_BatchCoordinator, "__init__", _capturing_init)

        original_register = _BatchCoordinator.register_program
        calls = {"n": 0}

        def _flaky(self, program_key):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom: simulated registration failure")
            return original_register(self, program_key)

        mocker.patch.object(_BatchCoordinator, "register_program", _flaky)

        baseline_alive = self._flush_threads_alive()

        with pytest.raises(RuntimeError, match="boom"):
            ensemble.run(blocking=False)

        coord = captured["coord"]
        # Load-bearing assertion: the orphaned-registration bug.
        assert coord._active_programs == set()
        assert coord._pending == {}
        # Coordinator + executor handle on the ensemble should be cleared.
        assert ensemble._coordinator is None
        assert ensemble._executor is None
        assert self._flush_threads_alive() == baseline_alive

    def test_run_recovers_after_registration_failure(self, dummy_simulator, mocker):
        """A second ``run()`` after a registration failure must succeed —
        no leftover state blocks re-entry."""
        ensemble = SampleProgramEnsemble(backend=dummy_simulator)
        ensemble.create_programs()

        original_register = _BatchCoordinator.register_program
        calls = {"n": 0}

        def _flaky_once(self, program_key):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return original_register(self, program_key)

        mocker.patch.object(_BatchCoordinator, "register_program", _flaky_once)

        with pytest.raises(RuntimeError, match="boom"):
            ensemble.run(blocking=False)

        # Subsequent register_program calls fall through to the original
        # implementation, so the second run() should succeed end-to-end.
        ensemble.run(blocking=True)
        assert ensemble.aggregate_results() == 15


def test_cancellation_event_is_shared_with_coordinator(dummy_simulator):
    """When the ensemble's cancellation event is set, the coordinator's
    ``_cancelled`` Event must also report ``is_set()``."""
    ensemble = SampleProgramEnsemble(backend=dummy_simulator)
    ensemble.create_programs()
    try:
        ensemble.run(blocking=False)
        assert (
            ensemble._coordinator is not None
        ), "expected coordinator under default BatchMode.MERGED"
        # Same identity, not just same value.
        assert ensemble._cancellation_event is ensemble._coordinator._cancelled
        # Setting the ensemble side propagates to the coordinator.
        ensemble._cancellation_event.set()
        assert ensemble._coordinator._cancelled.is_set()
    finally:
        ensemble.join()


class TestBatchConfig:
    """Ensemble-side smoke tests covering BatchConfig values used directly in
    ``ProgramEnsemble.run()``. Validation and defaults are owned by
    ``tests/qprog/test_batch_coordinator.py::TestBatchConfig``.
    """

    def test_valid_max_batch_size(self):
        config = BatchConfig(max_batch_size=10)
        assert config.max_batch_size == 10

    def test_max_batch_size_one(self):
        config = BatchConfig(max_batch_size=1)
        assert config.max_batch_size == 1

    def test_off_mode(self):
        config = BatchConfig(mode=BatchMode.OFF)
        assert config.mode is BatchMode.OFF
        assert config.max_batch_size is None

    def test_frozen(self):
        config = BatchConfig(max_batch_size=10)
        with pytest.raises(AttributeError):
            config.max_batch_size = 20


class _ParameterizedEnsemble(ProgramEnsemble):
    """Ensemble whose program count is configurable, for sizing tests."""

    def __init__(self, backend, n_programs):
        super().__init__(backend)
        self._n_programs = n_programs
        self.max_iterations = 1

    def create_programs(self):
        super().create_programs()
        self.programs = {
            f"prog_{i}": SimpleTestProgram(
                1, 0.0, backend=self.backend, program_id=f"prog_{i}"
            )
            for i in range(self._n_programs)
        }

    def aggregate_results(self):
        super().aggregate_results()
        return None


class _SubmittingProgram(_StubProgram):
    """Test program whose ``run()`` actually submits circuits through its
    backend — used to exercise the end-to-end ensemble → coordinator → flush
    pipeline (so flush-size assertions reflect real backend calls)."""

    _MINIMAL_QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n'

    def __init__(self, *, n_circuits: int = 1, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.n_circuits = n_circuits

    def run(self):
        circuits = {f"c{i}": self._MINIMAL_QASM for i in range(self.n_circuits)}
        self.backend.submit_circuits(circuits)
        self._total_circuit_count = self.n_circuits
        self._total_run_time = 0.0
        self._ran = True
        return self


class _SubmittingEnsemble(ProgramEnsemble):
    """Ensemble of :class:`_SubmittingProgram` instances; sizes flush
    integration tests."""

    def __init__(self, backend, n_programs: int, n_circuits_per_program: int = 1):
        super().__init__(backend)
        self._n_programs = n_programs
        self._n_circuits_per_program = n_circuits_per_program
        self.max_iterations = 1

    def create_programs(self):
        super().create_programs()
        self.programs = {
            f"prog_{i}": _SubmittingProgram(
                n_circuits=self._n_circuits_per_program,
                backend=self.backend,
                program_id=f"prog_{i}",
            )
            for i in range(self._n_programs)
        }

    def aggregate_results(self):
        super().aggregate_results()
        return None


class _AccumulatingProgram(_StubProgram):
    """Program whose run() *accumulates* fixed per-call increments into
    ``_total_circuit_count`` / ``_total_run_time``.

    Mirrors how real VQAs grow their counters monotonically across every
    dispatch they take part in (unlike :class:`SimpleTestProgram`, which
    assigns) — so re-dispatch exercises the ensemble's delta accounting.
    """

    def __init__(self, *, circ_per_call: int, time_per_call: float, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self._circ_per_call = circ_per_call
        self._time_per_call = time_per_call

    def run(self):
        self._total_circuit_count += self._circ_per_call
        self._total_run_time += self._time_per_call
        self._ran = True
        return self


class _AccumulatingEnsemble(ProgramEnsemble):
    """Ensemble of :class:`_AccumulatingProgram` with configurable per-program
    ``(circuits, runtime)`` increments, for exact count-accounting assertions.
    """

    def __init__(self, backend, specs: dict):
        super().__init__(backend)
        self._specs = specs

    def create_programs(self):
        super().create_programs()
        self.programs = {
            pid: _AccumulatingProgram(
                circ_per_call=circ,
                time_per_call=runtime,
                backend=self.backend,
                program_id=pid,
            )
            for pid, (circ, runtime) in self._specs.items()
        }

    def aggregate_results(self):
        super().aggregate_results()
        return None


class TestExecutorSizing:
    """Three-tier executor sizing in :meth:`ProgramEnsemble.run`.

    The default wait-for-all barrier needs one executor slot per program; the
    tests below pin each tier of the sizing decision so a regression in any
    tier — including the >256 fail-fast — is caught immediately.
    """

    @staticmethod
    def _spy_executor(mocker):
        return mocker.spy(ensemble_module, "ThreadPoolExecutor")

    def test_default_barrier_path_pool_at_least_n_programs(
        self, dummy_simulator, mocker
    ):
        """Default ``BatchConfig`` reserves one slot per registered program.

        The exact value is ``max(n_programs, cpu+4)``; pinning that exact
        formula ensures the test fails if the barrier-scaling branch is
        ever silently dropped on a host where ``cpu+4`` happens to dominate.
        """
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=10)
        ensemble.create_programs()
        ensemble.run(blocking=True)

        spy.assert_called_once()
        expected = max(10, (os.cpu_count() or 1) + 4)
        assert spy.call_args.kwargs["max_workers"] == expected

    def test_default_barrier_path_floors_at_cpu_default(self, dummy_simulator, mocker):
        """Small ensembles still get the cpu+4 default — never under-provisioned."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=2)
        ensemble.create_programs()
        ensemble.run(blocking=True)

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] >= (os.cpu_count() or 1) + 4

    def test_max_batch_size_pool_aligns_with_batch(self, dummy_simulator, mocker):
        """``max_batch_size`` sizes the pool to ``min(max_batch_size, n_programs)``
        so the barrier predicate can fill the batch in one wave (instead of
        firing prematurely at ``cpu+4``)."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=20)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=4))

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 4

    def test_max_batch_size_pool_capped_at_n_programs(self, dummy_simulator, mocker):
        """When ``max_batch_size > len(programs)``, the pool falls back to
        ``len(programs)`` — never spawn more threads than there is work for."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=8)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=512))

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 8

    def test_predicate1_flushes_align_with_max_batch_size(
        self, dummy_simulator, mocker
    ):
        """End-to-end: with one circuit per program, the pool-fills predicate
        (predicate1) fires at exactly ``max_batch_size`` — so each merged
        backend call carries that many circuits and the flush count matches
        ``ceil(n_programs / max_batch_size)``.  Regresses the bug where the
        pool was capped at ``cpu+4`` and flushes fired prematurely."""
        original = dummy_simulator.submit_circuits
        merged_sizes: list[int] = []

        def _spy(circuits, **kwargs):
            merged_sizes.append(len(circuits))
            return original(circuits, **kwargs)

        mocker.patch.object(dummy_simulator, "submit_circuits", _spy)

        ensemble = _SubmittingEnsemble(backend=dummy_simulator, n_programs=32)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=8))

        assert sum(merged_sizes) == 32
        assert all(size == 8 for size in merged_sizes), merged_sizes
        assert len(merged_sizes) == 4

    def test_predicate2_flush_can_exceed_max_batch_size(self, dummy_simulator, mocker):
        """``max_batch_size`` is a flush-trigger, not a hard cap.  When
        each program submits a multi-circuit batch in a single call,
        the circuit-count predicate (predicate2) fires the moment a
        program's submission carries pending past the threshold — and
        the flush takes everything pending, which can exceed
        ``max_batch_size``.

        Concretely: pool = ``min(10, 8) = 8``, so up to 8 programs run in
        parallel.  Each program submits 5 circuits in one call.  All 40
        circuits flush.  The combined merged-call sizes must sum to 40,
        and the trigger semantics permit (but do not require) any single
        flush to exceed 10.
        """
        original = dummy_simulator.submit_circuits
        merged_sizes: list[int] = []

        def _spy(circuits, **kwargs):
            merged_sizes.append(len(circuits))
            return original(circuits, **kwargs)

        mocker.patch.object(dummy_simulator, "submit_circuits", _spy)

        ensemble = _SubmittingEnsemble(
            backend=dummy_simulator, n_programs=8, n_circuits_per_program=5
        )
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=10))

        # All 40 circuits must be flushed across some number of merged calls.
        assert sum(merged_sizes) == 40
        # Each program contributes a 5-circuit chunk atomically, so every
        # flush should be a positive multiple of 5.
        assert merged_sizes, "expected at least one flush"
        assert all(size > 0 and size % 5 == 0 for size in merged_sizes), merged_sizes

    def test_minus_one_with_max_batch_size_caps_at_batch_size(
        self, dummy_simulator, mocker
    ):
        """``max_concurrent_programs=-1`` combined with ``max_batch_size`` caps
        the pool at ``min(max_batch_size, len(programs))`` instead of spawning
        one thread per program (which can exhaust OS thread limits on large
        ensembles)."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=300)
        ensemble.create_programs()
        ensemble.run(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=-1, max_batch_size=64),
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 64

    def test_off_mode_uses_default_pool(self, dummy_simulator, mocker):
        """``BatchMode.OFF`` has no barrier, so the cpu+4 default is sufficient."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=20)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == (os.cpu_count() or 1) + 4

    def test_exceeds_barrier_limit_raises(self, dummy_simulator):
        """Default config + >256 programs fails fast with an actionable message."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        with pytest.raises(RuntimeError) as excinfo:
            ensemble.run(blocking=True)

        msg = str(excinfo.value)
        assert "257" in msg
        assert "max_batch_size" in msg
        assert "BatchMode.OFF" in msg

    def test_exceeds_barrier_limit_succeeds_with_max_batch_size(self, dummy_simulator):
        """Same large ensemble runs cleanly once the user opts into early-flush."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=8))
        # All 257 programs ran (each contributes circ_count=1).
        assert ensemble.total_circuit_count == 257

    def test_exceeds_barrier_limit_succeeds_with_off_mode(self, dummy_simulator):
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))
        assert ensemble.total_circuit_count == 257

    def test_coordinator_n_workers_matches_executor_in_early_flush(
        self, program_ensemble
    ):
        """The coordinator's barrier cap matches executor capacity in early-flush."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False, batch_config=BatchConfig(max_batch_size=4))
        assert program_ensemble._coordinator is not None
        assert (
            program_ensemble._coordinator._n_workers
            == program_ensemble._executor._max_workers
        )
        program_ensemble.join()

    def test_coordinator_n_workers_matches_executor_on_barrier_path(
        self, program_ensemble
    ):
        """Default barrier path passes the executor capacity to the coordinator."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)
        assert program_ensemble._coordinator is not None
        assert (
            program_ensemble._coordinator._n_workers
            == program_ensemble._executor._max_workers
        )
        program_ensemble.join()

    def test_max_batch_size_exceeds_pool_runs_to_completion(self, dummy_simulator):
        """Regression: programs > pool with max_batch_size > pool must not deadlock.

        Each program submits 1 circuit, so the circuit-count cap can never
        fire before the barrier — the barrier predicate's ``n_workers`` cap
        is what keeps the run satisfiable.
        """
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=64)
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(max_batch_size=64))
        assert ensemble.total_circuit_count == 64

    def test_max_concurrent_programs_sizes_pool_directly(self, dummy_simulator, mocker):
        """``max_concurrent_programs`` on BatchConfig drives executor size."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=10)
        ensemble.create_programs()
        ensemble.run(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=10),
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 10

    def test_max_concurrent_programs_bypasses_barrier_limit(self, dummy_simulator):
        """Explicit ``max_concurrent_programs`` lifts the 256-program cap."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=300)
        ensemble.create_programs()
        ensemble.run(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=300),
        )
        assert ensemble.total_circuit_count == 300

    def test_max_concurrent_programs_above_soft_cap_warns(self, dummy_simulator):
        """Values above the advisory soft cap emit a UserWarning."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=2)
        ensemble.create_programs()
        with pytest.warns(UserWarning, match="max_concurrent_programs"):
            ensemble.run(
                blocking=True,
                batch_config=BatchConfig(max_concurrent_programs=2000),
            )

    def test_max_concurrent_programs_minus_one_resolves_to_ensemble_size(
        self, dummy_simulator, mocker
    ):
        """``-1`` resolves to ``len(programs)`` at run time."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=37)
        ensemble.create_programs()
        ensemble.run(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=-1),
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 37

    def test_max_concurrent_programs_minus_one_does_not_warn(
        self, dummy_simulator, recwarn
    ):
        """The ``-1`` sentinel is an explicit opt-in; no soft-cap warning
        even when the resolved value exceeds 1024."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=2000)
        ensemble.create_programs()
        ensemble.run(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=-1),
        )

        soft_cap_warnings = [
            w for w in recwarn.list if "max_concurrent_programs" in str(w.message)
        ]
        assert soft_cap_warnings == []


def beam_search_aggregate(
    programs,
    initial_solution,
    extend_fn,
    evaluate_fn,
    beam_width=None,
    n_partition_candidates=None,
):
    """Test-local shorthand wrapping _beam_search_aggregate_top_n for top_n=1."""
    return _beam_search_aggregate_top_n(
        programs,
        initial_solution,
        extend_fn,
        evaluate_fn,
        beam_width,
        n_partition_candidates,
        top_n=1,
    )[0][1]


# ──────────────────────────────────────────────────────────────────────
#  Helpers: lightweight mock VQA programs for testing
# ──────────────────────────────────────────────────────────────────────


class _MockProgram:
    """Minimal mock that implements get_top_solutions."""

    def __init__(self, candidates: list[SolutionEntry]):
        self._candidates = candidates

    def get_top_solutions(self, n=10, *, include_decoded=False):
        return self._candidates[:n]


# ──────────────────────────────────────────────────────────────────────
#  Simple extend / evaluate functions for testing
# ──────────────────────────────────────────────────────────────────────


def _sum_evaluate(solution):
    """Simple evaluator: sum of the solution vector (lower is better)."""
    return sum(solution)


def _neg_sum_evaluate(solution):
    """Evaluator where higher sums are better: returns -sum (lower is better)."""
    return -sum(solution)


def _write_extend(variable_maps):
    """Returns an extend_fn that writes decoded bits into global positions."""

    def extend(current, prog_id, candidate):
        result = list(current)
        for local_idx, global_idx in enumerate(variable_maps[prog_id]):
            result[global_idx] = int(candidate.decoded[local_idx])
        return result

    return extend


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────


class TestBeamSearchAggregateValidation:
    def test_beam_width_zero_raises(self):
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=0,
            )

    def test_beam_width_negative_raises(self):
        with pytest.raises(ValueError, match="beam_width must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=-1,
            )

    def test_n_partition_candidates_zero_raises(self):
        with pytest.raises(ValueError, match="n_partition_candidates must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=1,
                n_partition_candidates=0,
            )

    def test_n_partition_candidates_negative_raises(self):
        with pytest.raises(ValueError, match="n_partition_candidates must be >= 1"):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=1,
                n_partition_candidates=-3,
            )

    def test_n_partition_candidates_less_than_beam_width_raises(self):
        with pytest.raises(
            ValueError, match="n_partition_candidates.*must be >= beam_width"
        ):
            beam_search_aggregate(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=5,
                n_partition_candidates=2,
            )


class TestBeamSearchAggregateGreedy:
    """Test greedy mode (beam_width=1)."""

    def test_single_partition_single_candidate(self):
        """Greedy with one partition and one candidate returns that candidate."""
        candidates = [SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0])]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0, 1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        assert result == [1, 0]

    def test_two_partitions_greedy_picks_best_per_partition(self):
        """Greedy picks the single best candidate from each partition."""
        # Partition A: variables 0,1 — candidates: [1,0] (prob=0.9)
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
        ]
        # Partition B: variables 2,3 — candidates: [1,1] (prob=0.7)
        candidates_b = [
            SolutionEntry(bitstring="11", prob=0.7, decoded=[1, 1]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        # Greedy: After A, only [1,0,0,0]. After B, only [1,0,1,1].
        assert result == [1, 0, 1, 1]


class TestBeamSearchAggregateBeam:
    """Test standard beam search (beam_width > 1)."""

    def test_beam_width_2_explores_combinations(self):
        """Beam width 2 keeps two partial solutions and finds the best."""
        # Partition A (vars 0,1):
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.6, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.4, decoded=[0, 1]),
        ]
        # Partition B (vars 2,3):
        candidates_b = [
            SolutionEntry(bitstring="10", prob=0.7, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.3, decoded=[0, 1]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=2,
        )

        # With beam_width=2, both A candidates are kept.
        # Expanding B: 4 combinations. All have sum=2, so any is optimal.
        assert sum(result) == 2

    def test_beam_finds_better_than_greedy(self):
        """Beam search can find a solution that greedy misses."""
        weights = [10, 1, 1, 10]

        def weighted_evaluate(solution):
            return sum(s * w for s, w in zip(solution, weights))

        # Partition A (vars 0,1): beam_width=1 only sees first candidate
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.1, decoded=[0, 1]),
        ]
        # Partition B (vars 2,3):
        candidates_b = [
            SolutionEntry(bitstring="01", prob=0.9, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.1, decoded=[1, 0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}

        # Greedy (beam_width=1): only sees 1 candidate per partition
        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
        )

        # Beam (beam_width=2): sees 2 candidates per partition
        beam_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=2,
        )

        greedy_cost = weighted_evaluate(greedy_result)
        beam_cost = weighted_evaluate(beam_result)

        # Beam should find an equal or better solution
        assert beam_cost <= greedy_cost


class TestBeamSearchAggregateExhaustive:
    """Test exhaustive mode (beam_width=None)."""

    def test_exhaustive_explores_all_combinations(self):
        """With beam_width=None, all combinations are evaluated."""
        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.4, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.7, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0], "B": [1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=None,
        )

        # _sum_evaluate minimizes sum → best is [0, 0]
        assert result == [0, 0]

    def test_exhaustive_finds_global_optimum(self):
        """Exhaustive must find the true global optimum."""

        def tricky_evaluate(solution):
            """Only [0,1,0] has cost -100, everything else is >= 0."""
            if solution == [0, 1, 0]:
                return -100.0
            return sum(solution)

        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_c = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=tricky_evaluate,
            beam_width=None,
        )

        assert result == [0, 1, 0]


class TestBeamSearchAggregateEdgeCases:
    """Test edge cases."""

    def test_single_partition(self):
        """Single partition still works correctly."""
        candidates = [
            SolutionEntry(bitstring="01", prob=0.9, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.1, decoded=[1, 0]),
        ]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0, 1]}

        # beam_width=2 sees both candidates, picks highest sum
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=2,
        )

        assert result == [0, 1]

    def test_empty_programs_returns_initial(self):
        """No programs at all returns the initial solution."""
        result = beam_search_aggregate(
            programs={},
            initial_solution=[0, 0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        assert result == [0, 0, 0]

    def test_program_with_no_candidates_skipped(self):
        """A program returning no candidates is skipped without error."""
        programs = {"A": _MockProgram([])}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        assert result == [0, 0]

    def test_beam_width_limits_extraction(self):
        """beam_width limits both candidates extracted and beam size."""
        many_candidates = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        programs = {"A": _MockProgram(many_candidates)}
        var_maps = {"A": [0]}

        # beam_width=1 should only consider 1 candidate: [1]
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        # Only candidate [1] is considered (beam_width=1), so result is [1]
        assert result == [1]

    def test_non_zero_initial_solution_preserved(self):
        """Positions not touched by any partition retain their initial values."""
        # Partition only covers position 1; positions 0 and 2 should stay as-is
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [1]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[1, 0, 1],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            beam_width=1,
        )

        assert result == [1, 1, 1]

    def test_overlapping_partitions(self):
        """Partitions writing to overlapping global positions work correctly."""
        # Both partitions write to position 0; partition B overwrites A's value
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="0", prob=0.9, decoded=[0])]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0], "B": [0]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
        )

        # B runs after A and overwrites position 0 → final value is 0
        assert result == [0]

    def test_tie_breaking_is_stable(self):
        """When candidates have identical scores, a valid result is returned."""
        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.5, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.5, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates_a)}
        var_maps = {"A": [0]}

        # Both candidates have |sum|=1 or 0 under _sum_evaluate — either is valid
        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )

        assert result in ([0], [1])


class TestBeamSearchAggregatePruning:
    """Test that beam pruning behaves correctly."""

    def test_beam_prunes_to_width(self):
        """After each partition step, at most beam_width solutions are kept."""
        # 3 partitions × 3 candidates each.  With beam_width=2, the beam
        # should never exceed 2 partial solutions between steps, which limits
        # the total work and may exclude some global combinations.
        candidates = [
            SolutionEntry(bitstring="1", prob=0.5, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
            SolutionEntry(bitstring="1", prob=0.2, decoded=[1]),
        ]
        programs = {
            "A": _MockProgram(candidates),
            "B": _MockProgram(candidates),
            "C": _MockProgram(candidates),
        }
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )

        # With _sum_evaluate (minimize sum), the optimal is [0,0,0]
        assert result == [0, 0, 0]

    def test_monotonicity_exhaustive_leq_beam_leq_greedy(self):
        """Wider beam should always find equal or better solutions.

        Verifies the invariant: cost(exhaustive) <= cost(beam) <= cost(greedy).
        """
        weights = [10, 1, 1, 10, 5, 2]

        def weighted_evaluate(solution):
            return sum(s * w for s, w in zip(solution, weights))

        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.15, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.05, decoded=[0, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="01", prob=0.7, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.2, decoded=[1, 0]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        candidates_c = [
            SolutionEntry(bitstring="11", prob=0.6, decoded=[1, 1]),
            SolutionEntry(bitstring="10", prob=0.3, decoded=[1, 0]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
        var_maps = {"A": [0, 1], "B": [2, 3], "C": [4, 5]}

        kwargs = dict(
            programs=programs,
            initial_solution=[0] * 6,
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
        )

        greedy_cost = weighted_evaluate(beam_search_aggregate(**kwargs, beam_width=1))
        beam_cost = weighted_evaluate(beam_search_aggregate(**kwargs, beam_width=2))
        exhaustive_cost = weighted_evaluate(
            beam_search_aggregate(**kwargs, beam_width=None)
        )

        assert exhaustive_cost <= beam_cost <= greedy_cost

    def test_n_partition_candidates_widens_search(self):
        """More candidates per partition can find better solutions with narrow beam."""

        def weighted_evaluate(solution):
            return sum(solution) * 10

        # 3 candidates; greedy (beam_width=1) only sees the first one ([1])
        candidates = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
            SolutionEntry(bitstring="1", prob=0.1, decoded=[1]),
        ]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0]}

        # beam_width=1, default n_partition_candidates (=1): only sees [1]
        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
        )

        # beam_width=1, n_partition_candidates=3: sees all 3, picks best ([0])
        wider_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
            n_partition_candidates=3,
        )

        assert weighted_evaluate(wider_result) <= weighted_evaluate(greedy_result)


class TestBeamSearchAggregateTopN:
    """Test _beam_search_aggregate_top_n returning multiple ranked solutions."""

    def _make_two_partition_setup(self):
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.6, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.3, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="10", prob=0.7, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.2, decoded=[0, 1]),
            SolutionEntry(bitstring="00", prob=0.1, decoded=[0, 0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}
        return programs, var_maps

    def test_top_n_returns_n_results(self):
        programs, var_maps = self._make_two_partition_setup()
        results = _beam_search_aggregate_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=3,
            top_n=3,
        )
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_top_n_sorted_ascending(self):
        programs, var_maps = self._make_two_partition_setup()
        results = _beam_search_aggregate_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=3,
            top_n=3,
        )
        scores = [score for score, _sol in results]
        assert scores == sorted(scores)

    def test_top_n_1_matches_original(self):
        programs, var_maps = self._make_two_partition_setup()
        kwargs = dict(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=2,
        )
        original = beam_search_aggregate(**kwargs)
        top_1 = _beam_search_aggregate_top_n(**kwargs, top_n=1)
        assert top_1[0][1] == original

    def test_beam_width_bumped_to_n(self):
        """top_n=3 with beam_width=1 still returns 3 results."""
        programs, var_maps = self._make_two_partition_setup()
        results = _beam_search_aggregate_top_n(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
            top_n=3,
        )
        assert len(results) == 3

    def test_top_n_bump_validates_against_n_partition_candidates(self):
        """n_partition_candidates must be >= beam_width *after* the top_n bump."""
        with pytest.raises(
            ValueError, match="n_partition_candidates.*must be >= beam_width"
        ):
            _beam_search_aggregate_top_n(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                beam_width=2,
                n_partition_candidates=3,
                top_n=5,
            )

    def test_top_n_greater_than_beam_capped(self):
        """When fewer solutions exist than top_n, returns all available."""
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0]}
        results = _beam_search_aggregate_top_n(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            beam_width=1,
            top_n=10,
        )
        # Only 1 candidate per partition × 1 partition = at most beam_width solutions
        # beam_width bumped to 10 but only 1 candidate exists
        assert len(results) >= 1
        assert len(results) <= 10


# ──────────────────────────────────────────────────────────────────────
#  Hierarchical aggregation tests
# ──────────────────────────────────────────────────────────────────────


def hierarchical_aggregate(
    programs,
    initial_solution,
    extend_fn,
    evaluate_fn,
    group_size=4,
    k_per_partition=20,
    max_per_group=200,
):
    """Test-local shorthand wrapping _hierarchical_aggregate_top_n for top_n=1."""
    return _hierarchical_aggregate_top_n(
        programs,
        initial_solution,
        extend_fn,
        evaluate_fn,
        top_n=1,
        group_size=group_size,
        k_per_partition=k_per_partition,
        max_per_group=max_per_group,
    )[0][1]


class TestHierarchicalAggregateValidation:
    def test_top_n_zero_raises(self):
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            _hierarchical_aggregate_top_n(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                top_n=0,
            )

    def test_group_size_zero_raises(self):
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            _hierarchical_aggregate_top_n(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                group_size=0,
            )

    def test_k_per_partition_zero_raises(self):
        with pytest.raises(ValueError, match="k_per_partition must be >= 1"):
            _hierarchical_aggregate_top_n(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                k_per_partition=0,
            )

    def test_max_per_group_zero_raises(self):
        with pytest.raises(ValueError, match="max_per_group must be >= 1"):
            _hierarchical_aggregate_top_n(
                programs={},
                initial_solution=[0],
                extend_fn=lambda c, p, s: c,
                evaluate_fn=lambda s: 0.0,
                max_per_group=0,
            )


class TestHierarchicalAggregateBasic:
    """Test basic hierarchical aggregation functionality."""

    def test_single_partition_single_candidate(self):
        """Single partition with one candidate returns that candidate."""
        candidates = [SolutionEntry(bitstring="10", prob=0.8, decoded=[1, 0])]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0, 1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 0]

    def test_two_partitions_picks_best(self):
        """Two partitions should combine to produce best solution."""
        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="11", prob=0.7, decoded=[1, 1]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 0, 1, 1]

    def test_empty_programs_returns_initial(self):
        """No programs returns the initial solution."""
        result = hierarchical_aggregate(
            programs={},
            initial_solution=[0, 0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
        )

        assert result == [0, 0, 0]

    def test_program_with_no_candidates_skipped(self):
        """A program returning no candidates is effectively skipped."""
        programs = {"A": _MockProgram([])}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=lambda c, p, s: c,
            evaluate_fn=_sum_evaluate,
        )

        assert result == [0, 0]


class TestHierarchicalAggregateGrouping:
    """Test the grouping and pairwise merge logic."""

    def test_group_size_1_processes_each_partition_separately(self):
        """group_size=1 creates one group per partition, then merges pairwise."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0], "B": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 1]

    def test_group_size_larger_than_partitions(self):
        """group_size larger than the number of partitions is fine."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0], "B": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=100,
        )

        assert result == [1, 1]

    def test_odd_number_of_groups_last_carried_forward(self):
        """An odd number of groups carries the last group forward unpaired."""
        candidates_a = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        candidates_b = [SolutionEntry(bitstring="1", prob=0.7, decoded=[1])]
        candidates_c = [SolutionEntry(bitstring="1", prob=0.5, decoded=[1])]
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
            group_size=1,
        )

        assert result == [1, 1, 1]


class TestHierarchicalAggregateFindsOptimal:
    """Test that hierarchical aggregation finds global optima."""

    def test_finds_global_optimum(self):
        """Should find the true global optimum across all combinations."""

        def tricky_evaluate(solution):
            if solution == [0, 1, 0]:
                return -100.0
            return sum(solution)

        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        candidates_c = [
            SolutionEntry(bitstring="1", prob=0.9, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.1, decoded=[0]),
        ]
        programs = {
            "A": _MockProgram(candidates_a),
            "B": _MockProgram(candidates_b),
            "C": _MockProgram(candidates_c),
        }
        var_maps = {"A": [0], "B": [1], "C": [2]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=tricky_evaluate,
            group_size=4,
        )

        assert result == [0, 1, 0]

    def test_finds_better_than_greedy_beam(self):
        """Hierarchical can find solutions that beam_width=1 misses."""
        weights = [10, 1, 1, 10]

        def weighted_evaluate(solution):
            return sum(s * w for s, w in zip(solution, weights))

        candidates_a = [
            SolutionEntry(bitstring="10", prob=0.9, decoded=[1, 0]),
            SolutionEntry(bitstring="01", prob=0.1, decoded=[0, 1]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="01", prob=0.9, decoded=[0, 1]),
            SolutionEntry(bitstring="10", prob=0.1, decoded=[1, 0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0, 1], "B": [2, 3]}

        greedy_result = beam_search_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
            beam_width=1,
        )

        hierarchical_result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[0, 0, 0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=weighted_evaluate,
        )

        greedy_cost = weighted_evaluate(greedy_result)
        hierarchical_cost = weighted_evaluate(hierarchical_result)

        assert hierarchical_cost <= greedy_cost


class TestHierarchicalAggregateTopN:
    """Test _hierarchical_aggregate_top_n returning multiple ranked solutions."""

    def test_returns_multiple_solutions(self):
        """Should return up to top_n solutions."""
        candidates = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.4, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [0]}

        results = _hierarchical_aggregate_top_n(
            programs=programs,
            initial_solution=[0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            top_n=3,
        )

        assert len(results) >= 1
        assert len(results) <= 3
        # Results sorted ascending (best first)
        for i in range(len(results) - 1):
            assert results[i][0] <= results[i + 1][0]

    def test_top_n_caps_output(self):
        """Even with many combinations, only top_n are returned."""
        candidates_a = [
            SolutionEntry(bitstring="1", prob=0.6, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.4, decoded=[0]),
        ]
        candidates_b = [
            SolutionEntry(bitstring="1", prob=0.7, decoded=[1]),
            SolutionEntry(bitstring="0", prob=0.3, decoded=[0]),
        ]
        programs = {"A": _MockProgram(candidates_a), "B": _MockProgram(candidates_b)}
        var_maps = {"A": [0], "B": [1]}

        results = _hierarchical_aggregate_top_n(
            programs=programs,
            initial_solution=[0, 0],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_sum_evaluate,
            top_n=2,
        )

        assert len(results) == 2
        # Best (lowest sum) first
        assert results[0][0] <= results[1][0]

    def test_non_zero_initial_solution_preserved(self):
        """Positions not touched by any partition keep initial values."""
        candidates = [SolutionEntry(bitstring="1", prob=0.9, decoded=[1])]
        programs = {"A": _MockProgram(candidates)}
        var_maps = {"A": [1]}

        result = hierarchical_aggregate(
            programs=programs,
            initial_solution=[1, 0, 1],
            extend_fn=_write_extend(var_maps),
            evaluate_fn=_neg_sum_evaluate,
        )

        assert result == [1, 1, 1]


@pytest.fixture
def small_partitioning_ensemble(dummy_simulator):
    """A real PartitioningProgramEnsemble with two QAOA partitions."""
    graph = nx.path_graph(4)
    problem = MaxCutProblem(
        graph,
        config=GraphPartitioningConfig(
            minimum_n_clusters=2, partitioning_algorithm="spectral"
        ),
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=2,
        backend=dummy_simulator,
    )
    ensemble.create_programs()
    yield ensemble
    try:
        ensemble.reset()
    except Exception:
        pass


def _seed_best_params(ensemble):
    """Populate _best_params on every sub-program with a zero array of the right shape."""
    for program in ensemble.programs.values():
        program._best_params = np.zeros(program.n_layers * program.n_params_per_layer)


class TestEnsembleSampleSolutionPreflight:
    """Validation that fires before any executor / coordinator setup."""

    def test_no_programs_raises(self, program_ensemble):
        """sample_solution() with no programs raises ``RuntimeError``."""
        with pytest.raises(RuntimeError, match="No programs"):
            program_ensemble.sample_solution()

    def test_non_vqa_subprogram_raises(self, program_ensemble):
        """Sub-programs that are not VQAs raise ``TypeError``."""
        program_ensemble.create_programs()
        with pytest.raises(TypeError, match="VariationalQuantumAlgorithm"):
            program_ensemble.sample_solution()

    def test_unknown_key_raises(self, small_partitioning_ensemble):
        """Keys not in ``self._programs`` are rejected upfront."""
        with pytest.raises(ValueError, match="not in this ensemble"):
            small_partitioning_ensemble.sample_solution(
                params_per_program={"unknown_prog_id": np.zeros(2)}
            )

    def test_shape_mismatch_raises(self, small_partitioning_ensemble):
        """Wrong-shape params for any program raise ``ValueError``."""
        any_pid = next(iter(small_partitioning_ensemble.programs.keys()))
        with pytest.raises(ValueError, match="does not match"):
            small_partitioning_ensemble.sample_solution(
                params_per_program={any_pid: np.zeros(99)}
            )

    def test_empty_best_params_raises(self, small_partitioning_ensemble):
        """No dict + no trained params per program raises ``RuntimeError``."""
        with pytest.raises(RuntimeError, match="no parameters available"):
            small_partitioning_ensemble.sample_solution()


class TestEnsembleSampleSolution:
    """End-to-end behavior of the new sampling-only entry point."""

    def test_full_dict_populates_best_probs(self, small_partitioning_ensemble):
        """Full dict path runs measurement on every program."""
        params_per_program = {
            pid: np.zeros(p.n_layers * p.n_params_per_layer)
            for pid, p in small_partitioning_ensemble.programs.items()
        }
        small_partitioning_ensemble.sample_solution(
            params_per_program=params_per_program, blocking=True
        )
        for program in small_partitioning_ensemble.programs.values():
            assert program._best_probs
        assert small_partitioning_ensemble.total_circuit_count > 0

    def test_none_path_uses_existing_best_params(self, small_partitioning_ensemble):
        """params_per_program=None reads each sub-program's _best_params."""
        _seed_best_params(small_partitioning_ensemble)
        small_partitioning_ensemble.sample_solution(blocking=True)
        for program in small_partitioning_ensemble.programs.values():
            assert program._best_probs

    def test_partial_dict_warns_about_fallbacks(self, small_partitioning_ensemble):
        """Permissive subset emits a UserWarning naming the fallback program IDs."""
        _seed_best_params(small_partitioning_ensemble)
        pids = list(small_partitioning_ensemble.programs.keys())
        first = small_partitioning_ensemble.programs[pids[0]]
        partial = {pids[0]: np.zeros(first.n_layers * first.n_params_per_layer)}
        fallback_id = pids[1]
        with pytest.warns(
            UserWarning, match=rf"missing keys.*{re.escape(repr(fallback_id))}"
        ):
            small_partitioning_ensemble.sample_solution(
                params_per_program=partial, blocking=True
            )

    def test_suppress_strict_warning(self, small_partitioning_ensemble):
        """suppress_strict_warning=True silences the fallback warning."""
        _seed_best_params(small_partitioning_ensemble)
        pids = list(small_partitioning_ensemble.programs.keys())
        first = small_partitioning_ensemble.programs[pids[0]]
        partial = {pids[0]: np.zeros(first.n_layers * first.n_params_per_layer)}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            small_partitioning_ensemble.sample_solution(
                params_per_program=partial,
                blocking=True,
                suppress_strict_warning=True,
            )

    def test_does_not_mutate_best_params(self, small_partitioning_ensemble):
        """Explicit params do not overwrite each sub-program's _best_params."""
        _seed_best_params(small_partitioning_ensemble)
        original = {
            pid: program._best_params.copy()
            for pid, program in small_partitioning_ensemble.programs.items()
        }
        params_per_program = {
            pid: np.ones(p.n_layers * p.n_params_per_layer)
            for pid, p in small_partitioning_ensemble.programs.items()
        }
        small_partitioning_ensemble.sample_solution(
            params_per_program=params_per_program, blocking=True
        )
        for pid, program in small_partitioning_ensemble.programs.items():
            np.testing.assert_array_equal(program._best_params, original[pid])

    def test_run_then_sample_solution(self, small_partitioning_ensemble):
        """After run(blocking=True), sample_solution() repopulates _best_probs."""
        small_partitioning_ensemble.run(blocking=True)
        circuits_after_run = small_partitioning_ensemble.total_circuit_count
        for program in small_partitioning_ensemble.programs.values():
            program._best_probs = {}

        small_partitioning_ensemble.sample_solution(blocking=True)

        circuits_delta = (
            small_partitioning_ensemble.total_circuit_count - circuits_after_run
        )
        assert circuits_delta >= len(small_partitioning_ensemble.programs)
        for program in small_partitioning_ensemble.programs.values():
            assert program._best_probs

    def test_aggregate_results_after_sample_solution_only(
        self, small_partitioning_ensemble
    ):
        """sample_solution() alone makes the ensemble ready for aggregate_results."""
        params_per_program = {
            pid: np.zeros(p.n_layers * p.n_params_per_layer)
            for pid, p in small_partitioning_ensemble.programs.items()
        }
        small_partitioning_ensemble.sample_solution(
            params_per_program=params_per_program, blocking=True
        )
        small_partitioning_ensemble.aggregate_results()


class TestEnsembleRedispatchLifecycle:
    """Per-dispatch state must be reset/restored so a second dispatch (e.g.
    a MERGED ``run`` followed by an un-batched ``sample_solution``) does not
    inherit stale state from the first.
    """

    def test_merged_run_restores_program_backends(self, small_partitioning_ensemble):
        """After a batched dispatch each program's real backend is restored.

        Batched dispatch swaps in a ``_ProxyBackend``; leaving it in place
        would make the program submit into a shut-down coordinator later.
        """
        ensemble = small_partitioning_ensemble
        originals = {pid: p.backend for pid, p in ensemble.programs.items()}

        ensemble.run(blocking=True)  # default BatchMode.MERGED

        for pid, program in ensemble.programs.items():
            assert not isinstance(program.backend, _ProxyBackend)
            assert program.backend is originals[pid]

    def test_merged_run_then_unbatched_sample_solution(
        self, small_partitioning_ensemble
    ):
        """A MERGED ``run`` then an OFF ``sample_solution`` must not submit
        through the (by then shut-down) coordinator via a dangling proxy.
        """
        ensemble = small_partitioning_ensemble
        ensemble.run(blocking=True)  # MERGED; coordinator shut down in join()

        ensemble.sample_solution(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

        for program in ensemble.programs.values():
            assert program._best_probs

    def test_unbatched_dispatch_clears_stale_program_keys(
        self, small_partitioning_ensemble
    ):
        """An un-batched dispatch following a batched one must not inherit the
        ``_program_key_map`` populated by the batched dispatch.
        """
        ensemble = small_partitioning_ensemble
        ensemble.run(blocking=True)  # MERGED populates _program_key_map

        ensemble.sample_solution(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

        assert ensemble._program_key_map == {}


class TestEnsembleCountAccounting:
    """Exact circuit-count and run-time accounting across dispatches.

    Programs accumulate their ``_total_*`` counters monotonically, so the
    ensemble must add only each dispatch's *delta* — never the program's
    cumulative total — or repeated dispatches over-count.
    """

    @pytest.fixture
    def _reset_after(self):
        created = []
        yield created.append
        for ensemble in created:
            try:
                ensemble.reset()
            except Exception:
                pass

    def test_single_off_dispatch_counts_exactly(self, dummy_simulator, _reset_after):
        """First dispatch: baseline is zero, so totals equal the increments."""
        ensemble = _AccumulatingEnsemble(
            backend=dummy_simulator, specs={"a": (10, 2.0), "b": (5, 3.0)}
        )
        _reset_after(ensemble)
        ensemble.create_programs()

        ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

        assert ensemble.total_circuit_count == 15
        assert ensemble.total_run_time == 5.0

    def test_repeated_off_dispatches_accumulate_exact_counts(
        self, dummy_simulator, _reset_after
    ):
        """Three OFF dispatches: ensemble totals grow by the per-dispatch
        delta each time (15 circuits / 5.0s per dispatch), never doubling.
        """
        ensemble = _AccumulatingEnsemble(
            backend=dummy_simulator, specs={"a": (10, 2.0), "b": (5, 3.0)}
        )
        _reset_after(ensemble)
        ensemble.create_programs()

        for dispatch in range(1, 4):
            ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

            assert ensemble.total_circuit_count == 15 * dispatch
            assert ensemble.total_run_time == 5.0 * dispatch
            # Invariant: ensemble total == sum of program lifetime totals.
            assert ensemble.total_circuit_count == sum(
                p._total_circuit_count for p in ensemble.programs.values()
            )
            assert ensemble.total_run_time == sum(
                p._total_run_time for p in ensemble.programs.values()
            )

    def test_first_dispatch_excludes_preexisting_program_counts(
        self, dummy_simulator, _reset_after
    ):
        """A program that already carries counts from prior standalone work
        contributes only its in-ensemble delta, not its pre-existing totals.
        """
        ensemble = _AccumulatingEnsemble(
            backend=dummy_simulator, specs={"a": (10, 2.0)}
        )
        _reset_after(ensemble)
        ensemble.create_programs()
        program = next(iter(ensemble.programs.values()))
        program._total_circuit_count = 100
        program._total_run_time = 50.0

        ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

        # Only the +10 / +2.0 increment from this dispatch is credited.
        assert ensemble.total_circuit_count == 10
        assert ensemble.total_run_time == 2.0

    def test_mixed_mode_dispatches_count_exactly(self, dummy_simulator, _reset_after):
        """Switching modes between dispatches must not perturb the delta
        accounting (MERGED then OFF).
        """
        ensemble = _AccumulatingEnsemble(
            backend=dummy_simulator, specs={"a": (10, 2.0), "b": (5, 3.0)}
        )
        _reset_after(ensemble)
        ensemble.create_programs()

        ensemble.run(blocking=True)  # MERGED
        ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

        assert ensemble.total_circuit_count == 30
        assert ensemble.total_run_time == 10.0
