# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for a single ``ProgramEnsemble`` dispatch.

Covers executor sizing, batch coordination, the progress display, join,
cancellation and failure handling, solution sampling, circuit/runtime
accounting, and dry runs. The multi-round workflow loop that drives repeated
dispatches lives in ``test_ensemble_workflow.py``.
"""

import copy
import os
import pickle
import re
import threading
import warnings
from concurrent.futures import Future

import networkx as nx
import numpy as np
import pytest

import divi.qprog.ensemble as ensemble_module
from divi.backends import AsyncJobBackend, ExecutionResult
from divi.exceptions import ExecutionCancelledError
from divi.qprog._batch_coordinator import _BatchCoordinator, _ProxyBackend
from divi.qprog.ensemble import (
    BatchConfig,
    BatchMode,
    ProgramEnsemble,
    ReportingLevel,
)
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import GraphPartitioningConfig, MaxCutProblem
from divi.qprog.workflows import PartitioningProgramEnsemble
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
    discard_progress_event,
)
from divi.reporting._session import ProgressSession
from divi.reporting._state import ProgressState
from tests.qprog._helpers import (
    SimpleTestProgram,
    _FakeRunResult,
    _StubProgram,
)
from tests.qprog._program_contracts import verify_basic_program_ensemble_behaviour


class SampleProgramEnsemble(ProgramEnsemble):
    """A mock ProgramEnsemble for testing."""

    def __init__(self, backend, **kwargs):
        super().__init__(backend, **kwargs)
        self.max_iterations = 5

    def create_programs(self, state=None):
        """Creates a set of mock programs."""
        super().create_programs()
        self.programs = {
            "prog1": SimpleTestProgram(10, 5.5, backend=self.backend),
            "prog2": SimpleTestProgram(5, 10.0, backend=self.backend),
        }

    def aggregate_results(self):
        """A mock aggregation function."""
        # The super() call is important to trigger checks and the join()
        super().aggregate_results()
        return sum(p.circ_count for p in self.programs.values())


class _RecordingSession:
    """Synchronous session double that retains real reducer behaviour."""

    def __init__(self, state: ProgressState):
        self.state = state
        self.events: list[ProgressEvent] = []
        self.closed = False

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)
        self.state.apply(event)

    def close(self) -> None:
        self.closed = True


class _CancellationBlockingProgram(SimpleTestProgram):
    """Block the first run until its ensemble cancellation event is set."""

    def __init__(self, *, backend, started: threading.Event, finished: threading.Event):
        super().__init__(1, 0.1, backend=backend)
        self.started = started
        self.finished = finished
        self.run_count = 0
        self.emitter_when_cancelled = None

    def run(self):
        self.run_count += 1
        if self.run_count == 1:
            self.started.set()
            if self._cancellation_event is None:
                raise RuntimeError("cancellation event was not installed")
            if not self._cancellation_event.wait(timeout=5):
                raise RuntimeError("worker was not cancelled during setup cleanup")
            self.emitter_when_cancelled = self._progress_emitter
            self.finished.set()
        return super().run()


class _NestedEmitterBlockingProgram(SimpleTestProgram):
    """Hold a worker-owned emitter binding open until reset cancellation."""

    def __init__(
        self,
        *,
        backend,
        entered: threading.Event,
        cancelled: threading.Event,
        release: threading.Event,
        finished: threading.Event,
    ):
        super().__init__(1, 0.1, backend=backend)
        self.entered = entered
        self.cancelled = cancelled
        self.release = release
        self.finished = finished

    def run(self):
        outer_emitter = self._progress_emitter

        def emit_nested(event):
            outer_emitter(event)

        with self._bind_progress_emitter(emit_nested):
            self.entered.set()
            if self._cancellation_event is None:
                raise RuntimeError("cancellation event was not installed")
            if not self._cancellation_event.wait(timeout=5):
                raise RuntimeError("worker was not cancelled during reset")
            self.cancelled.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test did not release cancelled worker")
        self.finished.set()
        return self


@pytest.fixture
def program_ensemble(dummy_simulator):
    batch = SampleProgramEnsemble(backend=dummy_simulator)
    yield batch
    try:
        batch.reset()
    except Exception:
        pass  # Don't break test teardown due to a race condition


class TestProgramEnsemble:
    def test_correct_initialization(self, program_ensemble):
        assert program_ensemble._executor is None
        assert len(program_ensemble.programs) == 0
        assert program_ensemble.total_circuit_count == 0
        assert program_ensemble.total_run_time == 0.0
        assert program_ensemble._progress_session is None

    def test_public_import_surface(self):
        # Guards against __init__.py regressions: everything ensemble.py
        # advertises in __all__ must be importable from divi.qprog.
        import divi.qprog as qprog

        missing = [name for name in ensemble_module.__all__ if not hasattr(qprog, name)]
        assert not missing, f"not re-exported from divi.qprog: {missing}"

    def test_basic_program_ensemble_behaviour(self, program_ensemble, mocker):
        """Uses the contract to verify basic error handling and state checks."""
        verify_basic_program_ensemble_behaviour(program_ensemble, mocker)

    def test_programs_dict_is_correct(self, program_ensemble):
        program_ensemble.create_programs()
        assert len(program_ensemble.programs) == 2
        assert "prog1" in program_ensemble.programs
        assert "prog2" in program_ensemble.programs
        assert program_ensemble._progress_session is None

    def test_reset_closes_session_and_restores_program_emitters(self, program_ensemble):
        program_ensemble.create_programs()
        programs = tuple(program_ensemble.programs.values())
        original_emitters = tuple(program._progress_emitter for program in programs)
        program_ensemble.run_one_round(blocking=False)
        session = program_ensemble._progress_session

        assert session is not None
        assert all(
            program._progress_emitter is not original
            for program, original in zip(programs, original_emitters, strict=True)
        )

        program_ensemble.reset()

        assert program_ensemble._executor is None
        assert program_ensemble.futures == []
        assert all(
            program._progress_emitter is original
            for program, original in zip(programs, original_emitters, strict=True)
        )

    def test_reset_waits_for_worker_emitter_bindings_before_teardown(
        self, dummy_simulator
    ):
        entered = threading.Event()
        cancelled = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        reset_done = threading.Event()
        reset_errors = []
        program = _NestedEmitterBlockingProgram(
            backend=dummy_simulator,
            entered=entered,
            cancelled=cancelled,
            release=release,
            finished=finished,
        )
        original_emitter = program._progress_emitter
        ensemble = SampleProgramEnsemble(backend=dummy_simulator)
        ensemble.programs = {"blocking": program}
        ensemble.run_one_round(
            blocking=False,
            batch_config=BatchConfig(mode=BatchMode.OFF),
        )
        assert entered.wait(timeout=2)

        def _reset():
            try:
                ensemble.reset()
            except BaseException as exc:
                reset_errors.append(exc)
            finally:
                reset_done.set()

        reset_thread = threading.Thread(target=_reset)
        reset_thread.start()
        assert cancelled.wait(timeout=2)
        returned_before_worker = reset_done.wait(timeout=1)
        release.set()
        assert reset_done.wait(timeout=2)
        reset_thread.join(timeout=2)

        assert not returned_before_worker
        assert finished.is_set()
        assert reset_errors == []
        assert program._progress_emitter is original_emitter

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
        program_ensemble.run_one_round(blocking=False)
        assert len(program_ensemble.futures) == 2
        program_ensemble.join()

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
        program_ensemble.run_one_round(blocking=False)

        # Subsequent run should raise an exception
        with pytest.raises(RuntimeError, match="An ensemble is already being run."):
            program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=False)

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
        # blocking=True should execute and then join.
        result = program_ensemble.run_one_round(blocking=True)

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
        program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=False)  # This registers the hook

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
        program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=True)
        result = program_ensemble.aggregate_results()

        assert result == 15  # 10 + 5

    def test_one_queued_session_is_created_per_dispatch(self, program_ensemble, mocker):
        queued = mocker.spy(ProgressSession, "queued")
        program_ensemble.create_programs()

        program_ensemble.run_one_round(blocking=True)

        assert queued.call_count == 1

    def test_program_targets_register_before_executor_submission(
        self, program_ensemble, mocker
    ):
        program_ensemble.create_programs()
        session = _RecordingSession(ProgressState(hide_successful_programs=True))
        mocker.patch.object(ProgressSession, "queued", return_value=session)
        original_add = program_ensemble._add_program_to_executor

        def _assert_registered(program, task_fn):
            target = program._progress_key
            assert any(
                event.kind is EventKind.REGISTER and event.progress_key == target
                for event in session.events
            )
            return original_add(program, task_fn)

        mocker.patch.object(
            program_ensemble,
            "_add_program_to_executor",
            side_effect=_assert_registered,
        )

        program_ensemble.run_one_round(blocking=True)

    def test_compact_program_is_initially_not_visible(self, program_ensemble, mocker):
        program_ensemble.create_programs()
        sessions: list[_RecordingSession] = []

        def _record_session(state, **kwargs):
            del kwargs
            session = _RecordingSession(state)
            sessions.append(session)
            return session

        mocker.patch.object(ProgressSession, "queued", side_effect=_record_session)

        program_ensemble._start_progress_session(batching_enabled=False)

        compact_state = sessions[0].state
        program = program_ensemble.programs["prog1"]
        assert compact_state.get(program._progress_key).visible is False

    def test_failed_compact_program_becomes_visible(self, program_ensemble, mocker):
        program_ensemble.create_programs()
        sessions: list[_RecordingSession] = []

        def _record_session(state, **kwargs):
            del kwargs
            session = _RecordingSession(state)
            sessions.append(session)
            return session

        mocker.patch.object(ProgressSession, "queued", side_effect=_record_session)
        program_ensemble._start_progress_session(batching_enabled=False)
        program = program_ensemble.programs["prog1"]

        sessions[0].emit(
            ProgressEvent.finish(program._progress_key, TerminalStatus.FAILED)
        )

        failed_compact_state = sessions[0].state
        assert failed_compact_state.get(program._progress_key).visible is True

    def test_full_program_is_initially_visible(self, dummy_simulator, mocker):
        ensemble = SampleProgramEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.FULL
        )
        ensemble.create_programs()
        sessions: list[_RecordingSession] = []

        def _record_session(state, **kwargs):
            del kwargs
            session = _RecordingSession(state)
            sessions.append(session)
            return session

        mocker.patch.object(ProgressSession, "queued", side_effect=_record_session)

        try:
            ensemble._start_progress_session(batching_enabled=False)
            full_state = sessions[0].state
            program = ensemble.programs["prog1"]
            assert full_state.get(program._progress_key).visible is True
        finally:
            ensemble.reset()

    def test_successful_dispatch_restores_program_emitters(self, program_ensemble):
        program_ensemble.create_programs()
        programs = tuple(program_ensemble.programs.values())
        originals = tuple(program._progress_emitter for program in programs)

        program_ensemble.run_one_round(blocking=True)

        assert all(
            program._progress_emitter is original
            for program, original in zip(programs, originals, strict=True)
        )

    def test_off_binds_discard_emitter_without_touching_logger(
        self, dummy_simulator, mocker
    ):
        ensemble = SampleProgramEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.OFF
        )
        ensemble.create_programs()
        seen_emitters = []
        original_add = ensemble._add_program_to_executor

        def _record_emitter(program, task_fn):
            seen_emitters.append(program._progress_emitter)
            return original_add(program, task_fn)

        mocker.patch.object(
            ensemble, "_add_program_to_executor", side_effect=_record_emitter
        )
        logger = ensemble_module.logger
        level = logger.level
        handlers = tuple(logger.handlers)

        ensemble.run_one_round(blocking=True)

        assert seen_emitters == [discard_progress_event, discard_progress_event]
        assert ensemble._progress_session is None
        assert logger.level == level
        assert tuple(logger.handlers) == handlers

    def test_off_remains_silent_after_dispatch_teardown(self, dummy_simulator, caplog):
        ensemble = SampleProgramEnsemble(
            backend=dummy_simulator, reporting_level=ReportingLevel.OFF
        )
        ensemble.create_programs()
        ensemble.run_one_round(blocking=True)

        caplog.clear()
        with caplog.at_level("INFO", logger="divi"):
            ensemble._progress_emitter(
                ProgressEvent.show(("workflow", id(ensemble)), "late event")
            )

        assert caplog.records == []

    @pytest.mark.parametrize(
        "reporting_level", [ReportingLevel.COMPACT, ReportingLevel.FULL]
    )
    def test_env_var_suppresses_session_without_touching_logging(
        self, dummy_simulator, mocker, monkeypatch, reporting_level
    ):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        ensemble = SampleProgramEnsemble(
            backend=dummy_simulator,
            reporting_level=reporting_level,
        )
        ensemble.create_programs()
        seen_emitters = []
        original_add = ensemble._add_program_to_executor

        def _record_emitter(program, task_fn):
            seen_emitters.append(program._progress_emitter)
            return original_add(program, task_fn)

        mocker.patch.object(
            ensemble, "_add_program_to_executor", side_effect=_record_emitter
        )
        queued = mocker.spy(ProgressSession, "queued")
        logger = ensemble_module.logger
        level = logger.level
        handlers = tuple(logger.handlers)
        disabled = logger.disabled

        ensemble.run_one_round(blocking=True)

        queued.assert_not_called()
        assert seen_emitters == [discard_progress_event, discard_progress_event]
        assert ensemble._progress_session is None
        assert logger.level == level
        assert tuple(logger.handlers) == handlers
        assert logger.disabled is disabled

    def test_ensemble_owns_no_direct_rich_progress_objects(self, program_ensemble):
        program_ensemble.create_programs()
        program_ensemble.run_one_round(blocking=True)

        assert not hasattr(program_ensemble, "_progress_bar")
        assert not hasattr(program_ensemble, "_live_display")
        assert not hasattr(program_ensemble, "_listener_thread")
        assert not hasattr(program_ensemble, "_pb_task_map")
        assert not hasattr(program_ensemble, "_queue")
        assert not hasattr(program_ensemble, "_done_event")

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

    def test_typed_program_message_and_finish_use_the_bound_emitter(
        self, program_ensemble
    ):
        program_ensemble.create_programs()
        session = _RecordingSession(ProgressState())
        target = program_ensemble.programs["prog1"]._progress_key
        session.emit(
            ProgressEvent.register(
                target, ProgressScope.PROGRAM, "Program prog1", total=1
            )
        )
        program_ensemble._progress_emitter = session.emit

        program_ensemble._emit_progress_message(
            target, final_status=TerminalStatus.FAILED, message="Job failed"
        )

        assert session.events[-2:] == [
            ProgressEvent.show(target, "Job failed"),
            ProgressEvent.finish(target, TerminalStatus.FAILED, detail="Job failed"),
        ]

    def test_handle_cancellation_phases(self, program_ensemble, mocker):
        """Test all three phases of cancellation handling.

        Each phase emits a synthetic progress message via
        ``_emit_progress_message``; we spy on the helper to inspect the
        messages.
        """
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
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
        program_ensemble._start_progress_session(batching_enabled=False)
        program_ensemble._cancellation_event = mocker.MagicMock()
        render_failure = mocker.patch("divi.qprog.ensemble.render_failure")

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

        render_failure.assert_called_once()
        exc = render_failure.call_args.args[0]
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "boom"
        assert render_failure.call_args.kwargs == {
            "label": " (Program prog1)",
            "console": program_ensemble._progress_session.console,
        }

    def test_cancellation_without_failures_prints_no_failure_panels(
        self, program_ensemble, mocker
    ):
        """When every program either ran cleanly or cancelled cooperatively,
        no Rich failure panels should be printed — only the existing
        progress-row status updates."""
        program_ensemble.create_programs()
        program_ensemble._cancellation_event = mocker.MagicMock()
        render_failure = mocker.patch("divi.qprog.ensemble.render_failure")

        cancelled_future = Future()
        cancelled_future.set_exception(ExecutionCancelledError("Cancelled by user"))

        program_ensemble.futures = [cancelled_future]
        program_ensemble._future_to_program = {
            cancelled_future: program_ensemble.programs["prog1"]
        }
        mocker.patch("divi.qprog.ensemble.as_completed", return_value=[])

        program_ensemble._handle_cancellation()

        render_failure.assert_not_called()

    def test_handle_cancellation_unstoppable_futures(self, program_ensemble, mocker):
        """Test cancellation handling with unstoppable futures."""
        program_ensemble.create_programs()
        spy = mocker.spy(program_ensemble, "_emit_progress_message")
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
        program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=False)
        spy = mocker.spy(program_ensemble, "_emit_progress_message")

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
            call.args[0] == progs[0]._progress_key for call in failed_calls
        ), "the failed program's row was not emitted with final_status=Failed"

    def test_handle_failure_non_batched_cancels_jobs(self, program_ensemble, mocker):
        """Without coordinator, failure should call cancel_unfinished_job on running programs."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(blocking=False)
        program_ensemble._coordinator = None

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
        program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=False)
        spy = mocker.spy(program_ensemble, "_emit_progress_message")

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
        failed_targets = {
            call.args[0]
            for call in spy.call_args_list
            if call.kwargs.get("final_status") is TerminalStatus.FAILED
        }
        assert failed_targets == {program._progress_key for program in progs}

    def test_join_early_return_no_executor(self, program_ensemble):
        """Test join returns early when no executor."""
        program_ensemble._executor = None
        result = program_ensemble.join()
        assert result is None

    def test_join_keyboard_interrupt(self, program_ensemble, mocker):
        """Test join handles KeyboardInterrupt."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(blocking=False)

        mocker.patch(
            "divi.qprog.ensemble.as_completed", side_effect=KeyboardInterrupt()
        )
        mock_handle_cancellation = mocker.patch.object(
            program_ensemble, "_handle_cancellation"
        )

        result = program_ensemble.join()

        assert result is False
        mock_handle_cancellation.assert_called_once()
        # run() reads this to stop the workflow instead of starting a round.
        assert program_ensemble._round_cancelled is True

    def test_join_keyboard_interrupt_no_double_count(self, program_ensemble, mocker):
        """Results collected before KeyboardInterrupt are not double-counted."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(blocking=False)

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
        program_ensemble.run_one_round(blocking=False)

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
            f_bad: mocker.Mock(_progress_key="prog_bad"),
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

    def test_run_rejects_deepcopied_program_target_collision_before_session(
        self, program_ensemble, mocker
    ):
        program_ensemble.create_programs()
        original = program_ensemble.programs["prog1"]
        copied = copy.deepcopy(original)
        assert copied is not original
        assert copied._progress_key == original._progress_key
        program_ensemble.programs = {"original": original, "copied": copied}
        queued = mocker.spy(ProgressSession, "queued")

        with pytest.raises(RuntimeError, match="Duplicate progress keys"):
            program_ensemble.run_one_round(blocking=True)

        queued.assert_not_called()
        assert program_ensemble._executor is None

    def test_run_rejects_pickle_restored_target_collision_before_session(
        self, program_ensemble, mocker
    ):
        program_ensemble.create_programs()
        original = program_ensemble.programs["prog1"]
        restored = pickle.loads(pickle.dumps(original))
        assert restored is not original
        assert restored._progress_key == original._progress_key
        program_ensemble.programs = {"original": original, "restored": restored}
        queued = mocker.spy(ProgressSession, "queued")

        with pytest.raises(RuntimeError, match="Duplicate progress keys"):
            program_ensemble.run_one_round(blocking=True)

        queued.assert_not_called()
        assert program_ensemble._executor is None

    def test_atexit_unregister_failure(self, program_ensemble, mocker):
        """Test atexit unregister handles TypeError."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(blocking=False)
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
        program_ensemble.run_one_round(blocking=False)
        mock_join = mocker.spy(program_ensemble, "join")
        result = program_ensemble.aggregate_results()
        mock_join.assert_called_once()
        assert result == 15

    def test_run_passes_batch_config_to_coordinator(self, program_ensemble):
        """BatchConfig is forwarded to the coordinator."""
        program_ensemble.create_programs()
        config = BatchConfig(max_batch_size=50)
        program_ensemble.run_one_round(blocking=True, batch_config=config)
        assert program_ensemble.total_circuit_count == 15

    def test_run_with_batching_off(self, program_ensemble):
        """BatchMode.OFF disables the coordinator entirely."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(
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
            f"prog{i}": SimpleTestProgram(1, 0.1, backend=dummy_simulator)
            for i in range(1, 5)
        }
        programs = tuple(ensemble.programs.values())
        original_emitters = tuple(program._progress_emitter for program in programs)

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
            ensemble.run_one_round(blocking=False)

        coord = captured["coord"]
        # Load-bearing assertion: the orphaned-registration bug.
        assert coord._active_programs == set()
        assert coord._pending == {}
        # Coordinator + executor handle on the ensemble should be cleared.
        assert ensemble._coordinator is None
        assert ensemble._executor is None
        assert all(
            program._progress_emitter is original
            for program, original in zip(programs, original_emitters, strict=True)
        )
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
            ensemble.run_one_round(blocking=False)

        # Subsequent register_program calls fall through to the original
        # implementation, so the second run() should succeed end-to-end.
        ensemble.run_one_round(blocking=True)
        assert ensemble.aggregate_results() == 15

    def test_unbatched_partial_submission_waits_before_restoring_and_reuse(
        self, dummy_simulator, mocker
    ):
        ensemble = SampleProgramEnsemble(backend=dummy_simulator)
        ensemble.create_programs()
        started = threading.Event()
        finished = threading.Event()
        blocking = _CancellationBlockingProgram(
            backend=dummy_simulator,
            started=started,
            finished=finished,
        )
        other = SimpleTestProgram(1, 0.1, backend=dummy_simulator)
        ensemble.programs = {"blocking": blocking, "other": other}
        originals = tuple(
            program._progress_emitter for program in ensemble.programs.values()
        )
        original_add = ensemble._add_program_to_executor
        submission_count = 0

        def _fail_second_submission(program, task_fn):
            nonlocal submission_count
            submission_count += 1
            if submission_count == 2:
                raise RuntimeError("second submission failed")
            future = original_add(program, task_fn)
            if submission_count == 1:
                assert started.wait(timeout=2)
            return future

        mocker.patch.object(
            ensemble,
            "_add_program_to_executor",
            side_effect=_fail_second_submission,
        )

        try:
            with pytest.raises(RuntimeError, match="second submission failed"):
                ensemble.run_one_round(
                    blocking=False,
                    batch_config=BatchConfig(mode=BatchMode.OFF),
                )

            assert ensemble._cancellation_event.is_set()
            assert finished.is_set()
            assert blocking.emitter_when_cancelled is not originals[0]
            assert ensemble._executor is None
            assert ensemble.futures == []
            assert all(
                program._progress_emitter is original
                for program, original in zip(
                    ensemble.programs.values(), originals, strict=True
                )
            )

            ensemble.run_one_round(
                blocking=True,
                batch_config=BatchConfig(mode=BatchMode.OFF),
            )
            assert blocking.run_count == 2
            assert other.has_results()
        finally:
            ensemble._cancellation_event.set()
            finished.wait(timeout=2)
            ensemble.reset()


def test_cancellation_event_is_shared_with_coordinator(dummy_simulator):
    """When the ensemble's cancellation event is set, the coordinator's
    ``_cancelled`` Event must also report ``is_set()``."""
    ensemble = SampleProgramEnsemble(backend=dummy_simulator)
    ensemble.create_programs()
    try:
        ensemble.run_one_round(blocking=False)
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
            f"prog_{i}": SimpleTestProgram(1, 0.0, backend=self.backend)
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
            )
            for i in range(self._n_programs)
        }

    def aggregate_results(self):
        super().aggregate_results()
        return None


def _record_merged_sizes(mocker, backend) -> list[int]:
    """Patch ``backend.submit_circuits`` to record each merged call's size."""
    original = backend.submit_circuits
    merged_sizes: list[int] = []

    def _spy(payloads, **kwargs):
        merged_sizes.append(len(payloads))
        return original(payloads, **kwargs)

    mocker.patch.object(backend, "submit_circuits", _spy)
    return merged_sizes


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
        ensemble.run_one_round(blocking=True)

        spy.assert_called_once()
        expected = max(10, (os.cpu_count() or 1) + 4)
        assert spy.call_args.kwargs["max_workers"] == expected

    def test_default_barrier_path_floors_at_cpu_default(self, dummy_simulator, mocker):
        """Small ensembles still get the cpu+4 default — never under-provisioned."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=2)
        ensemble.create_programs()
        ensemble.run_one_round(blocking=True)

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] >= (os.cpu_count() or 1) + 4

    def test_max_batch_size_pool_aligns_with_batch(self, dummy_simulator, mocker):
        """``max_batch_size`` sizes the pool to ``min(max_batch_size, n_programs)``
        so the barrier predicate can fill the batch in one wave (instead of
        firing prematurely at ``cpu+4``)."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=20)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=4)
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 4

    def test_max_batch_size_pool_capped_at_n_programs(self, dummy_simulator, mocker):
        """When ``max_batch_size > len(programs)``, the pool falls back to
        ``len(programs)`` — never spawn more threads than there is work for."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=8)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=512)
        )

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
        merged_sizes = _record_merged_sizes(mocker, dummy_simulator)

        ensemble = _SubmittingEnsemble(backend=dummy_simulator, n_programs=32)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=8)
        )

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
        merged_sizes = _record_merged_sizes(mocker, dummy_simulator)

        ensemble = _SubmittingEnsemble(
            backend=dummy_simulator, n_programs=8, n_circuits_per_program=5
        )
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=10)
        )

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
        ensemble.run_one_round(
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
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == (os.cpu_count() or 1) + 4

    def test_exceeds_barrier_limit_raises(self, dummy_simulator):
        """Default config + >256 programs fails fast with an actionable message."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        with pytest.raises(RuntimeError) as excinfo:
            ensemble.run_one_round(blocking=True)

        msg = str(excinfo.value)
        assert "257" in msg
        assert "max_batch_size" in msg
        assert "BatchMode.OFF" in msg

    def test_exceeds_barrier_limit_succeeds_with_max_batch_size(self, dummy_simulator):
        """Same large ensemble runs cleanly once the user opts into early-flush."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=8)
        )
        # All 257 programs ran (each contributes circ_count=1).
        assert ensemble.total_circuit_count == 257

    def test_exceeds_barrier_limit_succeeds_with_off_mode(self, dummy_simulator):
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=257)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )
        assert ensemble.total_circuit_count == 257

    def test_coordinator_n_workers_matches_executor_in_early_flush(
        self, program_ensemble
    ):
        """The coordinator's barrier cap matches executor capacity in early-flush."""
        program_ensemble.create_programs()
        program_ensemble.run_one_round(
            blocking=False, batch_config=BatchConfig(max_batch_size=4)
        )
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
        program_ensemble.run_one_round(blocking=False)
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
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(max_batch_size=64)
        )
        assert ensemble.total_circuit_count == 64

    def test_max_concurrent_programs_sizes_pool_directly(self, dummy_simulator, mocker):
        """``max_concurrent_programs`` on BatchConfig drives executor size."""
        spy = self._spy_executor(mocker)
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=10)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=10),
        )

        spy.assert_called_once()
        assert spy.call_args.kwargs["max_workers"] == 10

    def test_max_concurrent_programs_bypasses_barrier_limit(self, dummy_simulator):
        """Explicit ``max_concurrent_programs`` lifts the 256-program cap."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=300)
        ensemble.create_programs()
        ensemble.run_one_round(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=300),
        )
        assert ensemble.total_circuit_count == 300

    def test_max_concurrent_programs_above_soft_cap_warns(self, dummy_simulator):
        """Values above the advisory soft cap emit a UserWarning."""
        ensemble = _ParameterizedEnsemble(backend=dummy_simulator, n_programs=2)
        ensemble.create_programs()
        with pytest.warns(UserWarning, match="max_concurrent_programs"):
            ensemble.run_one_round(
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
        ensemble.run_one_round(
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
        ensemble.run_one_round(
            blocking=True,
            batch_config=BatchConfig(max_concurrent_programs=-1),
        )

        soft_cap_warnings = [
            w for w in recwarn.list if "max_concurrent_programs" in str(w.message)
        ]
        assert soft_cap_warnings == []


def _small_partitioning_ensemble(backend, **kwargs):
    """Build a real PartitioningProgramEnsemble with two QAOA partitions."""
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
        backend=backend,
        **kwargs,
    )
    ensemble.create_programs()
    return ensemble


@pytest.fixture
def small_partitioning_ensemble(dummy_simulator):
    """A real PartitioningProgramEnsemble with two QAOA partitions."""
    ensemble = _small_partitioning_ensemble(dummy_simulator)
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

    def test_workflow_rejects_non_vqa_programs_before_training(
        self, dummy_simulator, make_dummy_simulator
    ):
        """A sampling backend cannot split a workflow with unsampleable children."""
        ensemble = SampleProgramEnsemble(
            backend=dummy_simulator,
            sampling_backend=make_dummy_simulator(100, seed=7),
        )

        with pytest.raises(TypeError, match="sampling_backend.*variational"):
            ensemble.run(max_rounds=1)

        assert all(not program._ran for program in ensemble.programs.values())

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

    def test_overridden_measurement_hook_receives_the_routed_backend(
        self, small_partitioning_ensemble, mocker
    ):
        """Ensemble routing calls an overridden hook with the swapped backend."""
        ensemble = small_partitioning_ensemble
        _seed_best_params(ensemble)
        hooks = [
            mocker.patch.object(
                program,
                "_run_solution_measurement_for",
                side_effect=lambda _param_sets, *, backend=None, program=program: setattr(
                    program, "_best_probs", {0: {"00": 1.0}}
                ),
            )
            for program in ensemble.programs.values()
        ]

        ensemble.sample_solution(blocking=True)

        assert all(hook.call_count == 1 for hook in hooks)

    def test_shared_sampling_backend_is_owned_by_the_ensemble(
        self, dummy_simulator, make_dummy_simulator
    ):
        """Children leave sampling routing to their ensemble coordinator."""
        sampling_backend = make_dummy_simulator(100, seed=7)
        ensemble = _small_partitioning_ensemble(
            dummy_simulator,
            sampling_backend=sampling_backend,
        )

        try:
            assert ensemble.sampling_backend is sampling_backend
            assert all(
                program.sampling_backend is None
                for program in ensemble.programs.values()
            )
        finally:
            ensemble.reset()

    def test_direct_sampling_uses_the_configured_sampling_backend(
        self, dummy_simulator, make_dummy_simulator, mocker
    ):
        """The ensemble-level default applies outside the workflow run path."""
        sampling_backend = make_dummy_simulator(100, seed=7)
        primary_submit = mocker.spy(dummy_simulator, "submit_circuits")
        sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
        ensemble = _small_partitioning_ensemble(
            dummy_simulator,
            sampling_backend=sampling_backend,
        )
        _seed_best_params(ensemble)

        try:
            ensemble.sample_solution(blocking=True)

            primary_submit.assert_not_called()
            sampling_submit.assert_called_once()
        finally:
            ensemble.reset()

    def test_workflow_run_batches_final_sampling_on_sampling_backend(
        self, dummy_simulator, make_dummy_simulator, mocker
    ):
        """Training completes before one merged sampling-only dispatch."""
        sampling_backend = make_dummy_simulator(100, seed=7)
        primary_submit = mocker.spy(dummy_simulator, "submit_circuits")
        sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
        ensemble = _small_partitioning_ensemble(
            dummy_simulator,
            sampling_backend=sampling_backend,
        )

        try:
            ensemble.run()

            assert primary_submit.call_count > 0
            sampling_submit.assert_called_once()
            assert all(program._best_probs for program in ensemble.programs.values())
        finally:
            ensemble.reset()

    def test_run_one_round_blocking_batches_final_sampling_on_sampling_backend(
        self, dummy_simulator, make_dummy_simulator, mocker
    ):
        """run_one_round(blocking=True) splits training from sampling too."""
        sampling_backend = make_dummy_simulator(100, seed=7)
        primary_submit = mocker.spy(dummy_simulator, "submit_circuits")
        sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
        ensemble = _small_partitioning_ensemble(
            dummy_simulator,
            sampling_backend=sampling_backend,
        )

        try:
            ensemble.run_one_round(blocking=True)

            assert primary_submit.call_count > 0
            sampling_submit.assert_called_once()
            assert all(program._best_probs for program in ensemble.programs.values())
        finally:
            ensemble.reset()

    def test_run_one_round_nonblocking_rejects_sampling_backend(
        self, dummy_simulator, make_dummy_simulator
    ):
        """run_one_round(blocking=False) can't split training from sampling."""
        ensemble = _small_partitioning_ensemble(
            dummy_simulator,
            sampling_backend=make_dummy_simulator(100, seed=7),
        )

        try:
            with pytest.raises(ValueError, match="blocking=False"):
                ensemble.run_one_round(blocking=False)
        finally:
            ensemble.reset()

    def test_backend_override_receives_one_merged_sampling_submission(
        self, small_partitioning_ensemble, make_dummy_simulator, mocker
    ):
        """The ensemble override wins over primary and child sampling backends."""
        ensemble = small_partitioning_ensemble
        sampling_backend = make_dummy_simulator(100, seed=7)
        child_sampling_backend = make_dummy_simulator(100, seed=11)
        primary_submit = mocker.spy(ensemble.backend, "submit_circuits")
        sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
        child_sampling_submit = mocker.spy(child_sampling_backend, "submit_circuits")
        for program in ensemble.programs.values():
            program._sampling_backend = child_sampling_backend
        _seed_best_params(ensemble)

        ensemble.sample_solution(
            blocking=True,
            backend=sampling_backend,
        )

        primary_submit.assert_not_called()
        child_sampling_submit.assert_not_called()
        sampling_submit.assert_called_once()
        assert all(
            program.backend is ensemble.backend
            for program in ensemble.programs.values()
        )

    def test_backend_override_routes_each_unbatched_sampling_submission(
        self, small_partitioning_ensemble, make_dummy_simulator, mocker
    ):
        """Unbatched sampling uses the override and restores child backends."""
        ensemble = small_partitioning_ensemble
        sampling_backend = make_dummy_simulator(100, seed=7)
        primary_submit = mocker.spy(ensemble.backend, "submit_circuits")
        sampling_submit = mocker.spy(sampling_backend, "submit_circuits")
        _seed_best_params(ensemble)

        ensemble.sample_solution(
            blocking=True,
            backend=sampling_backend,
            batch_config=BatchConfig(mode=BatchMode.OFF),
        )

        primary_submit.assert_not_called()
        assert sampling_submit.call_count == len(ensemble.programs)
        assert all(
            program.backend is ensemble.backend
            for program in ensemble.programs.values()
        )

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
        """The high-level workflow run populates VQE sampling probabilities."""
        small_partitioning_ensemble.run()
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

        ensemble.run_one_round(blocking=True)  # default BatchMode.MERGED

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
        ensemble.run_one_round(blocking=True)  # MERGED; coordinator shut down in join()

        ensemble.sample_solution(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

        for program in ensemble.programs.values():
            assert program._best_probs


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

        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

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
            ensemble.run_one_round(
                blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
            )

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

        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

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

        ensemble.run_one_round(blocking=True)  # MERGED
        ensemble.run_one_round(
            blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF)
        )

        assert ensemble.total_circuit_count == 30
        assert ensemble.total_run_time == 10.0


class TestProgramEnsembleDryRun:
    """``ProgramEnsemble.dry_run`` delegates to each sub-program and keys the
    result by program identifier, without touching the run-time machinery."""

    def test_report_keyed_by_program_id(self, program_ensemble):
        # Keying only; per-program content passthrough is covered by
        # test_keys_by_program_id_and_forwards_force_flag (with distinct
        # sentinel values, since the stub programs return {} here).
        program_ensemble.create_programs()
        reports = program_ensemble.dry_run()
        assert set(reports) == set(program_ensemble.programs)

    def test_without_programs_raises(self, program_ensemble):
        with pytest.raises(RuntimeError, match="create_programs"):
            program_ensemble.dry_run()

    def test_keys_by_program_id_and_forwards_force_flag(self, program_ensemble, mocker):
        program_ensemble.create_programs()
        sentinels = {}
        spies = {}
        for prog_id, program in program_ensemble.programs.items():
            sentinels[prog_id] = {"cost": mocker.sentinel.report}
            spies[prog_id] = mocker.patch.object(
                program, "dry_run", return_value=sentinels[prog_id]
            )

        reports = program_ensemble.dry_run(force_circuit_generation=True)

        assert reports == sentinels
        for spy in spies.values():
            spy.assert_called_once_with(force_circuit_generation=True)

    def test_failing_program_aborts_and_is_named(self, program_ensemble, mocker):
        program_ensemble.create_programs()
        programs = program_ensemble.programs
        mocker.patch.object(
            programs["prog2"], "dry_run", side_effect=ValueError("boom")
        )
        with pytest.raises(RuntimeError, match="prog2"):
            program_ensemble.dry_run()
