# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Event
from queue import Queue
from threading import Event as ThreadingEvent
from threading import Lock, Thread

import pytest
from rich.progress import Progress

from divi.backends import ExecutionResult
from divi.qprog.ensemble import BatchConfig, BatchMode, ProgramEnsemble
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting import queue_listener
from tests.qprog.qprog_contracts import verify_basic_program_ensemble_behaviour


class SimpleTestProgram(QuantumProgram):
    """A simple mock program for testing ProgramEnsemble execution."""

    def __init__(self, circ_count: int, run_time: float, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.circ_count = circ_count
        self.run_time = run_time
        # program_id is automatically set by the base class from kwargs["job_id"]
        # This attribute is checked by the base ProgramEnsemble.aggregate_results method
        self.losses_history = [1]

    def _build_pipelines(self) -> None:
        pass

    def run(self) -> tuple[int, float]:
        """A mock run that just returns the preset values."""
        return self.circ_count, self.run_time

    def _generate_circuits(self, **kwargs):
        """Dummy implementation for the abstract method."""
        return []

    def _post_process_results(self, results: dict):
        """Dummy implementation for the abstract method."""


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
        verify_basic_program_ensemble_behaviour(mocker, program_ensemble)

    def test_programs_dict_is_correct(self, program_ensemble):
        program_ensemble.create_programs()
        assert len(program_ensemble.programs) == 2
        assert "prog1" in program_ensemble.programs
        assert "prog2" in program_ensemble.programs
        assert hasattr(program_ensemble._queue, "get")  # Check if it's queue-like
        assert isinstance(program_ensemble._done_event, ThreadingEvent)

    def test_reset_cleans_up_all_resources(self, program_ensemble, mocker):
        """Tests that reset() correctly shuts down and cleans up all resources."""
        # First, create programs to initialize the manager, queue, etc.
        program_ensemble.create_programs()
        # Now, simulate a running state by creating an executor and futures
        program_ensemble._executor = mocker.MagicMock(spec=ThreadPoolExecutor)
        program_ensemble._listener_thread = mocker.MagicMock(spec=Thread)
        program_ensemble._progress_bar = mocker.MagicMock()
        program_ensemble._batch_progress = mocker.MagicMock()
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
        assert program_ensemble.futures is None

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

    def test_run_fails_if_no_programs(self, program_ensemble):
        with pytest.raises(RuntimeError, match="No programs to run."):
            program_ensemble.run()

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
        successful_future.set_result((5, 5.0))
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

    def testqueue_listener_exception_handling(self, mocker):
        """Test queue listener handles unexpected exceptions."""
        mock_queue = mocker.MagicMock()
        mock_progress_bar = mocker.MagicMock()
        mock_done_event = Event()
        pb_task_map = {}
        lock = Lock()
        mock_queue.get.side_effect = Exception("Unexpected queue error")
        mock_console = mocker.MagicMock()
        mock_progress_bar.console = mock_console

        listener_thread = Thread(
            target=queue_listener,
            args=(mock_queue, mock_progress_bar, pb_task_map, mock_done_event, lock),
        )
        listener_thread.start()
        time.sleep(0.2)
        mock_done_event.set()
        listener_thread.join(timeout=1)

        mock_console.log.assert_called()
        assert "Unexpected exception" in str(mock_console.log.call_args)

    def testqueue_listener_optional_message_fields(self, mocker):
        """Test queue listener handles all optional message fields."""
        mock_queue = Queue()
        mock_progress_bar = mocker.MagicMock()
        mock_done_event = Event()
        pb_task_map = {"job1": 1}
        lock = Lock()

        mock_queue.put(
            {
                "job_id": "job1",
                "progress": 1,
                "poll_attempt": 3,
                "max_retries": 5,
                "service_job_id": "service_123",
                "job_status": "running",
                "message": "Processing...",
                "final_status": "completed",
                "loss": -0.321,
            }
        )
        listener_thread = Thread(
            target=queue_listener,
            args=(mock_queue, mock_progress_bar, pb_task_map, mock_done_event, lock),
        )
        listener_thread.start()
        time.sleep(0.1)
        mock_done_event.set()
        listener_thread.join(timeout=1)

        mock_progress_bar.update.assert_called_once()
        call_kwargs = mock_progress_bar.update.call_args[1]
        assert call_kwargs["poll_attempt"] == 3
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["service_job_id"] == "service_123"
        assert call_kwargs["job_status"] == "running"
        assert call_kwargs["loss"] == -0.321

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
        program_ensemble._batch_progress = mocker.MagicMock()
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

    def test_handle_cancellation_phases(self, program_ensemble, mocker):
        """Test all three phases of cancellation handling."""
        program_ensemble.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}
        program_ensemble._cancellation_event = mocker.MagicMock()

        future1, future2, future3 = Future(), Future(), Future()
        future3.set_result((5, 2.0))  # already done
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
        assert mock_progress_bar.update.call_count >= 2

        # Verify specific update messages for each phase
        update_calls = mock_progress_bar.update.call_args_list
        messages = [call[1].get("message", "") for call in update_calls]
        assert "Cancelled by user" in messages
        assert "Finishing... ⏳" in messages
        assert "Stopped after current iteration" in messages

    def test_handle_cancellation_unstoppable_futures(self, program_ensemble, mocker):
        """Test cancellation handling with unstoppable futures."""
        program_ensemble.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
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
            for call in mock_progress_bar.update.call_args_list
            if call[1].get("message") == "Finishing... ⏳"
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

        # Mock backend cancel_job
        mock_backend = mocker.Mock()
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
        f_good.set_result((10, 5.0))
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
        """Failure path should update progress bars with failure status."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
        program_ensemble._pb_task_map = {"prog1": 1, "prog2": 2}

        f_bad = Future()
        f_bad.set_exception(RuntimeError("Job xyz has failed."))
        f_good = Future()
        f_good.set_result((10, 5.0))
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

        update_calls = mock_progress_bar.update.call_args_list
        final_statuses = [
            call[1].get("final_status")
            for call in update_calls
            if "final_status" in call[1]
        ]
        assert "Failed" in final_statuses

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
        f_running.set_result((3, 1.0))

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
        """When batch coordinator fails all futures, every program should be marked Failed."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        mock_progress_bar = mocker.MagicMock()
        program_ensemble._progress_bar = mock_progress_bar
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

        # Both programs should have a "Failed" final_status
        update_calls = mock_progress_bar.update.call_args_list
        failed_task_ids = {
            call[0][0]
            for call in update_calls
            if call[1].get("final_status") == "Failed"
        }
        assert failed_task_ids == {1, 2}

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
        mock_collect_results = mocker.patch.object(
            program_ensemble, "_collect_completed_results"
        )

        result = program_ensemble.join()

        assert result is False
        mock_handle_cancellation.assert_called_once()
        mock_collect_results.assert_called_once()

    def test_join_keyboard_interrupt_no_double_count(self, program_ensemble, mocker):
        """Results collected before KeyboardInterrupt are not double-counted."""
        program_ensemble.create_programs()
        program_ensemble.run(blocking=False)

        # Create two pre-resolved futures with known results
        f1 = Future()
        f1.set_result((10, 5.0))
        f2 = Future()
        f2.set_result((7, 3.0))
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
        f1.set_result((10, 5.0))
        f2 = Future()
        f2.set_result((7, 3.0))
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


class TestBatchConfig:
    def test_default_values(self):
        config = BatchConfig()
        assert config.mode is BatchMode.MERGED
        assert config.max_batch_size is None

    def test_valid_max_batch_size(self):
        config = BatchConfig(max_batch_size=10)
        assert config.max_batch_size == 10

    def test_max_batch_size_one(self):
        config = BatchConfig(max_batch_size=1)
        assert config.max_batch_size == 1

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchConfig(max_batch_size=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchConfig(max_batch_size=-5)

    def test_rejects_max_batch_size_with_off_mode(self):
        with pytest.raises(ValueError, match="max_batch_size has no effect"):
            BatchConfig(mode=BatchMode.OFF, max_batch_size=10)

    def test_off_mode(self):
        config = BatchConfig(mode=BatchMode.OFF)
        assert config.mode is BatchMode.OFF
        assert config.max_batch_size is None

    def test_frozen(self):
        config = BatchConfig(max_batch_size=10)
        with pytest.raises(AttributeError):
            config.max_batch_size = 20


def testqueue_listener(mocker):
    """Unit test for the queue_listener function."""
    mock_queue = Queue()
    mock_progress_bar = mocker.MagicMock(spec=Progress)
    mock_done_event = Event()
    lock = Lock()
    pb_task_map = {"job1": 1, "job2": 2}

    mock_queue.put(
        {
            "job_id": "job1",
            "progress": 1,
            "message": "step 1",
            "final_status": "running",
        }
    )
    mock_queue.put({"job_id": "job2", "progress": 1, "poll_attempt": 3})

    listener_thread = Thread(
        target=queue_listener,
        args=(mock_queue, mock_progress_bar, pb_task_map, mock_done_event, lock),
    )
    listener_thread.start()
    time.sleep(0.1)
    mock_done_event.set()
    listener_thread.join(timeout=1)
    assert not listener_thread.is_alive()

    expected_calls = [
        mocker.call(1, advance=1, message="step 1", final_status="running"),
        mocker.call(2, advance=1, poll_attempt=3),
    ]
    mock_progress_bar.update.assert_has_calls(expected_calls, any_order=True)
    assert mock_queue.empty()
