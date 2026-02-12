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
from divi.qprog.batch import ProgramBatch, _queue_listener
from divi.qprog.quantum_program import QuantumProgram
from tests.qprog.qprog_contracts import verify_basic_program_batch_behaviour


class SimpleTestProgram(QuantumProgram):
    """A simple mock program for testing ProgramBatch execution."""

    def __init__(self, circ_count: int, run_time: float, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.circ_count = circ_count
        self.run_time = run_time
        # program_id is automatically set by the base class from kwargs["job_id"]
        # This attribute is checked by the base ProgramBatch.aggregate_results method
        self.losses_history = [1]

    def run(self) -> tuple[int, float]:
        """A mock run that just returns the preset values."""
        return self.circ_count, self.run_time

    def _generate_circuits(self, **kwargs):
        """Dummy implementation for the abstract method."""
        return []

    def _post_process_results(self, results: dict):
        """Dummy implementation for the abstract method."""
        pass


class SampleProgramBatch(ProgramBatch):
    """A mock ProgramBatch for testing."""

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
def program_batch(dummy_simulator):
    batch = SampleProgramBatch(backend=dummy_simulator)
    yield batch
    try:
        batch.reset()
    except Exception:
        pass  # Don't break test teardown due to a race condition


@pytest.fixture(autouse=True)
def stop_live_display(program_batch):
    """Fixture to automatically stop any active Rich progress bars after a test."""
    yield
    if (
        hasattr(program_batch, "_progress_bar")
        and program_batch._progress_bar is not None
        and not program_batch._progress_bar.finished
    ):
        program_batch._progress_bar.stop()


class TestProgramBatch:
    def test_correct_initialization(self, program_batch):
        assert program_batch._executor is None
        assert len(program_batch.programs) == 0
        assert program_batch.total_circuit_count == 0
        assert program_batch.total_run_time == 0.0

    def test_basic_program_batch_behaviour(self, program_batch, mocker):
        """Uses the contract to verify basic error handling and state checks."""
        verify_basic_program_batch_behaviour(mocker, program_batch)

    def test_programs_dict_is_correct(self, program_batch):
        program_batch.create_programs()
        assert len(program_batch.programs) == 2
        assert "prog1" in program_batch.programs
        assert "prog2" in program_batch.programs
        assert hasattr(program_batch._queue, "get")  # Check if it's queue-like
        assert isinstance(program_batch._done_event, ThreadingEvent)

    def test_reset_cleans_up_all_resources(self, program_batch, mocker):
        """Tests that reset() correctly shuts down and cleans up all resources."""
        # First, create programs to initialize the manager, queue, etc.
        program_batch.create_programs()
        # Now, simulate a running state by creating an executor and futures
        program_batch._executor = mocker.MagicMock(spec=ThreadPoolExecutor)
        program_batch._listener_thread = mocker.MagicMock(spec=Thread)
        program_batch._progress_bar = mocker.MagicMock()
        program_batch.futures = [mocker.MagicMock()]
        program_batch._pb_task_map = {}

        # Configure the mock to behave like a stopped thread to prevent warnings
        program_batch._listener_thread.is_alive.return_value = False

        # Spy on the original objects before they are cleared
        mock_executor = program_batch._executor
        done_event_set_spy = mocker.spy(program_batch._done_event, "set")
        mock_listener_thread = program_batch._listener_thread
        mock_progress_bar = program_batch._progress_bar

        # Call the method under test
        program_batch.reset()

        # Assert that cleanup methods were called on the spied objects
        mock_executor.shutdown.assert_called_once_with(wait=False)
        done_event_set_spy.assert_called_once()
        mock_listener_thread.join.assert_called_once()
        mock_progress_bar.stop.assert_called_once()

        # Assert that state attributes are cleared
        assert program_batch._executor is None
        assert program_batch.futures is None

    def test_total_circuit_count_setter(self, program_batch):
        with pytest.raises(
            AttributeError,
            match="property 'total_circuit_count' of 'SampleProgramBatch'",
        ):
            program_batch.total_circuit_count = 100

    def test_total_run_time_setter(self, program_batch):
        with pytest.raises(
            AttributeError,
            match="property 'total_run_time' of 'SampleProgramBatch'",
        ):
            program_batch.total_run_time = 100

    def test_run_sets_executor_and_returns_expected_number_of_futures(
        self, program_batch
    ):
        program_batch.create_programs()
        program_batch.run()
        assert program_batch._executor is not None
        assert len(program_batch.futures) == 2

    def test_run_fails_if_no_programs(self, program_batch):
        with pytest.raises(RuntimeError, match="No programs to run."):
            program_batch.run()

    def test_run_fails_if_already_running(self, mocker, program_batch):
        program_batch.create_programs()

        # Mock ThreadPoolExecutor to simulate a long-running batch
        mock_executor = mocker.patch("divi.qprog.batch.ThreadPoolExecutor")
        mock_instance = mock_executor.return_value

        # Create futures that simulate a long-running process
        future1 = Future()
        future2 = Future()
        mock_instance.submit.side_effect = [future1, future2]

        # First run should work
        program_batch.run()

        # Subsequent run should raise an exception
        with pytest.raises(RuntimeError, match="A batch is already being run."):
            program_batch.run()

    def test_run_submits_correct_tasks(self, program_batch, mocker):
        """Tests that run() submits the correct function and arguments to the executor."""
        program_batch.create_programs()
        mock_executor_class = mocker.patch("divi.qprog.batch.ThreadPoolExecutor")
        mock_executor = mock_executor_class.return_value

        # The executor's submit method returns Future objects.
        mock_future_1 = mocker.MagicMock(spec=Future)
        mock_future_2 = mocker.MagicMock(spec=Future)
        mock_executor.submit.side_effect = [mock_future_1, mock_future_2]

        # Mock `as_completed` so that a later call to join() doesn't hang.
        mocker.patch(
            "divi.qprog.batch.as_completed", return_value=[mock_future_1, mock_future_2]
        )

        # Run non-blocking to inspect the state before it's cleaned up
        program_batch.run(blocking=False)

        assert mock_executor.submit.call_count == len(program_batch.programs)

        # The function submitted should be the internal _task_fn
        task_fn = program_batch._task_fn
        for program in program_batch.programs.values():
            mock_executor.submit.assert_any_call(task_fn, program)

        # Clean up the non-blocking run. This should now terminate correctly.
        program_batch.join()

    def test_blocking_run_executes_and_joins_correctly(self, program_batch):
        """Integration test for a standard blocking run."""
        program_batch.create_programs()
        # run(blocking=True) is the default, which should execute and then join.
        result = program_batch.run(blocking=True)

        assert result is program_batch
        assert program_batch.total_circuit_count == 15
        assert program_batch.total_run_time == 15.5
        # After a blocking run, the executor should be gone.
        assert program_batch._executor is None
        assert len(program_batch.futures) == 0

    def test_non_blocking_run_registers_atexit_hook(self, program_batch, mocker):
        """Tests that a non-blocking run correctly registers a cleanup hook."""
        mock_atexit_register = mocker.patch("atexit.register")
        program_batch.create_programs()
        program_batch.run(blocking=False)

        # Check that the executor is active and hook is registered
        assert program_batch._executor is not None
        mock_atexit_register.assert_called_once_with(program_batch._atexit_cleanup_hook)

        # Manually clean up to avoid side effects
        program_batch.join()

    def test_join_unregisters_atexit_hook(self, program_batch, mocker):
        """Tests that join() unregisters the cleanup hook after a non-blocking run."""
        mock_atexit_register = mocker.patch("atexit.register")
        mock_atexit_unregister = mocker.patch("atexit.unregister")

        program_batch.create_programs()
        program_batch.run(blocking=False)  # This registers the hook

        mock_atexit_register.assert_called_once()

        program_batch.join()  # This should unregister it

        mock_atexit_unregister.assert_called_once_with(
            program_batch._atexit_cleanup_hook
        )

    def test_check_all_done_true_when_all_futures_ready(self, program_batch):
        future_1 = Future()
        future_2 = Future()
        program_batch.futures = [future_1, future_2]

        # Test when no futures are done
        assert not program_batch.check_all_done()

        # Complete one future
        future_1.set_result(None)
        assert not program_batch.check_all_done()

        # Complete second future
        future_2.set_result(None)
        assert program_batch.check_all_done()

    def test_join_handles_task_exceptions(self, program_batch, mocker):
        """Ensures join() catches exceptions from futures and still cleans up."""
        program_batch.create_programs()
        program_batch.run(blocking=False)  # Start a non-blocking run

        # Mock the futures to simulate one failing task
        failing_future = Future()
        failing_future.set_exception(ValueError("Task failed"))
        successful_future = Future()
        successful_future.set_result((5, 5.0))
        program_batch.futures = [failing_future, successful_future]

        # Mock as_completed to yield the failing future first
        mocker.patch(
            "divi.qprog.batch.as_completed",
            return_value=[failing_future, successful_future],
        )
        # Mock executor shutdown to confirm it's called during cleanup
        mock_shutdown = mocker.spy(program_batch._executor, "shutdown")

        with pytest.raises(RuntimeError, match="Batch execution failed"):
            program_batch.join()

        mock_shutdown.assert_called_once_with(wait=True)
        assert program_batch._executor is None

    def test_aggregate_results_calls_join_and_aggregates(self, program_batch):
        """
        Tests that aggregate_results works correctly after a successful run,
        verifying the end-to-end data flow.
        """
        program_batch.create_programs()
        program_batch.run(blocking=True)
        result = program_batch.aggregate_results()

        assert result == 15  # 10 + 5

    def test_queue_listener_exception_handling(self, mocker):
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
            target=_queue_listener,
            args=(
                mock_queue,
                mock_progress_bar,
                pb_task_map,
                mock_done_event,
                False,
                lock,
            ),
        )
        listener_thread.start()
        time.sleep(0.2)
        mock_done_event.set()
        listener_thread.join(timeout=1)

        mock_console.log.assert_called()
        assert "Unexpected exception" in str(mock_console.log.call_args)

    def test_queue_listener_optional_message_fields(self, mocker):
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
            }
        )
        listener_thread = Thread(
            target=_queue_listener,
            args=(
                mock_queue,
                mock_progress_bar,
                pb_task_map,
                mock_done_event,
                False,
                lock,
            ),
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

    def test_reset_listener_thread_timeout(self, program_batch, mocker):
        """Test reset handles listener thread timeout warning."""
        program_batch.create_programs()
        program_batch._done_event = Event()
        mock_thread = mocker.MagicMock(spec=Thread)
        mock_thread.is_alive.return_value = True  # Simulate timeout
        program_batch._listener_thread = mock_thread

        with pytest.warns(UserWarning, match="Listener thread did not terminate"):
            program_batch.reset()
        mock_thread.join.assert_called_once_with(timeout=1)

    def test_reset_progress_bar_exception(self, program_batch, mocker):
        """Test reset handles progress bar stop exception."""
        program_batch.create_programs()
        mock_progress_bar = mocker.MagicMock()
        mock_progress_bar.stop.side_effect = Exception("Stop failed")
        program_batch._progress_bar = mock_progress_bar
        program_batch._pb_task_map = {}

        # Should not raise exception, just pass silently
        program_batch.reset()
        mock_progress_bar.stop.assert_called_once()

    def test_atexit_cleanup_warning(self, program_batch, mocker):
        """Test atexit cleanup hook issues warning."""
        program_batch.create_programs()
        mock_executor = mocker.MagicMock()
        program_batch._executor = mock_executor

        with pytest.warns(
            UserWarning, match="non-blocking ProgramBatch run was not explicitly closed"
        ):
            program_batch._atexit_cleanup_hook()

    def test_handle_cancellation_phases(self, program_batch, mocker):
        """Test all three phases of cancellation handling."""
        program_batch.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_batch._progress_bar = mock_progress_bar
        program_batch._pb_task_map = {"prog1": 1, "prog2": 2}
        program_batch._cancellation_event = mocker.MagicMock()

        future1, future2, future3 = Future(), Future(), Future()
        future3.set_result((5, 2.0))  # already done
        mocker.patch.object(
            future1, "cancel", return_value=True
        )  # pending, can be cancelled
        mocker.patch.object(
            future2, "cancel", return_value=False
        )  # running, cannot be cancelled

        program_batch.futures = [future1, future2, future3]
        program_batch._future_to_program = {
            future1: program_batch.programs["prog1"],
            future2: program_batch.programs["prog2"],
            future3: mocker.Mock(),  # Dummy program for completed future
        }
        mocker.patch("divi.qprog.batch.as_completed", return_value=[future2])

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program_batch._handle_cancellation()

        program_batch._cancellation_event.set.assert_called_once()
        assert mock_progress_bar.update.call_count >= 2

        # Verify specific update messages for each phase
        update_calls = mock_progress_bar.update.call_args_list
        messages = [call[1].get("message", "") for call in update_calls]
        assert "Cancelled by user" in messages
        assert "Finishing... ⏳" in messages
        assert "Completed during cancellation" in messages

    def test_handle_cancellation_unstoppable_futures(self, program_batch, mocker):
        """Test cancellation handling with unstoppable futures."""
        program_batch.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_batch._progress_bar = mock_progress_bar
        program_batch._pb_task_map = {"prog1": 1}
        program_batch._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_batch.futures = [future]
        program_batch._future_to_program = {future: program_batch.programs["prog1"]}
        mocker.patch("divi.qprog.batch.as_completed", return_value=[future])

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program_batch._handle_cancellation()

        finishing_calls = [
            call
            for call in mock_progress_bar.update.call_args_list
            if call[1].get("message") == "Finishing... ⏳"
        ]
        assert len(finishing_calls) > 0

    def test_handle_cancellation_calls_cancel_unfinished_job(
        self, program_batch, mocker
    ):
        """Test that _handle_cancellation calls cancel_unfinished_job for unstoppable futures."""
        program_batch.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_batch._progress_bar = mock_progress_bar
        program_batch._pb_task_map = {"prog1": 1}
        program_batch._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_batch.futures = [future]
        program = program_batch.programs["prog1"]
        program_batch._future_to_program = {future: program}

        # Mock cancel_unfinished_job - this prevents the warning since the actual method isn't called
        mock_cancel = mocker.patch.object(program, "cancel_unfinished_job")
        # Mock as_completed to return the future so Phase 3 doesn't hang
        mocker.patch("divi.qprog.batch.as_completed", return_value=[future])

        program_batch._handle_cancellation()

        # Verify cancel_unfinished_job was called
        mock_cancel.assert_called_once()

    def test_handle_cancellation_delegates_to_backend_cancel_job(
        self, program_batch, mocker
    ):
        """Test that _handle_cancellation delegates to backend.cancel_job via cancel_unfinished_job."""
        program_batch.create_programs()
        mock_progress_bar = mocker.MagicMock()
        program_batch._progress_bar = mock_progress_bar
        program_batch._pb_task_map = {"prog1": 1}
        program_batch._cancellation_event = mocker.MagicMock()

        future = Future()
        future.cancel = mocker.MagicMock(return_value=False)

        program_batch.futures = [future]
        program = program_batch.programs["prog1"]
        program_batch._future_to_program = {future: program}

        # Set up program with execution result
        execution_result = ExecutionResult(job_id="test_job_123")
        program._current_execution_result = execution_result

        # Mock backend cancel_job
        mock_backend = mocker.Mock()
        mock_backend.cancel_job = mocker.Mock()
        program.backend = mock_backend

        # Mock as_completed to return the future so Phase 3 doesn't hang
        mocker.patch("divi.qprog.batch.as_completed", return_value=[future])

        program_batch._handle_cancellation()

        # Verify cancel_job was called on the backend
        mock_backend.cancel_job.assert_called_once_with(execution_result)

    def test_join_early_return_no_executor(self, program_batch):
        """Test join returns early when no executor."""
        program_batch._executor = None
        result = program_batch.join()
        assert result is None

    def test_join_keyboard_interrupt(self, program_batch, mocker):
        """Test join handles KeyboardInterrupt."""
        program_batch.create_programs()
        program_batch.run(blocking=False)

        mocker.patch("divi.qprog.batch.as_completed", side_effect=KeyboardInterrupt())
        mock_handle_cancellation = mocker.patch.object(
            program_batch, "_handle_cancellation"
        )
        mock_collect_results = mocker.patch.object(
            program_batch, "_collect_completed_results"
        )

        result = program_batch.join()

        assert result is False
        mock_handle_cancellation.assert_called_once()
        mock_collect_results.assert_called_once()

    def test_atexit_unregister_failure(self, program_batch, mocker):
        """Test atexit unregister handles TypeError."""
        program_batch.create_programs()
        program_batch.run(blocking=False)
        mock_unregister = mocker.patch(
            "atexit.unregister", side_effect=TypeError("Not registered")
        )
        program_batch.join()
        mock_unregister.assert_called_once()

    def test_aggregate_results_with_running_executor(self, program_batch, mocker):
        """
        Tests that aggregate_results calls join() if the executor is still
        running and then correctly aggregates the results.
        """
        program_batch.create_programs()
        program_batch.run(blocking=False)
        mock_join = mocker.spy(program_batch, "join")
        result = program_batch.aggregate_results()
        mock_join.assert_called_once()
        assert result == 15


def test_queue_listener(mocker):
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
        target=_queue_listener,
        args=(mock_queue, mock_progress_bar, pb_task_map, mock_done_event, False, lock),
    )
    listener_thread.start()
    time.sleep(0.1)
    mock_done_event.set()
    listener_thread.join(timeout=1)
    assert not listener_thread.is_alive()

    common_kwargs = {"advance": 1, "refresh": False}
    expected_calls = [
        mocker.call(1, message="step 1", final_status="running", **common_kwargs),
        mocker.call(2, poll_attempt=3, **common_kwargs),
    ]
    mock_progress_bar.update.assert_has_calls(expected_calls, any_order=True)
    assert mock_queue.empty()
