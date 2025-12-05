# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from queue import Queue
from threading import Event

import pytest
import requests

from divi.backends import ExecutionResult, JobStatus
from divi.qprog.exceptions import _CancelledError
from divi.qprog.quantum_program import QuantumProgram


class ConcreteQuantumProgram(QuantumProgram):
    """Concrete implementation of QuantumProgram for testing."""

    def __init__(self, backend, seed=None, progress_queue=None, **kwargs):
        super().__init__(backend, seed, progress_queue, **kwargs)
        self._total_circuit_count = 0
        self._total_run_time = 0.0

    def run(self) -> tuple[int, float]:
        """Concrete implementation of run method."""
        return (5, 1.5)

    def _generate_circuits(self, **kwargs):
        """Concrete implementation of _generate_circuits method."""
        # Return empty list as this is a minimal test implementation
        return []

    def _post_process_results(self, results: dict):
        """Concrete implementation of _post_process_results method."""
        return {"processed": "results"}


class TestQuantumProgramBase:
    """Tests for QuantumProgram abstract base class contract and core functionality."""

    def test_initialization_with_all_params(self, mocker):
        """Test QuantumProgram initialization with all parameters."""
        mock_backend = mocker.Mock()
        mock_queue = Queue()

        program = ConcreteQuantumProgram(
            backend=mock_backend, seed=42, progress_queue=mock_queue
        )

        assert program.backend == mock_backend
        assert program._seed == 42
        assert program._progress_queue == mock_queue

    def test_initialization_with_kwargs(self, mocker):
        """Test QuantumProgram initialization with additional kwargs."""
        mock_backend = mocker.Mock()

        program = ConcreteQuantumProgram(
            backend=mock_backend, custom_param="test_value", another_param=123
        )

        assert program.backend == mock_backend
        assert program._seed is None
        assert program._progress_queue is None

    def test_abstract_class_behavior(self, mocker):
        """Test abstract class instantiation behavior."""
        mock_backend = mocker.Mock()

        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantumProgram(backend=mock_backend)

        # Test that concrete implementations can be instantiated
        program = ConcreteQuantumProgram(backend=mock_backend)
        assert isinstance(program, QuantumProgram)
        assert program.backend == mock_backend

    def test_abstract_methods_must_be_implemented(self, mocker):
        """Test that all abstract methods must be implemented in subclasses."""
        mock_backend = mocker.Mock()

        # Test missing run method
        class IncompleteProgram1(QuantumProgram):
            def _generate_circuits(self, **kwargs):
                return []

            def _post_process_results(self, results):
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram1(backend=mock_backend)

        # Test missing _generate_circuits method
        class IncompleteProgram2(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _post_process_results(self, results):
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram2(backend=mock_backend)

        # Test missing _post_process_results method
        class IncompleteProgram3(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _generate_circuits(self, **kwargs):
                return []

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram3(backend=mock_backend)

    def test_cancellation_event(self, mocker):
        """Test _set_cancellation_event method."""
        mock_backend = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        event = Event()
        program._set_cancellation_event(event)

        assert hasattr(program, "_cancellation_event")
        assert program._cancellation_event == event

    def test_total_circuit_count_property(self, mocker):
        """Test total_circuit_count property."""
        mock_backend = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        program._total_circuit_count = 15
        assert program.total_circuit_count == 15

    def test_total_run_time_property(self, mocker):
        """Test total_run_time property."""
        mock_backend = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        program._total_run_time = 3.7
        assert program.total_run_time == 3.7

    def test_properties_default_to_zero(self, mocker):
        """Test that properties default to zero when not set."""
        mock_backend = mocker.Mock()

        class DefaultProgram(QuantumProgram):
            def run(self):
                return (0, 0.0)

            def _generate_circuits(self, **kwargs):
                return []

            def _post_process_results(self, results):
                return {}

        program = DefaultProgram(mock_backend)
        assert program.total_circuit_count == 0
        assert program.total_run_time == 0.0


class TestQuantumProgramAsyncExecution:
    """Tests for QuantumProgram asynchronous execution paths and callbacks."""

    @pytest.fixture
    def mock_async_backend(self, mocker):
        """Fixture for a mock asynchronous backend."""
        backend = mocker.Mock()
        backend.is_async = True
        backend.submit_circuits.return_value = ExecutionResult(
            results=None, job_id="fake_job_id"
        )

        # Use a side effect to properly simulate the callback behavior
        def poll_side_effect(*args, **kwargs):
            # Simulate the callback being called once
            if (
                "progress_callback" in kwargs
                and kwargs["progress_callback"] is not None
            ):
                kwargs["progress_callback"](1, "RUNNING")
            return JobStatus.COMPLETED

        backend.poll_job_status.side_effect = poll_side_effect
        backend.get_job_results.return_value = ExecutionResult(
            results=[{"label": "circuit_1", "results": "final_results"}],
            job_id="fake_job_id",
        )
        return backend

    @pytest.fixture
    def async_test_program_class(self, mocker):
        """Fixture providing a reusable async test program class."""

        def _create_program_class():
            class AsyncTestProgram(ConcreteQuantumProgram):
                def run(self, **kwargs):
                    self._curr_circuits = self._generate_circuits()
                    return self._dispatch_circuits_and_process_results(**kwargs)

                def _generate_circuits(self, **kwargs):
                    executable = mocker.Mock()
                    executable.tag = "circuit_1"
                    executable.qasm = "fake_qasm"
                    bundle = mocker.Mock()
                    bundle.executables = [executable]
                    return [bundle]

                def _post_process_results(self, results: dict, **kwargs):
                    return results

            return AsyncTestProgram

        return _create_program_class

    def test_async_workflow_with_reporter(
        self, mock_async_backend, mocker, async_test_program_class
    ):
        """Tests that reporter callback is invoked during async workflow."""
        AsyncTestProgram = async_test_program_class()
        mock_queue = Queue()
        program = AsyncTestProgram(
            backend=mock_async_backend, progress_queue=mock_queue
        )
        program.reporter = mocker.Mock()

        program.run()

        program.reporter.info.assert_called()
        call_kwargs = program.reporter.info.call_args.kwargs
        assert call_kwargs.get("service_job_id") == "fake_job_id"
        assert "poll_attempt" in call_kwargs

    def test_async_workflow_uses_job_id(
        self, mock_async_backend, async_test_program_class
    ):
        """Tests that job_id from ExecutionResult is used for async polling."""
        AsyncTestProgram = async_test_program_class()
        program = AsyncTestProgram(backend=mock_async_backend)

        program.run()

        # Verify that poll_job_status was called with the ExecutionResult
        mock_async_backend.poll_job_status.assert_called_once()
        call_args = mock_async_backend.poll_job_status.call_args[0]
        assert call_args[0].job_id == "fake_job_id"

    def test_async_workflow_processes_results(
        self, mock_async_backend, async_test_program_class
    ):
        """Tests that async workflow correctly processes and returns results."""
        AsyncTestProgram = async_test_program_class()
        program = AsyncTestProgram(backend=mock_async_backend)

        result = program.run()

        assert result == {"circuit_1": "final_results"}

    def test_async_workflow_without_reporter(
        self, mock_async_backend, async_test_program_class
    ):
        """Tests that the async workflow works correctly even without a reporter."""
        AsyncTestProgram = async_test_program_class()
        # No progress_queue means no reporter attribute
        program = AsyncTestProgram(backend=mock_async_backend)

        result = program.run()

        mock_async_backend.submit_circuits.assert_called_once()
        mock_async_backend.poll_job_status.assert_called_once()
        assert result == {"circuit_1": "final_results"}

    def test_async_workflow_tracks_runtime(
        self, mock_async_backend, async_test_program_class
    ):
        """Tests that the async workflow correctly tracks runtime via the on_complete callback."""
        AsyncTestProgram = async_test_program_class()

        def poll_with_runtime(*args, **kwargs):
            if "on_complete" in kwargs and kwargs["on_complete"] is not None:
                kwargs["on_complete"]({"run_time": "2.5"})
            if (
                "progress_callback" in kwargs
                and kwargs["progress_callback"] is not None
            ):
                kwargs["progress_callback"](1, "RUNNING")
            return JobStatus.COMPLETED

        mock_async_backend.poll_job_status.side_effect = poll_with_runtime

        program = AsyncTestProgram(backend=mock_async_backend)
        program.run()

        assert program.total_run_time == 2.5

    def test_prepare_and_send_circuits_increments_count(self, mocker):
        """
        Tests that _prepare_and_send_circuits correctly increments _total_circuit_count.
        """

        class CircuitCountTestProgram(ConcreteQuantumProgram):
            def _generate_circuits(self, **kwargs):
                # Create multiple bundles with multiple executables
                executable1 = mocker.Mock()
                executable1.tag = "circuit_1"
                executable1.qasm = "qasm_1"

                executable2 = mocker.Mock()
                executable2.tag = "circuit_2"
                executable2.qasm = "qasm_2"

                executable3 = mocker.Mock()
                executable3.tag = "circuit_3"
                executable3.qasm = "qasm_3"

                bundle1 = mocker.Mock()
                bundle1.executables = [executable1, executable2]

                bundle2 = mocker.Mock()
                bundle2.executables = [executable3]

                return [bundle1, bundle2]

        mock_backend = mocker.Mock()
        mock_backend.is_async = False
        mock_backend.submit_circuits.return_value = ExecutionResult(
            results=[
                {"label": "circuit_1", "results": {}},
                {"label": "circuit_2", "results": {}},
                {"label": "circuit_3", "results": {}},
            ]
        )

        program = CircuitCountTestProgram(backend=mock_backend)
        program._curr_circuits = program._generate_circuits()

        # Initial count should be 0
        assert program.total_circuit_count == 0

        # Call _prepare_and_send_circuits
        program._prepare_and_send_circuits()

        # Should have incremented by 3 (one for each executable)
        assert program.total_circuit_count == 3

        # Call again to verify cumulative behavior
        program._prepare_and_send_circuits()
        assert program.total_circuit_count == 6

    def test_async_job_failure_raises_exception(
        self, mock_async_backend, async_test_program_class
    ):
        """Tests that an exception is raised if the async job fails."""
        AsyncTestProgram = async_test_program_class()

        mock_async_backend.poll_job_status.side_effect = None
        mock_async_backend.poll_job_status.return_value = JobStatus.FAILED

        program = AsyncTestProgram(backend=mock_async_backend)

        with pytest.raises(Exception, match="Job.*has failed"):
            program.run()

    def test_cancel_unfinished_job_no_execution_result(self, mocker):
        """Test cancel_unfinished_job when _current_execution_result is None."""
        mock_backend = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_not_called()

    def test_cancel_unfinished_job_no_job_id(self, mocker):
        """Test cancel_unfinished_job when execution result has no job_id."""
        mock_backend = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id=None)

        with pytest.warns(
            UserWarning, match="Cannot cancel job: execution result has no job_id"
        ):
            program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_not_called()

    def test_cancel_unfinished_job_success(self, mocker):
        """Test cancel_unfinished_job successfully cancels job."""
        mock_backend = mocker.Mock()
        mock_backend.cancel_job = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once_with(
            program._current_execution_result
        )

    def test_cancel_unfinished_job_409_conflict(self, mocker):
        """Test cancel_unfinished_job handles 409 Conflict gracefully."""
        mock_backend = mocker.Mock()
        mock_response = mocker.Mock()
        mock_response.status_code = HTTPStatus.CONFLICT
        mock_error = requests.exceptions.HTTPError("409 Conflict")
        mock_error.response = mock_response
        mock_backend.cancel_job = mocker.Mock(side_effect=mock_error)

        program = ConcreteQuantumProgram(backend=mock_backend)
        program.reporter = mocker.Mock()
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        program.reporter.info.assert_called_once_with(
            "Job test_job_123 already completed or cancelled"
        )

    def test_cancel_unfinished_job_other_error(self, mocker):
        """Test cancel_unfinished_job reports other errors."""
        mock_backend = mocker.Mock()
        mock_response = mocker.Mock()
        mock_response.status_code = HTTPStatus.FORBIDDEN
        mock_error = requests.exceptions.HTTPError("403 Forbidden")
        mock_error.response = mock_response
        mock_backend.cancel_job = mocker.Mock(side_effect=mock_error)

        program = ConcreteQuantumProgram(backend=mock_backend)
        program.reporter = mocker.Mock()
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        program.reporter.info.assert_called_once()
        assert "Failed to cancel job test_job_123" in str(
            program.reporter.info.call_args[0][0]
        )

    def test_cancel_unfinished_job_no_reporter(self, mocker):
        """Test cancel_unfinished_job works without reporter."""
        mock_backend = mocker.Mock()
        mock_backend.cancel_job = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        # Should not raise even without reporter
        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once()

    def test_wait_for_qoro_job_completion_cancelled_with_event(self, mocker):
        """Test _wait_for_qoro_job_completion raises _CancelledError when cancelled with event set."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        program = ConcreteQuantumProgram(backend=mock_backend)
        program._cancellation_event = Event()
        program._cancellation_event.set()

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(_CancelledError, match="Job test_job was cancelled"):
            program._wait_for_qoro_job_completion(execution_result)

    def test_wait_for_qoro_job_completion_cancelled_without_event(self, mocker):
        """Test _wait_for_qoro_job_completion raises RuntimeError when cancelled without event."""
        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        program = ConcreteQuantumProgram(backend=mock_backend)
        # No cancellation event set

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(RuntimeError, match="Job test_job was cancelled"):
            program._wait_for_qoro_job_completion(execution_result)

    def test_current_execution_result_lifecycle(self, mocker):
        """Test that _current_execution_result is set and cleared correctly."""
        mock_backend = mocker.Mock()
        mock_backend.is_async = False
        mock_backend.submit_circuits.return_value = ExecutionResult(
            results=[{"label": "circuit_1", "results": {}}]
        )

        program = ConcreteQuantumProgram(backend=mock_backend)
        program._curr_circuits = []

        # Initially None
        assert program._current_execution_result is None

        # After _prepare_and_send_circuits, should be set
        program._prepare_and_send_circuits()
        # Note: _current_execution_result is set in _dispatch_circuits_and_process_results
        # So let's test the full flow
        program._dispatch_circuits_and_process_results()

        # Should be cleared after processing
        assert program._current_execution_result is None

    def test_current_execution_result_cleared_on_exception(self, mocker):
        """Test that _current_execution_result is cleared even when exception occurs."""
        mock_backend = mocker.Mock()
        mock_backend.is_async = True
        mock_backend.submit_circuits.return_value = ExecutionResult(
            results=None, job_id="test_job"
        )
        mock_backend.poll_job_status.side_effect = RuntimeError("Polling failed")
        mock_backend.max_retries = 100

        program = ConcreteQuantumProgram(backend=mock_backend)
        program._curr_circuits = []

        # Should raise exception but clear _current_execution_result
        with pytest.raises(RuntimeError):
            program._dispatch_circuits_and_process_results()

        assert program._current_execution_result is None
