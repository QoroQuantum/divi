# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
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

    def _build_pipelines(self) -> None:
        pass

    def run(self) -> tuple[int, float]:
        """Concrete implementation of run method."""
        return (5, 1.5)


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
        """Test that run() must be implemented in subclasses."""
        mock_backend = mocker.Mock()

        # Test missing abstract methods (run and _build_pipelines)
        class IncompleteProgram(QuantumProgram):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram(backend=mock_backend)

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
        program = ConcreteQuantumProgram(mock_backend)
        assert program.total_circuit_count == 0
        assert program.total_run_time == 0.0


class TestQuantumProgramJobManagement:
    """Tests for QuantumProgram job management and cancellation."""

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

    def test_wait_for_async_result_cancelled_with_event(self, mocker):
        """Test _wait_for_async_result raises _CancelledError when cancelled with event set."""
        from divi.pipeline._core import _wait_for_async_result
        from divi.pipeline.abc import PipelineEnv

        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        cancel_event = Event()
        cancel_event.set()
        env = PipelineEnv(backend=mock_backend, cancellation_event=cancel_event)

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(_CancelledError, match="Job test_job was cancelled"):
            _wait_for_async_result(mock_backend, execution_result, env)

    def test_wait_for_async_result_cancelled_without_event(self, mocker):
        """Test _wait_for_async_result raises RuntimeError when cancelled without event."""
        from divi.pipeline._core import _wait_for_async_result
        from divi.pipeline.abc import PipelineEnv

        mock_backend = mocker.Mock()
        mock_backend.poll_job_status.return_value = JobStatus.CANCELLED
        mock_backend.max_retries = 100

        env = PipelineEnv(backend=mock_backend)
        # No cancellation event set

        execution_result = ExecutionResult(job_id="test_job")

        with pytest.raises(RuntimeError, match="Job test_job was cancelled"):
            _wait_for_async_result(mock_backend, execution_result, env)
