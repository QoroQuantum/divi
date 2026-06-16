# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from queue import Queue
from threading import Event

import pytest
import requests

from divi.backends import AsyncJobBackend, ExecutionResult
from divi.circuits import DEFAULT_PRECISION
from divi.pipeline import PipelineSet
from divi.qprog.quantum_program import QuantumProgram
from tests.conftest import DummySimulator


class ConcreteQuantumProgram(QuantumProgram):
    """Concrete implementation of QuantumProgram for testing."""

    def __init__(self, backend, seed=None, progress_queue=None, **kwargs):
        super().__init__(backend, seed, progress_queue, **kwargs)
        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._ran = False

    def _build_pipelines(self) -> PipelineSet:
        return PipelineSet({})

    def has_results(self) -> bool:
        return self._ran

    def run(self):
        """Concrete implementation of run method."""
        self._total_circuit_count = 5
        self._total_run_time = 1.5
        self._ran = True
        return self


class TestQuantumProgramBase:
    """Tests for QuantumProgram abstract base class contract and core functionality."""

    def test_initialization_with_all_params(self, mocker):
        """Test QuantumProgram initialization with all parameters."""
        mock_backend = DummySimulator(shots=1)
        mock_queue = Queue()

        program = ConcreteQuantumProgram(
            backend=mock_backend, seed=42, progress_queue=mock_queue
        )

        assert program.backend == mock_backend
        assert program._seed == 42
        assert program._progress_queue == mock_queue

    def test_initialization_with_program_id(self, mocker):
        """Test QuantumProgram initialization with explicit program_id."""
        mock_backend = DummySimulator(shots=1)

        program = ConcreteQuantumProgram(
            backend=mock_backend, program_id="test_program"
        )

        assert program.backend == mock_backend
        assert program._seed is None
        assert program._progress_queue is None
        assert program.program_id == "test_program"

    def test_initialization_with_unexpected_kwargs_raises(self, mocker):
        """Unexpected constructor kwargs should fail fast."""
        mock_backend = DummySimulator(shots=1)

        with pytest.raises(
            TypeError,
            match="Unexpected keyword argument\\(s\\): another_param, custom_param",
        ):
            ConcreteQuantumProgram(
                backend=mock_backend, custom_param="test_value", another_param=123
            )

    def test_abstract_class_behavior(self, mocker):
        """Test abstract class instantiation behavior."""
        mock_backend = DummySimulator(shots=1)

        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantumProgram(backend=mock_backend)

        # Test that concrete implementations can be instantiated
        program = ConcreteQuantumProgram(backend=mock_backend)
        assert isinstance(program, QuantumProgram)
        assert program.backend == mock_backend

    def test_abstract_methods_must_be_implemented(self, mocker):
        """Test that abstract methods must be implemented in subclasses."""
        mock_backend = DummySimulator(shots=1)

        # Test missing abstract methods (run and _build_pipelines)
        class IncompleteProgram(QuantumProgram):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProgram(backend=mock_backend)

    def test_build_pipelines_returns_pipeline_set(self):
        program = ConcreteQuantumProgram(backend=DummySimulator(shots=1))

        assert isinstance(program._build_pipelines(), PipelineSet)

    def test_cancellation_event(self, mocker):
        """Test _set_cancellation_event method."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)

        event = Event()
        program._set_cancellation_event(event)

        assert hasattr(program, "_cancellation_event")
        assert program._cancellation_event == event

    def test_total_circuit_count_property(self, mocker):
        """Test total_circuit_count property."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)

        program._total_circuit_count = 15
        assert program.total_circuit_count == 15

    def test_total_run_time_property(self, mocker):
        """Test total_run_time property."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)

        program._total_run_time = 3.7
        assert program.total_run_time == 3.7

    def test_properties_default_to_zero(self, mocker):
        """Test that properties default to zero when not set."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(mock_backend)
        assert program.total_circuit_count == 0
        assert program.total_run_time == 0.0

    def test_precision_property_defaults_to_module_constant(self, mocker):
        """``QuantumProgram.precision`` defaults to ``DEFAULT_PRECISION``."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)
        assert program.precision == DEFAULT_PRECISION
        assert program._precision == DEFAULT_PRECISION

    def test_precision_property_reflects_explicit_value(self, mocker):
        """Explicit ``precision=`` kwarg is exposed verbatim."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend, precision=5)
        assert program.precision == 5
        assert program._precision == 5


class TestQuantumProgramJobManagement:
    """Tests for QuantumProgram job management and cancellation."""

    def test_cancel_unfinished_job_no_execution_result(self, mocker):
        """Test cancel_unfinished_job when _current_execution_result is None."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program.cancel_unfinished_job()

    def test_cancel_unfinished_job_no_job_id(self, mocker):
        """Test cancel_unfinished_job when execution result has no job_id."""
        mock_backend = DummySimulator(shots=1)
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id=None)

        with pytest.warns(
            UserWarning, match="Cannot cancel job: execution result has no job_id"
        ):
            program.cancel_unfinished_job()

    def test_cancel_unfinished_job_success(self, mocker):
        """Test cancel_unfinished_job successfully cancels job."""
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_backend.cancel_job = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once_with(
            program._current_execution_result
        )

    def test_cancel_unfinished_job_409_conflict_is_silently_swallowed(self, mocker):
        """409 from the scheduler means the job already reached a terminal
        state — a normal race outcome of CTRL-C arriving as the job finishes.
        The error must not propagate and must not spam the user-facing
        reporter; it is logged at DEBUG for developers only."""
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_response = mocker.Mock()
        mock_response.status_code = HTTPStatus.CONFLICT
        mock_error = requests.exceptions.HTTPError("409 Conflict")
        mock_error.response = mock_response
        mock_backend.cancel_job = mocker.Mock(side_effect=mock_error)

        program = ConcreteQuantumProgram(backend=mock_backend)
        program.reporter = mocker.Mock()
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once()
        program.reporter.info.assert_not_called()

    def test_cancel_unfinished_job_other_error_is_silently_swallowed(self, mocker):
        """Non-409 HTTP errors (403, 404, network) during cleanup are
        diagnostic-only — they belong in ``logger.debug``, not on the
        user-facing reporter that's currently displaying cancellation
        status."""
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_response = mocker.Mock()
        mock_response.status_code = HTTPStatus.FORBIDDEN
        mock_error = requests.exceptions.HTTPError("403 Forbidden")
        mock_error.response = mock_response
        mock_backend.cancel_job = mocker.Mock(side_effect=mock_error)

        program = ConcreteQuantumProgram(backend=mock_backend)
        program.reporter = mocker.Mock()
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once()
        program.reporter.info.assert_not_called()

    def test_cancel_unfinished_job_no_reporter(self, mocker):
        """Test cancel_unfinished_job works without reporter."""
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_backend.cancel_job = mocker.Mock()
        program = ConcreteQuantumProgram(backend=mock_backend)
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        # Should not raise even without reporter
        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once()
