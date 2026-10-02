# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from http import HTTPStatus
from threading import Event

import pytest
import requests

from divi.backends import (
    AsyncJobBackend,
    ExecutionResult,
    JobCancelledError,
    JobStatus,
    JobTimedOutError,
)
from divi.circuits import DEFAULT_PRECISION
from divi.exceptions import ExecutionCancelledError
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline._core import _wait_for_async_result
from divi.pipeline.stages import MeasurementStage
from divi.qprog.quantum_program import QuantumProgram
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    TerminalStatus,
)
from divi.reporting._logging import log_progress_event
from tests.pipeline._helpers import DummySpecStage, two_group_meta


class TerminalPollingBackend:
    """Protocol-shaped backend that polls once before a terminal job error."""

    def __init__(self, error_type, status: JobStatus) -> None:
        self.error_type = error_type
        self.status = status

    @property
    def shots(self) -> int:
        return 100

    def submit_circuits(self, payloads, *, cancellation_event=None, **kwargs):
        del payloads, cancellation_event, kwargs
        return ExecutionResult(job_id="job-terminal")

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        del loop_until_complete, verbose, cancellation_event
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        raise self.error_type(execution_result.job_id)

    def get_job_results(self, execution_result):
        raise AssertionError(f"results requested for terminal job {execution_result}")

    def cancel_job(self, execution_result):
        del execution_result
        return None


class ConcreteQuantumProgram(QuantumProgram):
    """Concrete implementation of QuantumProgram for testing."""

    def __init__(self, backend, seed=None, **kwargs):
        super().__init__(backend, seed, **kwargs)
        self._ran = False

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

    def test_completed_checkpointing_is_unsupported_by_default(
        self, dummy_simulator, tmp_path
    ):
        program = ConcreteQuantumProgram(dummy_simulator)

        assert program._make_checkpoint(tmp_path) is None
        assert program._restore_checkpoint("{}", tmp_path) is False
        assert program.has_results() is False

    def test_quantum_program_uses_logging_emitter_by_default(
        self, default_test_simulator
    ):
        program = ConcreteQuantumProgram(backend=default_test_simulator)

        assert program._progress_emitter is log_progress_event

    def test_initialization_preserves_backend_and_seed(self, dummy_simulator):
        program = ConcreteQuantumProgram(backend=dummy_simulator, seed=42)

        assert program.backend == dummy_simulator
        assert program._seed == 42

    def test_bound_progress_emitter_is_restored_on_normal_exit(self, dummy_simulator):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        previous = program._progress_emitter
        events = []
        emitter = events.append

        with program._bind_progress_emitter(emitter):
            assert program._progress_emitter is emitter

        assert program._progress_emitter is previous

    def test_bound_progress_emitter_is_restored_on_exception(self, dummy_simulator):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        previous = program._progress_emitter
        events = []
        emitter = events.append

        with pytest.raises(RuntimeError, match="operation failed"):
            with program._bind_progress_emitter(emitter):
                raise RuntimeError("operation failed")

        assert program._progress_emitter is previous

    def test_ensure_progress_session_preserves_prebound_emitter(self, dummy_simulator):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        events = []
        emitter = events.append

        with program._bind_progress_emitter(emitter):
            with program._ensure_progress_session(label="Optimising", total=3):
                assert program._progress_emitter is emitter

        assert events == []

    def test_ensure_progress_session_uses_direct_session_and_restores_default(
        self, dummy_simulator, caplog
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        default_emitter = program._progress_emitter

        with caplog.at_level(logging.INFO, logger="divi"):
            with program._ensure_progress_session(label="Optimising", total=3):
                assert program._progress_emitter is not default_emitter

        assert program._progress_emitter is default_emitter
        assert [record.getMessage() for record in caplog.records] == [
            "Optimising: 0/3",
            "Optimising: 3/3 • Success! ✅",
        ]

    def test_env_var_suppresses_standalone_progress(
        self, dummy_simulator, caplog, monkeypatch
    ):
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        default_emitter = program._progress_emitter

        with caplog.at_level(logging.INFO, logger="divi"):
            with program._ensure_progress_session(label="Optimising", total=3):
                program._progress_emitter(
                    ProgressEvent.advance(program._progress_key, amount=1)
                )

        assert program._progress_emitter is default_emitter
        assert caplog.records == []

    def test_ensure_progress_session_adds_exactly_one_success_terminal(
        self, dummy_simulator, recording_direct_sessions
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        with program._ensure_progress_session(label="Optimising", total=3):
            pass

        session = recording_direct_sessions[0]
        terminal_events = [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ]
        assert terminal_events == [
            ProgressEvent.finish(program._progress_key, TerminalStatus.SUCCESS)
        ]
        target = session.state.get(program._progress_key)
        assert target.completed == 3
        assert target.terminal_status is TerminalStatus.SUCCESS

    def test_ensure_progress_session_adds_exactly_one_failure_terminal_and_reraises(
        self, dummy_simulator, recording_direct_sessions
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        with pytest.raises(RuntimeError, match="operation failed"):
            with program._ensure_progress_session(label="Optimising", total=3):
                raise RuntimeError("operation failed")

        session = recording_direct_sessions[0]
        terminal_events = [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ]
        assert terminal_events == [
            ProgressEvent.finish(
                program._progress_key,
                TerminalStatus.FAILED,
                detail="RuntimeError: operation failed",
            )
        ]
        assert session.state.get(program._progress_key).terminal_status is (
            TerminalStatus.FAILED
        )

    def test_ensure_progress_session_preserves_an_explicit_terminal_event(
        self, dummy_simulator, recording_direct_sessions
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        explicit = ProgressEvent.finish(
            program._progress_key,
            TerminalStatus.CANCELLED,
            detail="producer cancelled",
        )

        with program._ensure_progress_session(label="Optimising", total=3):
            program._progress_emitter(explicit)

        session = recording_direct_sessions[0]
        assert [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ] == [explicit]
        target = session.state.get(program._progress_key)
        assert target.terminal_status is TerminalStatus.CANCELLED
        assert target.detail == "producer cancelled"

    @pytest.mark.parametrize(
        ("exception", "terminal_status"),
        [
            (
                ExecutionCancelledError("cancelled while polling"),
                TerminalStatus.CANCELLED,
            ),
            (KeyboardInterrupt(), TerminalStatus.ABORTED),
        ],
    )
    def test_ensure_progress_session_maps_control_flow_exceptions_to_terminal_status(
        self,
        exception,
        terminal_status,
        dummy_simulator,
        recording_direct_sessions,
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        with pytest.raises(type(exception)):
            with program._ensure_progress_session(label="Executing", total=None):
                raise exception

        session = recording_direct_sessions[0]
        terminal_events = [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0].terminal_status is terminal_status

    @pytest.mark.parametrize(
        ("error_type", "job_status", "terminal_status"),
        [
            (JobTimedOutError, JobStatus.TIMED_OUT, TerminalStatus.FAILED),
            (JobCancelledError, JobStatus.CANCELLED, TerminalStatus.CANCELLED),
        ],
    )
    def test_direct_pipeline_failure_preserves_terminal_job_status(
        self,
        error_type,
        job_status,
        terminal_status,
        dummy_simulator,
        recording_direct_sessions,
    ):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        backend = TerminalPollingBackend(error_type, job_status)
        assert isinstance(backend, AsyncJobBackend)

        with pytest.raises(error_type):
            with program._ensure_progress_session(label="Executing", total=None):
                env = PipelineEnv(backend=backend)
                env._bind_progress(
                    program._progress_emitter,
                    program._progress_key,
                )
                _wait_for_async_result(
                    backend, ExecutionResult(job_id="job-terminal"), env
                )

        session = recording_direct_sessions[0]
        terminal_events = [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0].terminal_status is terminal_status
        assert terminal_events[0].job_status is job_status
        assert session.state.get(program._progress_key).job_status is job_status

    def test_initialization_with_unexpected_kwargs_raises(
        self, mocker, dummy_simulator
    ):
        """Unexpected constructor kwargs should fail fast."""

        with pytest.raises(
            TypeError,
            match="Unexpected keyword argument\\(s\\): another_param, custom_param",
        ):
            ConcreteQuantumProgram(
                backend=dummy_simulator, custom_param="test_value", another_param=123
            )

    def test_preprocessors_default_empty(self, dummy_simulator):
        """The base exposes no measurement routines until a subclass declares them."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        assert program._preprocessors() == ()

    def test_cancellation_event(self, mocker, dummy_simulator):
        """Test _set_cancellation_event method."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        event = Event()
        program._set_cancellation_event(event)

        assert hasattr(program, "_cancellation_event")
        assert program._cancellation_event == event

    def test_total_circuit_count_property(self, mocker, dummy_simulator):
        """Test total_circuit_count property."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        program._total_circuit_count = 15
        assert program.total_circuit_count == 15

    def test_total_run_time_property(self, mocker, dummy_simulator):
        """Test total_run_time property."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        program._total_run_time = 3.7
        assert program.total_run_time == 3.7

    def test_properties_default_to_zero(self, mocker, dummy_simulator):
        """Test that properties default to zero when not set."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        assert program.total_circuit_count == 0
        assert program.total_run_time == 0.0

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            pytest.param({}, DEFAULT_PRECISION, id="defaults_to_module_constant"),
            pytest.param({"precision": 5}, 5, id="explicit_value_verbatim"),
        ],
    )
    def test_precision_property(self, dummy_simulator, kwargs, expected):
        """``QuantumProgram.precision`` defaults to ``DEFAULT_PRECISION``; an
        explicit ``precision=`` kwarg is exposed verbatim."""
        program = ConcreteQuantumProgram(backend=dummy_simulator, **kwargs)
        assert program.precision == expected
        assert program._precision == expected


class TestQuantumProgramJobManagement:
    """Tests for QuantumProgram job management and cancellation."""

    def test_cancel_unfinished_job_no_execution_result(self, mocker, dummy_simulator):
        """Test cancel_unfinished_job when _current_execution_result is None."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)

        with pytest.warns(
            UserWarning, match="Cannot cancel job: no current execution result"
        ):
            program.cancel_unfinished_job()

    def test_cancel_unfinished_job_no_job_id(self, mocker, dummy_simulator):
        """Test cancel_unfinished_job when execution result has no job_id."""
        program = ConcreteQuantumProgram(backend=dummy_simulator)
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

    def test_cancel_unfinished_job_targets_the_last_submitted_job(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        backend.supports_expval = False
        backend.resolves_parameters = False
        backend.shots = 100
        backend.max_retries = 1
        backend.submit_circuits.return_value = ExecutionResult(job_id="job_7")
        backend.poll_job_status.return_value = JobStatus.COMPLETED
        backend.get_job_results.return_value = ExecutionResult(results=[])
        program = ConcreteQuantumProgram(backend=backend)
        pipeline = CircuitPipeline(
            stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
        )

        program._execute(pipeline, "x")
        program.cancel_unfinished_job()

        backend.cancel_job.assert_called_once_with(backend.submit_circuits.return_value)

    @pytest.mark.parametrize("status_code", [HTTPStatus.CONFLICT, HTTPStatus.FORBIDDEN])
    def test_cancel_unfinished_job_http_error_is_silently_swallowed(
        self, mocker, status_code
    ):
        """An HTTP error while cancelling neither propagates nor emits progress."""
        mock_backend = mocker.Mock(spec=AsyncJobBackend)
        mock_response = mocker.Mock()
        mock_response.status_code = status_code
        mock_error = requests.exceptions.HTTPError(f"{status_code.value} error")
        mock_error.response = mock_response
        mock_backend.cancel_job = mocker.Mock(side_effect=mock_error)

        program = ConcreteQuantumProgram(backend=mock_backend)
        events = []
        program._progress_emitter = events.append
        program._current_execution_result = ExecutionResult(job_id="test_job_123")

        program.cancel_unfinished_job()

        mock_backend.cancel_job.assert_called_once()
        assert events == []
