# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from threading import Event

import numpy as np
import pytest

from divi.backends import (
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
from tests._helpers import FakeAsyncBackend, exact_match, raise_after_running
from tests.pipeline._helpers import DummySpecStage, two_group_meta


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
        backend = FakeAsyncBackend(poll=raise_after_running(error_type))

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
        assert backend.fetched == []

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

    def test_qem_protocol_that_is_not_a_protocol_is_rejected(self, dummy_simulator):
        """Otherwise it fails at dispatch, on the first attribute the pipeline
        reads from it."""
        with pytest.raises(
            TypeError, match=exact_match("qem_protocol must be a QEMProtocol; got str.")
        ):
            ConcreteQuantumProgram(backend=dummy_simulator, qem_protocol="zne")

    @pytest.mark.parametrize("precision", [0, -1, 1.5, "8", True, np.True_])
    def test_precision_that_is_not_a_positive_integer_is_rejected(
        self, dummy_simulator, precision
    ):
        """It sets a format-spec width, so a bad value would otherwise surface
        only when the first circuit is rendered."""
        message = f"precision must be an integer of at least 1; got {precision!r}."
        with pytest.raises(ValueError, match=exact_match(message)):
            ConcreteQuantumProgram(backend=dummy_simulator, precision=precision)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
    def test_numpy_precision_is_stored_as_a_plain_int(self, dummy_simulator, dtype):
        program = ConcreteQuantumProgram(backend=dummy_simulator, precision=dtype(6))

        assert program.precision == 6
        assert type(program._precision) is int

    @pytest.mark.parametrize(
        "seed", [np.uint64(2**63 + 5), np.int64(2**60 + 1), 2**64 + 3]
    )
    def test_seed_keeps_every_digit_of_an_integer_seed(self, dummy_simulator, seed):
        program = ConcreteQuantumProgram(backend=dummy_simulator, seed=seed)

        assert program._seed == int(seed)
        assert type(program._seed) is int
        assert program._base_seed == int(seed)

    @pytest.mark.parametrize("seed", [1.5, "7", True, np.True_, np.float64(3.0)])
    def test_seed_that_is_not_an_integer_is_rejected(self, dummy_simulator, seed):
        message = f"seed must be an integer or None; got {seed!r}."
        with pytest.raises(ValueError, match=exact_match(message)):
            ConcreteQuantumProgram(backend=dummy_simulator, seed=seed)

    def test_unseeded_base_seed_is_one_stable_int(self, dummy_simulator):
        program = ConcreteQuantumProgram(backend=dummy_simulator)
        base_seed = program._base_seed

        assert type(base_seed) is int
        assert 0 <= base_seed < 2**63
        assert program._build_pipeline_env().base_seed == base_seed


def _two_group_pipeline() -> CircuitPipeline:
    return CircuitPipeline(
        stages=[DummySpecStage(meta=two_group_meta()), MeasurementStage()]
    )


def test_execute_totals_start_at_zero_and_accumulate(dummy_simulator, mocker):
    program = ConcreteQuantumProgram(backend=dummy_simulator)
    pipeline = _two_group_pipeline()

    def fake_run(initial_spec, env):
        env.artifacts.update(
            circuit_count=3,
            device_shots=300,
            backend_jobs=2,
            run_time=0.5,
        )
        return {}

    mocker.patch.object(pipeline, "run", side_effect=fake_run)

    def totals():
        return (
            program.total_circuit_count,
            program.total_device_shots,
            program.total_backend_jobs,
            program.total_run_time,
        )

    assert totals() == (0, 0, 0, 0.0)
    program._execute(pipeline, "x")
    program._execute(pipeline, "x")

    assert totals() == (6, 600, 4, 1.0)


def test_pipeline_stage_events_carry_the_program_key(dummy_simulator):
    program = ConcreteQuantumProgram(backend=dummy_simulator)
    program._progress_key = "program-key"
    events = []
    program._progress_emitter = events.append

    program._execute(_two_group_pipeline(), "x")

    stage_events = [
        event
        for event in events
        if event.message and event.message.startswith("Pipeline:")
    ]
    assert stage_events
    assert {event.progress_key for event in stage_events} == {"program-key"}
