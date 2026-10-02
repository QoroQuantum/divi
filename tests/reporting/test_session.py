# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural contracts for direct and queued progress sessions."""

import logging
from collections.abc import Callable
from io import StringIO
from queue import Queue
from threading import Event, Thread

import pytest
from rich.console import Console

from divi.reporting import _session as session_module
from divi.reporting._events import (
    EventKind,
    ProgressEvent,
    ProgressScope,
    TerminalStatus,
)
from divi.reporting._logging import (
    diagnose_reporting_failure,
    log_progress_event,
    log_progress_state,
)
from divi.reporting._session import ProgressSession
from divi.reporting._state import ProgressState

SessionFactory = Callable[[], ProgressSession]


def non_interactive_console() -> Console:
    """Return a console whose live-rendering capabilities are disabled."""
    return Console(file=StringIO(), force_terminal=False, force_jupyter=False)


def terminal_console() -> Console:
    """Return a console that reports terminal rendering support."""
    return Console(file=StringIO(), force_terminal=True, force_jupyter=False)


def make_direct() -> ProgressSession:
    """Create a direct session without a live Rich display."""
    return ProgressSession.direct(ProgressState(), console=non_interactive_console())


def make_queued() -> ProgressSession:
    """Create a queued session without a live Rich display."""
    return ProgressSession.queued(ProgressState(), console=non_interactive_console())


class ListenerThreadDouble:
    """Thread double that records start and join without executing its target."""

    instance: "ListenerThreadDouble | None" = None
    alive_at_start = True
    alive_after_join = True

    def __init__(self, *, target, args, name: str, daemon: bool) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.join_timeout: float | None = None
        self.started = False
        self.alive = self.alive_at_start
        type(self).instance = self

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout
        self.alive = self.alive_after_join

    def is_alive(self) -> bool:
        return self.alive


class StuckListenerThread(ListenerThreadDouble):
    """Listener double that never terminates."""


class DrainThenJoinListener(ListenerThreadDouble):
    """Listener double that terminates when joined."""

    alive_after_join = False


class DeadListenerThread(DrainThenJoinListener):
    """Listener double that has already terminated."""

    alive_at_start = False


def blocking_renderer() -> tuple[Callable[..., None], Event, Event]:
    """Return a renderer that blocks until released, and its (started, release)."""
    render_started = Event()
    release_render = Event()

    def render(state, affected) -> None:
        del state, affected
        render_started.set()
        release_render.wait(timeout=2.0)

    return render, render_started, release_render


def close_in_thread(session: ProgressSession) -> tuple[Thread, Event, list]:
    """Close ``session`` on a daemon thread; return it, its finish flag and errors."""
    close_finished = Event()
    close_failures: list[BaseException] = []

    def close_session() -> None:
        try:
            session.close()
        except BaseException as exc:
            close_failures.append(exc)
        finally:
            close_finished.set()

    closer = Thread(target=close_session, daemon=True)
    closer.start()
    return closer, close_finished, close_failures


class DrainCompletionEvent:
    """Acknowledge queued work when shutdown waits for drain progress."""

    def __init__(self, event_queue: Queue[ProgressEvent]) -> None:
        self._event = Event()
        self._event_queue = event_queue

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        if self._event_queue.unfinished_tasks > 0:
            self._event_queue.get_nowait()
            self._event_queue.task_done()
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class BlockingAdvanceQueue(Queue[ProgressEvent]):
    """Queue that pauses one advance before it becomes visible to consumers."""

    def __init__(self) -> None:
        super().__init__()
        self.block_advances = False
        self.put_started = Event()
        self.release_put = Event()

    def put(
        self,
        item: ProgressEvent,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        if self.block_advances and item.kind is EventKind.ADVANCE:
            self.put_started.set()
            if not self.release_put.wait(timeout=2.0):
                raise TimeoutError("test did not release the blocked queue put")
        super().put(item, block=block, timeout=timeout)


@pytest.mark.parametrize("session_factory", [make_direct, make_queued])
def test_session_applies_the_same_event_sequence(session_factory: SessionFactory):
    with session_factory() as session:
        session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 2))
        session.emit(ProgressEvent.advance("p"))
        session.emit(ProgressEvent.finish("p", TerminalStatus.SUCCESS))

    target = session.state.get("p")
    assert target.completed == 2
    assert target.terminal_status is TerminalStatus.SUCCESS


def test_queued_session_acknowledges_each_dequeued_event_once(monkeypatch, mocker):
    event_queue: Queue[ProgressEvent] = Queue()
    task_done = mocker.spy(event_queue, "task_done")
    monkeypatch.setattr(session_module, "Queue", lambda: event_queue)

    with make_queued() as session:
        session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
        session.emit(ProgressEvent.finish("p", TerminalStatus.SUCCESS))

    assert task_done.call_count == 2
    assert event_queue.unfinished_tasks == 0


@pytest.mark.parametrize(
    ("mode", "factory_name"),
    [("direct", "make_standalone_view"), ("queued", "make_ensemble_view")],
)
def test_session_closes_its_selected_view_exactly_once(
    mode, factory_name, monkeypatch, mocker
):
    render = mocker.stub(name="render")
    close_view = mocker.stub(name="close_view")
    monkeypatch.setattr(
        session_module,
        factory_name,
        lambda *args, **kwargs: (render, close_view),
    )

    if mode == "direct":
        session = ProgressSession.direct(ProgressState(), console=terminal_console())
    else:
        session = ProgressSession.queued(ProgressState(), console=terminal_console())

    with session:
        session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
    session.close()

    close_view.assert_called_once_with()


@pytest.mark.parametrize("session_factory", [make_direct, make_queued])
def test_events_emitted_after_close_are_ignored(session_factory: SessionFactory):
    session = session_factory()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 2))
    session.close()

    session.emit(ProgressEvent.advance("p"))

    assert session.state.get("p").completed == 0


def test_close_can_retry_after_interrupted_listener_shutdown(monkeypatch):
    shutdown_attempts = 0
    view_closed = []

    def interrupt_once() -> None:
        nonlocal shutdown_attempts
        shutdown_attempts += 1
        if shutdown_attempts == 1:
            raise KeyboardInterrupt

    session = ProgressSession(
        ProgressState(),
        lambda state, affected: None,
        lambda: view_closed.append(True),
        console=non_interactive_console(),
    )
    monkeypatch.setattr(session, "_shutdown_listener", interrupt_once)

    with pytest.raises(KeyboardInterrupt):
        session.close()
    session.close()

    assert shutdown_attempts == 2
    assert view_closed == [True]


def test_non_interactive_sessions_select_logging_renderer_without_closer(
    monkeypatch, mocker
):
    render = mocker.stub(name="render")
    monkeypatch.setattr(
        session_module,
        "log_progress_state",
        render,
    )
    make_standalone = mocker.patch.object(session_module, "make_standalone_view")
    make_ensemble = mocker.patch.object(session_module, "make_ensemble_view")

    direct = ProgressSession.direct(ProgressState(), console=non_interactive_console())
    queued = ProgressSession.queued(ProgressState(), console=non_interactive_console())
    direct.close()
    queued.close()

    assert direct._render is render
    assert queued._render is render
    assert direct._close_view is None
    assert queued._close_view is None
    make_standalone.assert_not_called()
    make_ensemble.assert_not_called()


def test_terminal_and_jupyter_sessions_select_the_corresponding_rich_views(
    monkeypatch, mocker
):
    render = mocker.stub(name="render")
    close_view = mocker.stub(name="close_view")
    make_standalone = mocker.patch.object(
        session_module, "make_standalone_view", return_value=(render, close_view)
    )
    make_ensemble = mocker.patch.object(
        session_module, "make_ensemble_view", return_value=(render, close_view)
    )
    logging_renderer = mocker.patch.object(session_module, "log_progress_state")
    console = terminal_console()
    jupyter_console = non_interactive_console()

    direct = ProgressSession.direct(ProgressState(), console=console)
    queued = ProgressSession.queued(
        ProgressState(), console=jupyter_console, is_jupyter=True
    )
    direct.close()
    queued.close()

    make_standalone.assert_called_once_with(console)
    make_ensemble.assert_called_once_with(jupyter_console, is_jupyter=True)
    logging_renderer.assert_not_called()


@pytest.mark.parametrize("session_factory", [make_direct, make_queued])
@pytest.mark.parametrize(
    ("level", "include_traceback"), [(logging.INFO, False), (logging.DEBUG, True)]
)
def test_processing_failure_uses_debug_level_to_select_traceback(
    session_factory: SessionFactory, level, include_traceback, caplog, mocker
):
    caplog.set_level(level, logger="divi")
    diagnose = mocker.patch.object(session_module, "diagnose_reporting_failure")

    with session_factory() as session:
        session.emit(ProgressEvent.advance("unknown"))

    diagnosed_exception = diagnose.call_args.args[0]
    assert isinstance(diagnosed_exception, KeyError)
    assert diagnose.call_args.kwargs == {"include_traceback": include_traceback}


@pytest.mark.parametrize("debug_enabled", [False, True])
def test_diagnostic_includes_traceback_only_under_debug(debug_enabled, capsys):
    try:
        raise RuntimeError("renderer unavailable")
    except RuntimeError as exc:
        diagnose_reporting_failure(exc, include_traceback=debug_enabled)

    stderr = capsys.readouterr().err
    assert stderr.count("Progress reporting failed: renderer unavailable") == 1
    traceback_start = stderr.find("Traceback (most recent call last)")
    assert (traceback_start >= 0) is debug_enabled
    if debug_enabled:
        assert "RuntimeError: renderer unavailable" in stderr[traceback_start:]


@pytest.mark.parametrize("value", ["TRUE", " yes ", "On"])
def test_environment_flag_disables_progress_case_and_space_insensitively(
    monkeypatch, value
):
    monkeypatch.setenv("DIVI_DISABLE_PROGRESS", value)

    assert session_module._environment_disables_progress() is True


@pytest.mark.parametrize(
    ("factory", "console"),
    [
        (ProgressSession.direct, non_interactive_console()),
        (ProgressSession.queued, non_interactive_console()),
    ],
    ids=["direct", "queued"],
)
def test_session_exposes_the_console_it_was_given(factory, console):
    with factory(ProgressState(), console=console) as session:
        assert session.console is console


@pytest.mark.parametrize(
    "factory",
    [ProgressSession.direct, ProgressSession.queued],
    ids=["direct", "queued"],
)
def test_session_creates_a_console_when_none_is_given(factory):
    with factory(ProgressState()) as session:
        assert isinstance(session.console, Console)


def test_queued_close_uses_bounded_join_and_diagnoses_a_stuck_listener(
    monkeypatch, mocker, caplog
):
    caplog.set_level(logging.DEBUG, logger="divi")
    StuckListenerThread.instance = None
    monkeypatch.setattr(session_module, "Thread", StuckListenerThread)
    diagnose = mocker.patch.object(session_module, "diagnose_reporting_failure")

    session = make_queued()
    session.close()

    listener = StuckListenerThread.instance
    assert listener is not None
    assert listener.started
    assert listener.daemon
    assert listener.join_timeout is not None
    assert 0 <= listener.join_timeout <= session_module._LISTENER_JOIN_TIMEOUT
    diagnosed_exception = diagnose.call_args.args[0]
    assert isinstance(diagnosed_exception, RuntimeError)
    assert "did not terminate" in str(diagnosed_exception)
    assert diagnose.call_args.kwargs == {"include_traceback": False}


def test_queued_close_diagnoses_a_listener_that_died_with_pending_events(
    monkeypatch, mocker, caplog
):
    caplog.set_level(logging.DEBUG, logger="divi")
    monkeypatch.setattr(session_module, "Thread", DeadListenerThread)
    diagnose = mocker.patch.object(session_module, "diagnose_reporting_failure")

    session = make_queued()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
    session.close()

    diagnose.assert_called_once()
    diagnosed_exception = diagnose.call_args.args[0]
    assert isinstance(diagnosed_exception, RuntimeError)
    assert str(diagnosed_exception) == (
        "Progress listener terminated before acknowledging all events"
    )
    assert diagnose.call_args.kwargs == {"include_traceback": False}


def test_close_diagnoses_a_failing_view_closer(mocker):
    failure = RuntimeError("view teardown failed")
    close_view = mocker.Mock(side_effect=failure)
    diagnose = mocker.patch.object(session_module, "diagnose_reporting_failure")
    session = ProgressSession(
        ProgressState(),
        lambda state, affected: None,
        close_view,
        console=non_interactive_console(),
    )

    session.close()

    diagnose.assert_called_once()
    assert diagnose.call_args.args[0] is failure


def test_queued_close_grants_join_a_fresh_budget_after_near_complete_drain(
    monkeypatch,
):
    event_queue: Queue[ProgressEvent] = Queue()
    done_event = DrainCompletionEvent(event_queue)
    DrainThenJoinListener.instance = None
    clock_values = iter([0.0, 0.19, 0.19])
    monkeypatch.setattr(session_module, "Queue", lambda: event_queue)
    monkeypatch.setattr(session_module, "Event", lambda: done_event)
    monkeypatch.setattr(session_module, "Thread", DrainThenJoinListener)
    monkeypatch.setattr(session_module, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(session_module, "_LISTENER_DRAIN_TIMEOUT", 0.2)
    monkeypatch.setattr(session_module, "_LISTENER_JOIN_TIMEOUT", 0.2)

    session = make_queued()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
    session.close()

    listener = DrainThenJoinListener.instance
    assert listener is not None
    assert event_queue.unfinished_tasks == 0
    assert listener.join_timeout == 0.2


def test_queued_close_is_bounded_when_render_blocks_after_dequeue(monkeypatch, mocker):
    render, render_started, release_render = blocking_renderer()
    monkeypatch.setattr(session_module, "log_progress_state", render)
    monkeypatch.setattr(session_module, "_LISTENER_DRAIN_TIMEOUT", 0.05)
    monkeypatch.setattr(session_module, "_LISTENER_JOIN_TIMEOUT", 0.05)
    diagnose = mocker.patch.object(session_module, "diagnose_reporting_failure")
    session = make_queued()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
    assert render_started.wait(timeout=1.0)

    closer, close_finished, close_failures = close_in_thread(session)
    try:
        assert close_finished.wait(timeout=0.5)
    finally:
        release_render.set()
        closer.join(timeout=1.0)
        if session._listener_thread is not None:
            session._listener_thread.join(timeout=1.0)

    assert close_failures == []
    diagnosed_exception = diagnose.call_args.args[0]
    assert isinstance(diagnosed_exception, RuntimeError)
    assert "did not terminate" in str(diagnosed_exception)


def test_queued_listener_drains_every_event_queued_before_shutdown_signal(
    monkeypatch,
):
    event_queue: Queue[ProgressEvent] = Queue()
    render, render_started, release_render = blocking_renderer()
    monkeypatch.setattr(session_module, "Queue", lambda: event_queue)
    monkeypatch.setattr(session_module, "log_progress_state", render)
    monkeypatch.setattr(session_module, "_LISTENER_DRAIN_TIMEOUT", 0.05)
    session = make_queued()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 2))
    session.emit(ProgressEvent.advance("p"))
    assert render_started.wait(timeout=1.0)

    closer, close_finished, close_failures = close_in_thread(session)
    done_event = session._done_event
    assert done_event is not None
    assert done_event.wait(timeout=1.0)
    release_render.set()
    closer.join(timeout=2.0)

    assert close_finished.is_set()
    assert close_failures == []
    assert event_queue.unfinished_tasks == 0
    assert session.state.get("p").completed == 1


def test_queued_close_linearizes_with_an_emit_blocked_before_queue_acceptance(
    monkeypatch,
):
    event_queue = BlockingAdvanceQueue()
    monkeypatch.setattr(session_module, "Queue", lambda: event_queue)
    session = make_queued()
    session.emit(ProgressEvent.register("p", ProgressScope.PROGRAM, "P", 1))
    event_queue.join()
    event_queue.block_advances = True

    emitter = Thread(
        target=session.emit,
        args=(ProgressEvent.advance("p"),),
        daemon=True,
    )
    emitter.start()
    assert event_queue.put_started.wait(timeout=1.0)
    closer, close_finished, close_failures = close_in_thread(session)
    close_finished.wait(timeout=0.5)
    event_queue.release_put.set()
    emitter.join(timeout=1.0)
    closer.join(timeout=1.0)

    assert close_failures == []
    assert not emitter.is_alive()
    assert not closer.is_alive()
    assert session.state.get("p").completed == 1
    assert event_queue.unfinished_tasks == 0


def test_logging_helpers_do_not_change_divi_handlers(monkeypatch):
    divi_logger = logging.getLogger("divi")
    application_handler = logging.NullHandler()
    monkeypatch.setattr(divi_logger, "handlers", [application_handler])
    state = ProgressState()
    affected = state.apply(
        ProgressEvent.register("p", ProgressScope.PROGRAM, "Program", 1)
    )
    log_progress_event(ProgressEvent.show("p", "Preparing"))
    log_progress_state(state, affected)

    assert divi_logger.handlers == [application_handler]
