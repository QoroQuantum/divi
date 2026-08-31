# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Operation-scoped direct and queued progress delivery."""

import logging
import os
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from types import TracebackType
from typing import Self

from rich.console import Console

from ._events import ProgressEvent
from ._logging import diagnose_reporting_failure, log_progress_state
from ._rich import (
    CloseViewFn,
    RenderFn,
    console_supports_live_rendering,
    make_ensemble_view,
    make_standalone_view,
)
from ._state import ProgressState

_LISTENER_DRAIN_TIMEOUT = 30.0
_LISTENER_JOIN_TIMEOUT = 5.0
_PROGRESS_DISABLE_ENV = "DIVI_DISABLE_PROGRESS"
_PROGRESS_DISABLE_TRUTHY = frozenset({"1", "true", "yes", "on"})

logger = logging.getLogger("divi")


def _environment_disables_progress() -> bool:
    """Return whether the process environment suppresses progress output."""
    value = os.getenv(_PROGRESS_DISABLE_ENV, "")
    return value.strip().lower() in _PROGRESS_DISABLE_TRUTHY


class ProgressSession:
    """Own progress state, delivery, rendering, and deterministic teardown."""

    def __init__(
        self,
        state: ProgressState,
        render: RenderFn,
        close_view: CloseViewFn | None,
        *,
        console: Console | None = None,
        event_queue: Queue[ProgressEvent] | None = None,
        done_event: Event | None = None,
    ) -> None:
        self.state = state
        self._console = console if console is not None else Console()
        self._queue = event_queue
        self._done_event = done_event
        self._render = render
        self._close_view = close_view
        self._listener_thread: Thread | None = None
        self._lifecycle_lock = Lock()
        self._close_lock = Lock()
        self._accepting = True
        self._closed = False

    @property
    def console(self) -> Console:
        """Return the console owned by this rendering session."""
        return self._console

    @classmethod
    def direct(
        cls,
        state: ProgressState,
        *,
        console: Console | None = None,
    ) -> Self:
        """Create a session that applies events in the emitting thread."""
        resolved_console = console if console is not None else Console()
        if console_supports_live_rendering(resolved_console):
            render, close_view = make_standalone_view(resolved_console)
        else:
            render, close_view = log_progress_state, None
        return cls(
            state,
            render,
            close_view,
            console=resolved_console,
        )

    @classmethod
    def queued(
        cls,
        state: ProgressState,
        *,
        console: Console | None = None,
        is_jupyter: bool = False,
    ) -> Self:
        """Create a session with one daemon listener as its state writer."""
        resolved_console = console if console is not None else Console()
        render_in_jupyter = is_jupyter or resolved_console.is_jupyter
        if console_supports_live_rendering(
            resolved_console, is_jupyter=render_in_jupyter
        ):
            render, close_view = make_ensemble_view(
                resolved_console, is_jupyter=render_in_jupyter
            )
        else:
            render, close_view = log_progress_state, None

        event_queue: Queue[ProgressEvent] = Queue()
        done_event = Event()
        session = cls(
            state,
            render,
            close_view,
            console=resolved_console,
            event_queue=event_queue,
            done_event=done_event,
        )
        listener = Thread(
            target=session._listen,
            name="divi-progress-listener",
            daemon=True,
        )
        session._listener_thread = listener
        listener.start()
        return session

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def emit(self, event: ProgressEvent) -> None:
        """Deliver one event unless the operation has already closed."""
        with self._lifecycle_lock:
            if not self._accepting:
                return
            if self._queue is not None:
                self._queue.put(event)
                return
            self._process(event)

    def close(self) -> None:
        """Drain pending work, stop rendering, and close the session once."""
        with self._close_lock:
            if self._closed:
                return
            with self._lifecycle_lock:
                self._accepting = False

            self._shutdown_listener()

            if self._close_view is not None:
                try:
                    self._close_view()
                except Exception as exc:
                    self._record_failure(exc)
            self._closed = True

    def _shutdown_listener(self) -> None:
        if (
            self._queue is None
            or self._done_event is None
            or self._listener_thread is None
        ):
            return

        deadline = monotonic() + _LISTENER_DRAIN_TIMEOUT
        while self._queue.unfinished_tasks > 0:
            if not self._listener_thread.is_alive():
                break
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            self._done_event.wait(timeout=min(0.05, remaining))

        self._done_event.set()
        self._listener_thread.join(timeout=_LISTENER_JOIN_TIMEOUT)
        if self._listener_thread.is_alive():
            self._record_failure(
                RuntimeError("Progress listener did not terminate within timeout"),
                include_traceback=False,
            )
        elif self._queue.unfinished_tasks > 0:
            self._record_failure(
                RuntimeError(
                    "Progress listener terminated before acknowledging all events"
                ),
                include_traceback=False,
            )

    def _listen(self) -> None:
        if self._queue is None or self._done_event is None:
            return
        while not self._done_event.is_set() or self._queue.unfinished_tasks > 0:
            try:
                event = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._apply_and_render(event)
            except Exception as exc:
                self._record_failure(exc)
            finally:
                self._queue.task_done()

    def _process(self, event: ProgressEvent) -> None:
        try:
            self._apply_and_render(event)
        except Exception as exc:
            self._record_failure(exc)

    def _apply_and_render(self, event: ProgressEvent) -> None:
        affected = self.state.apply(event)
        self._render(self.state, affected)

    def _record_failure(
        self,
        exc: Exception,
        *,
        include_traceback: bool | None = None,
    ) -> None:
        if include_traceback is None:
            include_traceback = logger.isEnabledFor(logging.DEBUG)
        diagnose_reporting_failure(exc, include_traceback=include_traceback)


__all__ = ["ProgressSession"]
