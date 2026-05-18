# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared cancellation primitives used across pipeline, qprog, and backend layers."""

import logging
import signal
import threading
from contextlib import contextmanager
from threading import Event

from divi.backends._async_job_backend import AsyncJobBackend
from divi.exceptions import ExecutionCancelledError

logger = logging.getLogger(__name__)


@contextmanager
def _sigint_to_event(event: Event):
    """Funnel SIGINT into ``event``: first press sets the event, second
    re-raises :class:`KeyboardInterrupt`. No-op outside the main thread
    or when a non-default SIGINT handler is already installed."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    existing = signal.getsignal(signal.SIGINT)
    if existing not in (signal.default_int_handler, signal.SIG_DFL):
        yield
        return

    press_count = 0

    def _handler(signum, frame):
        nonlocal press_count
        press_count += 1
        if press_count >= 2:
            signal.signal(signal.SIGINT, existing)
            raise KeyboardInterrupt
        event.set()

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, existing)


def _best_effort_cancel_job(backend, execution_result) -> None:
    """Notify an async backend's scheduler that an in-flight job should stop.

    No-op when ``backend`` is not an :class:`AsyncJobBackend` or when
    ``execution_result`` lacks a ``job_id``. Swallows the predictable failure
    modes (409 already-cancelled, network hiccups) at DEBUG so cancellation
    cleanup never masks the user's original interrupt.
    """
    if not isinstance(backend, AsyncJobBackend):
        return
    if execution_result is None or getattr(execution_result, "job_id", None) is None:
        return
    try:
        backend.cancel_job(execution_result)
    except Exception:
        logger.debug(
            "Best-effort cancel_job failed for %s",
            execution_result.job_id,
            exc_info=True,
        )


@contextmanager
def _auto_cancellation_scope(backend, execution_result):
    """SIGINT funnel + remote-job cleanup for direct callers of an
    :class:`~divi.backends.AsyncJobBackend`'s long-running poll loop.

    Yields a fresh :class:`threading.Event` that the wrapped poll loop should
    honour. On cancellation, best-effort cancels the remote job before
    re-raising. Implementers of :meth:`AsyncJobBackend.poll_job_status` should
    open this scope when ``loop_until_complete`` is ``True`` and the caller
    didn't supply their own cancellation event; when the caller did, they
    own the cleanup contract and this helper must not be opened.
    """
    event = Event()
    with _sigint_to_event(event):
        try:
            yield event
        except ExecutionCancelledError:
            _best_effort_cancel_job(backend, execution_result)
            raise
