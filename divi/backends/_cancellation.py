# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared cancellation primitives used across pipeline, qprog, and backend layers."""

import logging
import signal
import threading
from contextlib import contextmanager
from http import HTTPStatus
from threading import Event

import requests

from divi.exceptions import ExecutionCancelledError

from ._base import AsyncJobBackend, CircuitRunner, ExecutionResult
from ._job_status import QoroJobError

logger = logging.getLogger(__name__)


def _require_async_backend(backend: CircuitRunner) -> AsyncJobBackend:
    """Return ``backend``, which returned a job handle and so must be asynchronous."""
    if not isinstance(backend, AsyncJobBackend):
        raise RuntimeError(
            f"Backend {type(backend).__name__} returned an ExecutionResult "
            "with a job_id but is not an AsyncJobBackend."
        )
    return backend


def raise_if_cancelled(event: Event | None, message: str) -> None:
    """Raise :class:`~divi.exceptions.ExecutionCancelledError` if *event* is set."""
    if event is not None and event.is_set():
        raise ExecutionCancelledError(message)


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


def _best_effort_cancel_job(
    backend: CircuitRunner, execution_result: ExecutionResult | None
) -> None:
    """Notify an async backend's scheduler that an in-flight job should stop.

    No-op when ``backend`` is not an :class:`AsyncJobBackend` or when
    ``execution_result`` lacks a ``job_id``. Logs a failed cancel as a warning
    instead of raising; a 409 means the job already stopped.
    """
    if not isinstance(backend, AsyncJobBackend):
        return
    if execution_result is None or execution_result.job_id is None:
        return
    try:
        backend.cancel_job(execution_result)
    except Exception as exc:
        already_stopped = (
            isinstance(exc, requests.HTTPError)
            and exc.response is not None
            and exc.response.status_code == HTTPStatus.CONFLICT
        )
        level = logging.DEBUG if already_stopped else logging.WARNING
        logger.log(level, "Could not cancel job %s: %s", execution_result.job_id, exc)


@contextmanager
def _cancel_job_on_abort(
    backend: CircuitRunner, execution_result: ExecutionResult | None
):
    """Cancel the job if the block raises anything except a terminal ``QoroJobError``."""
    try:
        yield
    except QoroJobError:
        raise
    except BaseException:
        _best_effort_cancel_job(backend, execution_result)
        raise
