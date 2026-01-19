# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import os
from abc import ABC, abstractmethod
from queue import Queue

from rich.console import Console

logger = logging.getLogger(__name__)


class ProgressReporter(ABC):
    """An abstract base class for reporting progress of a quantum program."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Provides a progress update."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Provides a simple informational message.

        Args:
            message: The message to display.
            **kwargs: Additional keyword arguments for subclasses.
        """
        pass


class QueueProgressReporter(ProgressReporter):
    """Reports progress by putting structured dictionaries onto a Queue."""

    def __init__(self, job_id: str, progress_queue: Queue):
        self._job_id = job_id
        self._queue = progress_queue

    def update(self, **kwargs):
        payload = {"job_id": self._job_id, "progress": 1}
        self._queue.put(payload)

    def info(self, message: str, **kwargs):
        payload = {"job_id": self._job_id, "progress": 0, "message": message}

        if "Finished successfully!" in message:
            payload["final_status"] = "Success"

        if "poll_attempt" in kwargs:
            # For polling, remove the message key so the last message persists.
            del payload["message"]
            payload["poll_attempt"] = kwargs["poll_attempt"]
            payload["max_retries"] = kwargs["max_retries"]
            payload["service_job_id"] = kwargs["service_job_id"]
            payload["job_status"] = kwargs["job_status"]
        else:
            # For any other message, explicitly reset the polling attempt counter.
            payload["poll_attempt"] = 0

        self._queue.put(payload)


class LoggingProgressReporter(ProgressReporter):
    """Reports progress by logging messages to the console."""

    _atexit_registered = False

    def __init__(self):
        # Use the same console instance that RichHandler uses to avoid interference
        self._console = Console(file=None)  # file=None uses stdout, same as RichHandler
        self._status = None  # Track active status for overwriting messages
        self._current_msg = None  # Track current main message
        self._polling_msg = None  # Track current polling message
        self._disable_progress = self._should_disable_progress()

    def _ensure_atexit_hook(self):
        if self._disable_progress or LoggingProgressReporter._atexit_registered:
            return
        atexit.register(self._close_status)
        LoggingProgressReporter._atexit_registered = True

    @staticmethod
    def _should_disable_progress() -> bool:
        disable_env = os.getenv("DIVI_DISABLE_PROGRESS", "").strip().lower()
        return disable_env in {"1", "true", "yes", "on"}

    def _close_status(self):
        """Close any active status."""
        if self._status:
            self._status.__exit__(None, None, None)
            self._status = None
        self._current_msg = None
        self._polling_msg = None

    def _build_status_msg(self) -> str:
        """Build combined status message from current message and polling info."""
        parts = []
        if self._current_msg:
            parts.append(self._current_msg)
        if self._polling_msg:
            parts.append(self._polling_msg)
        return " - ".join(parts) if parts else ""

    def _update_or_create_status(self):
        """Update existing status or create a new one with combined message."""
        if self._disable_progress:
            return
        status_msg = self._build_status_msg()
        if not status_msg:
            return
        self._ensure_atexit_hook()
        if self._status:
            self._status.update(status_msg)
        else:
            self._status = self._console.status(status_msg, spinner="aesthetic")
            self._status.__enter__()

    def update(self, **kwargs):
        # Close any active status before logging
        self._close_status()
        logger.info(f"Finished Iteration #{kwargs['iteration']}")

    def info(self, message: str, overwrite: bool = False, **kwargs):
        if self._disable_progress:
            logger.info(message)
            return
        # A special check for iteration updates to use Rich's status for overwriting
        if "poll_attempt" in kwargs:
            self._polling_msg = (
                f"Job [cyan]{kwargs['service_job_id'].split('-')[0]}[/cyan] is "
                f"{kwargs['job_status']}. Polling attempt {kwargs['poll_attempt']} / "
                f"{kwargs['max_retries']}"
            )
            self._update_or_create_status()
            return

        # Use Rich's status for iteration messages to enable overwriting
        if "iteration" in kwargs:
            self._current_msg = f"Iteration #{kwargs['iteration'] + 1}: {message}"
            self._update_or_create_status()
            return

        # Use Rich's status for messages that should overwrite
        if overwrite:
            # Set current message, keep polling state so it can be concatenated
            self._current_msg = message
            self._update_or_create_status()
            return

        # Close status for normal messages
        self._close_status()
        logger.info(message)
