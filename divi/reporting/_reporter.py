# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from queue import Queue

logger = logging.getLogger(__name__)


class ProgressReporter(ABC):
    """An abstract base class for reporting progress of a quantum program."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Provides a progress update."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Provides a simple informational message."""
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

    # Define ANSI color codes
    CYAN = "\033[36m"
    RESET = "\033[0m"

    def update(self, **kwargs):
        # You can decide how to format the update for logging
        logger.info(f"Finished Iteration #{kwargs['iteration']}\r\n")

    def info(self, message: str, **kwargs):
        # A special check for iteration updates to mimic old behavior
        if "poll_attempt" in kwargs:
            logger.info(
                f"Job {self.CYAN}{kwargs['service_job_id'].split('-')[0]}{self.RESET} is {kwargs['job_status']}. Polling attempt {kwargs['poll_attempt']} / {kwargs['max_retries']}\r",
                extra={"append": True},
            )
            return

        if "iteration" in kwargs:
            logger.info(f"Iteration #{kwargs['iteration'] + 1}: {message}\r")
            return

        logger.info(message)
