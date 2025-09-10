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
    def update(self, **kwargs):
        """Provides a progress update."""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs):
        """
        Provides a simple informational message.
        No changes to progress or state.
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
        payload = {"job_id": self._job_id, "progress": 0}

        if "Finished Optimization" in message or "Computed Final Solution" in message:
            payload["final_status"] = "Success"
        elif "poll_attempt" in kwargs:
            payload["poll_attempt"] = kwargs["poll_attempt"]
            payload["max_retries"] = kwargs["max_retries"]
            payload["service_job_id"] = kwargs["service_job_id"]
            payload["job_status"] = kwargs["job_status"]
        else:
            payload["message"] = message

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
                rf"Job {self.CYAN}{kwargs['service_job_id'].split('-')[0]}{self.RESET} is {kwargs['job_status']}. Polling attempt {kwargs['poll_attempt']} / {kwargs['max_retries']}\r",
                extra={"append": True},
            )
            return

        if "iteration" in kwargs:
            logger.info(
                f"Running Iteration #{kwargs['iteration'] + 1} circuits: {message}\r"
            )
            return

        logger.info(message)
