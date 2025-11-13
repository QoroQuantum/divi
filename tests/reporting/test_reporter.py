# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from divi.reporting._reporter import (
    LoggingProgressReporter,
    ProgressReporter,
    QueueProgressReporter,
)


class TestAbstractProgressReporter:
    """
    Tests the abstract base class ProgressReporter.
    """

    def test_abc_enforcement(self):
        """
        Check that subclasses must implement the abstract methods.
        """

        # A class that implements none of the abstract methods
        class BadReporter(ProgressReporter):
            pass

        with pytest.raises(TypeError):
            BadReporter()

        # A class that implements only one of the two abstract methods
        class IncompleteReporter(ProgressReporter):
            def update(self, **kwargs) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteReporter()

        # A class that implements all abstract methods should instantiate without error
        class GoodReporter(ProgressReporter):
            def update(self, **kwargs) -> None:
                pass

            def info(self, message: str, **kwargs) -> None:
                pass

        try:
            GoodReporter()
        except TypeError:
            pytest.fail("Instantiating a complete subclass should not raise TypeError")


class TestQueueProgressReporter:
    """
    Tests for the QueueProgressReporter class.
    """

    @pytest.fixture
    def mock_queue(self):
        """
        Pytest fixture for a mock queue.
        """
        return MagicMock(spec=Queue)

    def test_update(self, mock_queue):
        """
        Test the update method.
        """
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        reporter.update()
        mock_queue.put.assert_called_once_with({"job_id": "test_job", "progress": 1})

    def test_info_simple_message(self, mock_queue):
        """
        Test the info method with a simple message.
        """
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        reporter.info("A test message")
        expected_payload = {
            "job_id": "test_job",
            "progress": 0,
            "message": "A test message",
            "poll_attempt": 0,
        }
        mock_queue.put.assert_called_once_with(expected_payload)

    def test_info_finished_message(self, mock_queue):
        """
        Test the info method with a 'Finished successfully!' message.
        """
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        reporter.info("Finished successfully!")
        expected_payload = {
            "job_id": "test_job",
            "progress": 0,
            "message": "Finished successfully!",
            "final_status": "Success",
            "poll_attempt": 0,
        }
        mock_queue.put.assert_called_once_with(expected_payload)

    def test_info_with_polling(self, mock_queue):
        """
        Test the info method with polling information.
        """
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        polling_kwargs = {
            "poll_attempt": 1,
            "max_retries": 5,
            "service_job_id": "service_123",
            "job_status": "RUNNING",
        }
        reporter.info("Polling...", **polling_kwargs)
        expected_payload = {
            "job_id": "test_job",
            "progress": 0,
            "poll_attempt": 1,
            "max_retries": 5,
            "service_job_id": "service_123",
            "job_status": "RUNNING",
        }
        mock_queue.put.assert_called_once_with(expected_payload)


class TestLoggingProgressReporter:
    """
    Tests for the LoggingProgressReporter class.
    """

    @patch("divi.reporting._reporter.logger")
    def test_update(self, mock_logger):
        """
        Test the update method.
        """
        reporter = LoggingProgressReporter()
        reporter.update(iteration=5)
        mock_logger.info.assert_called_once()
        logged_message = mock_logger.info.call_args[0][0]
        assert "Finished Iteration #5" in logged_message
        assert "\r\n" in logged_message

    @patch("divi.reporting._reporter.logger")
    def test_info_simple_message(self, mock_logger):
        """
        Test the info method with a simple message.
        """
        reporter = LoggingProgressReporter()
        reporter.info("Hello World")
        mock_logger.info.assert_called_once_with("Hello World")

    @patch("divi.reporting._reporter.logger")
    def test_info_with_polling(self, mock_logger):
        """
        Test the info method with polling information.
        """
        reporter = LoggingProgressReporter()
        polling_kwargs = {
            "poll_attempt": 2,
            "max_retries": 10,
            "service_job_id": "service_abc-123",
            "job_status": "PENDING",
        }
        reporter.info("Polling...", **polling_kwargs)
        mock_logger.info.assert_called_once()
        # Check for essential content without being brittle about exact formatting/color
        call_args, call_kwargs = mock_logger.info.call_args
        logged_message = call_args[0]
        assert "service_abc" in logged_message
        assert "PENDING" in logged_message
        assert "Polling attempt 2 / 10" in logged_message
        # Check for behaviorally important parts
        assert logged_message.endswith("\r")
        assert call_kwargs.get("extra") == {"append": True}

    @patch("divi.reporting._reporter.logger")
    def test_info_with_iteration(self, mock_logger):
        """
        Test the info method with iteration information.
        """
        reporter = LoggingProgressReporter()
        reporter.info("Doing something", iteration=0)
        mock_logger.info.assert_called_once()
        logged_message = mock_logger.info.call_args[0][0]
        assert "Iteration #1" in logged_message
        assert "Doing something" in logged_message
        assert logged_message.endswith("\r")
