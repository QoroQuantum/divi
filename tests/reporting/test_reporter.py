# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from queue import Queue

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
    def mock_queue(self, mocker):
        """
        Pytest fixture for a mock queue.
        """
        return mocker.MagicMock(spec=Queue)

    def test_update(self, mock_queue):
        """
        Test the update method.
        """
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        reporter.update()
        mock_queue.put.assert_called_once_with({"job_id": "test_job", "progress": 1})

    def test_update_with_loss(self, mock_queue):
        """Test the update method forwards loss when available."""
        reporter = QueueProgressReporter(job_id="test_job", progress_queue=mock_queue)
        reporter.update(loss=-0.123456789)
        mock_queue.put.assert_called_once_with(
            {"job_id": "test_job", "progress": 1, "loss": -0.123456789}
        )

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

    def test_update(self, mocker):
        """
        Test the update method.
        """
        mock_logger = mocker.patch("divi.reporting._reporter.logger")
        reporter = LoggingProgressReporter()
        reporter.update(iteration=5)
        mock_logger.info.assert_called_once()
        logged_message = mock_logger.info.call_args[0][0]
        assert "Finished Iteration #5" in logged_message

    def test_update_with_loss(self, mocker):
        """Test update logs iteration and formatted loss."""
        mock_logger = mocker.patch("divi.reporting._reporter.logger")
        reporter = LoggingProgressReporter()
        reporter.update(iteration=5, loss=-0.123456789)
        mock_logger.info.assert_called_once_with(
            "Finished Iteration #5 (loss=-0.123457)"
        )

    def test_info_simple_message(self, mocker):
        """
        Test the info method with a simple message.
        """
        mock_logger = mocker.patch("divi.reporting._reporter.logger")
        reporter = LoggingProgressReporter()
        reporter.info("Hello World")
        mock_logger.info.assert_called_once_with("Hello World")

    def test_info_with_overwrite(self, mocker):
        """
        Test the info method with overwrite=True (should use Rich's status).
        """
        mock_console_class = mocker.patch("divi.reporting._reporter.Console")
        mock_console = mock_console_class.return_value
        mock_status = mocker.MagicMock()
        mock_console.status.return_value = mock_status

        reporter = LoggingProgressReporter()
        reporter.info("Computing something", overwrite=True)

        # Overwrite creates a status with the message
        mock_console.status.assert_called_once()
        status_msg = mock_console.status.call_args[0][0]
        assert "Computing something" in status_msg
        mock_status.__enter__.assert_called_once()

        # Second overwrite updates the existing status (not a new one)
        reporter.info("Still computing", overwrite=True)
        mock_status.update.assert_called_once()
        update_msg = mock_status.update.call_args[0][0]
        assert "Still computing" in update_msg
        mock_status.__exit__.assert_not_called()

        # Normal message closes the status
        mock_logger = mocker.patch("divi.reporting._reporter.logger")
        reporter.info("Done!")
        mock_status.__exit__.assert_called_once()
        mock_logger.info.assert_called_once_with("Done!")

    def test_info_with_polling(self, mocker):
        """
        Test the info method with polling information.
        """
        mock_console_class = mocker.patch("divi.reporting._reporter.Console")
        mock_console = mock_console_class.return_value
        mock_status = mocker.MagicMock()
        mock_console.status.return_value = mock_status

        reporter = LoggingProgressReporter()
        polling_kwargs = {
            "poll_attempt": 2,
            "max_retries": 10,
            "service_job_id": "service_abc-123",
            "job_status": "PENDING",
        }
        reporter.info("Polling...", **polling_kwargs)

        # Status shows job ID prefix, job status, and attempt progress
        mock_console.status.assert_called_once()
        status_msg = mock_console.status.call_args[0][0]
        assert "service_abc" in status_msg
        assert "PENDING" in status_msg
        assert "2" in status_msg and "10" in status_msg
        mock_status.__enter__.assert_called_once()

        # Subsequent poll updates the existing status with new attempt number
        reporter.info("Polling...", **{**polling_kwargs, "poll_attempt": 3})
        mock_status.update.assert_called_once()
        update_msg = mock_status.update.call_args[0][0]
        assert "3" in update_msg and "10" in update_msg

    def test_info_with_iteration(self, mocker):
        """
        Test the info method with iteration information.
        """
        mock_console_class = mocker.patch("divi.reporting._reporter.Console")
        mock_console = mock_console_class.return_value
        mock_status = mocker.MagicMock()
        mock_console.status.return_value = mock_status

        reporter = LoggingProgressReporter()
        reporter.info("Doing something", iteration=0)

        # Status shows 1-indexed iteration number and the message
        mock_console.status.assert_called_once()
        status_msg = mock_console.status.call_args[0][0]
        assert "1" in status_msg
        assert "Doing something" in status_msg
        mock_status.__enter__.assert_called_once()

        # Subsequent iteration call updates the existing status
        reporter.info("Doing something else", iteration=0)
        mock_status.update.assert_called_once()
        update_msg = mock_status.update.call_args[0][0]
        assert "Doing something else" in update_msg

    def test_info_iteration_and_polling_concatenation(self, mocker):
        """
        Test that iteration and polling messages are concatenated together.
        """
        mock_console_class = mocker.patch("divi.reporting._reporter.Console")
        mock_console = mock_console_class.return_value
        mock_status = mocker.MagicMock()
        mock_console.status.return_value = mock_status

        reporter = LoggingProgressReporter()

        # First, set an iteration message
        reporter.info("Optimizing parameters", iteration=0)
        mock_console.status.assert_called_once()
        initial_msg = mock_console.status.call_args[0][0]
        assert "Iteration #1" in initial_msg
        assert "Optimizing parameters" in initial_msg

        # Then add polling info - should concatenate
        polling_kwargs = {
            "poll_attempt": 4,
            "max_retries": 5000,
            "service_job_id": "e4b1b59f-123",
            "job_status": "PENDING",
        }
        reporter.info("", **polling_kwargs)

        # Should have updated with concatenated message
        assert mock_status.update.call_count >= 1
        update_msg = mock_status.update.call_args[0][0]
        assert "Iteration #1" in update_msg
        assert "Optimizing parameters" in update_msg
        assert "Job" in update_msg
        assert "e4b1b59f" in update_msg
        assert "PENDING" in update_msg
        assert "Polling attempt 4 / 5000" in update_msg
        assert " - " in update_msg  # Should be concatenated with separator


class TestDisableProgress:
    """Tests for DIVI_DISABLE_PROGRESS env var behaviour."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " True "])
    def test_truthy_values_disable_progress(self, monkeypatch, value):
        """Truthy env var values disable the progress spinner."""
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", value)
        reporter = LoggingProgressReporter()
        assert reporter._disable_progress is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "random"])
    def test_falsy_values_keep_progress_enabled(self, monkeypatch, value):
        """Non-truthy env var values leave progress enabled."""
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", value)
        reporter = LoggingProgressReporter()
        assert reporter._disable_progress is False

    def test_unset_env_var_keeps_progress_enabled(self, monkeypatch):
        """When the env var is not set at all, progress is enabled."""
        monkeypatch.delenv("DIVI_DISABLE_PROGRESS", raising=False)
        reporter = LoggingProgressReporter()
        assert reporter._disable_progress is False

    def test_disabled_info_logs_instead_of_spinner(self, monkeypatch, caplog):
        """When progress is disabled, info() logs the message without creating a spinner."""
        monkeypatch.setenv("DIVI_DISABLE_PROGRESS", "1")
        reporter = LoggingProgressReporter()
        with caplog.at_level("INFO", logger="divi.reporting._reporter"):
            reporter.info("test message")
        assert "test message" in caplog.text
        assert reporter._status is None
