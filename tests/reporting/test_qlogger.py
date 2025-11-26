# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from divi.reporting import disable_logging, enable_logging
from divi.reporting._qlogger import CustomRichFormatter

LIBRARY_ROOT_LOGGER_NAME = "divi"


@pytest.fixture(autouse=True)
def reset_logging_config():
    """
    Fixture to reset logging configuration before each test to ensure isolation.
    """
    root_logger = logging.getLogger(LIBRARY_ROOT_LOGGER_NAME)
    root_logger.handlers.clear()
    root_logger.setLevel(logging.NOTSET)  # Reset to default


@pytest.fixture
def root_logger():
    """Fixture providing the root logger for divi."""
    return logging.getLogger(LIBRARY_ROOT_LOGGER_NAME)


def test_enable_logging_default_level(caplog, root_logger):
    """
    Test that enable_logging sets the correct level and adds a handler,
    and logs messages at INFO level.
    """
    enable_logging()

    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) == 1
    from rich.logging import RichHandler

    assert isinstance(root_logger.handlers[0], RichHandler)

    # Test logging output
    with caplog.at_level(logging.INFO, logger=LIBRARY_ROOT_LOGGER_NAME):
        root_logger.info("This is an info message.")

        # Should not be captured at INFO level
        root_logger.debug("This is a debug message.")

        assert "This is an info message." in caplog.text
        assert "This is a debug message." not in caplog.text
        assert caplog.records[0].levelname == "INFO"


def test_enable_logging_custom_level(caplog, root_logger):
    """
    Test that enable_logging sets a custom log level correctly.
    """
    enable_logging(level=logging.DEBUG)

    assert root_logger.level == logging.DEBUG

    # Test logging output at DEBUG level
    with caplog.at_level(logging.DEBUG, logger=LIBRARY_ROOT_LOGGER_NAME):
        root_logger.debug("A debug test message.")
        assert "A debug test message." in caplog.text
        assert caplog.records[0].levelname == "DEBUG"


def test_disable_logging(caplog, root_logger):
    """
    Test that disable_logging clears handlers and sets a level that
    effectively disables all logging.
    """

    # First, enable logging to ensure there's something to disable
    enable_logging(level=logging.DEBUG)

    # Should have handlers before disabling
    assert len(root_logger.handlers) > 0
    assert root_logger.level == logging.DEBUG

    disable_logging()

    assert len(root_logger.handlers) == 0
    assert root_logger.level == (logging.CRITICAL + 1)

    # Temporarily set propagate to False for this logger to ensure messages
    # do not reach the root logger (where caplog's handler is usually attached).
    # This prevents the messages from being captured by caplog when the logger
    # itself is "disabled" via level and cleared handlers.
    initial_propagate = root_logger.propagate
    root_logger.propagate = False

    try:
        # Test that no messages are logged after disabling
        # We need to capture from the *specific* logger we disabled.
        # caplog.at_level applies to the specific logger passed to it.
        with caplog.at_level(logging.DEBUG, logger=LIBRARY_ROOT_LOGGER_NAME):
            root_logger.info("This message should not appear.")
            root_logger.critical("This critical message should also not appear.")
            # Assert that nothing was captured by caplog *for this logger*
            assert "This message should not appear." not in caplog.text
            assert "This critical message should also not appear." not in caplog.text
            assert len(caplog.records) == 0
    finally:
        # Always restore propagation to its original state after the test
        root_logger.propagate = initial_propagate


def test_enable_logging_idempotency(caplog, root_logger):
    """
    Test that calling enable_logging multiple times doesn't add multiple handlers.
    """
    enable_logging()

    # Call again with a different level
    enable_logging(level=logging.WARNING)

    assert len(root_logger.handlers) == 1
    # Should reflect the last call's level
    assert root_logger.level == logging.WARNING

    with caplog.at_level(logging.WARNING, logger=LIBRARY_ROOT_LOGGER_NAME):
        root_logger.warning("This is a warning.")
        # Should not be captured at WARNING
        root_logger.info("This is an info.")
        assert "This is a warning." in caplog.text
        assert "This is an info." not in caplog.text


class TestCustomRichFormatter:
    """Tests for CustomRichFormatter."""

    @pytest.fixture
    def formatter(self):
        """Fixture providing a CustomRichFormatter instance."""
        return CustomRichFormatter("%(name)s - %(message)s")

    def _create_log_record(self, name, msg="Test message"):
        """Helper to create a LogRecord."""
        return logging.LogRecord(
            name=name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_format_removes_reporter_suffix(self, formatter):
        """Test that CustomRichFormatter removes '._reporter' suffix."""
        record = self._create_log_record("divi.reporting._reporter")
        formatted = formatter.format(record)

        assert record.name == "divi.reporting"
        assert "divi.reporting" in formatted
        assert "._reporter" not in formatted

    def test_format_preserves_other_names(self, formatter):
        """Test that CustomRichFormatter doesn't modify non-reporter logger names."""
        record = self._create_log_record("divi.reporting")
        original_name = record.name

        formatter.format(record)

        assert record.name == original_name
