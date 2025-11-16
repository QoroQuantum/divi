# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import io
import logging

import pytest

from divi.reporting import disable_logging, enable_logging
from divi.reporting._qlogger import (
    CustomFormatter,
    OverwriteStreamHandler,
    _is_jupyter,
)

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
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)

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


class TestIsJupyter:
    """Tests for _is_jupyter function."""

    @pytest.mark.parametrize(
        "patch_target,shell_name,side_effect,expected",
        [
            ("IPython.get_ipython", "ZMQInteractiveShell", None, True),
            ("IPython.get_ipython", "TerminalInteractiveShell", None, False),
            ("IPython.get_ipython", "OtherShell", None, False),
            ("IPython.get_ipython", None, NameError, False),
            (
                "builtins.__import__",
                None,
                ImportError("No module named 'IPython'"),
                False,
            ),
        ],
    )
    def test_is_jupyter(self, mocker, patch_target, shell_name, side_effect, expected):
        """Test _is_jupyter with various IPython configurations."""
        if side_effect:
            mocker.patch(patch_target, side_effect=side_effect)
        else:
            mock_get_ipython = mocker.patch(patch_target)
            mock_shell = mocker.Mock()
            mock_shell.__class__.__name__ = shell_name
            mock_get_ipython.return_value = mock_shell

        assert _is_jupyter() is expected


class TestCustomFormatter:
    """Tests for CustomFormatter."""

    @pytest.fixture
    def formatter(self):
        """Fixture providing a CustomFormatter instance."""
        return CustomFormatter("%(name)s - %(message)s")

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
        """Test that CustomFormatter removes '._reporter' suffix."""
        record = self._create_log_record("divi.reporting._reporter")
        formatted = formatter.format(record)

        assert record.name == "divi.reporting"
        assert "divi.reporting" in formatted
        assert "._reporter" not in formatted

    def test_format_preserves_other_names(self, formatter):
        """Test that CustomFormatter doesn't modify non-reporter logger names."""
        record = self._create_log_record("divi.reporting")
        original_name = record.name

        formatter.format(record)

        assert record.name == original_name


class TestOverwriteStreamHandler:
    """Tests for OverwriteStreamHandler."""

    @pytest.fixture
    def handler_with_stream(self):
        """Fixture providing a handler with a StringIO stream."""
        stream = io.StringIO()
        handler = OverwriteStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler, stream

    def _create_log_record(self, msg, **kwargs):
        """Helper to create a LogRecord with common defaults."""
        return logging.LogRecord(
            name=kwargs.get("name", "test"),
            level=kwargs.get("level", logging.INFO),
            pathname=kwargs.get("pathname", ""),
            lineno=kwargs.get("lineno", 0),
            msg=msg,
            args=kwargs.get("args", ()),
            exc_info=kwargs.get("exc_info", None),
        )

    @pytest.mark.parametrize(
        "last_record,expected_in_output",
        [
            ("Previous message", ["Previous message", "New message"]),
            ("", ["New message"]),
        ],
    )
    def test_emit_append(self, handler_with_stream, last_record, expected_in_output):
        """Test emit with append=True."""
        handler, stream = handler_with_stream
        handler._last_record = last_record

        record = self._create_log_record("New message\r")
        record.append = True

        handler.emit(record)

        output = stream.getvalue()
        for expected in expected_in_output:
            assert expected in output

    def test_emit_jupyter_clear_length(self, mocker, handler_with_stream):
        """Test that jupyter environment uses extended clear_length."""
        mocker.patch("divi.reporting._qlogger._is_jupyter", return_value=True)
        handler, stream = handler_with_stream
        handler._is_jupyter = True
        handler._last_message = "Previous message with some content"

        record = self._create_log_record("New message\r")
        handler.emit(record)

        output = stream.getvalue()
        assert "\r" in output

    def test_emit_normal_message(self, handler_with_stream):
        """Test emit with normal message (no overwriting)."""
        handler, stream = handler_with_stream

        record = self._create_log_record("Normal message")
        handler.emit(record)

        output = stream.getvalue()
        assert "Normal message" in output
        assert output.endswith("\n")

    @pytest.mark.parametrize(
        "msg_suffix,expected_end,last_message_empty",
        [
            ("\r\n", "\n", True),
            ("\r", "\r", False),
        ],
    )
    def test_emit_overwrite(
        self, handler_with_stream, msg_suffix, expected_end, last_message_empty
    ):
        """Test emit with overwrite endings."""
        handler, stream = handler_with_stream

        record = self._create_log_record(f"Overwrite message{msg_suffix}")
        handler.emit(record)

        output = stream.getvalue()
        assert "Overwrite message" in output
        assert output.endswith(expected_end)
        assert (handler._last_message == "") == last_message_empty
