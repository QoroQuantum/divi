# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

from rich.logging import RichHandler


def _ensure_unbuffered_stdout() -> None:
    """Ensure stdout flushes after every write so Rich progress bars display correctly.

    On Windows PowerShell, Python uses line buffering by default. Rich's progress
    bars overwrite the same line (no newline), so updates stay buffered until
    completion. Setting write_through=True flushes after each write.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(write_through=True)
        except (OSError, ValueError):
            pass


class CustomRichFormatter(logging.Formatter):
    """
    A custom log formatter that removes '._reporter' from the logger name.
    Works with RichHandler.
    """

    def format(self, record):
        # Modify the record's name attribute in place
        if record.name.endswith("._reporter"):
            record.name = record.name.removesuffix("._reporter")
        return super().format(record)


def enable_logging(level=logging.INFO):
    """
    Enable logging for the divi package with Rich formatting.

    Sets up a RichHandler that provides colorized, formatted log output
    and removes the '._reporter' suffix from logger names.

    Args:
        level (int, optional): Logging level to set (e.g., logging.INFO,
            logging.DEBUG). Defaults to logging.INFO.

    Note:
        This function clears any existing handlers and sets up a new handler
        with custom formatting.
    """
    _ensure_unbuffered_stdout()
    root_logger = logging.getLogger(__name__.split(".")[0])

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )

    # Use a simpler formatter since RichHandler handles time display
    formatter = CustomRichFormatter(
        "%(name)s - %(levelname)s - %(message)s",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def disable_logging():
    """
    Disable all logging for the divi package.

    Removes all handlers and sets the logging level to above CRITICAL,
    effectively suppressing all log messages. This is useful when using
    progress bars that provide visual feedback.
    """
    root_logger = logging.getLogger(__name__.split(".")[0])
    root_logger.handlers.clear()
    root_logger.setLevel(logging.CRITICAL + 1)
