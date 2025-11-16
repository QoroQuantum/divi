# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import sys


def _is_jupyter():
    """
    Checks if the code is running inside a Jupyter Notebook or IPython environment.
    """
    try:
        from IPython import get_ipython

        # Check if get_ipython() returns a shell instance (not None)
        # and if the shell class is 'ZMQInteractiveShell' for Jupyter notebooks/qtconsole
        # or 'TerminalInteractiveShell' for IPython console.
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # IPython terminal
        else:
            return False  # Other IPython environment (less common for typical Jupyter detection)
    except NameError:
        return False  # Not running in IPython
    except ImportError:
        return False  # IPython is not installed


class CustomFormatter(logging.Formatter):
    """
    A custom log formatter that removes '._reporter' from the logger name.
    """

    def format(self, record):
        # Modify the record's name attribute in place
        if record.name.endswith("._reporter"):
            record.name = record.name.removesuffix("._reporter")
        return super().format(record)


class OverwriteStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

        self._last_record = ""
        self._last_message = ""

        # Worst case: 2 complex emojis (8 chars each) + buffer = 21 extra chars
        self._emoji_buffer = 21

        self._is_jupyter = _is_jupyter()

    def emit(self, record):
        msg = self.format(record)
        append = getattr(record, "append", False)

        if append:
            space = " " if self._last_record else ""
            message_without_cr = record.message.removesuffix("\r")
            msg = f"{msg[:msg.index(record.message)]}{self._last_record}{space}[{message_without_cr}]\r"

        if msg.endswith("\r\n"):
            overwrite_and_newline = True
            clean_msg = msg.removesuffix("\r\n")

            if not append:
                self._last_record = record.message.removesuffix("\r\n")
        elif msg.endswith("\r"):
            overwrite_and_newline = False
            clean_msg = msg.removesuffix("\r")

            if not append:
                self._last_record = record.message.removesuffix("\r")
        else:
            # Normal message - no overwriting
            self.stream.write(msg + "\n")
            self.stream.flush()
            return

        # Clear previous line if needed
        if len(self._last_message) > 0:
            if self._is_jupyter:
                clear_length = len(self._last_message) + self._emoji_buffer + 50
            else:
                clear_length = min(
                    len(self._last_message) + self._emoji_buffer,
                    shutil.get_terminal_size().columns,
                )

            self.stream.write("\r" + " " * clear_length + "\r")
            self.stream.flush()

        # Write message with appropriate ending
        if overwrite_and_newline:
            self.stream.write(clean_msg + "\n")
            self._last_message = ""
        else:
            self.stream.write(clean_msg + "\r")
            self._last_message = self._strip_ansi(clean_msg)

        self.stream.flush()

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences for accurate length calculation"""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)


def enable_logging(level=logging.INFO):
    """
    Enable logging for the divi package with custom formatting.

    Sets up a custom logger with an OverwriteStreamHandler that supports
    message overwriting (for progress updates) and removes the '._reporter'
    suffix from logger names.

    Args:
        level (int, optional): Logging level to set (e.g., logging.INFO,
            logging.DEBUG). Defaults to logging.INFO.

    Note:
        This function clears any existing handlers and sets up a new handler
        with custom formatting.
    """
    root_logger = logging.getLogger(__name__.split(".")[0])

    formatter = CustomFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = OverwriteStreamHandler(sys.stdout)
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
