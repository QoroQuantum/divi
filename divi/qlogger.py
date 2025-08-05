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

        if record.message.startswith(r"\c"):
            sep = r"\c"
            msg = f"{msg.split(sep)[0]}{self._last_record} [{record.message[2:-2]}]\r"

        if msg.endswith("\r\n"):
            overwrite_and_newline = True
            clean_msg = msg[:-2]

            if not record.message.startswith("\c"):
                self._last_record = record.message[:-2]
        elif msg.endswith("\r"):
            overwrite_and_newline = False
            clean_msg = msg[:-1]

            if not record.message.startswith(r"\c"):
                self._last_record = record.message[:-1]
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
    root_logger = logging.getLogger(__name__.split(".")[0])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = OverwriteStreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def disable_logging():
    root_logger = logging.getLogger(__name__.split(".")[0])
    root_logger.handlers.clear()
    root_logger.setLevel(logging.CRITICAL + 1)
