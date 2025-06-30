import logging
import shutil
import sys


class OverwriteStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

        self._last_line_length = 0
        # Worst case: 2 complex emojis (8 chars each) + buffer = 21 extra chars
        self._emoji_buffer = 21

    def emit(self, record):
        msg = self.format(record)
        # breakpoint()
        if msg.endswith("\r\n"):
            overwrite_and_newline = True
            clean_msg = msg[:-2]
        elif msg.endswith("\r"):
            overwrite_and_newline = False
            clean_msg = msg[:-1]
        else:
            # Normal message - no overwriting
            self.stream.write(msg + "\n")
            self.stream.flush()
            self._last_line_length = 0
            return

        # Clear previous line if needed
        if self._last_line_length > 0:
            clear_length = min(
                self._last_line_length + self._emoji_buffer,
                shutil.get_terminal_size().columns,
            )
            self.stream.write("\r" + " " * clear_length + "\r")

        # Write message with appropriate ending
        if overwrite_and_newline:
            self.stream.write(clean_msg + "\n")
            self._last_line_length = 0
        else:
            self.stream.write(clean_msg + "\r")
            self._last_line_length = len(self._strip_ansi(clean_msg))

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
