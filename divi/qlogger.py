import logging
import sys


def enable_logging(level=logging.INFO):
    root_logger = logging.getLogger(__name__.split(".")[0])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def disable_logging():
    root_logger = logging.getLogger(__name__.split(".")[0])
    root_logger.handlers.clear()
    root_logger.setLevel(logging.CRITICAL + 1)
