# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for Divi's opt-in logging integration."""

import io
import logging
import subprocess
import sys

import pytest
from rich.logging import RichHandler

from divi.reporting import _logging as logging_module
from divi.reporting import disable_logging, enable_logging
from divi.reporting._events import ProgressEvent, ProgressScope
from divi.reporting._state import ProgressState

LIBRARY_LOGGER_NAME = "divi"


def test_reporting_public_surface_is_logging_only():
    import divi.reporting as reporting

    assert reporting.__all__ == ["disable_logging", "enable_logging"]
    assert not hasattr(reporting, "ProgressReporter")
    assert not hasattr(reporting, "queue_listener")


def test_progress_state_renderer_logs_affected_targets(caplog):
    state = ProgressState()
    affected = state.apply(
        ProgressEvent.register("p", ProgressScope.PROGRAM, "Program", 1)
    )
    with caplog.at_level(logging.INFO, logger=LIBRARY_LOGGER_NAME):
        logging_module.log_progress_state(state, affected)

    assert [record.getMessage() for record in caplog.records] == ["Program: 0/1"]


@pytest.fixture
def library_logger():
    """Provide an isolated Divi logger and restore its prior state afterwards."""
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    original_handlers = list(logger.handlers)
    original_level = logger.level
    for handler in original_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)

    try:
        yield logger
    finally:
        disable_logging()
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)


def test_importing_divi_does_not_configure_logging():
    """Importing Divi leaves its library logger at the standard library default."""
    code = """
import logging
import divi
logger = logging.getLogger("divi")
assert logger.level == logging.NOTSET
assert len(logger.handlers) == 1
assert isinstance(logger.handlers[0], logging.NullHandler)
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_enable_logging_makes_info_effective_in_a_fresh_process():
    """The convenience helper makes its default INFO stream observable."""
    code = """
import logging
from divi.reporting import enable_logging

enable_logging()
logger = logging.getLogger("divi.probe")
assert logger.isEnabledFor(logging.INFO)
logger.info("managed-info-visible")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "managed-info-visible" in result.stdout + result.stderr


def test_enable_logging_preserves_application_handlers_and_owns_logger_threshold(
    library_logger,
):
    """Enabling owns only the threshold change needed by the requested level."""
    application_handler = logging.StreamHandler(io.StringIO())
    library_logger.addHandler(application_handler)
    library_logger.setLevel(logging.ERROR)

    enable_logging()

    assert library_logger.level == logging.INFO
    assert application_handler in library_logger.handlers
    assert (
        sum(isinstance(handler, RichHandler) for handler in library_logger.handlers)
        == 1
    )


def test_repeated_enable_logging_updates_one_managed_handler(library_logger):
    """Repeated enable calls update the owned handler instead of adding another."""
    enable_logging()
    enable_logging(level=logging.WARNING)

    managed_handlers = [
        handler
        for handler in library_logger.handlers
        if isinstance(handler, RichHandler)
    ]
    assert len(managed_handlers) == 1
    assert managed_handlers[0].level == logging.WARNING


def test_disable_logging_removes_only_divis_managed_handler(library_logger):
    """Disabling retains application handlers and restores Divi's old threshold."""
    application_handler = logging.StreamHandler(io.StringIO())
    library_logger.addHandler(application_handler)
    library_logger.setLevel(logging.ERROR)
    enable_logging()

    disable_logging()

    assert library_logger.handlers == [application_handler]
    assert library_logger.level == logging.ERROR


def test_divi_handler_formatting_does_not_mutate_records_for_other_handlers(
    library_logger,
):
    """A later handler observes the source logger name after Divi formats a record."""
    observed_records: list[logging.LogRecord] = []

    class ObservingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            observed_records.append(record)

    enable_logging()
    library_logger.addHandler(ObservingHandler())

    logging.getLogger("divi.reporting.worker").warning("A reporting warning")

    assert observed_records[0].name == "divi.reporting.worker"
