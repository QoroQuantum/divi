# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for Divi's opt-in logging integration."""

import io
import logging
import subprocess
import sys

import pytest
from rich.console import Console
from rich.logging import RichHandler

import divi.reporting as reporting
from divi.backends._job_status import JobStatus
from divi.reporting import _logging as logging_module
from divi.reporting import disable_logging, enable_logging
from divi.reporting._events import ProgressEvent, ProgressScope, TerminalStatus
from divi.reporting._state import ProgressState

LIBRARY_LOGGER_NAME = "divi"


def logged_messages(caplog) -> list[str]:
    """Return the formatted messages of every captured record."""
    return [record.getMessage() for record in caplog.records]


def test_reporting_public_surface_is_logging_only():
    assert reporting.__all__ == ["disable_logging", "enable_logging"]
    assert not hasattr(reporting, "ProgressReporter")
    assert not hasattr(reporting, "queue_listener")


@pytest.mark.parametrize(
    ("scope", "label", "total", "expected"),
    [
        (ProgressScope.PROGRAM, "Program", 1, "Program: 0/1"),
        (ProgressScope.WORKFLOW, "Workflow", None, "Workflow"),
    ],
)
def test_progress_state_renderer_logs_affected_targets(
    caplog, scope, label, total, expected
):
    state = ProgressState()
    affected = state.apply(ProgressEvent.register("p", scope, label, total))
    with caplog.at_level(logging.INFO, logger=LIBRARY_LOGGER_NAME):
        logging_module.log_progress_state(state, affected)

    assert logged_messages(caplog) == [expected]


def polling_event(limit: int | None) -> ProgressEvent:
    """Create a running-job polling event at attempt three."""
    return ProgressEvent.polling(
        "p", job_id="abc-def", status=JobStatus.RUNNING, attempt=3, limit=limit
    )


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (ProgressEvent.register("p", ProgressScope.PROGRAM, "Program", 3), ["Program"]),
        (ProgressEvent.advance("p", amount=2), ["Progress advanced by 2"]),
        (
            ProgressEvent.advance("p", loss=-0.5),
            ["Progress advanced by 1 (loss=-0.500000)"],
        ),
        (ProgressEvent.show("p", "Preparing"), ["Preparing"]),
        (ProgressEvent.show("p", ""), []),
        (polling_event(None), ["Job abc-def is RUNNING. Polling attempt 3 / ∞"]),
        (polling_event(5), ["Job abc-def is RUNNING. Polling attempt 3 / 5"]),
        (ProgressEvent.finish("p", TerminalStatus.SUCCESS), ["Success"]),
        (
            ProgressEvent.finish("p", TerminalStatus.FAILED, detail="boom"),
            ["Failed (boom)"],
        ),
    ],
)
def test_log_progress_event_logs_one_line_per_event(caplog, event, expected):
    with caplog.at_level(logging.INFO, logger=LIBRARY_LOGGER_NAME):
        logging_module.log_progress_event(event)

    assert logged_messages(caplog) == expected


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


def render_through_managed_handler(message: str) -> str:
    """Log ``message`` at INFO through Divi's handler and return the printed text."""
    enable_logging()
    handler = logging_module._managed_handler
    assert handler is not None
    output = io.StringIO()
    handler.console = Console(file=output, width=200, color_system=None)
    logging.getLogger("divi.probe").info("%s", message)
    return output.getvalue()


def test_managed_handler_prints_bracketed_text_verbatim(library_logger):
    message = "Program: 1/3 [next phase] [loss: -0.123457] [Job abc is RUNNING]"

    assert message in render_through_managed_handler(message)


def test_managed_handler_prints_the_level_once(library_logger):
    rendered = render_through_managed_handler("Preparing")

    assert "divi.probe - Preparing" in rendered
    assert rendered.count("INFO") == 1


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
