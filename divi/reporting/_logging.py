# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Logging fallbacks and opt-in Rich formatting for Divi."""

import logging
import sys
import traceback
from collections.abc import Hashable

from rich.logging import RichHandler

from ._events import EventKind, ProgressEvent
from ._rich import render_status_text
from ._state import ProgressState

logger = logging.getLogger("divi")
_managed_handler: RichHandler | None = None
_previous_logger_level: int | None = None


def enable_logging(level: int = logging.INFO) -> None:
    """Enable Divi's Rich logging handler at ``level``."""
    global _managed_handler, _previous_logger_level

    if _managed_handler is None:
        _previous_logger_level = logger.level
        _managed_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        _managed_handler.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(_managed_handler)
    logger.setLevel(level)
    _managed_handler.setLevel(level)


def disable_logging() -> None:
    """Remove Divi's handler and restore the preceding logger level."""
    global _managed_handler, _previous_logger_level

    if _managed_handler is None:
        return
    logger.removeHandler(_managed_handler)
    _managed_handler.close()
    _managed_handler = None
    if _previous_logger_level is not None:
        logger.setLevel(_previous_logger_level)
        _previous_logger_level = None


def log_progress_event(event: ProgressEvent) -> None:
    """Log one progress event without configuring the application logger."""
    if event.kind is EventKind.REGISTER:
        if event.label is not None:
            logger.info("%s", event.label)
        return

    if event.kind is EventKind.ADVANCE:
        message = f"Progress advanced by {event.amount}"
        if event.loss is not None:
            message += f" (loss={float(event.loss):.6f})"
        logger.info("%s", message)
        return

    if event.kind is EventKind.SHOW:
        if event.message is not None:
            logger.info("%s", event.message)
        return

    if event.kind is EventKind.POLLING:
        limit = "∞" if event.max_retries is None else event.max_retries
        logger.info(
            "Job %s is %s. Polling attempt %s / %s",
            event.service_job_id,
            event.job_status,
            event.poll_attempt,
            limit,
        )
        return

    if event.kind is EventKind.FINISH:
        detail = f" ({event.detail})" if event.detail else ""
        logger.info("%s%s", event.terminal_status, detail)


def diagnose_reporting_failure(exc: Exception, *, include_traceback: bool) -> None:
    """Write one immediate diagnostic that cannot be swallowed by logging."""
    message = f"Progress reporting failed: {exc}"
    print(message, file=sys.stderr, flush=True)
    if include_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def log_progress_state(state: ProgressState, affected: set[Hashable]) -> None:
    """Render affected progress targets through logging."""
    for target_id in affected:
        target = state.get(target_id)
        parts: list[str] = []
        if target.total is not None:
            parts.append(f"{target.completed}/{target.total}")
        status = str(render_status_text(target)).strip()
        if status:
            parts.append(status)
        suffix = f": {' '.join(parts)}" if parts else ""
        logger.info("%s%s", target.label, suffix)


__all__ = [
    "disable_logging",
    "enable_logging",
    "diagnose_reporting_failure",
    "log_progress_event",
    "log_progress_state",
]
