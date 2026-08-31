# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable progress events exchanged between producers and renderers."""

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from divi.backends._job_status import JobStatus


class EventKind(StrEnum):
    """The operation represented by a progress event."""

    REGISTER = "register"
    ADVANCE = "advance"
    SHOW = "show"
    POLLING = "polling"
    FINISH = "finish"


class ProgressScope(StrEnum):
    """The reporting scope that owns a progress row."""

    PROGRAM = "program"
    BATCH = "batch"
    PREPARATION = "preparation"
    WORKFLOW = "workflow"


class TerminalStatus(StrEnum):
    """The terminal outcome of a progress row."""

    SUCCESS = "Success"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    ABORTED = "Aborted"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressEvent:
    """A single immutable request produced through the named factories."""

    kind: EventKind
    progress_key: Hashable
    scope: ProgressScope | None = None
    label: str | None = None
    total: int | None = None
    visible: bool = True
    amount: int | None = None
    loss: float | None = None
    message: str | None = None
    service_job_id: str | None = None
    job_status: JobStatus | None = None
    poll_attempt: int | None = None
    max_retries: int | None = None
    terminal_status: TerminalStatus | None = None
    detail: str | None = None
    batch_color: str = ""
    program_keys: tuple[Hashable, ...] = ()

    @classmethod
    def register(
        cls,
        progress_key: Hashable,
        scope: ProgressScope,
        label: str,
        total: int | None,
        *,
        visible: bool = True,
        batch_color: str = "",
        program_keys: Iterable[Hashable] = (),
    ) -> "ProgressEvent":
        """Create an event that registers a progress row."""
        if total is not None and total < 0:
            raise ValueError("total must be non-negative or None")
        members = tuple(program_keys)
        if scope is not ProgressScope.BATCH and (batch_color or members):
            raise ValueError("batch metadata requires batch scope")
        return cls(
            kind=EventKind.REGISTER,
            progress_key=progress_key,
            scope=scope,
            label=label,
            total=total,
            visible=visible,
            batch_color=batch_color,
            program_keys=members,
        )

    @classmethod
    def advance(
        cls,
        progress_key: Hashable,
        *,
        amount: int = 1,
        loss: float | None = None,
    ) -> "ProgressEvent":
        """Create an event that advances a progress row."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        return cls(
            kind=EventKind.ADVANCE,
            progress_key=progress_key,
            amount=amount,
            loss=loss,
        )

    @classmethod
    def show(cls, progress_key: Hashable, message: str) -> "ProgressEvent":
        """Create an event that displays a row message."""
        return cls(kind=EventKind.SHOW, progress_key=progress_key, message=message)

    @classmethod
    def polling(
        cls,
        progress_key: Hashable,
        *,
        job_id: str,
        status: JobStatus | str,
        attempt: int,
        limit: int | None,
    ) -> "ProgressEvent":
        """Create a polling event, preserving unknown backend status text."""
        if not job_id:
            raise ValueError("job_id must not be empty")
        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        if limit is not None and attempt > limit:
            raise ValueError("attempt must not exceed limit")
        known_status = JobStatus.coerce(status)
        if known_status is None:
            return cls.show(
                progress_key,
                f"Backend status {status} for job {job_id} (attempt {attempt})",
            )
        return cls(
            kind=EventKind.POLLING,
            progress_key=progress_key,
            service_job_id=job_id,
            job_status=known_status,
            poll_attempt=attempt,
            max_retries=limit,
        )

    @classmethod
    def finish(
        cls,
        progress_key: Hashable,
        status: TerminalStatus,
        *,
        job_status: JobStatus | None = None,
        detail: str | None = None,
    ) -> "ProgressEvent":
        """Create an event that marks a progress row as terminal."""
        return cls(
            kind=EventKind.FINISH,
            progress_key=progress_key,
            job_status=job_status,
            terminal_status=status,
            detail=detail,
        )


ProgressEmitter = Callable[[ProgressEvent], None]


def discard_progress_event(event: ProgressEvent) -> None:
    """Ignore a progress event when reporting is disabled."""
    del event


__all__ = [
    "ProgressEmitter",
    "EventKind",
    "ProgressEvent",
    "ProgressScope",
    "TerminalStatus",
    "discard_progress_event",
]
