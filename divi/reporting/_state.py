# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pure state transitions for typed progress events."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from divi.backends._job_status import JobStatus

from ._events import EventKind, ProgressEvent, ProgressScope, TerminalStatus


@dataclass(slots=True)
class _TargetState:
    """The rendered state for one registered progress target."""

    scope: ProgressScope
    label: str
    total: int | None
    visible: bool
    completed: int = 0
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


class ProgressState:
    """Aggregate state for progress targets and their event transitions."""

    def __init__(self, *, hide_successful_programs: bool = False) -> None:
        """Create empty state with the requested successful-row visibility."""
        self._targets: dict[Hashable, _TargetState] = {}
        self._hide_successful_programs = hide_successful_programs

    @property
    def targets(self) -> Mapping[Hashable, _TargetState]:
        """Return a read-only mapping of isolated target-state snapshots."""
        return MappingProxyType(
            {
                target: replace(target_state)
                for target, target_state in self._targets.items()
            }
        )

    def get(self, progress_key: Hashable) -> _TargetState:
        """Return an isolated snapshot for a registered progress key."""
        return replace(self._stored(progress_key))

    def apply(self, event: ProgressEvent) -> set[Hashable]:
        """Apply one event and return every target whose rendered state changed."""
        if event.kind is EventKind.REGISTER:
            return self._register(event)

        target_state = self._stored(event.progress_key)
        if target_state.terminal_status is not None:
            return set()

        if event.kind is EventKind.ADVANCE:
            return self._advance(event, target_state)
        if event.kind is EventKind.SHOW:
            return self._show(event, target_state)
        if event.kind is EventKind.POLLING:
            return self._polling(event, target_state)
        if event.kind is EventKind.FINISH:
            return self._finish(event, target_state)
        raise ValueError(f"unsupported progress event kind: {event.kind!r}")

    def _stored(self, progress_key: Hashable) -> _TargetState:
        """Return the mutable record owned exclusively by this reducer."""
        return self._targets[progress_key]

    def _register(self, event: ProgressEvent) -> set[Hashable]:
        scope = cast(ProgressScope, event.scope)
        label = cast(str, event.label)

        existing = self._targets.get(event.progress_key)
        if existing is not None and existing.terminal_status is None:
            if existing.scope is not scope:
                raise ValueError("an active target cannot change scope")
            return set()

        members: list[_TargetState] = []
        if scope is ProgressScope.BATCH:
            members = [self._stored(key) for key in event.program_keys]

        target_state = _TargetState(
            scope=scope,
            label=label,
            total=event.total,
            visible=event.visible,
            batch_color=event.batch_color,
            program_keys=event.program_keys,
        )
        self._targets[event.progress_key] = target_state
        affected = {event.progress_key}
        for program_key, member in zip(event.program_keys, members, strict=True):
            member.batch_color = event.batch_color
            affected.add(program_key)
        return affected

    def _advance(
        self, event: ProgressEvent, target_state: _TargetState
    ) -> set[Hashable]:
        target_state.completed += cast(int, event.amount)
        if event.loss is not None:
            target_state.loss = event.loss
        return {event.progress_key}

    def _show(self, event: ProgressEvent, target_state: _TargetState) -> set[Hashable]:
        if (
            target_state.message == event.message
            and target_state.service_job_id is None
            and target_state.job_status is None
            and target_state.poll_attempt is None
            and target_state.max_retries is None
        ):
            return set()
        self._clear_polling(target_state)
        target_state.message = event.message
        return {event.progress_key}

    def _polling(
        self, event: ProgressEvent, target_state: _TargetState
    ) -> set[Hashable]:
        if (
            target_state.message,
            target_state.service_job_id,
            target_state.job_status,
            target_state.poll_attempt,
            target_state.max_retries,
        ) == (
            None,
            event.service_job_id,
            event.job_status,
            event.poll_attempt,
            event.max_retries,
        ):
            return set()
        target_state.message = None
        target_state.service_job_id = event.service_job_id
        target_state.job_status = event.job_status
        target_state.poll_attempt = event.poll_attempt
        target_state.max_retries = event.max_retries
        return {event.progress_key}

    def _finish(
        self, event: ProgressEvent, target_state: _TargetState
    ) -> set[Hashable]:
        terminal_status = cast(TerminalStatus, event.terminal_status)

        self._clear_polling(target_state)
        target_state.message = None
        target_state.job_status = event.job_status
        target_state.terminal_status = terminal_status
        target_state.detail = event.detail
        if terminal_status is TerminalStatus.SUCCESS and target_state.total is not None:
            target_state.completed = target_state.total
        if target_state.scope is ProgressScope.PROGRAM:
            target_state.visible = not (
                self._hide_successful_programs
                and terminal_status is TerminalStatus.SUCCESS
            )

        affected = {event.progress_key}
        if target_state.scope is ProgressScope.BATCH:
            for program_key in target_state.program_keys:
                self._stored(program_key).batch_color = ""
                affected.add(program_key)
        return affected

    @staticmethod
    def _clear_polling(target_state: _TargetState) -> None:
        target_state.service_job_id = None
        target_state.job_status = None
        target_state.poll_attempt = None
        target_state.max_retries = None


__all__ = ["ProgressState"]
