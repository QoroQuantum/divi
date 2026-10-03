# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends._cancellation."""

import logging
from threading import Event

import pytest

from divi.backends import AsyncJobBackend, ExecutionResult
from divi.backends._cancellation import (
    _auto_cancellation_scope,
    _best_effort_cancel_job,
)
from divi.exceptions import ExecutionCancelledError


class TestBestEffortCancelJob:
    """Tests for the helper that funnels in-flight async-job cancellation."""

    def test_calls_backend_cancel_for_async_backend_with_job_id(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")
        _best_effort_cancel_job(backend, result)
        backend.cancel_job.assert_called_once_with(result)

    def test_noop_for_sync_backend(self, mocker):
        sync_backend = mocker.Mock(spec=["cancel_job"])
        _best_effort_cancel_job(sync_backend, ExecutionResult(job_id="job_y"))
        sync_backend.cancel_job.assert_not_called()

    def test_noop_when_no_job_id(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        _best_effort_cancel_job(backend, ExecutionResult(results=[]))
        backend.cancel_job.assert_not_called()

    def test_swallows_cancel_job_exception(self, mocker, caplog):
        backend = mocker.Mock(spec=AsyncJobBackend)
        error = RuntimeError("server says no")
        backend.cancel_job.side_effect = error
        result = ExecutionResult(job_id="job_z")
        with caplog.at_level(logging.DEBUG, logger="divi.backends._cancellation"):
            _best_effort_cancel_job(backend, result)
        backend.cancel_job.assert_called_once_with(result)
        (record,) = caplog.records
        assert record.levelno == logging.DEBUG
        assert record.getMessage() == "Best-effort cancel_job failed for job_z"
        assert record.exc_info[1] is error


class TestAutoCancellationScope:
    """Tests for ``_auto_cancellation_scope``: bundles the SIGINT funnel with
    best-effort remote-job cleanup for direct callers of ``poll_job_status``."""

    def test_cancels_backend_on_execution_cancelled(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")

        with pytest.raises(ExecutionCancelledError):
            with _auto_cancellation_scope(backend, result):
                raise ExecutionCancelledError("polling cancelled")

        backend.cancel_job.assert_called_once_with(result)

    def test_does_not_cancel_on_unrelated_exceptions(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")

        with pytest.raises(RuntimeError, match="not a cancellation"):
            with _auto_cancellation_scope(backend, result):
                raise RuntimeError("not a cancellation")

        backend.cancel_job.assert_not_called()

    def test_yields_a_fresh_unset_event(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)
        with _auto_cancellation_scope(
            backend, ExecutionResult(job_id="job_x")
        ) as event:
            assert isinstance(event, Event)
            assert not event.is_set()
