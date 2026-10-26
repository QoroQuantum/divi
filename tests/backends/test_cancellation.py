# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for divi.backends._cancellation."""

import logging

import pytest

from divi.backends import AsyncJobBackend, ExecutionResult, JobFailedError
from divi.backends._cancellation import (
    _best_effort_cancel_job,
    _cancel_job_on_abort,
    _require_async_backend,
)
from divi.exceptions import ExecutionCancelledError
from tests._helpers import exact_match


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


class TestCancelJobOnAbort:
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RuntimeError("poll failed"), id="error"),
            pytest.param(ExecutionCancelledError("stop"), id="cancelled"),
            pytest.param(KeyboardInterrupt(), id="interrupt"),
        ],
    )
    def test_cancels_the_job_on_a_non_terminal_exit(self, mocker, error):
        backend = mocker.Mock(spec=AsyncJobBackend)
        result = ExecutionResult(job_id="job_x")

        with pytest.raises(type(error)):
            with _cancel_job_on_abort(backend, result):
                raise error

        backend.cancel_job.assert_called_once_with(result)

    def test_leaves_a_terminal_job_error_alone(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)

        with pytest.raises(JobFailedError):
            with _cancel_job_on_abort(backend, ExecutionResult(job_id="job_x")):
                raise JobFailedError("job_x")

        backend.cancel_job.assert_not_called()

    def test_does_not_cancel_when_the_block_succeeds(self, mocker):
        backend = mocker.Mock(spec=AsyncJobBackend)

        with _cancel_job_on_abort(backend, ExecutionResult(job_id="job_x")):
            pass

        backend.cancel_job.assert_not_called()


class TestRequireAsyncBackend:
    def test_returns_an_async_backend(self, async_backend):
        assert _require_async_backend(async_backend) is async_backend

    def test_rejects_a_backend_that_is_not_asynchronous(self, dummy_simulator):
        with pytest.raises(
            RuntimeError,
            match=exact_match(
                "Backend DummySimulator returned an ExecutionResult with a job_id "
                "but is not an AsyncJobBackend."
            ),
        ):
            _require_async_backend(dummy_simulator)
