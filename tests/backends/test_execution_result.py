# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from divi.backends import ExecutionResult


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        res = ExecutionResult()
        assert res.results is None
        assert res.job_id is None
        assert res.is_async() is False

    def test_init_sync(self):
        """Test initialization for synchronous result."""
        results = [{"label": "test", "results": {}}]
        res = ExecutionResult(results=results)
        assert res.results == results
        assert res.job_id is None
        assert res.is_async() is False

    def test_init_async(self):
        """Test initialization for asynchronous result."""
        res = ExecutionResult(job_id="job-123")
        assert res.results is None
        assert res.job_id == "job-123"
        assert res.is_async() is True

    def test_immutability(self):
        """Test that ExecutionResult is immutable."""
        res = ExecutionResult(job_id="job-123")
        with pytest.raises(FrozenInstanceError):
            res.job_id = "job-456"
        with pytest.raises(FrozenInstanceError):
            res.results = []

    def test_with_results(self):
        """Test with_results method."""
        res_async = ExecutionResult(job_id="job-123")
        results = [{"label": "test", "results": {}}]

        res_completed = res_async.with_results(results)

        # Check new instance
        assert res_completed is not res_async
        assert res_completed.job_id == "job-123"
        assert res_completed.results == results
        assert res_completed.is_async() is False

        # Check original instance is unchanged
        assert res_async.results is None
        assert res_async.is_async() is True

    def test_is_async_edge_cases(self):
        """Test is_async with various combinations."""
        # Both None -> False (Sync with no results? Or just empty)
        assert ExecutionResult(results=None, job_id=None).is_async() is False

        # Both set -> False (Async job that has finished and results fetched)
        assert ExecutionResult(results=[], job_id="id").is_async() is False
