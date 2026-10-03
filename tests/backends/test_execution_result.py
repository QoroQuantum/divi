# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends import ExecutionResult

_RESULTS = [{"label": "test", "results": {}}]


@pytest.mark.parametrize(
    "kwargs, is_async",
    [
        pytest.param({}, False, id="defaults"),
        pytest.param({"results": _RESULTS}, False, id="sync"),
        pytest.param({"job_id": "job-123"}, True, id="async"),
        pytest.param({"results": [], "job_id": "id"}, False, id="async-fetched"),
    ],
)
def test_construction(kwargs, is_async):
    """Only a job id without results is still asynchronous; one backend job by default."""
    res = ExecutionResult(**kwargs)

    assert res.results == kwargs.get("results")
    assert res.job_id == kwargs.get("job_id")
    assert res.backend_jobs == 1
    assert res.is_async() is is_async


def test_with_results():
    res_async = ExecutionResult(job_id="job-123", backend_jobs=3)

    res_completed = res_async.with_results(_RESULTS)

    assert res_completed is not res_async
    assert res_completed.job_id == "job-123"
    assert res_completed.backend_jobs == 3
    assert res_completed.results == _RESULTS
    assert res_completed.run_time == 0.0
    assert res_completed.is_async() is False
    assert res_async.with_results(_RESULTS, run_time=2.5).run_time == 2.5

    # The original instance is unchanged.
    assert res_async.results is None
    assert res_async.is_async() is True
