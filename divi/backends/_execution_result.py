# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExecutionResult:
    """Result container for circuit execution.

    This class provides a unified return type for all CircuitRunner.submit_circuits()
    methods. For synchronous backends, it contains the results directly. For
    asynchronous backends, it contains the job_id that can be used to fetch results later.

    The class is frozen (immutable) to ensure data integrity. Use the `with_results()`
    method to create a new instance with results populated from an async ExecutionResult.

    Attributes:
        results (list[dict] | None): For sync backends or after fetching: List of result
            dictionaries, each containing 'label' and 'results' keys. Format:
            [{"label": str, "results": dict}, ...]
        job_id (str | None): For async backends: Job identifier that can be used
            to poll and retrieve results from the backend.

    Examples:
        >>> # Synchronous backend
        >>> result = ExecutionResult(results=[{"label": "circuit_0", "results": {"00": 100}}])
        >>> result.is_async()
        False

        >>> # Asynchronous backend
        >>> result = ExecutionResult(job_id="job-12345")
        >>> result.is_async()
        True
        >>> # After fetching results
        >>> result = backend.get_job_results(result)
        >>> result.results is not None
        True
    """

    results: list[dict] | None = None
    """Results for synchronous backends: [{"label": str, "results": dict}, ...]"""

    job_id: str | None = None
    """Job identifier for asynchronous backends."""

    def is_async(self) -> bool:
        """Check if this result represents an async job.

        Returns:
            bool: True if job_id is not None and results are None (async backend),
                False otherwise (sync backend or results already fetched).
        """
        return self.job_id is not None and self.results is None

    def with_results(self, results: list[dict]) -> "ExecutionResult":
        """Create a new ExecutionResult with results populated.

        This method creates a new instance with results set, effectively converting
        an async ExecutionResult to a completed one.

        Args:
            results: The job results to populate.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results populated
                and job_id preserved.
        """
        return replace(self, results=results)
