# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from abc import ABC, abstractmethod
from http import HTTPStatus
from queue import Queue
from threading import Event
from typing import Any
from warnings import warn

import requests

from divi.backends import CircuitRunner, JobStatus
from divi.backends._execution_result import ExecutionResult
from divi.circuits import CircuitBundle, CircuitTag, format_circuit_tag
from divi.qprog.exceptions import _CancelledError
from divi.reporting import LoggingProgressReporter, QueueProgressReporter


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    This class defines the interface and provides common functionality for quantum algorithms.
    It handles circuit execution, result processing, and data persistence.

    Subclasses must implement:
        - run(): Execute the quantum algorithm
        - _generate_circuits(): Generate quantum circuits for execution
        - _post_process_results(): Process execution results

    Attributes:
        backend (CircuitRunner): The quantum circuit execution backend.
        _seed (int | None): Random seed for reproducible results.
        _progress_queue (Queue | None): Queue for progress reporting.
        _circuits (list): List of circuits to be executed.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        **kwargs,
    ):
        """Initialize the QuantumProgram.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            seed (int | None): Random seed for reproducible results. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.
            **kwargs: Additional keyword arguments for subclasses.
                program_id (str | None): Program identifier for progress reporting in batch
                operations. If provided along with progress_queue, enables queue-based
                progress reporting.
        """
        if backend is None:
            raise ValueError("QuantumProgram requires a backend.")

        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue
        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._curr_circuits = []
        self._current_execution_result = None

        # --- Progress Reporting ---
        self.program_id = kwargs.get("program_id", None)
        if progress_queue and self.program_id is not None:
            self.reporter = QueueProgressReporter(self.program_id, progress_queue)
        else:
            self.reporter = LoggingProgressReporter()

    @abstractmethod
    def run(self, **kwargs) -> tuple[int, float]:
        """Execute the quantum algorithm.

        Args:
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            tuple[int, float]: A tuple containing:
                - int: Total number of circuits executed
                - float: Total runtime in seconds
        """
        pass

    @abstractmethod
    def _generate_circuits(self, **kwargs) -> list[CircuitBundle]:
        """Generate quantum circuits for execution.

        This method should generate and return a list of CircuitBundle objects based on
        the current algorithm state. The circuits will be executed by the backend.

        Args:
            **kwargs: Additional keyword arguments for circuit generation.

        Returns:
            list[CircuitBundle]: List of CircuitBundle objects to be executed.
        """
        pass

    @abstractmethod
    def _post_process_results(self, results: dict, **kwargs) -> Any:
        """Process execution results.

        Args:
            results (dict): Raw results from circuit execution.

        Returns:
            Any: Processed results specific to the algorithm.
        """
        pass

    def _set_cancellation_event(self, event: Event):
        """Set a cancellation event for graceful program termination.

        This method is called by batch runners to provide a mechanism
        for stopping the optimization loop cleanly when requested.

        Args:
            event (Event): Threading Event object that signals cancellation when set.
        """
        self._cancellation_event = event

    @property
    def total_circuit_count(self) -> int:
        """Get the total number of circuits executed.

        Returns:
            int: Cumulative count of circuits submitted for execution.
        """
        return self._total_circuit_count

    @property
    def total_run_time(self) -> float:
        """Get the total runtime across all circuit executions.

        Returns:
            float: Cumulative execution time in seconds.
        """
        return self._total_run_time

    def _prepare_and_send_circuits(self, **kwargs) -> ExecutionResult:
        """Prepare circuits for execution and submit them to the backend.

        Returns:
            ExecutionResult: Result from circuit submission. For async backends,
                contains job_id. For sync backends, contains results directly.
        """
        job_circuits = {}
        self._reset_tag_cache()

        for bundle in self._curr_circuits:
            for executable in bundle.executables:
                job_circuits[self._encode_tag(executable.tag)] = executable.qasm

        self._total_circuit_count += len(job_circuits)

        execution_result = self.backend.submit_circuits(job_circuits, **kwargs)

        return execution_result

    def _track_runtime(self, response):
        """Extract and track runtime from a backend response.

        Args:
            response: Backend response containing runtime information.
                Can be a dict or a list of responses.
        """
        if isinstance(response, dict):
            self._total_run_time += float(response["run_time"])
        elif isinstance(response, list):
            self._total_run_time += sum(float(r.json()["run_time"]) for r in response)

    def _wait_for_qoro_job_completion(
        self, execution_result: ExecutionResult
    ) -> list[dict]:
        """Wait for a QoroService job to complete and return results.

        Args:
            execution_result: The ExecutionResult from circuit submission.

        Returns:
            list[dict]: The job results from the backend.

        Raises:
            Exception: If job fails or doesn't complete.
        """
        job_id = execution_result.job_id
        if job_id is None:
            raise ValueError("ExecutionResult must have a job_id for async completion")

        # Build the poll callback if reporter is available
        if hasattr(self, "reporter"):
            update_function = lambda n_polls, status: self.reporter.info(
                message="",
                poll_attempt=n_polls,
                max_retries=self.backend.max_retries,
                service_job_id=job_id,
                job_status=status,
            )
        else:
            update_function = None

        # Poll until complete
        status = self.backend.poll_job_status(
            execution_result,
            loop_until_complete=True,
            on_complete=self._track_runtime,
            verbose=False,  # Disable the default logger in QoroService
            progress_callback=update_function,
        )

        if status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job_id} has failed")

        if status == JobStatus.CANCELLED:
            # If cancellation was requested (e.g., by ProgramBatch), raise _CancelledError
            # so it's handled gracefully. Otherwise, raise RuntimeError for unexpected cancellation.
            if (
                hasattr(self, "_cancellation_event")
                and self._cancellation_event
                and self._cancellation_event.is_set()
            ):
                raise _CancelledError(f"Job {job_id} was cancelled")
            raise RuntimeError(f"Job {job_id} was cancelled")

        if status != JobStatus.COMPLETED:
            raise Exception("Job has not completed yet, cannot post-process results")
        completed_result = self.backend.get_job_results(execution_result)
        return completed_result.results

    def cancel_unfinished_job(self):
        """Cancel the currently running cloud job if one exists.

        This method attempts to cancel the job associated with the current
        ExecutionResult. It is best-effort and will log warnings for any errors
        (e.g., job already completed, permission denied) without raising exceptions.

        This is typically called by ProgramBatch when handling cancellation
        to ensure cloud jobs are cancelled before local threads terminate.
        """

        if self._current_execution_result is None:
            warn("Cannot cancel job: no current execution result", stacklevel=2)
            return

        if self._current_execution_result.job_id is None:
            warn("Cannot cancel job: execution result has no job_id", stacklevel=2)
            return

        try:
            self.backend.cancel_job(self._current_execution_result)
        except requests.exceptions.HTTPError as e:
            # Check if this is an expected error (job already completed/failed/cancelled)
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == HTTPStatus.CONFLICT
            ):
                # 409 Conflict means job is already in a terminal state - this is expected
                # in race conditions where job completes before we can cancel it.
                if hasattr(self, "reporter"):
                    self.reporter.info(
                        f"Job {self._current_execution_result.job_id} already completed or cancelled"
                    )
            else:
                # Unexpected error (403 Forbidden, 404 Not Found, etc.) - report it
                if hasattr(self, "reporter"):
                    self.reporter.info(
                        f"Failed to cancel job {self._current_execution_result.job_id}: {e}"
                    )
        except Exception as e:
            # Other unexpected errors - report them
            if hasattr(self, "reporter"):
                self.reporter.info(
                    f"Failed to cancel job {self._current_execution_result.job_id}: {e}"
                )

    def _dispatch_circuits_and_process_results(self, **kwargs):
        """Run an iteration of the program.

        The outputs are stored in the Program object.

        Args:
            **kwargs: Additional keyword arguments for circuit submission and result processing.

        Returns:
            Any: Processed results from _post_process_results.
        """
        execution_result = self._prepare_and_send_circuits(**kwargs)

        # Store the execution result for potential cancellation
        self._current_execution_result = execution_result

        try:
            # For async backends, poll for results
            if execution_result.job_id is not None:
                results = self._wait_for_qoro_job_completion(execution_result)
            else:
                # For sync backends, results are already available
                results = execution_result.results
                if results is None:
                    raise ValueError("ExecutionResult has neither results nor job_id")

            results = {r["label"]: r["results"] for r in results}
            results = {self._parse_tag(k): v for k, v in results.items()}

            result = self._post_process_results(results, **kwargs)

            return result
        finally:
            # Clear the execution result after processing
            self._current_execution_result = None

    def _reset_tag_cache(self) -> None:
        """Hook to reset per-run tag caches. Default is no-op."""

    @staticmethod
    def _parse_tag(tag: str) -> CircuitTag:
        """Parse a tag string to CircuitTag. Raises ValueError if format is invalid."""
        m = re.match(r"^(\d+)_([^:]+):(\d+)_ham:(-?\d+)_(\d+)$", str(tag))

        if not m:
            raise ValueError(f"Cannot parse tag: {tag!r}")
        return CircuitTag(
            param_id=int(m.group(1)),
            qem_name=m.group(2),
            qem_id=int(m.group(3)),
            meas_id=int(m.group(5)),
            hamiltonian_id=int(m.group(4)),
        )

    @staticmethod
    def _encode_tag(tag: CircuitTag) -> str:
        """Convert a tag to a backend-safe string."""
        return format_circuit_tag(tag)

    @staticmethod
    def _group_results_by_tag(
        results: dict[CircuitTag, dict[str, int]],
    ) -> dict[int, dict[int, dict[tuple[str, int], list[dict[str, int]]]]]:
        """Group results by parameter id, Hamiltonian sample id, and QEM key.

        Returns:
            dict[int, dict[int, dict[tuple[str, int], list[dict[str, int]]]]]:
                Nested mapping in the shape
                ``{param_id: {hamiltonian_id: {(qem_name, qem_id): [shots_by_meas_id]}}}``.
        """
        grouped: dict[
            int,
            dict[int, dict[tuple[str, int], list[tuple[int, dict[str, int]]]]],
        ] = {}
        for tag, shots in results.items():
            qem_key = (tag.qem_name, tag.qem_id)
            grouped.setdefault(tag.param_id, {}).setdefault(
                tag.hamiltonian_id, {}
            ).setdefault(qem_key, []).append((tag.meas_id, shots))

        return {
            param_id: {
                ham_id: {
                    qem_key: [
                        shots
                        for _, shots in sorted(meas_shots, key=lambda item: item[0])
                    ]
                    for qem_key, meas_shots in qem_dict.items()
                }
                for ham_id, qem_dict in ham_dict.items()
            }
            for param_id, ham_dict in grouped.items()
        }

    @staticmethod
    def _merge_shot_histograms(shots_dicts: list[dict[str, int]]) -> dict[str, int]:
        """Merge multiple shot histograms into a single histogram."""
        merged_counts: dict[str, int] = {}
        for shots_dict in shots_dicts:
            for bitstring, count in shots_dict.items():
                merged_counts[bitstring] = merged_counts.get(bitstring, 0) + count
        return merged_counts

    @staticmethod
    def _average_probabilities(
        probs_per_group: list[dict[str, float]],
    ) -> dict[str, float]:
        """Average probability dictionaries over all observed bitstrings."""
        if not probs_per_group:
            return {}

        all_bitstrings = set()
        for probs in probs_per_group:
            all_bitstrings.update(probs.keys())

        n_groups = len(probs_per_group)
        return {
            bitstring: sum(p.get(bitstring, 0.0) for p in probs_per_group) / n_groups
            for bitstring in all_bitstrings
        }
