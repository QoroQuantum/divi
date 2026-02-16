# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from http import HTTPStatus
from queue import Queue
from threading import Event
from warnings import warn

import requests

from divi.backends import CircuitRunner
from divi.pipeline import PipelineEnv
from divi.reporting import LoggingProgressReporter, QueueProgressReporter


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    Subclasses must implement:
        - run(): Execute the quantum algorithm

    Attributes:
        backend (CircuitRunner): The quantum circuit execution backend.
        _seed (int | None): Random seed for reproducible results.
        _progress_queue (Queue | None): Queue for progress reporting.
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

    def cancel_unfinished_job(self):
        """Cancel the currently running cloud job if one exists.

        This method attempts to cancel the job associated with the current
        ExecutionResult. It is best-effort and will log warnings for any errors
        (e.g., job already completed, permission denied) without raising exceptions.

        This is typically called by ProgramBatch when handling cancellation
        to ensure cloud jobs are cancelled before local threads terminate.
        """
        result = self._current_execution_result

        if result is None:
            warn("Cannot cancel job: no current execution result", stacklevel=2)
            return

        if result.job_id is None:
            warn("Cannot cancel job: execution result has no job_id", stacklevel=2)
            return

        try:
            self.backend.cancel_job(result)
        except requests.exceptions.HTTPError as e:
            # Check if this is an expected error (job already completed/failed/cancelled)
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == HTTPStatus.CONFLICT
            ):
                # 409 Conflict means job is already in a terminal state - this is expected
                # in race conditions where job completes before we can cancel it.
                self.reporter.info(
                    f"Job {result.job_id} already completed or cancelled"
                )
            else:
                # Unexpected error (403 Forbidden, 404 Not Found, etc.) - report it
                self.reporter.info(f"Failed to cancel job {result.job_id}: {e}")
        except Exception as e:
            # Other unexpected errors - report them
            self.reporter.info(f"Failed to cancel job {result.job_id}: {e}")

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_pipelines(self) -> None:
        """Construct all :class:`CircuitPipeline` instances for this program.

        Every subclass must implement this to set up its pipeline attributes
        (e.g. ``self._pipeline``, ``self._cost_pipeline``).
        """
        ...

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a :class:`PipelineEnv` from the current program state.

        Subclasses may override to inject additional fields (e.g. ``param_sets``
        in :class:`VariationalQuantumAlgorithm`).
        """
        return PipelineEnv(
            backend=self.backend,
            reporter=getattr(self, "reporter", None),
            cancellation_event=getattr(self, "_cancellation_event", None),
            **overrides,
        )
