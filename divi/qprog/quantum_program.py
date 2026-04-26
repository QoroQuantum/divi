# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from http import HTTPStatus
from queue import Queue
from threading import Event
from typing import Any
from warnings import warn

import requests

from divi.backends import AsyncJobBackend, CircuitRunner
from divi.circuits.qem import _NoMitigation
from divi.pipeline import CircuitPipeline, DryRunReport, PipelineEnv, dry_run_pipeline
from divi.reporting import (
    LoggingProgressReporter,
    ProgressReporter,
    QueueProgressReporter,
)


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    Subclasses must implement:
        - run(): Execute the quantum algorithm

    Attributes:
        backend: The quantum circuit execution backend.
        _seed: Random seed for reproducible results.
        _progress_queue: Queue for progress reporting.
    """

    def __init__(
        self,
        backend: CircuitRunner,
        seed: int | None = None,
        progress_queue: Queue | None = None,
        program_id: str | None = None,
        **kwargs,
    ):
        """Initialize the QuantumProgram.

        Args:
            backend (CircuitRunner): Quantum circuit execution backend.
            seed (int | None): Random seed for reproducible results. Defaults to None.
            progress_queue (Queue | None): Queue for progress reporting. Defaults to None.
            program_id (str | None): Program identifier for progress reporting in
                batch operations. If provided along with progress_queue, enables
                queue-based progress reporting.
        """
        if backend is None:
            raise ValueError("QuantumProgram requires a backend.")

        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue

        qem_protocol = kwargs.pop("qem_protocol", None)
        self._qem_protocol = _NoMitigation() if qem_protocol is None else qem_protocol

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._curr_circuits = []
        self._current_execution_result = None
        self._cancellation_event = None

        # --- Progress Reporting ---
        self.program_id = program_id
        self.reporter: ProgressReporter
        if progress_queue and self.program_id is not None:
            self.reporter = QueueProgressReporter(self.program_id, progress_queue)
        else:
            self.reporter = LoggingProgressReporter()

    @abstractmethod
    def run(self, **kwargs) -> "QuantumProgram":
        """Execute the quantum algorithm.

        Args:
            **kwargs: Additional keyword arguments for subclasses.

        Returns:
            QuantumProgram: Returns ``self`` for method chaining.
        """

    @abstractmethod
    def has_results(self) -> bool:
        """Return True once the program has produced results."""

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

        This is typically called by ProgramEnsemble when handling cancellation
        to ensure cloud jobs are cancelled before local threads terminate.
        """
        result = self._current_execution_result

        if result is None:
            warn("Cannot cancel job: no current execution result", stacklevel=2)
            return

        if result.job_id is None:
            warn("Cannot cancel job: execution result has no job_id", stacklevel=2)
            return

        if not isinstance(self.backend, AsyncJobBackend):
            warn(
                f"Cannot cancel job: backend {type(self.backend).__name__} "
                "does not implement the AsyncJobBackend protocol.",
                stacklevel=2,
            )
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
    def _build_pipelines(self) -> dict[str, CircuitPipeline]:
        """Build and return this program's pipelines keyed by stable name.

        The returned dict is the **single source of truth** for which
        pipelines this program owns — :meth:`dry_run` and every other
        introspection surface iterate it, using the dict key as the
        pipeline's label.  Subclasses call this method from ``__init__``
        and assign the result to ``self._pipelines``.

        Pipelines should be **pure structural** here (stage composition
        only); anything that depends on program state resolved at run
        time — observables, parameter-bound meta-circuits, Hamiltonians —
        belongs in :meth:`_get_initial_spec`, which is called lazily.
        """
        ...

    def _get_initial_spec(self, name: str) -> Any:
        """Return the ``initial_spec`` for pipeline ``name`` — typed to match
        the input expected by that pipeline's :class:`~divi.pipeline.abc.SpecStage`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_initial_spec() "
            f"to support dry_run()."
        )

    def dry_run(
        self, *, force_circuit_generation: bool = False
    ) -> dict[str, DryRunReport]:
        """Run forward pass on all pipelines and return a fan-out analysis.

        Traverses each pipeline stage without executing circuits and collects
        the fan-out factor, per-stage metadata, and total circuit count into a
        dict of :class:`~divi.pipeline.DryRunReport` objects keyed by the
        names registered in :meth:`_build_pipelines`. Pass the returned dict
        to :func:`~divi.pipeline.format_dry_run` for the pretty tree output.

        Uses the analytic dry path by default; see the user guide for how
        that trades circuit generation for multiplicative-factor
        bookkeeping.

        Args:
            force_circuit_generation: If ``True``, force every stage to run
                its full ``expand`` path so that the trace contains real
                DAGs and QASM strings. Useful when inspecting actual
                pipeline output (e.g. debugging a stage's circuit
                transformation). Defaults to ``False``.

        Example:
            >>> from divi.pipeline import format_dry_run
            >>> reports = program.dry_run()
            >>> format_dry_run(reports)  # pretty-print to stdout
        """
        self._pipelines = self._build_pipelines()
        reports: dict[str, DryRunReport] = {}
        for name, pipeline in self._pipelines.items():
            env = self._build_pipeline_env()
            trace = pipeline.run_forward_pass(
                self._get_initial_spec(name),
                env,
                force_forward_sweep=True,
                dry=not force_circuit_generation,
            )
            reports[name] = dry_run_pipeline(name, trace, pipeline.stages, env)
        return reports

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a :class:`PipelineEnv` from the current program state.

        Subclasses may override to inject additional fields (e.g. ``param_sets``
        in :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`).
        """
        return PipelineEnv(
            backend=self.backend,
            reporter=self.reporter,
            cancellation_event=self._cancellation_event,
            **overrides,
        )
