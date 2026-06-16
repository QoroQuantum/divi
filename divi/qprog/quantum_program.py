# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import contextmanager
from queue import Queue
from threading import Event
from typing import Any
from warnings import warn

from divi.backends import AsyncJobBackend, CircuitRunner
from divi.backends._cancellation import _best_effort_cancel_job, _sigint_to_event
from divi.circuits import DEFAULT_PRECISION
from divi.circuits.qem import _NoMitigation
from divi.pipeline import (
    CircuitPipeline,
    DryRunReport,
    PipelineEnv,
    PipelineSet,
    ResultFormat,
    Stage,
    dry_run_pipeline,
)
from divi.pipeline.stages import PauliTwirlStage, QEMStage
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
        precision: int = DEFAULT_PRECISION,
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
            precision (int): Decimal places for numeric parameter values in
                QASM emission.  Higher values produce longer QASM strings;
                lower values shrink them at the cost of parameter resolution.
                Defaults to :data:`~divi.circuits.DEFAULT_PRECISION`.
        """
        if backend is None:
            raise ValueError("QuantumProgram requires a backend.")

        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue
        self._precision = precision

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
        self._evaluation_counter = 0

        # --- Progress Reporting ---
        self.program_id = program_id
        self.reporter: ProgressReporter
        if progress_queue and self.program_id is not None:
            self.reporter = QueueProgressReporter(self.program_id, progress_queue)
        else:
            self.reporter = LoggingProgressReporter()

    @property
    def precision(self) -> int:
        """Decimal places used for numeric parameter values in QASM emission."""
        return self._precision

    @property
    def total_circuit_count(self) -> int:
        """Cumulative count of circuits submitted for execution across all
        runs of this program."""
        return self._total_circuit_count

    @property
    def total_run_time(self) -> float:
        """Cumulative backend execution time in seconds across all runs of
        this program."""
        return self._total_run_time

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

    @contextmanager
    def _install_cancellation_handler(self):
        """Funnel SIGINT into :attr:`_cancellation_event` for the duration of ``run()``.

        First Ctrl+C sets the event; second Ctrl+C hard-aborts via
        ``KeyboardInterrupt``. No-op outside the main thread or when a
        non-default SIGINT handler is already installed — defers to
        debuggers, Jupyter, and enclosing ensemble handlers.
        """
        if self._cancellation_event is None:
            self._cancellation_event = Event()
        with _sigint_to_event(self._cancellation_event):
            yield

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

        _best_effort_cancel_job(self.backend, result)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_pipelines(self) -> PipelineSet:
        """Build this program's named pipeline registry.

        Subclasses and mixins extend a base set cooperatively via
        ``super()._build_pipelines().with_(name, pipeline, seed_factory)``.
        """
        ...

    def _assemble_pipeline(
        self,
        spec_stage: Stage,
        terminal_stage: Stage,
        *,
        result_format: ResultFormat,
        extra_stages: tuple[Stage, ...] = (),
    ) -> CircuitPipeline:
        """Assemble one pipeline from this program's shared error-mitigation config.

        Ordering: ``spec_stage`` → ``extra_stages`` → [QEM (→ PauliTwirl) when
        applicable] → ``terminal_stage``.

        Parameter-binding policy belongs to parameterized program classes, not
        this shared assembler.
        """
        stages = [
            spec_stage,
            *extra_stages,
            *self._mitigation_stages(result_format),
            terminal_stage,
        ]
        return CircuitPipeline(stages=stages)

    def _mitigation_stages(self, result_format: ResultFormat) -> tuple[Stage, ...]:
        """Build the QEM and optional Pauli-twirling stages for a result format."""
        if isinstance(self._qem_protocol, _NoMitigation):
            return ()
        if not self._qem_protocol.applies_to(result_format):
            return ()

        stages: list[Stage] = [QEMStage(protocol=self._qem_protocol)]
        if self._qem_protocol.n_twirls > 0:
            stages.append(PauliTwirlStage(n_twirls=self._qem_protocol.n_twirls))
        return tuple(stages)

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
                self._pipelines.spec_for(name),
                env,
                bypass_cache=True,
                dry=not force_circuit_generation,
            )
            reports[name] = dry_run_pipeline(name, trace, pipeline.stages, env)
        return reports

    def _build_pipeline_env(self, **overrides) -> PipelineEnv:
        """Construct a :class:`PipelineEnv` from the current program state.

        Subclasses may override to inject additional fields (e.g. ``param_sets``
        in :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`).
        """
        env_kwargs = {
            "backend": self.backend,
            "reporter": self.reporter,
            "cancellation_event": self._cancellation_event,
            "evaluation_counter": self._evaluation_counter,
        }
        env_kwargs.update(overrides)  # caller-supplied values win
        return PipelineEnv(**env_kwargs)

    def _execute(self, pipeline: CircuitPipeline, initial_spec: Any, **env_overrides):
        """Run ``pipeline`` and fold its execution artifacts into program totals.

        The shared run+accounting core. ``env_overrides`` are forwarded to
        :meth:`_build_pipeline_env`. Returns the raw pipeline result.
        """
        env = self._build_pipeline_env(**env_overrides)
        result = pipeline.run(initial_spec=initial_spec, env=env)
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)
        self._current_execution_result = env.artifacts.get("_current_execution_result")
        return result

    def _run_pipeline(self, name: str, *, initial_spec: Any = None, **env_overrides):
        """Run a registered pipeline by ``name`` (resolving its seed from the
        registry unless ``initial_spec`` is given). Thin wrapper over :meth:`_execute`.
        """
        if initial_spec is None:
            initial_spec = self._pipelines.spec_for(name)
        return self._execute(self._pipelines[name], initial_spec, **env_overrides)

    def _pipeline_source_batch(self, name: str):
        """Return a registered pipeline's spec-stage circuit batch.

        Recomputed deterministically from ``env.evaluation_counter`` so it
        reproduces the cost pipeline's sampled batch for the current evaluation.
        """
        pipeline = self._pipelines[name]
        env = self._build_pipeline_env()
        return pipeline.run_spec_stage(self._pipelines.spec_for(name), env).batch
