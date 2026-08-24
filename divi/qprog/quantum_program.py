# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Hashable
from contextlib import contextmanager
from queue import Queue
from threading import Event
from typing import Any, Self
from warnings import warn

import numpy as np

from divi.backends import AsyncJobBackend, CircuitRunner
from divi.backends._cancellation import _best_effort_cancel_job, _sigint_to_event
from divi.circuits import DEFAULT_PRECISION
from divi.circuits.qem import _NoMitigation
from divi.pipeline import (
    CircuitPipeline,
    CircuitPreprocessor,
    DryRunReport,
    PipelineEnv,
    PipelineResult,
    ResultFormat,
    Stage,
    dry_run_pipeline,
)
from divi.pipeline._result_keys_operations import extract_param_set_idx
from divi.pipeline.stages import (
    CircuitSpecStage,
    MeasurementStage,
    PauliTwirlStage,
    PreprocessStage,
    QEMStage,
)
from divi.reporting import (
    LoggingProgressReporter,
    ProgressReporter,
    QueueProgressReporter,
)


def reject_unclaimed_run_kwargs(program: object, kwargs: dict[str, Any]) -> None:
    """Reject leftover ``run`` keywords, before anything is submitted.

    ``run`` forwards ``**kwargs`` through several layers, each popping what it
    consumes, so whatever remains is claimed by nobody — a misspelling, or a flag
    the caller believes is doing something. Call this once the layer has taken its
    own arguments and before any circuit goes out: checking afterwards would
    report the mistake having already paid for the run it invalidates.
    """
    if not kwargs:
        return
    message = (
        f"{type(program).__name__}.run() got unexpected keyword argument(s): "
        f"{', '.join(sorted(kwargs))}."
    )
    if "dry_run" in kwargs:
        # A flag on the action is the near-universal convention, so this is the
        # reflex to catch: silently ignoring it would execute the run the caller
        # was trying to avoid.
        message += (
            " To preview without executing anything, call the separate method "
            "instead: program.dry_run()."
        )
    raise TypeError(message)


class QuantumProgram(ABC):
    """Abstract base class for quantum programs.

    Subclasses must implement:
        - run(): Execute the quantum algorithm

    **Program-author extension hooks** (override as needed):

    - ``_spec_stage()`` — Returns the :class:`~divi.pipeline.SpecStage` that
      converts the seed into a :class:`~divi.circuits.MetaCircuit` batch.
      Default returns :class:`~divi.pipeline.stages.CircuitSpecStage`;
      override with :class:`~divi.pipeline.stages.TrotterSpecStage` for
      Hamiltonian-seeded programs (QAOA, TimeEvolution).

    - ``_initial_spec()`` — Returns the seed passed into ``_spec_stage()``.
      The base raises ``NotImplementedError``; any subclass that calls
      :meth:`evaluate` must implement it.
      :class:`~divi.qprog.VariationalQuantumAlgorithm` returns its cost
      ansatz; QAOA / TimeEvolution return their Hamiltonian.  Programs that
      never call ``evaluate`` (e.g. those that assemble their own pipeline
      inside ``run``) do not need to implement this method.

    Note: ``_spec_stage`` and ``_initial_spec`` are intentionally not
    abstract — ``_spec_stage`` has a working default, and ``_initial_spec``
    is only required by programs that use ``evaluate``.

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
        suppress_performance_warnings: bool = False,
        **kwargs,
    ):
        """Initialise the QuantumProgram.

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
            suppress_performance_warnings (bool): Silence
                :class:`~divi.pipeline.DiviPerformanceWarning` from this
                program's pipelines (e.g. exhaustive-QuEPP scaling or
                bind-before-mitigation hints). Defaults to False.
        """
        if backend is None:
            raise ValueError("QuantumProgram requires a backend.")

        self.backend = backend
        self._seed = seed
        self._progress_queue = progress_queue
        self._precision = precision
        self._suppress_performance_warnings = suppress_performance_warnings

        qem_protocol = kwargs.pop("qem_protocol", None)
        self._qem_protocol = _NoMitigation() if qem_protocol is None else qem_protocol

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        self._total_circuit_count = 0
        self._total_run_time = 0.0
        self._current_execution_result = None
        self._last_cost_variance = None
        self._cancellation_event = None
        self._evaluation_counter = 0

        # Pipelines memoized per preprocessor ``cache_key`` so each one's
        # forward-pass cache persists across optimizer iterations. Preprocessors
        # with ``cache_key is None`` are never stored here.
        self._preprocessor_pipeline_cache: dict[Hashable, CircuitPipeline] = {}

        # Fixed seed for unseeded stochastic stages: the explicit ``seed`` when
        # given, otherwise drawn once so it stays stable across a program's
        # evaluations (random across separate programs/runs).
        self._base_seed = (
            self._seed
            if self._seed is not None
            else int(np.random.default_rng().integers(0, 2**63))
        )

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
    def run(self, **kwargs) -> Self:
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
        for stopping the optimisation loop cleanly when requested.

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

    def _preprocessors(self) -> tuple[CircuitPreprocessor, ...]:
        """The measurement routines this program exposes for introspection
        (:meth:`dry_run`). Subclasses and mixins extend cooperatively via
        ``(*super()._preprocessors(), ...)``. Per-evaluation metric/overlap routines
        are intentionally excluded — they are driven directly by optimizers, not
        enumerated here.
        """
        return ()

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

        Parameter-binding policy belongs to parameterised program classes, not
        this shared assembler.
        """
        stages = [
            spec_stage,
            *extra_stages,
            *self._mitigation_stages(result_format),
            terminal_stage,
        ]
        return CircuitPipeline(
            stages=stages,
            suppress_performance_warnings=self._suppress_performance_warnings,
        )

    def _mitigation_stages(self, result_format: ResultFormat) -> tuple[Stage, ...]:
        """Build the QEM and optional Pauli-twirling stages for a result format."""
        if isinstance(self._qem_protocol, _NoMitigation):
            return ()
        if not self._qem_protocol.applies_to(result_format):
            return ()

        stages: list[Stage] = [QEMStage(protocol=self._qem_protocol)]
        if self._qem_protocol.n_twirls > 0:
            stages.append(
                PauliTwirlStage(
                    n_twirls=self._qem_protocol.n_twirls,
                    # Twirl labels are sampled; the program's seed makes the draw
                    # reproducible across calls.
                    seed=self._base_seed,
                )
            )
        return tuple(stages)

    def dry_run(
        self, *, force_circuit_generation: bool = False
    ) -> dict[str, DryRunReport]:
        """Traverse each pipeline without executing circuits and return a fan-out analysis.

        Collects the fan-out factor, per-stage metadata, and total circuit
        count into a dict of :class:`~divi.pipeline.DryRunReport` objects keyed
        by pipeline name. Covers the program's own routines and any auxiliary
        metric/overlap pipelines the optimizer drives per evaluation (see
        :meth:`_preprocessors`). Pass the returned dict to
        :func:`~divi.pipeline.format_dry_run` for the tree output.

        Uses the analytic dry path by default, trading circuit generation for
        multiplicative-factor bookkeeping (see the pipeline guide). The preview is
        side-effect free: it runs against a seeded throwaway RNG and touches no
        execution counters, so a later ``run()`` is unaffected.

        Args:
            force_circuit_generation: If ``True``, force every stage to run
                its full ``expand`` path so the trace contains real DAGs and
                QASM strings. Defaults to ``False``.

        Example:
            >>> from divi.pipeline import format_dry_run
            >>> reports = program.dry_run()
            >>> format_dry_run(reports)  # pretty-print to stdout
        """
        self._validate_before_preview()
        reports: dict[str, DryRunReport] = {}
        preprocessors = self._preprocessors()
        if not preprocessors:
            # No registered routines: skip _initial_spec() per its opt-out contract.
            return reports
        initial_spec = self._initial_spec()
        # Seeded throwaway RNG so the preview never advances the live generator.
        dry_rng = np.random.default_rng(self._base_seed)
        for preprocessor in preprocessors:
            key = preprocessor.name
            if key in reports:
                # The name is the report key, so a duplicate would hide a routine.
                suffix = 2
                while f"{key}#{suffix}" in reports:
                    suffix += 1
                warn(
                    f"Two routines on {type(self).__name__} are both named "
                    f"{preprocessor.name!r}; reporting the second as "
                    f"'{key}#{suffix}'. Give each routine a distinct name to "
                    "keep the report readable.",
                    UserWarning,
                    stacklevel=2,
                )
                key = f"{key}#{suffix}"
            pipeline = self._build_preprocessor_pipeline(preprocessor)
            env = self._dry_run_env(preprocessor, dry_rng)
            trace = pipeline.run_forward_pass(
                initial_spec,
                env,
                bypass_cache=True,
                dry=not force_circuit_generation,
            )
            reports[key] = dry_run_pipeline(
                key,
                trace,
                pipeline.stages,
                env,
                cadence=preprocessor.cadence,
            )
        return reports

    def _validate_before_preview(self) -> None:
        """Re-check state a preview should catch. Default checks nothing.

        Constructor validation can be outrun by later assignment, and catching a
        mismatch here is the whole point: the run that follows would fail after
        submitting circuits.
        """

    def _dry_run_env(
        self, preprocessor: CircuitPreprocessor, rng: "np.random.Generator"
    ) -> PipelineEnv:
        """The env used to preview one pipeline.

        ``reporter=None`` keeps the preview silent. Subclasses override to narrow
        the env per routine, since not every pipeline is driven with the same
        inputs (a one-time readout binds different parameters than the optimizer's
        working set).

        A parametric seed gets one zero-filled vector of its own width; without it
        the binding stage sees an empty one and reports ``n_params: 0``.
        """
        overrides: dict[str, Any] = {}
        n_params = self._routine_parameter_count(preprocessor)
        if n_params:
            overrides["param_sets"] = np.zeros((1, n_params))
        return self._build_pipeline_env(rng=rng, reporter=None, **overrides)

    def _routine_parameter_count(self, preprocessor: CircuitPreprocessor) -> int:
        """Free parameters the binding stage expects for ``preprocessor``.

        A routine that rewrites the circuit into one with a different parameter
        count (an overlap circuit carries two concatenated vectors) binds that
        count; the binding stage rejects any other width. A routine that leaves the
        count alone binds whatever the program itself declares, which is not the
        same as every parameter on the seed — a data-bound circuit carries data
        parameters that a different stage binds. ``0`` for a non-circuit seed.
        """
        spec = self._initial_spec()
        raw = getattr(spec, "parameters", None)
        if not raw:
            return 0
        transformed = len(preprocessor.preprocess(spec).parameters)
        return (
            self._bindable_parameter_count() if transformed == len(raw) else transformed
        )

    def _bindable_parameter_count(self) -> int:
        """Parameters the program's own binding stage varies. Subclasses that
        separate trainable weights from other bound values override this."""
        return len(self._initial_spec().parameters)

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
            "base_seed": self._base_seed,
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
        self._last_cost_variance = env.artifacts.get("cost_variance")
        return result

    # ------------------------------------------------------------------ #
    # Spec stage + seed (the program contract)
    # ------------------------------------------------------------------ #

    def _spec_stage(self) -> Stage:
        """The pipeline's first stage — emits this program's seed circuit.

        Default wraps a pre-built ``MetaCircuit``; programs that build their
        circuit from a Hamiltonian (QAOA, TimeEvolution) override this with a
        :class:`~divi.pipeline.stages.TrotterSpecStage`.
        """
        return CircuitSpecStage()

    def _initial_spec(self) -> Any:
        """The seed fed into :meth:`_spec_stage` — a prepared
        ``MetaCircuit`` for circuit-seeded programs, a Hamiltonian for
        Trotter-seeded ones.

        Has no universal default (the base does not assume a circuit factory):
        programs that drive :meth:`evaluate` must implement it.
        :class:`~divi.qprog.VariationalQuantumAlgorithm` returns its cost ansatz;
        QAOA / TimeEvolution return their Hamiltonian.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare a pipeline seed; "
            "implement _initial_spec to use evaluate()."
        )

    def _make_measurement_stage(self) -> Stage:
        """The default ``handles_measurement`` terminal. Overridden by
        :class:`~divi.qprog.ObservableMeasuringMixin` to carry the program's
        grouping / shot-distribution strategy."""
        return MeasurementStage()

    def _build_preprocessor_pipeline(
        self,
        preprocessor: CircuitPreprocessor,
    ) -> CircuitPipeline:
        """Resolve the pipeline for ``preprocessor``.

        Pipelines are memoized by ``preprocessor.cache_key`` so their
        forward-pass cache persists across optimizer iterations. A ``None``
        key rebuilds the pipeline each call and never retains it.
        """
        key = preprocessor.cache_key
        pipeline = None if key is None else self._preprocessor_pipeline_cache.get(key)
        if pipeline is None:
            terminal = preprocessor.terminal_stage or self._make_measurement_stage()
            pipeline = self._assemble_pipeline(
                self._spec_stage(),
                terminal,
                result_format=preprocessor.result_format,
                extra_stages=(PreprocessStage(preprocessor),),
            )
            if key is not None:
                self._preprocessor_pipeline_cache[key] = pipeline

        return pipeline

    def evaluate(
        self,
        params: "np.ndarray",
        preprocessor: CircuitPreprocessor,
        *,
        shots: int | None = None,
        return_variance: bool = False,
        preserve_keys: bool = False,
        axes_to_preserve: tuple[str, ...] = (),
    ) -> dict[int, Any] | PipelineResult | tuple[dict[int, Any], dict[int, float]]:
        """Measure this program's prepared state under ``preprocessor`` for ``params``.

        The single entry point external callers (optimizers, metric estimators)
        use instead of assembling pipelines themselves. ``preprocessor`` supplies the
        post-spec ``MetaCircuit`` transform, the result format, and an optional
        custom terminal stage.

        Args:
            params: Parameter set(s) to evaluate; coerced to 2D ``(n_sets, n_params)``.
            preprocessor: Selects the routine (cost / sample / metric / ...).
            shots: Per-evaluation shot-count override. ``None`` (default) uses the
                backend's own shot count; shot-adaptive optimizers (e.g. SPSA's
                ``M_k`` schedule) pass an explicit budget the static backend
                cannot supply.
            return_variance: Also return per-set shot-noise variance.
            preserve_keys: Output control. When ``True``, return the raw
                pipeline-result dict (keyed by full ``(axis, value)`` keys)
                instead of collapsing it to ``{param_set_idx: value}``.
            axes_to_preserve: Pipeline-reduce control. Spec axes named here are
                kept in the result keys instead of being averaged away (e.g. the
                per-branch ``ham`` axis for the QDrift metric cohort). Operates on
                a different layer than ``preserve_keys`` and requires it: preserved
                axes would otherwise collide when the result is collapsed by
                ``param_set``.

        Returns:
            ``{param_set_idx: value}`` (value shape per ``preprocessor.result_format``),
            the raw pipeline-result dict when ``preserve_keys=True``, or
            ``(values, shot_variances)`` when ``return_variance=True``.
        """
        if preserve_keys and return_variance:
            raise ValueError(
                "preserve_keys=True is incompatible with return_variance=True."
            )
        if axes_to_preserve and not preserve_keys:
            raise ValueError(
                "axes_to_preserve requires preserve_keys=True; otherwise the "
                "preserved axes collide during param_set collapse."
            )

        env_overrides: dict[str, Any] = {"param_sets": np.atleast_2d(params)}
        if axes_to_preserve:
            env_overrides["axes_to_preserve"] = tuple(axes_to_preserve)
        if shots is not None:
            env_overrides["shots_override"] = int(shots)
        if return_variance:
            env_overrides["collect_variance"] = True

        result = self._execute(
            self._build_preprocessor_pipeline(preprocessor),
            self._initial_spec(),
            **env_overrides,
        )

        if preserve_keys:
            return result

        values = dict(
            sorted(
                (extract_param_set_idx(key, default=0), value)
                for key, value in result.items()
            )
        )
        if not return_variance:
            return values

        return values, self._cost_shot_variances(values)

    def _cost_shot_variances(self, values: dict[int, float]) -> dict[int, float]:
        """Map the last pipeline run's shot-noise variance to each param-set index.

        The variance is computed at the counts stage with only the obs_group axis
        stripped. For a plain expval cost that leaves exactly the param_set axis,
        so each index maps to one variance. A pipeline with extra reduce axes
        (e.g. ZNE scales, a QNN data batch) yields several entries per index;
        collapsing them to a single scalar would be wrong, so such indices — and
        any value absent from the artifact (e.g. native-expval backends produce
        none) — are reported as ``nan`` so the optimizer falls back to its
        variance-free path.
        """
        raw_var = self._last_cost_variance or {}
        by_idx: dict[int, float] = {}
        collided: set[int] = set()
        for key, variance in raw_var.items():
            idx = extract_param_set_idx(key)
            if idx in by_idx:
                collided.add(idx)
            by_idx[idx] = variance
        for idx in collided:
            by_idx[idx] = float("nan")
        return {idx: by_idx.get(idx, float("nan")) for idx in values}
