# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from threading import Event
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from divi.backends._cancellation import _sigint_to_event
from divi.exceptions import VJPUnavailableError
from divi.hamiltonians.trotterization import ExactTrotterization
from divi.reporting._events import ProgressEvent

from ._compilation import (
    _insert_unmeasured_rows,
    batch_lineage,
    reject_colliding_body_tags,
)
from ._execution import (
    _default_execute_fn,
    _execute_incremental,
    _execute_vjp,
    _measurement_artifacts,
    _record_no_submission,
)
from ._postprocessing import (
    _check_histogram,
    _counts_to_cost_variance,
    _counts_to_expvals,
    _counts_to_probs,
    _expval_dicts_to_indexed,
)
from ._result_keys_operations import (
    FOREIGN_KEY_ATTR,
    OBS_GROUP_AXIS,
    strip_axis_from_label,
)
from .abc import (
    BundleStage,
    ChildResults,
    DiviPerformanceWarning,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    PipelineResult,
    PipelineTrace,
    ResultFormat,
    SpecStage,
    Stage,
    StageOutput,
    StageToken,
    _ScopedStageToken,
)
from .stages import (
    CircuitSpecStage,
    MeasurementStage,
    ParameterBindingStage,
    PreprocessStage,
    TrotterSpecStage,
)

logger = logging.getLogger(__name__)

_DIFFERENTIABLE_STAGES = (
    CircuitSpecStage,
    TrotterSpecStage,
    PreprocessStage,
    MeasurementStage,
    ParameterBindingStage,
)

#: ``(id(initial_spec), stage_instance_ids, per_stage_cache_key_extras)``.
_ForwardCacheKey = tuple[
    int,
    tuple[int, ...],
    tuple[tuple[Hashable, ...], ...],
]


def _report_pipeline_stage(env: "PipelineEnv", stage_name: str | None) -> None:
    """Emit a classical-pipeline phase event when reporting is bound.

    Passing ``None`` clears any lingering phase once the forward pass is
    complete and execution is about to begin.
    """
    if env._progress_emitter is None or env._progress_key is None:
        return
    message = f"Pipeline: {stage_name}" if stage_name else ""
    env._progress_emitter(ProgressEvent.show(env._progress_key, message))


@contextmanager
def _sigint_to_cancellation(env: "PipelineEnv"):
    """Pipeline-scoped wrapper around :func:`_sigint_to_event` that ensures
    ``env.cancellation_event`` exists."""
    if env.cancellation_event is None:
        env.cancellation_event = Event()
    with _sigint_to_event(env.cancellation_event):
        yield


def _has_custom_dry_expand(stage: Stage) -> bool:
    """True iff ``stage``'s class overrides :meth:`Stage.dry_expand`."""
    return type(stage).dry_expand is not Stage.dry_expand


def _unsafe_downstream_dag_consumers(stages: Sequence[Stage], idx: int) -> list[Stage]:
    """Downstream stages that would corrupt ``stages[idx]``'s dry placeholders.

    Returns stages after ``idx`` that both (a) declare
    ``consumes_dag_bodies=True`` — so their real :meth:`~Stage.expand` may
    mutate circuit-body DAGs in place — and (b) have not overridden
    :meth:`~Stage.dry_expand` — so they would run that real expand over the
    shared DAG references an upstream dry path emits.  When this list is
    non-empty for some upstream stage, the pipeline demotes that upstream
    stage to its real ``expand`` under :meth:`CircuitPipeline.run_forward_pass`
    dry mode, so downstream mutation stays safely sandboxed per branch.
    """
    return [
        s
        for s in stages[idx + 1 :]
        if (not isinstance(s, BundleStage) or s.consumes_dag_bodies)
        and not _has_custom_dry_expand(s)
    ]


def _warn_dry_fallback(stage: Stage, unsafe_stages: Sequence[Stage]) -> None:
    """Emit a :class:`DiviPerformanceWarning` for a demoted dry-mode stage.

    Called by :meth:`CircuitPipeline._resolve_expand_fns` when ``stage``'s
    analytic ``dry_expand`` can't be used safely because one or more
    downstream stages lack a ``dry_expand`` override while claiming to
    consume DAG bodies — see :func:`_unsafe_downstream_dag_consumers`.
    """
    culprits = ", ".join(type(s).__name__ for s in unsafe_stages)
    stage_name = type(stage).__name__
    warnings.warn(
        f"Dry-run analytic path disabled for {stage_name}: downstream "
        f"stage(s) {culprits} consume DAG bodies but don't override "
        f"dry_expand. Circuit counts stay correct, only the speedup for "
        f"{stage_name} is lost. To fix: implement dry_expand on "
        f"{culprits}, or set consumes_dag_bodies=False if they don't "
        f"mutate DAGs.",
        DiviPerformanceWarning,
        stacklevel=4,
    )


def _validate_stage_order(stages: Sequence[Stage]) -> None:
    """Ensure non-empty, exactly one spec stage first, then bundle stages."""
    if not stages:
        raise ValueError("stages cannot be empty")

    if not (
        isinstance(stages[0], SpecStage)
        and all(isinstance(s, BundleStage) for s in stages[1:])
    ):
        raise ValueError(
            "Pipeline must have exactly one 'spec' stage and it must come before "
            "any 'bundle' stage"
        )

    if not any(isinstance(s, BundleStage) and s.handles_measurement for s in stages):
        raise ValueError(
            "Pipeline must contain at least one stage that handles measurement "
            "(a stage with handles_measurement=True)"
        )

    meas_stages = [
        s for s in stages if isinstance(s, BundleStage) and s.handles_measurement
    ]
    if len(meas_stages) > 1:
        names = [s.name for s in meas_stages]
        raise ValueError(
            f"Multiple measurement-handling stages: {names}. "
            "Use exactly one measurement-handling stage."
        )

    axis_counts = Counter(stage.axis_name for stage in stages)
    duplicates = sorted(name for name, count in axis_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate stage axis names are not allowed: {', '.join(duplicates)}"
        )


def _measurement_cost_variance(
    stages: Sequence[Stage], raw: ChildResults, trace: PipelineTrace
) -> dict[tuple, float]:
    """Delegate variance semantics to the pipeline's measurement stage."""
    measurement_idx, measurement_stage = next(
        (idx, stage)
        for idx, stage in enumerate(stages)
        if isinstance(stage, BundleStage) and stage.handles_measurement
    )
    cost_variance = measurement_stage._estimate_cost_variance(
        raw, trace.final_batch, trace.stage_tokens[measurement_idx]
    )
    return (
        cost_variance
        if cost_variance is not None
        else _counts_to_cost_variance(raw, trace.final_batch)
    )


def _execute_measured(
    plan: PipelineTrace,
    env: PipelineEnv,
    execute_fn: Callable[[PipelineTrace, PipelineEnv], Any],
    no_submission: Any,
) -> Any:
    """Run ``execute_fn`` on ``plan``'s measured nodes.

    Returns ``no_submission`` when every node is unmeasured.
    """
    measured = replace(
        plan,
        final_batch={
            key: node
            for key, node in plan.final_batch.items()
            if not node.is_unmeasured
        },
    )
    if not measured.final_batch:
        _record_no_submission(env)
        return no_submission
    with _sigint_to_cancellation(env):
        return execute_fn(measured, env)


class CircuitPipeline:
    """
    Single ordered pipeline: one spec stage, then bundle stages.
    All stages pass keyed MetaCircuit batches.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        suppress_performance_warnings: bool = False,
    ) -> None:
        """
        Args:
            stages: Ordered sequence of stages (non-empty). Must contain exactly one
                SpecStage first, then zero or more BundleStages.
            suppress_performance_warnings: When True, silence any
                :class:`~divi.pipeline.DiviPerformanceWarning` emitted by
                individual stages' ``validate`` hooks during pipeline
                construction. Hard validation errors still raise.
        """
        _validate_stage_order(stages)
        with warnings.catch_warnings():
            if suppress_performance_warnings:
                warnings.simplefilter("ignore", DiviPerformanceWarning)
            for i, stage in enumerate(stages):
                stage.validate(before=tuple(stages[:i]), after=tuple(stages[i + 1 :]))
        self._stages = list(stages)
        self._forward_cache: dict[_ForwardCacheKey, PipelineTrace] = {}

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Read-only view of the pipeline stages."""
        return tuple(self._stages)

    def run(
        self,
        initial_spec: Any,
        env: PipelineEnv,
        *,
        bypass_cache: bool = False,
        execute_fn: Callable[
            [PipelineTrace, PipelineEnv], ChildResults
        ] = _default_execute_fn,
    ) -> PipelineResult:
        """
        Run the pipeline: spec expand → bundle expand → (substitute params, generate QASM) → execute → reduce.

        1. Run ``expand`` on the single spec stage: input any (e.g. Hamiltonian), output batch of MetaCircuits.
        2. Run ``expand`` on each bundle stage: each takes a MetaCircuit batch, modifies body/measurement, returns MetaCircuit batch.
        3. Execute: ``execute_fn(meta_circuit_batch, env)`` (or default: param substitution + backend run) → raw results.
        4. Convert raw results according to the result format recorded on the circuits.
        5. Run ``reduce`` on each stage in reverse order.

        Args:
            initial_spec: Input for the spec stage (typically a Hamiltonian).
            env: Pipeline environment (backend, progress emitter, etc.).
            bypass_cache: When True, ignore any cached forward trace and
                recompute the full forward pass from the beginning.
            execute_fn: (trace, env) → raw_results. Defaults to the built-in
                lowering of tagged MetaCircuit body/measurement QASMs and
                backend execution.

        Returns:
            A ``PipelineResult`` dict keyed by ``NodeKey`` tuples.
            For single-circuit pipelines, use ``result.value`` to get the
            result directly.
        """

        try:
            plan = self.run_forward_pass(initial_spec, env, bypass_cache=bypass_cache)

            # Forward pass is done — clear the classical-pipeline indicator so
            # the spinner shows only execution/polling state from here on.
            _report_pipeline_stage(env, None)

            raw = _execute_measured(plan, env, execute_fn, {})
            return self._build_result(raw, env, plan)
        finally:
            # A pipeline run owns only its transient stage/polling phase;
            # clear it so no run leaks a live spinner.
            _report_pipeline_stage(env, None)

    def run_vjp(
        self, initial_spec: Any, env: PipelineEnv, cotangents: np.ndarray
    ) -> tuple[PipelineResult, dict[int, np.ndarray]]:
        """Run the pipeline, submitting its circuits for vector-Jacobian products.

        The circuits go to ``env.backend``'s
        :meth:`~divi.backends.CircuitRunner.submit_vjp`.

        Args:
            initial_spec: Input for the spec stage.
            env: Pipeline environment.
            cotangents: ``(n_param_sets, n_observables)`` weights of each
                parameter set's expectation values.

        Returns:
            The result :meth:`run` gives, and ``cotangents[i] @ d<O>/dθ`` for
            each parameter set ``i``, one entry per parameter in the circuit's
            order.

        Raises:
            VJPUnavailableError: Before the forward pass, if a stage has no
                vector-Jacobian product (stochastic Trotterization has none);
                after it, if the circuits it submits are not one
                expectation-value circuit per parameter set, or the backend
                refuses.
        """
        foreign = [
            type(s).__name__
            for s in self._stages
            if type(s) not in _DIFFERENTIABLE_STAGES
            or (
                isinstance(s, TrotterSpecStage)
                and not isinstance(s._trotterization_strategy, ExactTrotterization)
            )
        ]
        if foreign:
            raise VJPUnavailableError(
                f"Stages {foreign} have no vector-Jacobian product."
            )
        binding = self._stages[-1]
        if not isinstance(binding, ParameterBindingStage):
            raise VJPUnavailableError(
                "The pipeline must bind parameters in its last stage."
            )
        vjps: dict[int, np.ndarray] = {}
        execute = partial(
            _execute_vjp, binding=binding, cotangents=cotangents, vjps=vjps
        )
        result = self.run(initial_spec, env, execute_fn=execute)
        # Identity-only observables run no circuit and have zero derivatives.
        n_params = len(env.param_sets[0])
        for idx in range(len(cotangents)):
            vjps.setdefault(idx, np.zeros(n_params))
        return result, vjps

    def run_incremental(
        self,
        initial_spec: Any,
        env: PipelineEnv,
        initial_state: QuantumCircuit,
        steps: Sequence[int],
    ) -> list[PipelineResult]:
        """Run the pipeline, evolving one state through repeats of its circuit.

        The pipeline's single bound circuit is one step, applied to
        ``initial_state`` by ``env.backend``'s
        :meth:`~divi.backends.CircuitRunner.submit_incremental_evolution`.

        Args:
            initial_spec: Input for the spec stage.
            env: Pipeline environment.
            initial_state: Preparation of the state the first step acts on.
            steps: Step counts to measure at, strictly increasing.

        Returns:
            One result per entry of ``steps``, each as :meth:`run` gives it.

        Raises:
            IncrementalEvolutionUnavailableError: If the circuits it submits are
                not backend expectation values of exactly one bound circuit, or
                the backend refuses.
        """
        try:
            plan = self.run_forward_pass(initial_spec, env)
            _report_pipeline_stage(env, None)
            execute = partial(
                _execute_incremental, initial_state=initial_state, steps=list(steps)
            )
            per_step = _execute_measured(plan, env, execute, [{} for _ in steps])
            return [self._build_result(raw, env, plan) for raw in per_step]
        finally:
            _report_pipeline_stage(env, None)

    def _build_result(
        self, raw: ChildResults, env: PipelineEnv, plan: PipelineTrace
    ) -> PipelineResult:
        """Convert ``raw`` to the recorded result format and reduce it."""
        unmeasured = {
            key: node for key, node in plan.final_batch.items() if node.is_unmeasured
        }
        result_format = next(
            (
                meta.result_format
                for meta in plan.final_batch.values()
                if meta.result_format is not None
            ),
            None,
        )
        if result_format is ResultFormat.PROBS:
            raw = _counts_to_probs(raw, env.backend.shots)
        elif result_format is ResultFormat.COUNTS:
            raw = {key: _check_histogram(key, counts) for key, counts in raw.items()}
        elif result_format is ResultFormat.EXPVALS:
            ham_ops = plan.env_artifacts.get("ham_ops")
            if ham_ops is not None:
                # Backend-native expval: no shot counts, so a shot-noise
                # variance is undefined; cost_variance is left unset and
                # consumers default to nan.
                raw = _expval_dicts_to_indexed(raw, ham_ops)
            else:
                if env.collect_variance:
                    # Constants have no shot noise; several observables, no scalar.
                    env.artifacts["cost_variance"] = _measurement_cost_variance(
                        self._stages, raw, plan
                    ) | {
                        strip_axis_from_label(branch_key, OBS_GROUP_AXIS): (
                            0.0 if len(node.observable or ()) == 1 else float("nan")
                        )
                        for key, node in unmeasured.items()
                        for branch_key in batch_lineage(
                            {key: node}, len(env.param_sets)
                        ).values()
                    }
                raw = _counts_to_expvals(raw, plan.final_batch)
        if unmeasured:
            raw = _insert_unmeasured_rows(raw, plan.final_batch, len(env.param_sets))

        result = PipelineResult(self._reduce(raw, env, plan.stage_tokens))
        if any(meta._was_multi_obs for meta in plan.initial_batch.values()):
            result._squeeze = False
        return result

    def run_forward_pass(
        self,
        initial_spec: Any,
        env: PipelineEnv,
        *,
        bypass_cache: bool = False,
        dry: bool = False,
    ) -> PipelineTrace:
        """Run only the forward expansion pass and return lineage metadata.

        When ``dry=True``, each stage is invoked via :meth:`~Stage.dry_expand`
        instead of :meth:`~Stage.expand`, skipping expensive per-item content
        generation (DAG copies, QASM rendering, classical simulations). Dry
        traces are never cached and never read from the cache — they carry
        placeholder bodies that would corrupt real runs if replayed.

        If a dry-aware stage has a downstream neighbour that (a) declares
        ``consumes_dag_bodies=True`` and (b) has not overridden
        :meth:`~Stage.dry_expand`, the pipeline conservatively demotes the
        upstream stage to its real ``expand`` for this run and emits a
        :class:`~divi.pipeline.DiviPerformanceWarning` naming both stages.
        The circuit count stays correct; only the analytic speedup is
        lost for the affected stage.
        """

        # Pair each stage with the expand callable to use for this run.
        # In dry mode, some stages may be demoted back to their real expand
        # to keep the trace correct when a non-dry-aware downstream stage
        # would otherwise mutate shared DAG references.  Keeping stage and
        # callable zipped together eliminates any index-drift footgun when
        # the list is sliced below.
        plan: list[tuple[Stage, Callable[[Any, PipelineEnv], StageOutput[Any]]]] = list(
            zip(self._stages, self._resolve_expand_fns(dry=dry))
        )

        def run_bundle_stages(
            data: Any,
            bundle_plan: Sequence[
                tuple[Stage, Callable[[Any, PipelineEnv], StageOutput[Any]]]
            ],
        ) -> tuple[Any, list[StageToken], list[ExpansionResult]]:
            tokens: list[StageToken] = []
            expansions: list[ExpansionResult] = []

            for stage, stage_expand in bundle_plan:
                _report_pipeline_stage(env, stage.name)
                output = stage_expand(data, env)
                data = output.batch
                reject_colliding_body_tags(type(stage).__name__, output.batch)
                tokens.append(output.token)
                expansions.append(
                    ExpansionResult(batch=output.batch, stage_name=stage.name)
                )
            return data, tokens, expansions

        stage_ids = tuple(id(stage) for stage in self._stages)
        stage_extras = tuple(stage.cache_key_extras(env) for stage in self._stages)
        cache_key = (id(initial_spec), stage_ids, stage_extras)

        # Dry traces carry placeholder bodies — never cache them, and never
        # serve a real-run request from a dry trace (or vice versa).
        cached = None if (bypass_cache or dry) else self._forward_cache.get(cache_key)

        recompute_from_idx = next(
            (
                idx
                for idx, stage in enumerate(self._stages)
                if stage.is_volatile_for(env)
            ),
            None,
        )

        if cached is None or recompute_from_idx == 0:
            spec_stage, spec_expand = plan[0]
            _report_pipeline_stage(env, spec_stage.name)
            data, spec_token = spec_expand(initial_spec, env)

            initial_batch_snapshot = data

            final_batch, bundle_tokens, expansions = run_bundle_stages(data, plan[1:])
            trace = PipelineTrace(
                initial_batch=initial_batch_snapshot,
                final_batch=final_batch,
                stage_expansions=tuple(expansions),
                stage_tokens=(spec_token, *bundle_tokens),
                env_artifacts=_measurement_artifacts(final_batch),
            )
            if not dry:
                self._forward_cache[cache_key] = trace
            return trace

        # A cached trace carries its measurement metadata on each MetaCircuit.
        if recompute_from_idx is None:
            return cached

        if recompute_from_idx == 1:
            data = cached.initial_batch
        else:
            data = cached.stage_expansions[recompute_from_idx - 2].batch

        final_batch, rerun_tokens, rerun_expansions = run_bundle_stages(
            data, plan[recompute_from_idx:]
        )
        prefix_tokens = list(cached.stage_tokens[:recompute_from_idx])
        prefix_expansions = list(cached.stage_expansions[: recompute_from_idx - 1])

        trace = PipelineTrace(
            initial_batch=cached.initial_batch,
            final_batch=final_batch,
            stage_expansions=(*prefix_expansions, *rerun_expansions),
            stage_tokens=(*prefix_tokens, *rerun_tokens),
            env_artifacts=_measurement_artifacts(final_batch),
        )

        if not dry:
            self._forward_cache[cache_key] = trace

        return trace

    def run_spec_stage(
        self,
        initial_spec: Any,
        env: PipelineEnv,
    ) -> StageOutput[MetaCircuitBatch]:
        """Expand only the pipeline's spec stage.

        Reuses the spec output of a forward pass already cached under the
        current key (e.g. the cost evaluation that precedes the metric within
        one optimizer step), so the metric measures on the cost's sampled
        batch. With no cached trace it expands directly; deterministic
        per-evaluation seeding means that recompute reproduces the same batch.
        """
        stage = self._stages[0]
        stage_ids = tuple(id(s) for s in self._stages)
        stage_extras = tuple(s.cache_key_extras(env) for s in self._stages)
        cache_key = (id(initial_spec), stage_ids, stage_extras)
        cached = self._forward_cache.get(cache_key)
        if cached is not None:
            return StageOutput(cached.initial_batch, cached.stage_tokens[0])
        _report_pipeline_stage(env, stage.name)
        return stage.expand(initial_spec, env)

    def _resolve_expand_fns(
        self, *, dry: bool
    ) -> list[Callable[[Any, PipelineEnv], StageOutput[Any]]]:
        """Build the per-stage expand callables for one forward pass.

        Real runs always route through :meth:`Stage.expand`.  Dry runs route
        through :meth:`Stage.dry_expand`, except that any stage with a custom
        ``dry_expand`` whose downstream neighbours would mutate its shared
        placeholder DAGs is demoted back to the real path — with a
        :class:`DiviPerformanceWarning` naming the upstream stage and the
        downstream culprit(s).  The circuit count stays correct regardless;
        only the affected stage loses its analytic speedup.
        """
        if not dry:
            return [stage.expand for stage in self._stages]

        fns: list[Callable[[Any, PipelineEnv], StageOutput[Any]]] = []
        for idx, stage in enumerate(self._stages):
            if not _has_custom_dry_expand(stage):
                # No analytic override — the default ``dry_expand`` already
                # delegates to ``expand``; call ``expand`` directly for a
                # cleaner stack trace under debugging.
                fns.append(stage.expand)
                continue

            unsafe = _unsafe_downstream_dag_consumers(self._stages, idx)
            if unsafe:
                _warn_dry_fallback(stage, unsafe)
                fns.append(stage.expand)
            else:
                fns.append(stage.dry_expand)
        return fns

    def _reduce(
        self,
        raw_results: ChildResults,
        env: PipelineEnv,
        tokens: Sequence[StageToken],
    ) -> Any:
        reduced: ChildResults = raw_results
        bundle_stages = list(self._stages[1:])
        bundle_tokens = list(tokens[1:])

        # Collect axis names for all bundle stages (in expand order).
        all_bundle_axes = {
            s.axis_name for s in bundle_stages if s.axis_name is not None
        }

        # Reduce in reverse expand order (bottom-up).
        for stage, token in reversed(list(zip(bundle_stages, bundle_tokens))):
            # Axes from ALL OTHER bundle stages — both upstream and downstream.
            # Branch keys include tag axes from _compile_batch that may come from
            # stages on either side of this one.
            other_axes = all_bundle_axes - {stage.axis_name}

            if not other_axes:
                reduced = stage.reduce(reduced, env, token)
            else:
                reduced = _reduce_with_isolated_axes(
                    stage, reduced, env, token, other_axes
                )

        # Spec stage reduce (outermost).
        reduced = self._stages[0].reduce(reduced, env, tokens[0])
        return reduced


def _scope_token(
    token: StageToken,
    foreign_key: tuple,
    foreign_axes: AbstractSet[str],
) -> StageToken:
    """Scope a dict token to match a specific foreign-key group.

    When a stage token is a dict keyed by pipeline label tuples (e.g.
    QEMStage contexts), entries whose foreign-axis components are a
    subset of *foreign_key* are retained with those axes stripped from
    the keys.  Non-dict tokens are returned as-is.

    Subset matching is needed because a stage's token may not contain
    axes from other body-level sources (e.g. QEMStage tokens carry
    circuit-body axes like ``param_set`` but not measurement-body axes
    like ``obs_group``).
    """
    if isinstance(token, _ScopedStageToken):
        return token.scoped_to(foreign_key, foreign_axes)
    if not isinstance(token, dict):
        return token

    scoped: dict = {}
    foreign_set = set(foreign_key)
    for key, value in token.items():
        if not isinstance(key, tuple):
            return token  # not a label-keyed dict — return unchanged

        kf = tuple(ax for ax in key if isinstance(ax, tuple) and ax[0] in foreign_axes)
        if set(kf) <= foreign_set:
            ko = tuple(
                ax
                for ax in key
                if not (isinstance(ax, tuple) and ax[0] in foreign_axes)
            )
            scoped[ko] = value

    if not scoped:
        raise KeyError(
            f"_scope_token: no token entries matched foreign_key={foreign_key!r}. "
            f"Token keys: {list(token.keys())!r}, foreign_axes: {foreign_axes!r}"
        )
    return scoped


def _reduce_with_isolated_axes(
    stage: Stage,
    results: ChildResults,
    env: PipelineEnv,
    token: StageToken,
    foreign_axes: AbstractSet[str],
) -> ChildResults:
    """Group results by foreign axes, reduce each group, and re-attach foreign keys.

    This ensures the stage's reduce only sees keys from its own axis and the spec axis,
    never axes from other stages. After reduce, foreign key parts are re-attached
    so they survive to the final output.
    """
    groups: dict[tuple, ChildResults] = defaultdict(dict)

    for full_key, value in results.items():
        own_key = tuple(ax for ax in full_key if ax[0] not in foreign_axes)
        foreign_key = tuple(ax for ax in full_key if ax[0] in foreign_axes)
        groups[foreign_key][own_key] = value

    out: ChildResults = {}
    for foreign_key, group_results in groups.items():
        scoped_token = _scope_token(token, foreign_key, foreign_axes)
        if isinstance(scoped_token, dict):
            # Inject the foreign_key so the stage can identify which
            # foreign-axis group (e.g. param_set index) is being reduced.
            scoped_token = {**scoped_token, FOREIGN_KEY_ATTR: foreign_key}
        group_reduced = stage.reduce(group_results, env, scoped_token)
        for own_key, value in group_reduced.items():
            out[own_key + foreign_key] = value

    return out
