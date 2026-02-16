# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from rich.console import Console
from rich.tree import Tree

from divi.backends import JobStatus
from divi.pipeline._compilation import _collapse_to_parent_results, _compile_batch
from divi.pipeline._postprocessing import _counts_to_expvals, _counts_to_probs
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    PipelineEnv,
    PipelineTrace,
    ResultFormat,
    SpecStage,
    Stage,
    StageToken,
)
from divi.pipeline.stages import MeasurementStage


def _path_children(keys: list[Any]) -> dict[str, list[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        if isinstance(key, tuple) and key and isinstance(key[0], tuple):
            key_str = "/".join(f"{p[0]}:{p[1]}" for p in key)
        elif isinstance(key, tuple):
            key_str = "/".join(str(part) for part in key)
        else:
            key_str = str(key)
        parts = key_str.split("/")
        for i in range(1, len(parts) + 1):
            child = "/".join(parts[:i])
            parent = "/".join(parts[: i - 1]) if i > 1 else ""
            children[parent].add(child)
    return {p: sorted(c) for p, c in children.items()}


def format_pipeline_tree(trace: PipelineTrace) -> None:
    """Print the full pipeline expansion tree to the terminal."""
    _, lineage = _compile_batch(trace.final_batch)
    keys = sorted(lineage.values(), key=str)
    if not keys:
        print("(empty)")
        return
    children = _path_children(keys)
    roots = children.get("", [])
    s = lambda p: p.split("/")[-1]

    def add(tree: Tree, path: str) -> None:
        for c in children.get(path, []):
            add(tree.add(s(c)), c)

    root = Tree(s(roots[0])) if len(roots) == 1 else Tree("")
    for r in roots:
        add(root if len(roots) == 1 else root.add(s(r)), r)

    Console(no_color=True).print(root)


def _wait_for_async_result(backend, execution_result, env):
    """Poll an async backend until job completes, then fetch results.

    Faithful mirror of ``QuantumProgram._wait_for_qoro_job_completion``.
    """
    job_id = execution_result.job_id
    if job_id is None:
        raise ValueError("ExecutionResult must have a job_id for async completion")

    # Lazy import to avoid circular dependency: pipeline → qprog → pipeline
    from divi.qprog.exceptions import _CancelledError

    # Build the poll callback if reporter is available
    progress_callback = None
    if env.reporter is not None:
        progress_callback = lambda n_polls, status: env.reporter.info(
            message="",
            poll_attempt=n_polls,
            max_retries=backend.max_retries,
            service_job_id=job_id,
            job_status=status,
        )

    # Runtime tracking via env.artifacts
    def _track_runtime(response):
        if isinstance(response, dict):
            env.artifacts["run_time"] = env.artifacts.get("run_time", 0.0) + float(
                response.get("run_time", 0)
            )
        elif isinstance(response, list):
            env.artifacts["run_time"] = env.artifacts.get("run_time", 0.0) + sum(
                float(r.json()["run_time"]) for r in response
            )

    # Poll until complete
    status = backend.poll_job_status(
        execution_result,
        loop_until_complete=True,
        on_complete=_track_runtime,
        verbose=progress_callback is None,
        progress_callback=progress_callback,
    )

    if status == JobStatus.FAILED:
        raise RuntimeError(f"Job {job_id} has failed")

    if status == JobStatus.CANCELLED:
        # If cancellation was requested (e.g., by ProgramBatch), raise _CancelledError
        # so it's handled gracefully. Otherwise, raise RuntimeError for unexpected cancellation.
        if env.cancellation_event and env.cancellation_event.is_set():
            raise _CancelledError(f"Job {job_id} was cancelled")
        raise RuntimeError(f"Job {job_id} was cancelled")

    if status != JobStatus.COMPLETED:
        raise RuntimeError("Job has not completed yet, cannot post-process results")

    return backend.get_job_results(execution_result)


def _default_execute_fn(
    trace: PipelineTrace,
    env: PipelineEnv,
) -> ChildResults:
    """Default execute: lower MetaCircuit batch to QASM circuits, then backend run."""
    circuits, lineage_by_label = _compile_batch(trace.final_batch)

    env.artifacts["circuit_count"] = len(circuits)

    submit_kwargs = {}
    ham_ops = env.artifacts.get("ham_ops")
    if ham_ops is not None:
        submit_kwargs["ham_ops"] = ham_ops

    result = env.backend.submit_circuits(circuits, **submit_kwargs)

    # Store for cancellation support (read by cancel_unfinished_job)
    env.artifacts["_current_execution_result"] = result

    if result.is_async():
        result = _wait_for_async_result(env.backend, result, env)

    if result.results is None:
        raise RuntimeError("Backend returned no results")

    raw_by_label = {r["label"]: r["results"] for r in result.results}

    return _collapse_to_parent_results(raw_by_label, lineage_by_label)


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

    if not any(isinstance(s, MeasurementStage) for s in stages):
        raise ValueError(
            "Pipeline must contain at least one MeasurementStage "
            "(or a subclass of it)"
        )

    axis_counts = Counter(stage.axis_name for stage in stages)
    duplicates = sorted(name for name, count in axis_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate stage axis names are not allowed: {', '.join(duplicates)}"
        )


class CircuitPipeline:
    """
    Single ordered pipeline: one spec stage, then bundle stages.
    All stages pass keyed MetaCircuit batches.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        """
        Args:
            stages: Ordered sequence of stages (non-empty). Must contain exactly one
                SpecStage first, then zero or more BundleStages.
        """
        _validate_stage_order(stages)
        self._stages = list(stages)
        self._forward_cache: dict[tuple[int, tuple[int, ...]], PipelineTrace] = {}

    def run(
        self,
        initial_spec: Any,
        env: PipelineEnv,
        *,
        force_forward_sweep: bool = False,
        execute_fn: Callable[
            [PipelineTrace, PipelineEnv], ChildResults
        ] = _default_execute_fn,
    ) -> Any:
        """
        Run the pipeline: spec expand → bundle expand → (substitute params, generate QASM) → execute → reduce.

        1. Run ``expand`` on the single spec stage: input any (e.g. Hamiltonian), output batch of MetaCircuits.
        2. Run ``expand`` on each bundle stage: each takes a MetaCircuit batch, modifies body/measurement, returns MetaCircuit batch.
        3. Execute: ``execute_fn(meta_circuit_batch, env)`` (or default: param substitution + backend run) → raw results.
        4. Convert raw results according to ``env.result_format`` (set by the measurement stage).
        5. Run ``reduce`` on each stage in reverse order.

        Args:
            initial_spec: Input for the spec stage (typically a Hamiltonian).
            env: Pipeline environment (backend, reporter, etc.).
            force_forward_sweep: When True, ignore any cached forward trace and
                recompute the full forward pass from the beginning.
            execute_fn: (trace, env) → raw_results. Defaults to the built-in
                lowering of tagged MetaCircuit body/measurement QASMs and
                backend execution.

        Returns:
            Reduced result for a single batch.
        """

        plan = self.run_forward_pass(
            initial_spec, env, force_forward_sweep=force_forward_sweep
        )

        # Restore result_format and stage-produced artifacts from the
        # cached trace onto the (possibly fresh) PipelineEnv.  When the
        # forward pass is cached, expand() doesn't re-run, so these
        # env fields would otherwise be empty/None.
        if plan.result_format is not None:
            env.result_format = plan.result_format

        env.artifacts.update(plan.env_artifacts)

        raw = execute_fn(plan, env)

        # Convert raw backend results into the canonical format declared
        # by the measurement stage during expand.  This runs *before* the
        # reduce chain so that downstream stages (QEM, etc.) receive
        # values in the expected type.
        if env.result_format is not None:
            if env.result_format is ResultFormat.PROBS:
                raw = _counts_to_probs(raw, env.backend.shots)
            elif env.result_format is ResultFormat.EXPVALS:
                if not env.backend.supports_expval:
                    raw = _counts_to_expvals(raw, plan.final_batch)

        return self._reduce(raw, env, plan.stage_tokens)

    def run_forward_pass(
        self, initial_spec: Any, env: PipelineEnv, *, force_forward_sweep: bool = False
    ) -> PipelineTrace:
        """Run only the forward expansion pass and return lineage metadata."""

        def run_bundle_stages(
            data: Any, bundle_stages: Sequence[Stage]
        ) -> tuple[Any, list[StageToken], list[ExpansionResult]]:
            tokens: list[StageToken] = []
            expansions: list[ExpansionResult] = []

            for stage in bundle_stages:
                expansion_result, token = stage.expand(data, env)
                data = expansion_result.batch
                tokens.append(token)
                expansions.append(
                    ExpansionResult(
                        # Deep-copy the batch snapshot so that the next stage's
                        # in-place mutations (via object.__setattr__) do not
                        # corrupt this snapshot.  Required for partial-rerun
                        # caching when first_stateful_idx > 1.
                        batch=copy.deepcopy(expansion_result.batch),
                        stage_name=stage.name,
                    )
                )
            return data, tokens, expansions

        stage_ids = tuple(id(stage) for stage in self._stages)
        cache_key = (id(initial_spec), stage_ids)
        cached = None if force_forward_sweep else self._forward_cache.get(cache_key)

        first_stateful_idx = next(
            (idx for idx, stage in enumerate(self._stages) if stage.stateful),
            None,
        )

        if cached is not None and first_stateful_idx is None:
            return cached

        if cached is None or first_stateful_idx == 0:
            spec_stage = self._stages[0]
            data, spec_token = spec_stage.expand(initial_spec, env)

            # Snapshot the spec-stage output *before* bundle stages can mutate
            # the MetaCircuit objects via object.__setattr__ (e.g.
            # ParameterBindingStage.set_circuit_bodies).  Without this copy the
            # cached initial_batch would be the same Python object that
            # downstream stages later modify, corrupting the cache.
            initial_batch_snapshot = copy.deepcopy(data)

            final_batch, bundle_tokens, expansions = run_bundle_stages(
                data, self._stages[1:]
            )
            trace = PipelineTrace(
                initial_batch=initial_batch_snapshot,
                final_batch=final_batch,
                stage_expansions=tuple(expansions),
                stage_tokens=tuple([spec_token, *bundle_tokens]),
                result_format=env.result_format,
                env_artifacts=dict(env.artifacts),
            )
            self._forward_cache[cache_key] = trace
            return trace

        if first_stateful_idx is None or first_stateful_idx <= 0:
            raise ValueError("stateful stage index must be >= 1 for partial rerun.")

        # Deep-copy the intermediate batch so that in-place mutations by
        # downstream stages (e.g. MetaCircuit.set_circuit_bodies) do not
        # corrupt the cached expansion data for future partial reruns.
        if first_stateful_idx == 1:
            data = copy.deepcopy(cached.initial_batch)
        else:
            data = copy.deepcopy(cached.stage_expansions[first_stateful_idx - 2].batch)

        final_batch, rerun_tokens, rerun_expansions = run_bundle_stages(
            data, self._stages[first_stateful_idx:]
        )
        prefix_tokens = list(cached.stage_tokens[:first_stateful_idx])
        prefix_expansions = list(cached.stage_expansions[: first_stateful_idx - 1])

        trace = PipelineTrace(
            initial_batch=cached.initial_batch,
            final_batch=final_batch,
            stage_expansions=tuple([*prefix_expansions, *rerun_expansions]),
            stage_tokens=tuple([*prefix_tokens, *rerun_tokens]),
            # Carry forward result_format and env_artifacts from the cached
            # trace — the pre-stateful stages (including MeasurementStage)
            # didn't re-run, so their env side-effects aren't on this env.
            result_format=cached.result_format,
            env_artifacts={**cached.env_artifacts, **env.artifacts},
        )

        self._forward_cache[cache_key] = trace

        return trace

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


def _reduce_with_isolated_axes(
    stage: Stage,
    results: ChildResults,
    env: PipelineEnv,
    token: StageToken,
    foreign_axes: set[str],
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
        group_reduced = stage.reduce(group_results, env, token)
        for own_key, value in group_reduced.items():
            out[own_key + foreign_key] = value

    return out
