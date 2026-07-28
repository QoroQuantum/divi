# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Dry-run analysis for circuit pipelines.

Reports the per-stage factor (fan-out or reduction) introduced by a
:class:`PipelineTrace`, without executing any circuits. Observable
grouping in :class:`~divi.pipeline.stages.MeasurementStage` is counted
as a reduction (``factor < 1``), since grouping N Pauli terms into
M ≤ N commuting groups saves circuits.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, TypeAlias

from qiskit.dagcircuit import DAGCircuit

from divi.circuits import MetaCircuit
from divi.pipeline._compilation import _effective_bodies
from divi.pipeline._preprocessor import PipelineCadence
from divi.pipeline.abc import (
    MetaCircuitBatch,
    PipelineEnv,
    PipelineTrace,
    Stage,
)

#: Keys the renderers index without a guard, so a report carrying a partial
#: ``circuit_stats`` is rejected rather than failing mid-render. ``mean_depth`` and
#: the ``std_*`` keys are absent on purpose: a merged group omits them when its
#: members differ, and the renderer guards them.
_REQUIRED_CIRCUIT_STATS = frozenset(
    {
        "min_depth",
        "max_depth",
        "min_width",
        "max_width",
        "mean_width",
        "mean_2q_depth",
    }
)


@dataclass(frozen=True)
class StageInfo:
    """Per-stage dry-run report.

    ``factor`` is the ratio of circuits after this stage to circuits before it:
    above 1 is a fan-out, below 1 a reduction (observable grouping). The three
    identifiers differ — ``name`` is the stage class, ``label`` its constructor
    ``name=``, ``axis`` the result-key axis the tree shows in brackets.
    """

    name: str
    """Stage identifier — ``type(stage).__name__`` (e.g. ``"CircuitSpecStage"``,
    ``"MeasurementStage"``). This is the value to match when checking a
    pipeline's composition via ``[s.name for s in report.stages]``."""

    axis: str | None
    """The result-key axis this stage fans out over — the stage's ``axis_name``
    (``"param_set"``, ``"twirl"``, ``"obs_group"``), and what the rendered tree
    shows in brackets.

    A stage that declares no axis falls back to its own name, so ``None`` is
    rare."""

    factor: float
    """Ratio of logical circuits after this stage to circuits before it.
    ``factor > 1`` is a fan-out (e.g.
    :class:`~divi.pipeline.stages.ParameterBindingStage`,
    :class:`~divi.pipeline.stages.PauliTwirlStage`); ``factor < 1`` is a
    reduction (e.g. observable grouping in
    :class:`~divi.pipeline.stages.MeasurementStage` collapsing N Pauli terms
    into M ≤ N commuting groups yields ``factor = M / N``)."""

    metadata: Mapping[str, Any]
    """Stage-specific introspection rendered by
    :func:`~divi.pipeline.format_dry_run` — e.g. ``strategy`` / ``n_groups``
    for :class:`~divi.pipeline.stages.MeasurementStage`, ``n_twirls`` for
    :class:`~divi.pipeline.stages.PauliTwirlStage`."""

    label: str | None = None
    """The name the stage instance was constructed with.

    Distinct from :attr:`name`, which is the stage *class*; for built-in stages the
    two coincide, since their constructors pass the class name. ``None`` when the
    stage exposes no ``name``."""

    def __post_init__(self) -> None:
        # Its own copy: the caller's dict must not keep changing under the report.
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DryRunReport:
    """Complete dry-run report for a single pipeline.

    Every count describes **one evaluation** — one call to the objective the
    optimizer is minimizing. That covers one parameter vector for most optimizers,
    but a population optimizer binds its whole working set per call, so the count
    covers all ``n_param_sets`` of them.

    An evaluation is not an iteration: an optimizer may evaluate the objective
    several times per step, and the number of steps depends on the landscape. To
    reach a whole-run total, multiply by your own figures. :attr:`cadence` tells a
    recurring routine from a one-time one.
    """

    pipeline_name: str
    """Name of the routine this report describes — the key it is filed under in
    :meth:`~divi.qprog.QuantumProgram.dry_run`'s result (``"cost"``, ``"sample"``).
    """

    stages: tuple[StageInfo, ...]
    """Ordered per-stage reports for the pipeline's forward pass — one
    :class:`StageInfo` per stage, in execution order."""

    total_circuits: int
    """Circuits this pipeline submits per evaluation (see above)."""

    total_shots: int = 0
    """Shots per evaluation — ``total_circuits`` weighted by each circuit's
    configured shot budget (backend ``shots``, or a per-group
    ``shot_distribution`` allocation).

    A configured budget, not a consumption forecast: an analytic-expval backend
    reports the shots it was configured with while consuming none, and ``0``
    means the backend exposes no shot count at all."""

    cadence: PipelineCadence = PipelineCadence.PER_EVALUATION
    """How often this routine runs over a ``run()`` — recurring
    (:attr:`~divi.pipeline.PipelineCadence.PER_EVALUATION`: cost, metric/overlap)
    or one-time (:attr:`~divi.pipeline.PipelineCadence.ONCE`: sample).

    A ``ONCE`` routine runs *at most* once: solution sampling is skipped entirely
    under ``run(perform_final_computation=False)``, so subtract it when budgeting
    that path."""

    objective_fingerprint: Hashable | None = None
    """Opaque fingerprint of what this pipeline optimizes — derived from the
    observable's Pauli labels and coefficients, or from the terminal stage's
    settings for a sampled readout.

    Compare two reports' fingerprints to tell whether they share an objective; do
    not parse the value, its contents are not a stable format."""

    env_artifacts: Mapping[str, Any] = field(default_factory=dict)
    """Stage-produced artifacts captured during the forward pass — e.g.
    ``per_group_shots`` (when a ``shot_distribution`` is configured on
    :class:`~divi.pipeline.stages.MeasurementStage`), ``ham_ops`` (for
    expval-native backends).  These are the same artifacts the pipeline
    would produce on a real run, so a dry-run report is the canonical
    surface for ``"what would my pipeline do?"`` introspection — no
    need to drop into private helpers or rerun the forward pass manually.
    """

    circuit_stats: Mapping[str, float] = field(default_factory=dict)
    """Aggregate depth/width stats across the post-fan-out final batch's
    DAG bodies — the pre-execution analogue of
    :attr:`~divi.backends.CircuitRunner.depth_history`.  Empty when the
    final batch has no DAG bodies (e.g. probability-mode pipelines that
    only carry bound QASM strings). Populated keys: ``min_depth``,
    ``max_depth``, ``mean_width``, ``min_width``, ``max_width``,
    ``mean_2q_depth``, and ``total_2q_gates`` (entangling gates summed over
    every submitted circuit, counting each body once per measurement circuit),
    plus ``mean_depth``, ``std_depth`` and ``std_width`` for a single program.
    A report merged from a group of programs omits the mean and the standard
    deviations where its members disagree, since a figure describing none of
    them is worse than none at all.

    On the default analytic path a stage that rewrites circuits (QEM variants,
    Pauli twirling) is previewed as placeholders, so these figures describe the
    circuits *entering* it — the counts stay exact, the shape does not. Pass
    ``force_circuit_generation=True`` to expand and measure for real.
    """

    def __post_init__(self) -> None:
        # Its own copies: the trace's dicts must not keep changing under the report.
        for name in ("env_artifacts", "circuit_stats"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        # Compared by identity, so a plain string from json.loads must be coerced.
        if not isinstance(self.cadence, PipelineCadence):
            try:
                object.__setattr__(self, "cadence", PipelineCadence(self.cadence))
            except ValueError:
                raise ValueError(
                    f"cadence must be one of "
                    f"{[member.value for member in PipelineCadence]}; got "
                    f"{self.cadence!r}."
                ) from None
        if self.circuit_stats:
            missing = _REQUIRED_CIRCUIT_STATS - set(self.circuit_stats)
            if missing:
                raise ValueError(
                    f"circuit_stats is missing {', '.join(sorted(missing))}. Pass "
                    f"either all of {', '.join(sorted(_REQUIRED_CIRCUIT_STATS))} "
                    "or no stats at all."
                )

    def __repr__(self) -> str:
        # The generated repr inlines every stage, artifact and stat, Pauli strings
        # included.
        stats = self.circuit_stats
        width = f", {stats['max_width']:g} qubits" if stats else ""
        # "…shots per evaluation" qualifies the counts; "…shots, once" stands apart.
        joiner = " " if self.cadence is PipelineCadence.PER_EVALUATION else ", "
        return (
            f"DryRunReport({self.pipeline_name}: "
            f"{_cost_headline(self.total_circuits, self.total_shots)}"
            f"{joiner}{_cadence_label(self.cadence)}{width}, "
            f"{len(self.stages)} stage{_plural(len(self.stages))})"
        )


EnsembleReports: TypeAlias = dict[Hashable, dict[str, DryRunReport]]


def _two_qubit_depth(dag: DAGCircuit) -> int:
    """Longest chain of two-qubit gates connected by shared qubits.

    Mirrors ``DAGCircuit.depth()`` semantics but restricted to gates
    whose ``qargs`` length is 2.  Single-qubit gates are ignored, even
    when they would extend a circuit's overall depth — the goal here is
    to surface the dominant fidelity predictor on superconducting
    hardware, where two-qubit gates are an order of magnitude noisier
    than single-qubit ones.

    Internal helper: shared with :class:`~divi.pipeline.stages.CircuitSpecStage`
    so its ``introspect()`` can report ``depth_2q`` without recomputing
    the same walk.
    """
    qubit_depth: dict = {}
    for node in dag.two_qubit_ops():
        qa, qb = node.qargs
        d = max(qubit_depth.get(qa, 0), qubit_depth.get(qb, 0)) + 1
        qubit_depth[qa] = qubit_depth[qb] = d
    return max(qubit_depth.values(), default=0)


def _aggregate_circuit_stats(batch: MetaCircuitBatch) -> dict[str, float]:
    """Compute mean/std/min/max for depth and width across a batch's DAG bodies.

    Pre-execution analogue of :attr:`~divi.backends.CircuitRunner.depth_history`
    aggregates: walks every ``circuit_bodies`` DAG in the batch and
    summarises its depth, two-qubit depth, and qubit count.  Returns an
    empty dict when no DAG bodies are reachable (e.g. probability-mode
    pipelines whose final batch carries only bound QASM strings).
    """
    depths: list[int] = []
    twoq_depths: list[int] = []
    widths: list[int] = []
    total_2q_gates = 0
    for mc in batch.values():
        # Stats come from the DAGs; qasm_bodies (QASM strings) are skipped —
        # binding doesn't change depth or width. Bodies sharing a tag collapse on
        # submission, so they are counted once, as the circuit total counts them.
        distinct = dict(mc.circuit_bodies)
        if not distinct:
            continue
        for dag in distinct.values():
            depths.append(dag.depth())
            twoq_depths.append(_two_qubit_depth(dag))
            widths.append(dag.num_qubits())
        # Depth and width are per-circuit, but the gate *total* scales with how
        # many circuits each body becomes. Parameter binding fans out into
        # ``qasm_bodies``, which this loop never sees, so take the scale from the
        # submitted count rather than from the measurement circuits alone.
        gates_per_body = sum(len(dag.two_qubit_ops()) for dag in distinct.values())
        total_2q_gates += gates_per_body * _submitted_count(mc) // len(distinct)

    if not depths:
        return {}

    return {
        "mean_depth": round(mean(depths), 2),
        "std_depth": round(pstdev(depths), 2),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "mean_2q_depth": round(mean(twoq_depths), 2),
        "total_2q_gates": total_2q_gates,
        "mean_width": round(mean(widths), 2),
        "std_width": round(pstdev(widths), 2),
        "min_width": min(widths),
        "max_width": max(widths),
    }


def _distinct_body_tags(mc: MetaCircuit) -> int:
    """Bodies that survive submission — execution keys each circuit by its tag
    (see ``_compile_batch``), so bodies sharing one collapse into a single job."""
    return len({tag for tag, _ in _effective_bodies(mc)})


def _logical_count(mc: MetaCircuit, *, count_observable_terms: bool = True) -> int:
    # Distinct tags, matching the submitted count: counting collapsed bodies here
    # would advertise a fan-out the total (correctly) omits.
    n_bodies = _distinct_body_tags(mc)
    if mc.measurement_qasms:
        return n_bodies * len(mc.measurement_qasms)
    if count_observable_terms and mc.observable is not None:
        # Naive expval baseline: one circuit per Pauli term, so that grouping
        # at the measurement stage shows up as a reduction.
        return n_bodies * sum(len(o) for o in mc.observable)
    return n_bodies


def _batch_logical_circuits(
    batch: MetaCircuitBatch, *, count_observable_terms: bool = True
) -> int:
    return sum(
        _logical_count(mc, count_observable_terms=count_observable_terms)
        for mc in batch.values()
    )


def _submitted_count(mc: MetaCircuit) -> int:
    """Circuits one final-batch MetaCircuit really submits — the logical count
    without the naive per-Pauli-term baseline."""
    return _logical_count(mc, count_observable_terms=False)


def _total_shots(batch: MetaCircuitBatch, env: PipelineEnv) -> int:
    """Total shots across the final batch.

    A ``shot_distribution`` allocates per-group shot budgets (``group_shots``),
    so those circuits sum their own allocation; otherwise every circuit runs at
    the backend's configured shots. Returns ``0`` when the backend reports no
    shot count (analytic expectation values consume none).
    """
    backend_shots = (
        env.shots_override
        if env.shots_override is not None
        else getattr(env.backend, "shots", None)
    )
    total = 0
    for mc in batch.values():
        if mc.group_shots:
            # Distinct tags, matching total_circuits: a collapsed body is never
            # submitted, so it is never billed shots either.
            total += _distinct_body_tags(mc) * sum(mc.group_shots.values())
        elif backend_shots is not None:
            total += _submitted_count(mc) * backend_shots
    return total


def _measures_observable_per_term(trace: PipelineTrace) -> bool:
    """Whether the pipeline reads the observable out term-by-term.

    ``MeasurementStage`` records QWC groups (``measurement_groups``) on every
    expectation-value path and leaves them empty for sampling or raw-bitstring
    terminals (e.g. PCE). A non-empty ``measurement_groups`` on the final batch
    is the signal — not the observable field, which lingers even where it is
    never measured per term.
    """
    return any(mc.measurement_groups for mc in trace.final_batch.values())


#: Coefficient precision for the objective fingerprint. Coarse enough that
#: recomputing a Hamiltonian doesn't split a group on float noise, fine enough to
#: separate objectives a user would call different.
_OBJECTIVE_COEFF_DIGITS = 9


def _observable_fingerprint(obs: Any) -> Hashable:
    if isinstance(obs, tuple):
        return tuple(_observable_fingerprint(o) for o in obs)
    paulis = getattr(obs, "paulis", None)
    coeffs = getattr(obs, "coeffs", None)
    if paulis is None or coeffs is None:
        return repr(obs)
    return (
        tuple(str(p) for p in paulis),
        tuple(
            (
                round(complex(c).real, _OBJECTIVE_COEFF_DIGITS),
                round(complex(c).imag, _OBJECTIVE_COEFF_DIGITS),
            )
            for c in coeffs
        ),
    )


#: Stage metadata that changes the objective without changing the observable.
_OBJECTIVE_SHAPING_KEYS = ("supervised",)


def _objective_fingerprint(
    trace: PipelineTrace,
    stages: Sequence[StageInfo],
    terminal_metadata: Mapping[str, Any],
) -> Hashable:
    """Fingerprint what the pipeline optimizes.

    Combines the observable's coefficients, the loss shaping applied over it, and
    the measuring stage's own settings — a sampled readout has no observable, and
    two readouts over identical Pauli terms can still compute different objectives.

    ``terminal_metadata`` must come from the stage that handles measurement, not
    from the last stage: ``ParameterBindingStage`` is required to go last, and its
    settings say nothing about the objective.
    """
    shaping = tuple(
        (key, value)
        for key in _OBJECTIVE_SHAPING_KEYS
        if (value := _stage_metadata_value_in(stages, key)) is not None
    )
    observables = tuple(
        _observable_fingerprint(mc.observable)
        for mc in trace.final_batch.values()
        if mc.observable is not None
    )
    settings = tuple(
        sorted(
            (k, v)
            for k, v in terminal_metadata.items()
            if isinstance(v, (str, int, float, bool, tuple, type(None)))
        )
    )
    key = (observables, shaping, settings)
    return key if any(key) else None


def _terminal_metadata(
    stages: Sequence[Stage], infos: Sequence[StageInfo]
) -> Mapping[str, Any]:
    """Metadata of the stage that handles measurement — the one that decides what a
    sampled readout is actually measuring.

    Falls back to the last stage's metadata only when no stage claims measurement,
    which a valid pipeline cannot be.
    """
    for stage, info in zip(stages, infos):
        # Only BundleStage declares it; a spec stage has no such attribute.
        if getattr(stage, "handles_measurement", False):
            return info.metadata
    return infos[-1].metadata if infos else {}


def _stage_metadata_value_in(stages: Sequence[StageInfo], *keys: str) -> Any:
    """The first of ``keys`` any stage reports, in the order given."""
    for key in keys:
        for stage in stages:
            if key in stage.metadata:
                return stage.metadata[key]
    return None


def _safe_introspect(
    stage: Stage, batch: MetaCircuitBatch, env: PipelineEnv, token: Any
) -> dict[str, Any]:
    """A stage's introspection metadata, or a note naming why it is missing.

    ``introspect`` is optional detail on the report, so one stage's failure is
    recorded against that stage rather than ending the preview.
    """
    try:
        meta = stage.introspect(batch, env, token)
    except Exception as exc:
        return {"introspect failed": f"{type(exc).__name__}: {exc}"}
    if not isinstance(meta, dict):
        return {"introspect failed": f"expected a dict, got {type(meta).__name__}"}
    return meta


def dry_run_pipeline(
    name: str,
    trace: PipelineTrace,
    stages: tuple[Stage, ...],
    env: PipelineEnv,
    cadence: PipelineCadence,
) -> DryRunReport:
    """Analyze a pipeline trace and compute per-stage factor.

    ``name`` and ``cadence`` describe the routine being previewed; neither is
    derivable from the trace.
    """
    infos: list[StageInfo] = []

    count_observable_terms = _measures_observable_per_term(trace)

    spec_stage = stages[0]
    spec_token = trace.stage_tokens[0] if trace.stage_tokens else None
    spec_meta = _safe_introspect(spec_stage, trace.initial_batch, env, spec_token)
    prev_logical = _batch_logical_circuits(
        trace.initial_batch, count_observable_terms=count_observable_terms
    )
    infos.append(
        StageInfo(
            name=type(spec_stage).__name__,
            axis=getattr(spec_stage, "axis_name", None),
            factor=float(prev_logical),
            metadata=spec_meta,
            label=getattr(spec_stage, "name", None),
        )
    )

    for i, expansion in enumerate(trace.stage_expansions):
        stage = stages[i + 1]
        cur_logical = _batch_logical_circuits(
            expansion.batch, count_observable_terms=count_observable_terms
        )

        if prev_logical:
            factor = cur_logical / prev_logical
        else:
            factor = float(cur_logical)

        token = trace.stage_tokens[i + 1] if i + 1 < len(trace.stage_tokens) else None
        meta = _safe_introspect(stage, expansion.batch, env, token)

        infos.append(
            StageInfo(
                name=type(stage).__name__,
                axis=getattr(stage, "axis_name", None),
                factor=factor,
                metadata=meta,
                label=getattr(stage, "name", None),
            )
        )
        prev_logical = cur_logical

    total = _batch_logical_circuits(trace.final_batch, count_observable_terms=False)

    return DryRunReport(
        pipeline_name=name,
        stages=tuple(infos),
        total_circuits=total,
        total_shots=_total_shots(trace.final_batch, env),
        cadence=cadence,
        objective_fingerprint=_objective_fingerprint(
            trace, infos, _terminal_metadata(stages, infos)
        ),
        env_artifacts=dict(trace.env_artifacts),
        circuit_stats=_aggregate_circuit_stats(trace.final_batch),
    )


def _plural(count: int) -> str:
    return "" if count == 1 else "s"


def _cost_headline(circuits: int, shots: int) -> str:
    """``"N circuits · M shots"`` — the shot clause is dropped when unknown (0)."""
    head = f"{circuits:,} circuit{_plural(circuits)}"
    if shots:
        head += f" · {shots:,} shot{_plural(shots)}"
    return head


def _cadence_label(cadence: PipelineCadence) -> str:
    return "once" if cadence is PipelineCadence.ONCE else "per evaluation"
