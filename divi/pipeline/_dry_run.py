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

from statistics import mean, pstdev
from typing import Any, NamedTuple

from qiskit.dagcircuit import DAGCircuit
from rich.console import Console
from rich.tree import Tree

from divi.circuits import MetaCircuit
from divi.pipeline.abc import (
    MetaCircuitBatch,
    PipelineEnv,
    PipelineTrace,
    Stage,
)


class StageInfo(NamedTuple):
    """Per-stage dry-run report.

    ``factor`` is the ratio of logical circuits after this stage to
    circuits before it. ``factor > 1`` is a fan-out (e.g.
    :class:`~divi.pipeline.stages.ParameterBindingStage`,
    :class:`~divi.pipeline.stages.PauliTwirlStage`); ``factor < 1`` is a
    reduction (e.g. observable grouping in
    :class:`~divi.pipeline.stages.MeasurementStage` collapsing N Pauli
    terms into M ≤ N commuting groups yields ``factor = M / N``).
    """

    name: str
    axis: str | None
    factor: float
    metadata: dict[str, Any]


class DryRunReport(NamedTuple):
    """Complete dry-run report for a single pipeline."""

    pipeline_name: str
    stages: tuple[StageInfo, ...]
    total_circuits: int
    env_artifacts: dict[str, Any] = {}
    """Stage-produced artifacts captured during the forward pass — e.g.
    ``per_group_shots`` (when a ``shot_distribution`` is configured on
    :class:`~divi.pipeline.stages.MeasurementStage`), ``ham_ops`` (for
    expval-native backends).  These are the same artifacts the pipeline
    would produce on a real run, so a dry-run report is the canonical
    surface for ``"what would my pipeline do?"`` introspection — no
    need to drop into private helpers or rerun the forward pass manually.
    """

    circuit_stats: dict[str, float] = {}
    """Aggregate depth/width stats across the post-fan-out final batch's
    DAG bodies — the pre-execution analogue of
    :attr:`~divi.backends.CircuitRunner.depth_history`.  Empty when the
    final batch has no DAG bodies (e.g. probability-mode pipelines that
    only carry bound QASM strings).  Populated keys: ``mean_depth``,
    ``std_depth``, ``min_depth``, ``max_depth``, ``mean_width``,
    ``std_width``, ``min_width``, ``max_width``, ``mean_2q_depth``.
    """


def _effective_bodies(mc: MetaCircuit) -> tuple:
    # Mirrors _compile_batch: bound bodies take priority over parametric DAGs.
    return mc.bound_circuit_bodies or mc.circuit_bodies or ()


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
    for mc in batch.values():
        # bound_circuit_bodies are QASM strings — re-parsing them only to
        # measure depth would be expensive, and in practice their structure
        # mirrors ``circuit_bodies`` modulo concrete parameter substitution,
        # which doesn't change depth or width.  Iterate the parametric DAGs
        # when available; fall back to nothing when a body has only bound
        # QASM (parameter-binding fast path stripped DAGs).
        for _tag, dag in mc.circuit_bodies:
            depths.append(dag.depth())
            twoq_depths.append(_two_qubit_depth(dag))
            widths.append(dag.num_qubits())

    if not depths:
        return {}

    return {
        "mean_depth": round(mean(depths), 2),
        "std_depth": round(pstdev(depths), 2),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "mean_2q_depth": round(mean(twoq_depths), 2),
        "mean_width": round(mean(widths), 2),
        "std_width": round(pstdev(widths), 2),
        "min_width": min(widths),
        "max_width": max(widths),
    }


def _logical_count(mc: MetaCircuit) -> int:
    n_bodies = len(_effective_bodies(mc))
    if mc.measurement_qasms:
        return n_bodies * len(mc.measurement_qasms)
    if mc.observable is not None:
        # Pre-measurement baseline: one circuit per Pauli term, so that
        # grouping at the measurement stage shows up as a reduction.
        return n_bodies * sum(len(o) for o in mc.observable)
    return n_bodies


def _batch_logical_circuits(batch: MetaCircuitBatch) -> int:
    return sum(_logical_count(mc) for mc in batch.values())


def dry_run_pipeline(
    name: str,
    trace: PipelineTrace,
    stages: tuple[Stage, ...],
    env: PipelineEnv,
) -> DryRunReport:
    """Analyze a pipeline trace and compute per-stage factor."""
    infos: list[StageInfo] = []

    spec_stage = stages[0]
    spec_token = trace.stage_tokens[0] if trace.stage_tokens else None
    spec_meta = spec_stage.introspect(trace.initial_batch, env, spec_token)
    prev_logical = _batch_logical_circuits(trace.initial_batch)
    infos.append(
        StageInfo(
            name=type(spec_stage).__name__,
            axis=getattr(spec_stage, "axis_name", None),
            factor=float(prev_logical),
            metadata=spec_meta,
        )
    )

    for i, expansion in enumerate(trace.stage_expansions):
        stage = stages[i + 1]
        cur_logical = _batch_logical_circuits(expansion.batch)

        if prev_logical:
            factor = cur_logical / prev_logical
        else:
            factor = float(cur_logical)

        token = trace.stage_tokens[i + 1] if i + 1 < len(trace.stage_tokens) else None
        meta = stage.introspect(expansion.batch, env, token)

        infos.append(
            StageInfo(
                name=type(stage).__name__,
                axis=getattr(stage, "axis_name", None),
                factor=factor,
                metadata=meta,
            )
        )
        prev_logical = cur_logical

    total = sum(
        len(_effective_bodies(mc)) * max(len(mc.measurement_qasms or ()), 1)
        for mc in trace.final_batch.values()
    )

    return DryRunReport(
        pipeline_name=name,
        stages=tuple(infos),
        total_circuits=total,
        env_artifacts=dict(trace.env_artifacts),
        circuit_stats=_aggregate_circuit_stats(trace.final_batch),
    )


def _format_factor(factor: float) -> tuple[str, str, str | None]:
    # Returns (line_token, total_token, total_op). total_op=None omits this
    # stage from the Total product line.
    if factor == 1:
        return "1", "", None
    if factor > 1:
        token = f"{factor:g}"
        return f"[bold yellow]×{token}[/bold yellow]", token, "×"
    reciprocal = 1.0 / factor if factor else 0.0
    if reciprocal and abs(reciprocal - round(reciprocal)) < 1e-9:
        token = f"{int(round(reciprocal))}"
    else:
        token = f"{reciprocal:.3g}"
    return f"[bold green]÷{token}[/bold green]", token, "÷"


def format_dry_run(reports: dict[str, DryRunReport]) -> None:
    """Print dry-run reports as rich trees with stage metadata."""
    console = Console()

    for report in reports.values():
        tree = Tree(f"[bold]{report.pipeline_name}[/bold]")
        factors: list[tuple[str, str]] = []

        for idx, stage in enumerate(report.stages):
            axis_str = f" [dim]\\[{stage.axis}][/dim]" if stage.axis else ""
            line_token, total_token, total_op = _format_factor(stage.factor)
            if total_op is not None:
                factors.append((total_op, total_token))

            # Spec stage is the source, not a multiplier — bare number on its line.
            display_token = total_token if idx == 0 and total_op == "×" else line_token
            node = tree.add(f"[cyan]{stage.name}[/cyan]{axis_str} → {display_token}")

            for key, val in stage.metadata.items():
                if isinstance(val, list) and len(val) == 2:
                    node.add(f"[green]{key}: {val[0]} .. {val[1]}[/green]")
                elif isinstance(val, float):
                    node.add(f"[green]{key}: {val:.4f}[/green]")
                else:
                    node.add(f"[green]{key}: {val}[/green]")

        if len(factors) > 1:
            total_str = factors[0][1]
            for op, tok in factors[1:]:
                total_str += f" {op} {tok}"
            tree.add(
                f"[bold]Total: {total_str} = {report.total_circuits:,} circuits[/bold]"
            )
        else:
            tree.add(f"[bold]Total: {report.total_circuits:,} circuits[/bold]")

        if report.circuit_stats:
            stats = report.circuit_stats
            # Depth: lead with the average so the reader knows which stat
            # this is. Add ± std and range only when there is spread.
            if stats["min_depth"] == stats["max_depth"]:
                depth_part = f"avg depth {stats['mean_depth']:g}"
            else:
                depth_part = (
                    f"avg depth {stats['mean_depth']:g} "
                    f"± {stats['std_depth']:g} "
                    f"(range {stats['min_depth']}-{stats['max_depth']})"
                )
            if stats["min_width"] == stats["max_width"]:
                width_part = f"width {stats['min_width']}"
            else:
                width_part = f"width {stats['min_width']}-{stats['max_width']}"
            tree.add(f"[bold]Summary: {depth_part}, {width_part}[/bold]")

        console.print(tree)
        console.print()
