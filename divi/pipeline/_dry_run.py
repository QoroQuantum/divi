# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Dry-run analysis for circuit pipelines.

Inspects a :class:`PipelineTrace` (produced by
:meth:`CircuitPipeline.run_forward_pass`) and reports the fan-out
factor introduced by each stage, without executing any circuits.
"""

from typing import Any, NamedTuple

from rich.console import Console
from rich.tree import Tree

from divi.pipeline.abc import (
    MetaCircuitBatch,
    PipelineEnv,
    PipelineTrace,
    Stage,
)


class StageInfo(NamedTuple):
    """Per-stage fan-out report."""

    name: str
    axis: str | None
    fan_out: int
    cumulative_bodies: int
    cumulative_measurements: int
    metadata: dict[str, Any]


class DryRunReport(NamedTuple):
    """Complete dry-run report for a single pipeline."""

    pipeline_name: str
    stages: tuple[StageInfo, ...]
    total_circuits: int


def _count_qasms(batch: MetaCircuitBatch) -> tuple[int, int, int]:
    """Return (n_meta_circuits, total_bodies, total_measurements)."""
    n_mc = len(batch)
    total_bodies = sum(len(mc.circuit_body_qasms or ()) for mc in batch.values())
    total_meas = sum(len(mc.measurement_qasms or ()) for mc in batch.values())
    return n_mc, total_bodies, total_meas


def dry_run_pipeline(
    name: str,
    trace: PipelineTrace,
    stages: tuple[Stage, ...],
    env: PipelineEnv | None = None,
) -> DryRunReport:
    """Analyze a pipeline trace and compute per-stage fan-out."""
    infos: list[StageInfo] = []

    # SpecStage (stages[0])
    n_mc, prev_bodies, prev_meas = _count_qasms(trace.initial_batch)
    spec_stage = stages[0]
    spec_token = trace.stage_tokens[0] if trace.stage_tokens else None
    spec_meta = spec_stage.introspect(trace.initial_batch, env, spec_token)
    infos.append(
        StageInfo(
            name=type(spec_stage).__name__,
            axis=getattr(spec_stage, "axis_name", None),
            fan_out=n_mc,
            cumulative_bodies=prev_bodies,
            cumulative_measurements=prev_meas,
            metadata=spec_meta,
        )
    )

    # BundleStages (stages[1:], corresponding to trace.stage_expansions)
    for i, expansion in enumerate(trace.stage_expansions):
        stage = stages[i + 1]
        _, cur_bodies, cur_meas = _count_qasms(expansion.batch)

        if cur_meas > 0 and prev_meas == 0:
            body_fan = max(cur_bodies // prev_bodies, 1) if prev_bodies else 1
            meas_fan = cur_meas // max(n_mc, 1)
            fan_out = max(body_fan, meas_fan)
        elif cur_bodies != prev_bodies:
            fan_out = cur_bodies // prev_bodies if prev_bodies else cur_bodies
        elif cur_meas != prev_meas and prev_meas > 0:
            fan_out = cur_meas // prev_meas
        else:
            fan_out = 1

        # Token index: stage_tokens[0] = spec_token, stage_tokens[1+] = bundle tokens
        token = trace.stage_tokens[i + 1] if i + 1 < len(trace.stage_tokens) else None
        meta = stage.introspect(expansion.batch, env, token)

        infos.append(
            StageInfo(
                name=type(stage).__name__,
                axis=getattr(stage, "axis_name", None),
                fan_out=fan_out,
                cumulative_bodies=cur_bodies,
                cumulative_measurements=cur_meas,
                metadata=meta,
            )
        )
        prev_bodies, prev_meas = cur_bodies, cur_meas

    total = sum(
        len(mc.circuit_body_qasms or ()) * max(len(mc.measurement_qasms or ()), 1)
        for mc in trace.final_batch.values()
    )

    return DryRunReport(pipeline_name=name, stages=tuple(infos), total_circuits=total)


def format_dry_run(reports: dict[str, DryRunReport]) -> None:
    """Print dry-run reports as rich trees with stage metadata."""
    console = Console()

    for report in reports.values():
        tree = Tree(f"[bold]{report.pipeline_name}[/bold]")
        factors = []

        for stage in report.stages:
            axis_str = f" [dim]\\[{stage.axis}][/dim]" if stage.axis else ""
            if stage.fan_out > 1:
                fan_str = f"[bold yellow]×{stage.fan_out}[/bold yellow]"
                factors.append(str(stage.fan_out))
            else:
                fan_str = str(stage.fan_out)

            node = tree.add(f"[cyan]{stage.name}[/cyan]{axis_str} → {fan_str}")

            # Render metadata as sub-items
            for key, val in stage.metadata.items():
                if isinstance(val, list) and len(val) == 2:
                    node.add(f"[green]{key}: {val[0]} .. {val[1]}[/green]")
                elif isinstance(val, float):
                    node.add(f"[green]{key}: {val:.4f}[/green]")
                else:
                    node.add(f"[green]{key}: {val}[/green]")

        total_str = " × ".join(factors) if factors else "1"
        tree.add(
            f"[bold]Total: {total_str} = {report.total_circuits:,} circuits[/bold]"
        )

        console.print(tree)
        console.print()
