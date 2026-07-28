# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Rendering for dry-run reports.

Turns the :class:`~divi.pipeline.DryRunReport` objects produced by
:mod:`divi.pipeline._dry_run` into rich trees: one program's routines, or a whole
ensemble's, collapsed into compact rows or grouped by shape when there are many.
"""

import re
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import replace
from statistics import mean
from typing import Any, TextIO, cast
from warnings import warn

from rich.console import Console
from rich.markup import escape
from rich.tree import Tree

from divi.pipeline._dry_run import (
    DryRunReport,
    EnsembleReports,
    _cadence_label,
    _cost_headline,
    _plural,
    _stage_metadata_value_in,
)
from divi.pipeline._preprocessor import PipelineCadence

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_text(value: Any) -> str:
    """Render a caller-supplied label: escape rich markup, make control characters
    visible so a newline cannot forge a tree row and an escape byte cannot emit
    live ANSI."""
    text = _CONTROL_CHARS.sub(lambda m: repr(m.group(0))[1:-1], str(value))
    return escape(text)


#: Rendered for a program that registered no preprocessors (nothing to preview).
_NO_PREPROCESSORS_ROW = "[dim](no preprocessors — 0 circuits)[/dim]"

#: Auto-style thresholds (used when ``format_dry_run`` is called without an
#: explicit ``style``): up to this many programs render as full per-program
#: trees (``verbose``) — few enough to read in full.
_AUTO_VERBOSE_MAX = 3
#: Up to this many programs render one summary line each (``compact``); beyond
#: it, fall back to ``grouped`` so large sweeps/partitions collapse identical
#: programs instead of scrolling hundreds of rows.
_AUTO_COMPACT_MAX = 16


def _auto_style(n_programs: int) -> str:
    if n_programs <= _AUTO_VERBOSE_MAX:
        return "verbose"
    if n_programs <= _AUTO_COMPACT_MAX:
        return "compact"
    return "grouped"


def _declared_n_param_sets(report: DryRunReport) -> int | None:
    """Parameter sets a stage reported binding, if any stage reports one."""
    return _stage_metadata_value_in(report.stages, "n_param_sets")


def _format_factor(factor: float) -> tuple[str, str, str | None]:
    # Returns (line_token, total_token, total_op). total_op=None omits this
    # stage from the Total product line.
    if factor == 1:
        return "1", "", None
    if factor == 0:
        # Degenerate: a stage drove the logical count to zero. Show a bare
        # "0" rather than a nonsensical "÷0".
        return "0", "", None
    if factor > 1:
        token = f"{factor:g}"
        return f"[bold yellow]×{token}[/bold yellow]", token, "×"
    reciprocal = 1.0 / factor if factor else 0.0
    if reciprocal and abs(reciprocal - round(reciprocal)) < 1e-9:
        token = f"{int(round(reciprocal))}"
    else:
        token = f"{reciprocal:.3g}"
    return f"[bold green]÷{token}[/bold green]", token, "÷"


def _factor_token(factor: float) -> str:
    """A stage factor as it reads on its own row (``×10``, ``÷2.8``, ``1``)."""
    line_token, _, _ = _format_factor(factor)
    return re.sub(r"\[/?[a-z ]+\]", "", line_token)


def _expression_reconciles(report: DryRunReport) -> bool:
    """Whether the per-stage factors multiply out to the reported circuit count."""
    product = 1.0
    for stage in report.stages:
        product *= stage.factor or 1.0
    return round(product) == report.total_circuits


def _total_expression(report: DryRunReport) -> str:
    """Multiplicative fan-out expression, e.g. ``14 × 10 × 10 ÷ 2.8``.

    Empty when the pipeline has at most one contributing factor (the total
    circuit count already tells the whole story).
    """
    factors: list[tuple[str, str]] = []
    for stage in report.stages:
        _, total_token, total_op = _format_factor(stage.factor)
        if total_op is not None:
            factors.append((total_op, total_token))
    if len(factors) <= 1:
        return ""
    return factors[0][1] + "".join(f" {op} {tok}" for op, tok in factors[1:])


def _populate_pipeline_node(node: Tree, report: DryRunReport) -> None:
    """Attach per-stage rows, the total line, and the summary line to ``node``."""
    for idx, stage in enumerate(report.stages):
        axis_str = f" [dim]\\[{_safe_text(stage.axis)}][/dim]" if stage.axis else ""
        line_token, total_token, total_op = _format_factor(stage.factor)

        # Spec stage is the source, not a multiplier — bare number on its line.
        display_token = total_token if idx == 0 and total_op == "×" else line_token
        stage_node = node.add(
            f"[cyan]{_safe_text(stage.name)}[/cyan]{axis_str} → {display_token}"
        )

        for key, val in stage.metadata.items():
            label = _safe_text(key)
            # A two-element sequence renders as a range.
            if isinstance(val, (list, tuple)) and len(val) == 2:
                stage_node.add(
                    f"[green]{label}: {_safe_text(val[0])} .. "
                    f"{_safe_text(val[1])}[/green]"
                )
            elif isinstance(val, float):
                stage_node.add(f"[green]{label}: {val:.4f}[/green]")
            else:
                stage_node.add(f"[green]{label}: {_safe_text(val)}[/green]")

    expr = _total_expression(report)
    # Shown only when the factors multiply out to the total.
    lead = f"{expr} = " if expr and _expression_reconciles(report) else ""
    # A population optimizer binds its whole working set per evaluation; name the
    # count so the label doesn't imply a single vector.
    n_sets = _declared_n_param_sets(report)
    scope = (
        f"{_cadence_label(report.cadence)}, all {n_sets} parameter sets"
        if n_sets and n_sets > 1
        else _cadence_label(report.cadence)
    )
    node.add(
        f"[bold]Total ({scope}): {lead}"
        f"{_cost_headline(report.total_circuits, report.total_shots)}[/bold]"
    )

    if report.circuit_stats:
        stats = report.circuit_stats
        # A merged group omits mean_depth where its members differ; show the span.
        if "mean_depth" not in stats:
            depth_part = f"depth {stats['min_depth']}-{stats['max_depth']}"
        elif stats["min_depth"] == stats["max_depth"]:
            depth_part = f"avg depth {stats['mean_depth']:g}"
        else:
            depth_part = (
                f"avg depth {stats['mean_depth']:g} "
                f"(range {stats['min_depth']}-{stats['max_depth']})"
            )
        if stats["min_width"] == stats["max_width"]:
            width_part = f"width {stats['min_width']}"
        else:
            width_part = f"width {stats['min_width']}-{stats['max_width']}"
        summary = f"Summary: {depth_part}, {width_part}"
        if stats.get("total_2q_gates"):
            n_2q = int(stats["total_2q_gates"])
            summary += f", {n_2q} 2q-gate{_plural(n_2q)} total"
        elif stats.get("max_2q_gates"):
            summary += (
                f", {int(stats['min_2q_gates'])}-{int(stats['max_2q_gates'])} "
                "2q-gates total per program"
            )
        node.add(f"[bold]{summary}[/bold]")


def _render_reports(reports: dict[str, DryRunReport], console: Console) -> None:
    """Print one rich tree per pipeline — the single-program layout."""
    for report in reports.values():
        tree = Tree(f"[bold]{_safe_text(report.pipeline_name)}[/bold]")
        _populate_pipeline_node(tree, report)
        console.print(tree)
        console.print()


def _populate_program_tree(root: Tree, reports: dict[str, DryRunReport]) -> None:
    """Attach one pipeline sub-tree per report to ``root``."""
    if not reports:
        root.add(_NO_PREPROCESSORS_ROW)
        return
    for report in reports.values():
        node = root.add(f"[bold]{_safe_text(report.pipeline_name)}[/bold]")
        _populate_pipeline_node(node, report)


def _ensemble_max_stat(nested: EnsembleReports, key: str) -> float | None:
    """The largest ``key`` across every program. ``None`` when no pipeline carries
    DAG bodies, so there is nothing to measure."""
    values = [
        v
        for reports in nested.values()
        if (v := _program_max_stat(reports, key)) is not None
    ]
    return max(values) if values else None


def _cadence_totals(
    report_dicts: Iterable[dict[str, DryRunReport]], cadence: PipelineCadence
) -> tuple[int, int]:
    """(circuits, shots) for one evaluation, summed over pipelines with this
    cadence."""
    circuits = shots = 0
    for reports in report_dicts:
        for report in reports.values():
            if report.cadence is cadence:
                circuits += report.total_circuits
                shots += report.total_shots
    return circuits, shots


def _cadence_summary(
    label: str,
    report_dicts: Iterable[dict[str, DryRunReport]],
    *,
    multiplier: int = 1,
    width: float | None = None,
    depth: float | None = None,
) -> str:
    """A bold total line reporting recurring and one-time costs separately.
    Shared by the ensemble grand total and the grouped-view subtotal."""
    report_dicts = list(report_dicts)
    groups = []
    for cadence in (PipelineCadence.PER_EVALUATION, PipelineCadence.ONCE):
        circuits, shots = _cadence_totals(report_dicts, cadence)
        if circuits:
            groups.append((cadence, circuits * multiplier, shots * multiplier))
    if not groups:
        groups = [(PipelineCadence.PER_EVALUATION, 0, 0)]

    shape_bits = []
    if width is not None:
        shape_bits.append(f"widest {int(width)}q")
    if depth is not None:
        shape_bits.append(f"deepest {int(depth)}")
    width_suffix = f" · {' · '.join(shape_bits)}" if shape_bits else ""

    if len(groups) == 1:
        cadence, circuits, shots = groups[0]
        return (
            f"[bold]{label} ({_cadence_label(cadence)}): "
            f"{_cost_headline(circuits, shots)}{width_suffix}[/bold]"
        )
    body = "; ".join(
        f"{_cadence_label(cadence)}: {_cost_headline(circuits, shots)}"
        for cadence, circuits, shots in groups
    )
    return f"[bold]{label} — {body}{width_suffix}[/bold]"


def _ensemble_total_line(nested: EnsembleReports) -> str:
    return _cadence_summary(
        "Ensemble total",
        nested.values(),
        width=_ensemble_max_stat(nested, "max_width"),
        depth=_ensemble_max_stat(nested, "max_depth"),
    )


def _ids_preview(ids: list[Hashable]) -> str:
    if len(ids) <= 3:
        return ", ".join(_safe_text(i) for i in ids)
    return f"{_safe_text(ids[0])} … {_safe_text(ids[-1])}"


def _ensemble_header(nested: EnsembleReports) -> str:
    n = len(nested)
    return f"[bold]Ensemble Dry Run[/bold]  ({n} program{_plural(n)})"


#: Metadata keys reporting a parameter count, in preference order.
_PARAM_COUNT_KEYS = ("n_bound_params", "n_params")


def _declared_n_params(report: DryRunReport) -> int | None:
    """The parameter count a stage reported, if any stage reports one."""
    return _stage_metadata_value_in(report.stages, *_PARAM_COUNT_KEYS)


def _program_signature(reports: dict[str, DryRunReport]) -> tuple:
    """Fingerprint used to bucket programs that traverse their pipelines alike.

    Keys on pipeline *shape* (the ordered stages and the axes they fan out over)
    and on what the traversal produces. Deliberately excludes the per-stage
    factors: a compensating pair — a spec stage emitting one circuit per
    Hamiltonian term, then a measurement stage grouping them back down — differs
    numerically between programs while leaving the outcome identical, and
    splitting on it fragments a sweep that is uniform in every respect a reader
    cares about. Where members' factors do differ, :func:`_merge_group_report`
    surfaces the range. ``circuit_stats`` is likewise excluded (depth and width
    track qubit count) and enveloped on merge instead.
    """
    return tuple(
        (
            name,
            tuple((s.name, s.axis) for s in report.stages),
            report.total_circuits,
            report.total_shots,
            report.cadence,
            # Discrete structure only. Depth and gate counts vary across uniform
            # sweeps, so keying on them would put every member in its own group.
            report.circuit_stats.get("max_width"),
            _declared_n_params(report),
            report.objective_fingerprint,
        )
        for name, report in reports.items()
    )


def _program_max_stat(reports: dict[str, DryRunReport], key: str) -> float | None:
    """The largest ``key`` across one program's pipelines."""
    values = [r.circuit_stats[key] for r in reports.values() if r.circuit_stats]
    return max(values) if values else None


def _report_n_samples(reports: dict[str, DryRunReport]) -> float | None:
    """Data-axis size, if any pipeline fans out over one."""
    values = (_stage_metadata_value_in(r.stages, "n_samples") for r in reports.values())
    return next((v for v in values if v is not None), None)


def _render_compact(nested: EnsembleReports, console: Console) -> None:
    root = Tree(_ensemble_header(nested))
    for program_id, reports in nested.items():
        shape = [
            f"{int(v)}{unit}"
            for v, unit in (
                (_program_max_stat(reports, "max_width"), "q"),
                (_program_max_stat(reports, "max_depth"), " deep"),
                (_report_n_samples(reports), " samples"),
            )
            if v is not None
        ]
        shape_tag = f"  [dim]({', '.join(shape)})[/dim]" if shape else ""
        prog_node = root.add(f"{_safe_text(program_id)}{shape_tag}")
        if not reports:
            prog_node.add(_NO_PREPROCESSORS_ROW)
        for name, report in reports.items():
            one_time = report.cadence is PipelineCadence.ONCE
            headline = _cost_headline(report.total_circuits, report.total_shots)
            expr = _total_expression(report)
            # A one-time pipeline takes the cadence tag below instead of a scope.
            scope = "" if one_time else _cadence_label(PipelineCadence.PER_EVALUATION)
            bits = [b for b in (scope, expr) if b]
            detail = f"  [dim]{', '.join(bits)}[/dim]" if bits else ""
            # Flag one-time pipelines so their count isn't read as recurring.
            once = "  [dim]once[/dim]" if one_time else ""
            if _has_sampled_count(report):
                # Compact drops the stage metadata that says so elsewhere.
                once += "  [dim]sampled count[/dim]"
            prog_node.add(f"[cyan]{_safe_text(name)}[/cyan]: {headline}{detail}{once}")
    root.add(_ensemble_total_line(nested))
    console.print(root)


def _render_verbose(nested: EnsembleReports, console: Console) -> None:
    console.print(_ensemble_header(nested))
    for program_id, reports in nested.items():
        root = Tree(f"[bold]{_safe_text(program_id)}[/bold]")
        _populate_program_tree(root, reports)
        console.print(root)
        console.print()
    console.print(_ensemble_total_line(nested))


def _aggregate_group_stats(reports: list[DryRunReport]) -> Mapping[str, float]:
    """Combine the per-program ``circuit_stats`` of one pipeline across a group.

    Members of a group share a pipeline shape but not circuit content, so a
    single member's depth would misrepresent the group. Report the true
    envelope: min/max span the members' extremes.

    ``mean_depth`` is emitted only when the members agree. Averaging genuinely
    different circuits produces a figure describing none of them — an ansatz
    comparison whose members are 24 and 100 deep does not have an "average depth
    of 62" — so where they differ the renderer shows the span alone.

    No ``std_*`` is emitted for a multi-member group either: a pooled
    circuit-level std can't be recovered from per-program summaries, and
    reporting the std *of the means* under the same label as the single-program
    circuit-level std would mislead (it can read ``± 0`` beside a wide range).
    A singleton keeps its own (circuit-level) stats.
    """
    stats = [r.circuit_stats for r in reports if r.circuit_stats]
    if len(stats) <= 1:
        return stats[0] if stats else {}

    def col(key: str) -> list[float]:
        return [s[key] for s in stats]

    merged = {
        "min_depth": min(col("min_depth")),
        "max_depth": max(col("max_depth")),
        "mean_2q_depth": round(mean(col("mean_2q_depth")), 2),
        "mean_width": round(mean(col("mean_width")), 2),
        "min_width": min(col("min_width")),
        "max_width": max(col("max_width")),
    }
    if len(set(col("mean_depth"))) == 1:
        merged["mean_depth"] = stats[0]["mean_depth"]
    if all("total_2q_gates" in s for s in stats):
        # Per-program: the subtotal line applies the group size, once.
        gates = col("total_2q_gates")
        if len(set(gates)) == 1:
            merged["total_2q_gates"] = gates[0]
        else:
            merged["min_2q_gates"] = min(gates)
            merged["max_2q_gates"] = max(gates)
    return merged


def _metadata_equal(left: Any, right: Any) -> bool:
    """Compare two metadata values without assuming ``==`` yields a bool.

    A custom stage's ``introspect`` may return an array, whose elementwise ``==``
    is not truth-testable; falling back to the rendered form is enough, since a
    metadata value that renders identically also displays identically.
    """
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _has_sampled_count(report: DryRunReport) -> bool:
    """Whether any stage reported a fan-out it sampled rather than enumerated."""
    return any(
        str(stage.metadata.get("path_count", "")).startswith("sampled")
        for stage in report.stages
    )


def _shared_metadata(metas: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge group members' metadata, marking fields they disagree on.

    A field present and equal everywhere passes through unchanged. A field whose
    value differs between members is reported as ``mixed (a | b)`` rather than
    dropped: a vanished row is invisible unless the reader happens to have a
    uniform run to compare against, so silently omitting it hides exactly the
    divergence worth seeing (two different objectives in one sweep, say). A
    field missing from some members is dropped — it describes no shared fact.
    """
    merged: dict[str, Any] = {}
    for key, value in metas[0].items():
        if any(key not in m for m in metas):
            continue
        values = [m[key] for m in metas]
        if all(_metadata_equal(v, value) for v in values):
            merged[key] = value
            continue
        seen = list(dict.fromkeys(str(v) for v in values))
        preview = " | ".join(seen[:3])
        if len(seen) > 3:
            preview += f" | … (+{len(seen) - 3} more)"
        merged[key] = f"mixed ({preview})"
    return merged


def _merge_group_report(reports: list[DryRunReport]) -> DryRunReport:
    """Fold a group's per-pipeline reports into one representative report.

    Depth/width stats are enveloped across members and only metadata common to
    all is kept. Totals are identical by :func:`_program_signature`; a per-stage
    factor may not be, so a stage whose factor varies across the group reports
    the range alongside the representative's value rather than passing one
    member's number off as the group's.
    """
    rep = reports[0]
    merged_stages = []
    # Members share a stage sequence by construction: _program_signature keys on it.
    for aligned in zip(*(r.stages for r in reports)):
        metadata = _shared_metadata([stage.metadata for stage in aligned])
        factors = {stage.factor for stage in aligned}
        if len(factors) > 1:
            metadata = {
                **metadata,
                "factor_range": [
                    _factor_token(min(factors)),
                    _factor_token(max(factors)),
                ],
            }
        merged_stages.append(replace(aligned[0], metadata=metadata))
    return replace(
        rep, stages=tuple(merged_stages), circuit_stats=_aggregate_group_stats(reports)
    )


#: Traits the grouping key distinguishes, as ``(label, extractor)``. Split into
#: causes and consequences: a differing batch size also changes the circuit and
#: shot counts, and naming the consequence sends the reader hunting for a
#: difference in the wrong place.
_SIGNATURE_CAUSES = (
    ("the objective they optimize", lambda r: r.objective_fingerprint),
    ("batch size", lambda r: _stage_metadata_value_in(r.stages, "n_samples")),
    ("register width", lambda r: r.circuit_stats.get("max_width")),
    ("parameter count", lambda r: _declared_n_params(r)),
    ("pipeline shape", lambda r: tuple((s.name, s.axis) for s in r.stages)),
    ("cadence", lambda r: r.cadence),
)
_SIGNATURE_CONSEQUENCES = (
    ("circuit count", lambda r: r.total_circuits),
    ("shot count", lambda r: r.total_shots),
)


def _differing_traits(
    nested: EnsembleReports, traits: tuple[tuple[str, Any], ...]
) -> list[str]:
    # Over the routines every program has: differing routine *sets* would otherwise
    # make every trait differ at once.
    shared = set.intersection(*(set(reports) for reports in nested.values()))
    differing = []
    for label, extract in traits:
        seen = {
            tuple(
                sorted(
                    (
                        (name, extract(report))
                        for name, report in reports.items()
                        if name in shared
                    ),
                    key=lambda item: item[0],
                )
            )
            for reports in nested.values()
        }
        if len(seen) > 1:
            differing.append(label)
    return differing


def _distinguishing_trait(nested: EnsembleReports) -> str:
    """Name what actually separates these programs, for the grouped-view fallback.

    Guessing here is worse than saying nothing: a bond-length sweep agrees on every
    printed figure and differs only in its Hamiltonian, and a batch-size sweep
    agrees on the objective too. A blanket "they differ in shape, register,
    parameter count or objective" is then contradicted by the rows beside it.
    """
    routine_sets = {tuple(sorted(reports)) for reports in nested.values()}
    differing = ["the routines they run"] if len(routine_sets) > 1 else []
    differing += _differing_traits(nested, _SIGNATURE_CAUSES) or _differing_traits(
        nested, _SIGNATURE_CONSEQUENCES
    )
    if not differing:
        return "they differ in a trait the grouping key tracks"
    if len(differing) == 1:
        return f"they differ in {differing[0]}"
    return f"they differ in {', '.join(differing[:-1])} and {differing[-1]}"


def _render_grouped(nested: EnsembleReports, console: Console) -> None:
    groups: defaultdict[tuple, list[Hashable]] = defaultdict(list)
    for program_id, reports in nested.items():
        groups[_program_signature(reports)].append(program_id)

    if len(groups) == len(nested) > 1:
        # Nothing collapsed, so fall back rather than print a tree per program.
        console.print(
            f"[dim]These {len(nested)} programs are all distinct "
            f"({_distinguishing_trait(nested)}), so grouping would print one full "
            "tree each — showing the compact view instead.[/dim]"
        )
        _render_compact(nested, console)
        return

    console.print(_ensemble_header(nested))
    for ids in groups.values():
        members = [nested[i] for i in ids]
        rep_reports = members[0]
        count = len(ids)
        label = (
            f"[bold]{count} program{_plural(count)}[/bold]  "
            f"[dim]({_ids_preview(ids)})[/dim]"
        )
        root = Tree(label)
        if not rep_reports:
            root.add(_NO_PREPROCESSORS_ROW)
        for name in rep_reports:
            merged = _merge_group_report([m[name] for m in members])
            node = root.add(f"[bold]{_safe_text(merged.pipeline_name)}[/bold]")
            _populate_pipeline_node(node, merged)
        root.add(
            _cadence_summary(f"Subtotal (× {count})", [rep_reports], multiplier=count)
        )
        console.print(root)
        console.print()
    console.print(_ensemble_total_line(nested))


_ENSEMBLE_RENDERERS = {
    "compact": _render_compact,
    "grouped": _render_grouped,
    "verbose": _render_verbose,
}


def format_dry_run(
    reports: dict[str, DryRunReport] | EnsembleReports,
    *,
    style: str | None = None,
    file: TextIO | None = None,
    width: int | None = None,
) -> None:
    """Print dry-run reports as rich trees with stage metadata.

    Accepts either the flat single-program result from
    :meth:`~divi.qprog.QuantumProgram.dry_run`
    (``dict[str, DryRunReport]``) or the nested ensemble result from
    :meth:`~divi.qprog.ensemble.ProgramEnsemble.dry_run`
    (``dict[program_id, dict[str, DryRunReport]]``), dispatching on the shape
    of the values.

    Args:
        reports: A flat or nested dry-run report dict (see above).
        style: Ensemble rendering style — ``"compact"`` (one summary line per
            pipeline per program plus a grand total), ``"grouped"`` (one tree
            per distinct program shape with a count and subtotal), or
            ``"verbose"`` (a full tree per program). When ``None`` (the
            default), the style is chosen from the program count: ``verbose``
            for a handful, ``compact`` for a moderate number, and ``grouped``
            for large sweeps/partitions. The styles differ only in how they
            arrange *multiple programs*, so passing one with flat
            single-program input has nothing to select between and warns.
        file: Where to write. Defaults to stdout.
        width: Fixed output width in characters. Defaults to the terminal's, which
            falls back to 80 when output is redirected — narrow enough to wrap
            totals mid-number.
    """
    if isinstance(reports, DryRunReport):
        raise TypeError(
            "format_dry_run expects the dict returned by dry_run(), not a single "
            f"DryRunReport. Pass the whole result, or wrap it: "
            f'{{"{reports.pipeline_name}": report}}.'
        )
    if not isinstance(reports, Mapping):
        raise TypeError(
            f"format_dry_run expects the dict returned by dry_run(), got "
            f"{type(reports).__name__}. Call it first: "
            "format_dry_run(program.dry_run())."
        )
    if style is not None and style not in _ENSEMBLE_RENDERERS:
        raise ValueError(
            f"Unknown style {style!r}; expected one of "
            f"{', '.join(sorted(_ENSEMBLE_RENDERERS))}."
        )

    # Stated positively so NaN fails it too; a width of 0 renders an empty tree.
    if width is not None and not width >= 1:
        raise ValueError(f"width must be a positive number of columns, got {width}.")

    console = Console(file=file, width=width)
    if not reports:
        console.print(
            "[dim]No dry-run reports to display: this dict is empty. A program "
            "returns one entry per registered routine, so an empty result means "
            "none are registered — see _preprocessors().[/dim]"
        )
        return

    # Every value is checked before anything renders, so a bad one cannot leave
    # partial output above the traceback.
    first = next(iter(reports.values()))
    if not isinstance(first, (DryRunReport, Mapping)):
        raise TypeError(
            f"format_dry_run expects DryRunReport values (or, for an ensemble, "
            f"dicts of them), got {type(first).__name__}."
        )
    if isinstance(first, DryRunReport):
        for name, report in reports.items():
            if not isinstance(report, DryRunReport):
                raise TypeError(
                    f"format_dry_run expects DryRunReport values throughout, got "
                    f"{type(report).__name__} for {name!r}. A dict may not mix a "
                    "single program's reports with an ensemble's."
                )
        if style is not None:
            warn(
                f"style={style!r} was ignored: it selects between multi-program "
                "layouts, and this is a single program's report. Pass the nested "
                "result from ProgramEnsemble.dry_run() to use it.",
                UserWarning,
                stacklevel=2,
            )
        _render_reports(cast(dict[str, DryRunReport], reports), console)
        return

    nested = cast(EnsembleReports, reports)
    # ``EnsembleReports`` is a plain dict, so a hand-assembled one is expected.
    for program_id, program_reports in nested.items():
        if not isinstance(program_reports, Mapping):
            raise TypeError(
                f"format_dry_run expects each program's value to be a dict of "
                f"DryRunReport, got {type(program_reports).__name__} for program "
                f"{program_id!r}."
            )
        for name, report in program_reports.items():
            if not isinstance(report, DryRunReport):
                raise TypeError(
                    f"format_dry_run expects DryRunReport values, got "
                    f"{type(report).__name__} for {program_id!r}/{name!r}."
                )
    _ENSEMBLE_RENDERERS[style or _auto_style(len(nested))](nested, console)
