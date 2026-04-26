# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QUBO/HUBO validation result container and serialization helpers."""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass, field

import dimod
import numpy as np
import scipy.sparse as sps


def _serialize_qubo_for_wire(qubo) -> dict[str, float]:
    """Convert any Divi QUBO/HUBO type to the wire-format dict expected by Usher.

    The wire format uses string keys like ``"0,1"`` mapping to float weights.

    Accepts:
        - ``np.ndarray`` / ``scipy.sparse.spmatrix`` — extract nonzero entries
        - ``dimod.BinaryQuadraticModel`` — iterate linear + quadratic terms
        - ``dict`` with tuple keys (HUBO) — stringify keys
        - ``dimod.BinaryPolynomial`` — iterate terms
        - ``BinaryOptimizationProblem`` — use ``.canonical_problem.terms``

    Returns:
        Dictionary with comma-separated string keys and float values.
    """
    # Lazy import to avoid circular dependency
    from divi.qprog.problems._binary import BinaryOptimizationProblem

    if isinstance(qubo, BinaryOptimizationProblem):
        terms = qubo.canonical_problem.terms
        return {
            ",".join(str(idx) for idx in term_key): float(coeff)
            for term_key, coeff in terms.items()
            if coeff != 0
        }

    if isinstance(qubo, dimod.BinaryQuadraticModel):
        wire: dict[str, float] = {}
        for var, bias in qubo.linear.items():
            if bias != 0:
                wire[f"{var},{var}"] = float(bias)
        for (u, v), bias in qubo.quadratic.items():
            if bias != 0:
                wire[f"{u},{v}"] = float(bias)
        return wire

    if isinstance(qubo, dimod.BinaryPolynomial):
        wire = {}
        for term, coeff in qubo.items():
            if coeff == 0:
                continue
            key = ",".join(str(v) for v in sorted(term))
            wire[key] = float(coeff)
        return wire

    if isinstance(qubo, np.ndarray):
        if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
            raise ValueError(
                f"QUBO matrix must be square, got shape {qubo.shape}"
            )
        # Emit upper-triangular form: diagonal as-is, off-diagonal
        # entries folded as Q_wire[i,j] = Q[i,j] + Q[j,i] for i < j.
        # This avoids double-counting when the matrix is symmetric.
        n = qubo.shape[0]
        wire = {}
        for i in range(n):
            val = float(qubo[i, i])
            if val != 0:
                wire[f"{i},{i}"] = val
            for j in range(i + 1, n):
                val = float(qubo[i, j]) + float(qubo[j, i])
                if val != 0:
                    wire[f"{i},{j}"] = val
        return wire

    if sps.issparse(qubo):
        coo = qubo.tocoo()
        # Fold into upper-triangular form (same as ndarray path)
        wire = {}
        for r, c, val in zip(coo.row, coo.col, coo.data):
            if val == 0:
                continue
            if r == c:
                wire[f"{r},{c}"] = wire.get(f"{r},{c}", 0) + float(val)
            elif r < c:
                key = f"{r},{c}"
                wire[key] = wire.get(key, 0) + float(val)
            else:
                key = f"{c},{r}"
                wire[key] = wire.get(key, 0) + float(val)
        return wire

    if isinstance(qubo, dict):
        wire = {}
        for key, coeff in qubo.items():
            if float(coeff) == 0:
                continue
            if isinstance(key, tuple):
                str_key = ",".join(str(k) for k in key)
            elif isinstance(key, str):
                str_key = key
            else:
                str_key = str(key)
            wire[str_key] = float(coeff)
        return wire

    if isinstance(qubo, list):
        return _serialize_qubo_for_wire(np.asarray(qubo))

    raise TypeError(
        f"Cannot serialize QUBO of type {type(qubo).__name__} to wire format. "
        f"Supported types: ndarray, spmatrix, BQM, BinaryPolynomial, dict, "
        f"BinaryOptimizationProblem."
    )


@dataclass(frozen=True)
class QUBOValidationResult:
    """Result container for QUBO/HUBO validation.

    Returned by :meth:`~divi.backends.QoroService.validate_qubo` and
    :func:`~divi.validate`. Displays a rich HTML report when rendered
    in a Jupyter notebook.

    .. note::
        Credit cost scales with QUBO size.
    """

    job_id: str
    """Unique identifier for the validation job."""

    status: str
    """Job status (``COMPLETED``, ``FAILED``, etc.)."""

    hardness: dict | None = field(default=None, repr=False)
    """Hardness analysis — difficulty rating, spectral gap, condition number."""

    report: dict | None = field(default=None, repr=False)
    """Full validation report — quality score, state probabilities, etc."""

    created_at: str | None = None
    """ISO timestamp when the validation job was created."""

    completed_at: str | None = None
    """ISO timestamp when the validation job completed."""

    # ---- Convenience properties ----

    @property
    def quality_score(self) -> float | None:
        """Overall quality score (0–100) from the validation report.

        When a parameter sweep was run, returns the score at the best
        parameters found (``quality_at_best``), which is the actionable
        metric.  Falls back to the score at the user-supplied or default
        parameters.
        """
        if self.report:
            return self.report.get(
                "quality_at_best", self.report.get("quality_score")
            )
        return None

    @property
    def concentration_ratio(self) -> float | None:
        """Concentration ratio — how tightly probability mass centers on target states.

        Prefers the ratio at the best sweep parameters when available.
        """
        if self.report:
            return self.report.get(
                "concentration_at_best", self.report.get("concentration_ratio")
            )
        return None

    @property
    def best_parameters(self) -> dict | None:
        """Best QAOA parameters found during parameter sweep (if requested)."""
        if self.report:
            return self.report.get("best_parameters")
        return None

    @property
    def penalty_recommendation(self) -> float | None:
        """Recommended penalty multiplier for constrained problems."""
        if self.report:
            return self.report.get("penalty_recommendation")
        return None

    @property
    def is_well_tuned(self) -> bool | None:
        """Whether the penalty parameter is well-tuned based on the analysis."""
        pt = self.report.get("penalty_tuning") if self.report else None
        if pt and isinstance(pt, dict):
            return pt.get("is_well_tuned")
        return None

    @property
    def feasibility_rate(self) -> float | None:
        """Fraction of sampled states that satisfy all constraints."""
        if self.report:
            return self.report.get("feasibility_rate")
        return None

    @property
    def state_probabilities(self) -> list[dict] | None:
        """Per-state probability data from the validation report."""
        if self.report:
            return self.report.get("state_probabilities")
        return None

    @property
    def sensitivity(self) -> list | None:
        """Sensitivity analysis data (if sensitivity was requested)."""
        if self.report:
            return self.report.get("sensitivity")
        return None

    @property
    def approximation_ratio(self) -> float | None:
        """Approximation ratio achieved by the QAOA ansatz."""
        if self.report:
            return self.report.get("approximation_ratio")
        return None

    # ---- Display methods ----

    def summary(self) -> str:
        """Return a rich text summary of the validation result."""
        lines = [
            f"QUBO Validation Result — Job {self.job_id[:8]}...",
            f"  Status: {self.status}",
        ]
        if self.quality_score is not None:
            lines.append(f"  Quality Score: {self.quality_score:.2f} / 100")
        if self.concentration_ratio is not None:
            lines.append(f"  Concentration Ratio: {self.concentration_ratio:.2f}")
        if self.approximation_ratio is not None:
            lines.append(f"  Approximation Ratio: {self.approximation_ratio:.4f}")
        if self.hardness:
            difficulty = self.hardness.get("difficulty", "unknown")
            lines.append(f"  Hardness: {difficulty}")
            if "spectral_gap" in self.hardness:
                lines.append(f"    Spectral Gap: {self.hardness['spectral_gap']:.4f}")
            if "condition_number" in self.hardness:
                lines.append(
                    f"    Condition Number: {self.hardness['condition_number']:.2f}"
                )
        if self.best_parameters:
            lines.append(
                f"  Best Parameters: γ={self.best_parameters.get('gamma', '?')}, "
                f"β={self.best_parameters.get('beta', '?')}"
            )
        if self.penalty_recommendation is not None:
            lines.append(
                f"  Penalty Recommendation: λ={self.penalty_recommendation:.2f}"
            )
        if self.feasibility_rate is not None:
            lines.append(f"  Feasibility Rate: {self.feasibility_rate:.1%}")
        if self.created_at:
            lines.append(f"  Created: {self.created_at}")
        if self.completed_at:
            lines.append(f"  Completed: {self.completed_at}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def display(self, *, n_qubits: int | None = None) -> None:
        """Print a rich console report of the validation result.

        Uses the ``rich`` library to display styled panels, tables, and
        gauges in the terminal.  In Jupyter notebooks, prefer evaluating
        the result object directly (which triggers ``_repr_html_``).

        Parameters
        ----------
        n_qubits:
            Number of qubits in the problem (used to compute the uniform
            probability baseline).  If ``None``, inferred from the report
            metadata when available.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # --- Infer n_qubits ---
        if n_qubits is None and self.report:
            n_qubits = self.report.get("num_qubits")

        # --- Header ---
        status_color = {
            "COMPLETED": "green",
            "FAILED": "red",
        }.get(self.status, "yellow")

        console.print(
            Panel(
                self.summary(),
                title="[cyan bold]QUBO Validation Report[/cyan bold]",
                subtitle=f"[dim]Job {self.job_id[:12]}…[/dim]",
                border_style="cyan",
            )
        )

        # --- Quality gauge ---
        qs = self.quality_score
        if qs is not None:
            bar_len = 40
            filled = int(bar_len * qs / 100)
            if qs >= 75:
                bar_color = "green"
            elif qs >= 50:
                bar_color = "yellow"
            elif qs >= 25:
                bar_color = "bright_red"
            else:
                bar_color = "red"
            bar = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * (bar_len - filled)}[/dim]"
            console.print(f"  Quality: {bar} [bold]{qs:.2f}[/bold] / 100\n")

        # --- Hardness analysis ---
        if self.hardness:
            ht = Table(title="Hardness Analysis", border_style="yellow")
            ht.add_column("Metric", style="bold")
            ht.add_column("Value", justify="right")

            for key, value in self.hardness.items():
                label = key.replace("_", " ").title()
                if isinstance(value, float):
                    ht.add_row(label, f"{value:.4f}")
                else:
                    ht.add_row(label, str(value))

            console.print(ht)

        # --- Best parameters ---
        bp = self.best_parameters
        if bp:
            gamma = bp.get("gamma")
            beta = bp.get("beta")
            prob = bp.get("probability")
            parts = []
            if gamma is not None:
                parts.append(f"[bold green]γ = {gamma:.4f}[/bold green]")
            if beta is not None:
                parts.append(f"[bold green]β = {beta:.4f}[/bold green]")
            if prob is not None:
                parts.append(f"[dim]P(target) = {prob:.6f}[/dim]")
            console.print(
                Panel(
                    "  " + "\n  ".join(parts),
                    title="[green]Best Parameters[/green]",
                    border_style="green",
                )
            )

        # --- Penalty tuning ---
        pr = self.penalty_recommendation
        wt = self.is_well_tuned
        if pr is not None or wt is not None:
            items = []
            if pr is not None:
                items.append(f"Recommended λ = [bold]{pr:.2f}[/bold]")
            if wt is True:
                items.append("[green]✓ Well-tuned[/green]")
            elif wt is False:
                items.append("[red]✗ Needs adjustment[/red]")
            console.print(
                Panel(
                    "  " + "\n  ".join(items),
                    title="[yellow]Penalty Tuning[/yellow]",
                    border_style="yellow",
                )
            )

        # --- State probabilities ---
        sp = self.state_probabilities
        if sp:
            target_set = set()
            if self.report and "target_states" in self.report:
                target_set = set(self.report["target_states"])

            uniform_prob = (1.0 / (2**n_qubits)) if n_qubits else None

            st = Table(title="State Probabilities", border_style="magenta")
            st.add_column("State", style="bold")
            st.add_column("Target?", justify="center")
            st.add_column("Probability", justify="right")
            if uniform_prob:
                st.add_column("vs Uniform", justify="right")

            for s in sp[:20]:
                state = s.get("state", "?")
                is_target = s.get("is_target", state in target_set)
                marker = "[green]✓[/green]" if is_target else "[dim]✗[/dim]"
                prob = s.get("probability", 0)

                row = [str(state), marker, f"{prob:.6f}"]

                if uniform_prob:
                    boost = prob / uniform_prob if uniform_prob > 0 else 0
                    boost_str = f"{boost:.1f}×" if boost >= 1.0 else f"{boost:.2f}×"
                    if is_target:
                        row.append(f"[bold]{boost_str}[/bold]")
                    else:
                        row.append(boost_str)

                st.add_row(*row)

            console.print(st)
            if uniform_prob:
                console.print(
                    f"  [dim]Uniform: {uniform_prob:.6f} (1/{2**n_qubits})[/dim]"
                )

        # --- Sensitivity ---
        sens = self.sensitivity
        if sens:
            se = Table(
                title="Sensitivity Analysis (per-qubit fragility)",
                border_style="blue",
            )
            se.add_column("Qubit", justify="center")
            se.add_column("Sensitivity", justify="right")
            se.add_column("Assessment")

            for entry in sens[:16]:
                q = entry.get("qubit", "?")
                val = entry.get("score", entry.get("sensitivity", 0))
                if val > 0.5:
                    assessment = "[red]fragile[/red]"
                elif val > 0.2:
                    assessment = "[yellow]moderate[/yellow]"
                else:
                    assessment = "[green]stable[/green]"
                se.add_row(str(q), f"{val:.4f}", assessment)

            console.print(se)

        # --- Recommendations ---
        recs = _build_recommendations(self)
        if recs:
            # Convert HTML tags to Rich markup for terminal display
            import re

            def _html_to_rich(text: str) -> str:
                text = text.replace("<strong>", "[bold]").replace("</strong>", "[/bold]")
                text = re.sub(r"<[^>]+>", "", text)  # strip remaining HTML
                return text

            console.print(
                Panel(
                    "\n".join(f"  • {_html_to_rich(r)}" for r in recs),
                    title="[cyan]Recommendations[/cyan]",
                    border_style="cyan",
                )
            )

    def _repr_html_(self) -> str:
        """Jupyter HTML rendering with styled cards, progress bars, and tables."""
        return _build_validation_html(self)


# ---------------------------------------------------------------------------
# HTML rendering helpers (kept private — called only by _repr_html_)
# ---------------------------------------------------------------------------

_CSS = textwrap.dedent("""\
    <style>
    .qvr-root {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: #e2e8f0;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 28px;
        max-width: 820px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .qvr-root *, .qvr-root *::before, .qvr-root *::after {
        box-sizing: border-box;
    }
    .qvr-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
    }
    .qvr-title {
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .qvr-badge {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .qvr-badge-completed { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .qvr-badge-failed    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .qvr-badge-pending   { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
    .qvr-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .qvr-card {
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(8px);
    }
    .qvr-card-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .qvr-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .qvr-card-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    .qvr-gauge-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(148,163,184,0.15);
        margin-top: 10px;
        overflow: hidden;
    }
    .qvr-gauge-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    .qvr-section {
        margin-bottom: 18px;
    }
    .qvr-section-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(148,163,184,0.1);
    }
    .qvr-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .qvr-table th {
        text-align: left;
        padding: 8px 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(148,163,184,0.1);
    }
    .qvr-table td {
        padding: 7px 10px;
        border-bottom: 1px solid rgba(148,163,184,0.05);
        color: #cbd5e1;
    }
    .qvr-prob-bar {
        display: inline-block;
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        vertical-align: middle;
        margin-left: 6px;
    }
    .qvr-target { color: #4ade80; font-weight: 600; }
    .qvr-non-target { color: #64748b; }
    .qvr-rec {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
    }
    .qvr-rec strong { color: #e2e8f0; }
    .qvr-footer {
        font-size: 11px;
        color: #475569;
        margin-top: 16px;
        text-align: right;
    }
    </style>
""")


def _gauge_color(score: float) -> str:
    """Return a CSS gradient colour based on score 0-100."""
    if score >= 75:
        return "linear-gradient(90deg, #22c55e, #4ade80)"
    if score >= 50:
        return "linear-gradient(90deg, #eab308, #facc15)"
    if score >= 25:
        return "linear-gradient(90deg, #f97316, #fb923c)"
    return "linear-gradient(90deg, #ef4444, #f87171)"


def _difficulty_badge(difficulty: str) -> str:
    colors = {
        "easy": ("#4ade80", "rgba(34,197,94,0.12)"),
        "medium": ("#facc15", "rgba(250,204,21,0.12)"),
        "hard": ("#f87171", "rgba(239,68,68,0.12)"),
    }
    fg, bg = colors.get(difficulty.lower(), ("#94a3b8", "rgba(148,163,184,0.1)"))
    return (
        f'<span style="color:{fg};background:{bg};padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">'
        f"{html.escape(difficulty.capitalize())}</span>"
    )


def _build_validation_html(result: QUBOValidationResult) -> str:
    """Assemble the complete HTML report for a QUBOValidationResult."""
    parts: list[str] = [_CSS, '<div class="qvr-root">']

    # --- Header ---
    status_cls = {
        "COMPLETED": "qvr-badge-completed",
        "FAILED": "qvr-badge-failed",
    }.get(result.status, "qvr-badge-pending")

    parts.append(
        f'<div class="qvr-header">'
        f'<span class="qvr-title">QUBO Validation Report</span>'
        f'<span class="qvr-badge {status_cls}">{html.escape(result.status)}</span>'
        f"</div>"
    )

    # --- Top metric cards ---
    parts.append('<div class="qvr-cards">')

    # Quality score
    qs = result.quality_score
    if qs is not None:
        color = _gauge_color(qs)
        parts.append(
            f'<div class="qvr-card">'
            f'<div class="qvr-card-label">Quality Score</div>'
            f'<div class="qvr-card-value">{qs:.1f}<span style="font-size:14px;color:#64748b;"> / 100</span></div>'
            f'<div class="qvr-gauge-bar"><div class="qvr-gauge-fill" style="width:{qs}%;background:{color};"></div></div>'
            f"</div>"
        )

    # Concentration ratio
    cr = result.concentration_ratio
    if cr is not None:
        parts.append(
            f'<div class="qvr-card">'
            f'<div class="qvr-card-label">Concentration Ratio</div>'
            f'<div class="qvr-card-value">{cr:.2f}×</div>'
            f'<div class="qvr-card-sub">target vs. uniform</div>'
            f"</div>"
        )

    # Approximation ratio
    ar = result.approximation_ratio
    if ar is not None:
        parts.append(
            f'<div class="qvr-card">'
            f'<div class="qvr-card-label">Approximation Ratio</div>'
            f'<div class="qvr-card-value">{ar:.4f}</div>'
            f"</div>"
        )

    # Feasibility rate
    fr = result.feasibility_rate
    if fr is not None:
        parts.append(
            f'<div class="qvr-card">'
            f'<div class="qvr-card-label">Feasibility Rate</div>'
            f'<div class="qvr-card-value">{fr:.1%}</div>'
            f"</div>"
        )

    parts.append("</div>")  # close .qvr-cards

    # --- Hardness card ---
    if result.hardness:
        h = result.hardness
        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">Hardness Analysis</div>')
        parts.append('<div class="qvr-card">')
        difficulty = h.get("difficulty", "unknown")
        parts.append(
            f'<div style="margin-bottom:10px;">Difficulty: {_difficulty_badge(difficulty)}</div>'
        )
        metrics = []
        if "spectral_gap" in h:
            metrics.append(f"Spectral Gap: <strong>{h['spectral_gap']:.4f}</strong>")
        if "condition_number" in h:
            metrics.append(
                f"Condition Number: <strong>{h['condition_number']:.2f}</strong>"
            )
        if "degeneracy" in h:
            metrics.append(f"Degeneracy: <strong>{h['degeneracy']}</strong>")
        if metrics:
            parts.append(
                '<div style="font-size:13px;color:#94a3b8;">'
                + " &nbsp;·&nbsp; ".join(metrics)
                + "</div>"
            )
        parts.append("</div></div>")

    # --- State probabilities table ---
    sp = result.state_probabilities
    if sp:
        # Build target state set from either per-state flag or report root
        target_set = set()
        if result.report and "target_states" in result.report:
            target_set = set(result.report["target_states"])

        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">State Probabilities</div>')
        parts.append(
            '<table class="qvr-table"><thead><tr>'
            "<th>State</th><th>Target</th><th>Probability</th><th>Energy</th>"
            "</tr></thead><tbody>"
        )
        max_prob = max((s.get("probability", 0) for s in sp), default=1) or 1
        for s in sp[:20]:  # cap display at 20 states
            state_str = html.escape(str(s.get("state", "?")))
            is_target = s.get("is_target", s.get("state", "") in target_set)
            marker_cls = "qvr-target" if is_target else "qvr-non-target"
            marker = "✓" if is_target else "✗"
            prob = s.get("probability", 0)
            bar_w = max(1, int(100 * prob / max_prob))
            energy = s.get("energy", "—")
            energy_str = f"{energy:.4f}" if isinstance(energy, (int, float)) else str(energy)
            parts.append(
                f"<tr>"
                f'<td><code>{state_str}</code></td>'
                f'<td class="{marker_cls}">{marker}</td>'
                f'<td>{prob:.4f} <span class="qvr-prob-bar" style="width:{bar_w}px;"></span></td>'
                f"<td>{html.escape(energy_str)}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    # --- Best parameters ---
    bp = result.best_parameters
    if bp:
        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">Best Parameters</div>')
        parts.append('<div class="qvr-card">')
        params = []
        if "gamma" in bp:
            params.append(f"γ = <strong>{bp['gamma']:.4f}</strong>")
        if "beta" in bp:
            params.append(f"β = <strong>{bp['beta']:.4f}</strong>")
        if "probability" in bp:
            params.append(
                f"P(target) = <strong>{bp['probability']:.4f}</strong>"
            )
        parts.append(
            '<div style="font-size:14px;color:#cbd5e1;">'
            + " &nbsp;·&nbsp; ".join(params)
            + "</div>"
        )
        parts.append("</div></div>")

    # --- Penalty tuning ---
    pr = result.penalty_recommendation
    wt = result.is_well_tuned
    if pr is not None or wt is not None:
        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">Penalty Tuning</div>')
        parts.append('<div class="qvr-card">')
        items = []
        if pr is not None:
            items.append(f"Recommended λ = <strong>{pr:.2f}</strong>")
        if wt is not None:
            tuned_str = (
                '<span style="color:#4ade80;">✓ Well-tuned</span>'
                if wt
                else '<span style="color:#f87171;">✗ Needs adjustment</span>'
            )
            items.append(tuned_str)
        parts.append(
            '<div style="font-size:14px;color:#cbd5e1;">'
            + " &nbsp;·&nbsp; ".join(items)
            + "</div>"
        )
        parts.append("</div></div>")

    # --- Sensitivity ---
    sens = result.sensitivity
    if sens:
        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">Sensitivity Analysis</div>')
        parts.append(
            '<table class="qvr-table"><thead><tr>'
            "<th>Qubit</th><th>Sensitivity</th>"
            "</tr></thead><tbody>"
        )
        for entry in sens[:16]:
            q = entry.get("qubit", "?")
            val = entry.get("score", entry.get("sensitivity", 0))
            bar_color = _gauge_color(max(0, 100 - val * 100))
            parts.append(
                f"<tr>"
                f"<td>{q}</td>"
                f'<td>{val:.4f} <span class="qvr-prob-bar" style="width:{max(1,int(val*80))}px;background:{bar_color};"></span></td>'
                f"</tr>"
            )
        parts.append("</tbody></table></div>")

    # --- Recommendations ---
    recommendations = _build_recommendations(result)
    if recommendations:
        parts.append('<div class="qvr-section">')
        parts.append('<div class="qvr-section-title">Recommendations</div>')
        parts.append('<div class="qvr-rec">')
        for rec in recommendations:
            parts.append(f"• {rec}<br/>")
        parts.append("</div></div>")

    # --- Footer ---
    job_short = result.job_id[:8] if result.job_id else "?"
    timing = ""
    if result.created_at and result.completed_at:
        timing = (
            f" · {html.escape(str(result.created_at))} → {html.escape(str(result.completed_at))}"
        )
    parts.append(
        f'<div class="qvr-footer">Job {html.escape(job_short)}…{timing}</div>'
    )

    parts.append("</div>")  # close .qvr-root
    return "\n".join(parts)


def _build_recommendations(result: QUBOValidationResult) -> list[str]:
    """Generate actionable recommendations based on the validation report.

    Prefers server-side recommendations (returned by the Composer
    validation engine) when available, since the server has richer
    context (warm-start details, landscape shape, exploration budget).
    Falls back to client-generated heuristics when the report does not
    include server recommendations.
    """
    # --- Server-side recommendations (preferred) ---
    if result.report:
        server_recs = result.report.get("recommendations")
        if server_recs and isinstance(server_recs, list) and len(server_recs) > 0:
            return [str(r) for r in server_recs]

    # --- Client-side fallback ---
    recs: list[str] = []

    qs = result.quality_score
    if qs is not None:
        if qs >= 80:
            recs.append(
                "<strong>Quality is high</strong> — your formulation is well-suited for QAOA."
            )
        elif qs >= 50:
            recs.append(
                "<strong>Moderate quality</strong> — consider tuning penalty parameters "
                "or increasing QAOA layers."
            )
        else:
            recs.append(
                "<strong>Low quality score</strong> — review your QUBO formulation. "
                "Check constraint penalties and problem encoding."
            )

    if result.hardness:
        diff = result.hardness.get("difficulty", "").lower()
        if diff == "hard":
            recs.append(
                "Problem is <strong>hard</strong> — consider decomposition or "
                "classical pre-processing to reduce effective problem size."
            )

    if result.feasibility_rate is not None and result.feasibility_rate < 0.5:
        recs.append(
            f"Feasibility rate is only <strong>{result.feasibility_rate:.0%}</strong> — "
            f"increase the penalty multiplier to enforce constraints more strongly."
        )

    wt = result.is_well_tuned
    if wt is False and result.penalty_recommendation is not None:
        recs.append(
            f"Penalty is <strong>not well-tuned</strong>. "
            f"Try λ = <strong>{result.penalty_recommendation:.2f}</strong>."
        )

    cr = result.concentration_ratio
    if cr is not None and cr < 1.5:
        recs.append(
            "Low concentration ratio — probability is spread too uniformly. "
            "Try deeper circuits (more QAOA layers) or parameter sweeps."
        )

    return recs
