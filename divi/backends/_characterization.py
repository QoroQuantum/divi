# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QUBO/HUBO characterization: serialization, result container, and public API."""

import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from divi.backends._qoro_service import QoroService
from divi.qprog.problems import BinaryOptimizationProblem


def _serialize_qubo_for_wire(
    problem: BinaryOptimizationProblem,
) -> dict[str, float]:
    """Serialize a :class:`BinaryOptimizationProblem` to the Usher wire format.

    The wire format uses comma-joined string keys (``"0,1"``) mapping to
    float coefficients. Iteration goes through ``problem.canonical_problem``
    so the input may be any QUBO / HUBO shape supported by
    ``BinaryOptimizationProblem``'s constructor.
    """
    return {
        ",".join(str(idx) for idx in term_key): float(coeff)
        for term_key, coeff in problem.canonical_problem.terms.items()
        if coeff != 0
    }


_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Module-level constants used by ``_render``; pulled out to keep the
# rendering function focused on layout instead of palette bookkeeping.
_QUALITY_BAR_LEN = 40
_QUALITY_COLORS = ((75, "green"), (50, "yellow"), (25, "bright_red"))
_SENSITIVITY_LABELS = (
    (0.5, "[red]fragile[/red]"),
    (0.2, "[yellow]moderate[/yellow]"),
)
_RECOMMENDATION_BULLETS = {
    "action": "[red]•[/red]",
    "warn": "[yellow]•[/yellow]",
    "info": "[cyan]•[/cyan]",
}
_WELL_TUNED_LABELS = {
    True: "[green]✓ Well-tuned[/green]",
    False: "[red]✗ Needs adjustment[/red]",
}
_STATE_TABLE_CAP = 20
_SENSITIVITY_TABLE_CAP = 16


def _threshold_pick(
    value: float, thresholds: tuple[tuple[float, str], ...], default: str
) -> str:
    """Return the first label whose threshold ``value`` meets, else ``default``."""
    return next((label for cutoff, label in thresholds if value >= cutoff), default)


def _html_to_rich(text: str) -> str:
    """Convert a small subset of HTML to ``rich`` console markup."""
    text = text.replace("<strong>", "[bold]").replace("</strong>", "[/bold]")
    return _HTML_TAG_RE.sub("", text)


@dataclass
class CharacterizationResult:
    """Result container for QUBO/HUBO characterization.

    Returned by :meth:`~divi.backends.QoroService.characterize` and
    :func:`~divi.backends.characterize`. Displays a rich HTML report when
    rendered in a Jupyter notebook.

    .. note::
        Credit cost scales with QUBO size.
    """

    job_id: str
    """Unique identifier for the characterization job."""

    status: str
    """Job status (``COMPLETED``, ``FAILED``, etc.)."""

    hardness: dict | None = field(default=None, repr=False)
    """Hardness analysis — difficulty rating, spectral gap, condition number."""

    report: dict | None = field(default=None, repr=False)
    """Full characterization report — quality score, state probabilities, etc."""

    created_at: str | None = None
    """ISO timestamp when the characterization job was created."""

    completed_at: str | None = None
    """ISO timestamp when the characterization job completed."""

    # Server-rendered HTML report, fetched once at construction (see
    # ``_wrap_response``). ``kw_only=True`` lets it sit after the optional
    # fields without imposing a default.
    html: str = field(kw_only=True, repr=False, compare=False)

    def _field(self, key: str, *fallbacks: str):
        """Return ``self.report[key]`` (or first present fallback), else ``None``."""
        if not self.report:
            return None
        for k in (key, *fallbacks):
            if k in self.report:
                return self.report[k]
        return None

    @property
    def quality_score(self) -> float | None:
        """Overall quality score (0–100) from the characterization report.

        When a parameter sweep was run, returns the score at the best
        parameters found (``quality_at_best``); otherwise the score at
        the user-supplied or default parameters.
        """
        return self._field("quality_at_best", "quality_score")

    @property
    def concentration_ratio(self) -> float | None:
        """How tightly probability mass concentrates on target states.

        Prefers the value at the best sweep parameters when available.
        """
        return self._field("concentration_at_best", "concentration_ratio")

    @property
    def approximation_ratio(self) -> float | None:
        """Approximation ratio achieved by the QAOA ansatz."""
        return self._field("approximation_ratio")

    @property
    def best_parameters(self) -> dict | None:
        """Best QAOA parameters found during parameter sweep (if requested)."""
        return self._field("best_parameters")

    @property
    def state_probabilities(self) -> list[dict] | None:
        """Per-state probability data from the characterization report."""
        return self._field("state_probabilities")

    @property
    def sensitivity(self) -> list | None:
        """Per-qubit sensitivity analysis (if requested)."""
        return self._field("sensitivity")

    @property
    def feasibility_rate(self) -> float | None:
        """Fraction of sampled states that satisfy all constraints."""
        return self._field("feasibility_rate")

    @property
    def penalty_recommendation(self) -> float | None:
        """Recommended penalty multiplier for constrained problems."""
        return self._field("penalty_recommendation")

    @property
    def is_well_tuned(self) -> bool | None:
        """Whether the penalty parameter is well-tuned based on the analysis."""
        pt = self._field("penalty_tuning")
        return pt.get("is_well_tuned") if isinstance(pt, dict) else None

    @property
    def recommendations(self) -> list[dict]:
        """Server-supplied list of interpretive recommendations.

        Each entry is a dict with ``level`` (``info`` / ``warn`` / ``action``),
        ``metric`` (which report field triggered it), and ``html`` (a short
        message containing inline ``<strong>`` markup).
        """
        recs = self._field("recommendations")
        return recs if isinstance(recs, list) else []

    def summary(self) -> str:
        """Return a rich text summary of the characterization result."""
        lines = [
            f"QUBO Characterization Result — Job {self.job_id[:8]}...",
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
        if bp := self.best_parameters:
            lines.append(
                f"  Best Parameters: γ={bp.get('gamma', '?')}, "
                f"β={bp.get('beta', '?')}"
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

    def display(self) -> None:
        """Print a rich console report of the characterization result.

        Uses the ``rich`` library to display styled panels, tables, and
        gauges in the terminal.  In Jupyter notebooks, prefer evaluating
        the result object directly (which triggers ``_repr_html_``).
        """
        _render(self)

    def _repr_html_(self) -> str:
        """Return the server-rendered HTML report (Jupyter)."""
        return self.html


@dataclass
class CharacterizationOptions:
    """Configuration for :func:`~divi.backends.characterize`.

    All fields are optional; default-construct for a basic run with no
    sub-analyses. The dataclass validates field combinations at
    construction time (``__post_init__``), so misconfiguration surfaces
    before any API call.

    Examples:
        >>> CharacterizationOptions(parameter_sweep=True, sensitivity=True)
        >>> CharacterizationOptions(gamma=1.2, beta=0.7)
    """

    sensitivity: bool = False
    """Request per-qubit sensitivity analysis."""

    parameter_sweep: bool = False
    """Request a γ/β parameter sweep.

    Mutually exclusive with fixed ``gamma`` / ``beta``.
    """

    auto_tune: bool = False
    """Request automatic penalty tuning."""

    gamma: float | None = None
    """Fixed γ value. Mutually exclusive with ``parameter_sweep``."""

    beta: float | None = None
    """Fixed β value. Mutually exclusive with ``parameter_sweep``."""

    cost_qubo: BinaryOptimizationProblem | None = None
    """Cost-only :class:`BinaryOptimizationProblem` for penalty analysis."""

    penalty_qubo: BinaryOptimizationProblem | None = None
    """Penalty-only :class:`BinaryOptimizationProblem` for penalty analysis."""

    constraints: list | None = None
    """Constraint descriptors."""

    ansatz: dict | None = None
    """Ansatz configuration dict (e.g. ``{"mixer": "x", "layers": 1}``).

    The ``auto_warmstart`` key is reserved for the backend and rejected
    at construction time if supplied.
    """

    def __post_init__(self) -> None:
        if self.parameter_sweep and (self.gamma is not None or self.beta is not None):
            raise ValueError(
                "parameter_sweep=True is mutually exclusive with fixed "
                "gamma/beta — pick one."
            )
        if self.ansatz is not None and "auto_warmstart" in self.ansatz:
            raise ValueError(
                "ansatz['auto_warmstart'] is managed by the backend and "
                "cannot be set from the client."
            )

    def _to_wire(self) -> dict | None:
        """Serialize to the wire-format options dict (or ``None`` if empty)."""
        analysis = {
            k: v
            for k, v in {
                "gamma": self.gamma,
                "beta": self.beta,
                "sensitivity": self.sensitivity or None,
                "parameter_sweep": self.parameter_sweep or None,
                "auto_tune": self.auto_tune or None,
            }.items()
            if v is not None
        }
        options = {
            k: v
            for k, v in {
                "analysis": analysis or None,
                "ansatz": self.ansatz,
                "cost_qubo": (
                    _serialize_qubo_for_wire(self.cost_qubo)
                    if self.cost_qubo is not None
                    else None
                ),
                "penalty_qubo": (
                    _serialize_qubo_for_wire(self.penalty_qubo)
                    if self.penalty_qubo is not None
                    else None
                ),
                "constraints": self.constraints,
            }.items()
            if v is not None
        }
        return options or None


def _render(result: "CharacterizationResult") -> None:
    """Print a rich console report for ``result``.

    Free function rather than a method so that ``CharacterizationResult``
    stays focused on data + properties; the ~180 lines of styled-rendering
    code live here.
    """
    console = Console()

    # ``num_qubits`` is server-supplied in the report; used below for the
    # uniform-distribution baseline in the state-probabilities table.
    n_qubits = result.report.get("num_qubits") if result.report else None

    # --- Header ---
    console.print(
        Panel(
            result.summary(),
            title="[cyan bold]QUBO Characterization Report[/cyan bold]",
            subtitle=f"[dim]Job {result.job_id[:12]}…[/dim]",
            border_style="cyan",
        )
    )

    # --- Quality gauge ---
    qs = result.quality_score
    if qs is not None:
        color = _threshold_pick(qs, _QUALITY_COLORS, default="red")
        filled = int(_QUALITY_BAR_LEN * qs / 100)
        bar = (
            f"[{color}]{'█' * filled}[/{color}]"
            f"[dim]{'░' * (_QUALITY_BAR_LEN - filled)}[/dim]"
        )
        console.print(f"  Quality: {bar} [bold]{qs:.2f}[/bold] / 100\n")

    # --- Hardness analysis ---
    if result.hardness:
        ht = Table(title="Hardness Analysis", border_style="yellow")
        ht.add_column("Metric", style="bold")
        ht.add_column("Value", justify="right")
        for key, value in result.hardness.items():
            label = key.replace("_", " ").title()
            fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
            ht.add_row(label, fmt)
        console.print(ht)

    # --- Best parameters ---
    bp = result.best_parameters
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
    pr = result.penalty_recommendation
    wt = result.is_well_tuned
    if pr is not None or wt is not None:
        items = []
        if pr is not None:
            items.append(f"Recommended λ = [bold]{pr:.2f}[/bold]")
        if wt in _WELL_TUNED_LABELS:
            items.append(_WELL_TUNED_LABELS[wt])
        console.print(
            Panel(
                "  " + "\n  ".join(items),
                title="[yellow]Penalty Tuning[/yellow]",
                border_style="yellow",
            )
        )

    # --- State probabilities ---
    sp = result.state_probabilities
    if sp:
        target_set = set((result.report or {}).get("target_states") or ())
        uniform_prob = (1.0 / (2**n_qubits)) if n_qubits else None

        st = Table(title="State Probabilities", border_style="magenta")
        st.add_column("State", style="bold")
        st.add_column("Target?", justify="center")
        st.add_column("Probability", justify="right")
        if uniform_prob:
            st.add_column("vs Uniform", justify="right")

        for s in sp[:_STATE_TABLE_CAP]:
            state = s.get("state", "?")
            is_target = s.get("is_target", state in target_set)
            marker = "[green]✓[/green]" if is_target else "[dim]✗[/dim]"
            prob = s.get("probability", 0)
            row = [str(state), marker, f"{prob:.6f}"]
            if uniform_prob:
                boost = prob / uniform_prob
                boost_str = f"{boost:.1f}×" if boost >= 1.0 else f"{boost:.2f}×"
                row.append(f"[bold]{boost_str}[/bold]" if is_target else boost_str)
            st.add_row(*row)

        console.print(st)
        if uniform_prob and n_qubits is not None:
            console.print(f"  [dim]Uniform: {uniform_prob:.6f} (1/{2**n_qubits})[/dim]")

    # --- Sensitivity ---
    sens = result.sensitivity
    if sens:
        se = Table(
            title="Sensitivity Analysis (per-qubit fragility)",
            border_style="blue",
        )
        se.add_column("Qubit", justify="center")
        se.add_column("Sensitivity", justify="right")
        se.add_column("Assessment")
        for entry in sens[:_SENSITIVITY_TABLE_CAP]:
            val = entry.get("score", entry.get("sensitivity", 0))
            assessment = _threshold_pick(
                val, _SENSITIVITY_LABELS, default="[green]stable[/green]"
            )
            se.add_row(str(entry.get("qubit", "?")), f"{val:.4f}", assessment)
        console.print(se)

    # --- Recommendations (server-supplied) ---
    recs = result.recommendations
    if recs:
        default_bullet = _RECOMMENDATION_BULLETS["info"]
        lines = [
            f"  {_RECOMMENDATION_BULLETS.get(r.get('level', 'info'), default_bullet)}"
            f" {_html_to_rich(r.get('html', ''))}"
            for r in recs
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title="[cyan]Recommendations[/cyan]",
                border_style="cyan",
            )
        )


def _wrap_response(data: dict, service: QoroService) -> CharacterizationResult:
    # ``job_id`` and ``status`` are required fields in the server contract;
    # let ``KeyError`` surface noisily on a malformed payload rather than
    # silently fabricating defaults. Optional metadata stays as ``.get()``.
    job_id = data["job_id"]
    return CharacterizationResult(
        job_id=job_id,
        status=data["status"],
        hardness=data.get("hardness"),
        report=data.get("report"),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        html=service._fetch_characterization_html(job_id),
    )


def characterize(
    problem: BinaryOptimizationProblem,
    target_states: list[str],
    *,
    service: QoroService,
    options: CharacterizationOptions | None = None,
) -> CharacterizationResult:
    """One-call QUBO/HUBO characterization with rich notebook display.

    Converts the problem to wire format, submits it to the Qoro
    Characterization Service, and returns a :class:`CharacterizationResult`
    that renders a styled report in Jupyter.

    Args:
        problem: A :class:`~divi.qprog.problems.BinaryOptimizationProblem`.
            Wrap raw inputs (ndarray, sparse, BQM, HUBO dict, etc.) by
            constructing one — the constructor accepts every shape this
            function used to take directly.
        target_states: Bitstrings of the known optimal / target states.
        service: A :class:`~divi.backends.QoroService` instance to drive
            the API calls.
        options: Optional :class:`CharacterizationOptions` configuring
            sub-analyses, fixed parameters, ansatz, and constraints.
            Defaults to a no-op options object (server-side defaults).

    Returns:
        CharacterizationResult: Rich result object. Displaying it in
        Jupyter shows a styled HTML report.

    Raises:
        requests.exceptions.HTTPError: On API errors.

    Examples:
        >>> import numpy as np
        >>> from divi.backends import (
        ...     CharacterizationOptions,
        ...     QoroService,
        ...     characterize,
        ... )
        >>> from divi.qprog.problems import BinaryOptimizationProblem
        >>> problem = BinaryOptimizationProblem(np.array([[-1, 2], [0, -1]]))
        >>> result = characterize(
        ...     problem,
        ...     target_states=["01", "10"],
        ...     service=QoroService(),
        ...     options=CharacterizationOptions(parameter_sweep=True),
        ... )
        >>> result.quality_score
        78.5

    .. note::
        Credit cost scales with QUBO size.
    """
    options = options or CharacterizationOptions()
    data = service.characterize(
        qubo=_serialize_qubo_for_wire(problem),
        target_states=target_states,
        options=options._to_wire(),
    )
    return _wrap_response(data, service)


def get_characterization_result(
    job_id: str,
    *,
    service: QoroService,
) -> CharacterizationResult:
    """Re-fetch a previous characterization result by job ID.

    This does **not** cost any credits — it only retrieves the stored
    result from a previously completed characterization run.

    Args:
        job_id: Identifier of a previously submitted characterization job.
        service: A :class:`~divi.backends.QoroService` instance to drive
            the API call.

    Returns:
        CharacterizationResult: The full result including hardness,
        report, state probabilities, and any analysis data that was
        computed during the original run.

    Examples:
        >>> from divi.backends import QoroService, get_characterization_result
        >>> result = get_characterization_result(
        ...     "4d0550f5-ffb0-...", service=QoroService()
        ... )
        >>> result.display()   # rich console report
        >>> result.quality_score
        45.89
    """
    data = service.characterize(job_id=job_id)
    return _wrap_response(data, service)
