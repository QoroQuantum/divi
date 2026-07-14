# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QUBO/HUBO characterization: serialization, result container, and public API."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from divi.qprog.problems import BinaryOptimizationProblem

from .._qoro_service import QoroService

logger = logging.getLogger(__name__)


# Smallest QUBO size at which the factored encoding is even probed.
# Below this, the legacy comma-key dict is always smaller on the wire,
# so the two eigendecomposition probes would be pure overhead.
_FACTORED_PROBE_MIN_QUBITS = 64

# Minimum relative tolerance for treating an eigenvalue as zero.
# Combined with ``n · eps_machine`` at use sites; for an ``n × n``
# matrix the effective threshold is
# ``max(_EIGVAL_TOL_REL, n · eps) · |λ_max|``, which stays above the
# backward-error floor of ``eigh`` at all matrix sizes.
_EIGVAL_TOL_REL = 1e-12

# Minimum eigenvalue magnitude (relative to ``|λ_max|``) retained by
# the truncated decomposition. Eigenvalues below this threshold are
# treated as a baseline plateau and dropped into the diagonal residual.
# Choosing the cut by absolute magnitude — rather than by gap ratio or
# Frobenius energy — preserves structurally significant modes even when
# a single penalty eigenvalue dominates ``‖Q‖_F``.
_TRUNCATED_MAGNITUDE_THRESHOLD = 1e-2

# Hard upper bound on the JSON payload size emitted by the truncated
# candidate. Kept well under typical reverse-proxy body-size limits
# (e.g. nginx ``client_max_body_size 10m``).
_TRUNCATED_PAYLOAD_BUDGET_BYTES = 950_000

# Maximum acceptable ``‖Q_recon − Q‖_max / ‖Q‖_max`` from the truncated
# candidate. If reconstruction error exceeds this the candidate is
# discarded and a lossless encoding (or legacy) is shipped instead.
_TRUNCATED_REL_ERROR_MAX = 1e-3


def _serialize_qubo_legacy(canonical) -> dict[str, float]:
    """Serialize to the comma-key dict format, e.g. ``{"0": -1.0, "0,1": 2.0}``.

    Term keys are original variable *names* (which may be strings for a
    ``dimod.BinaryQuadraticModel``), so they are remapped to integer indices
    via ``variable_to_idx``. Accepts terms of any degree, so it is the only
    valid path for HUBO inputs.
    """
    idx = canonical.variable_to_idx
    return {
        ",".join(str(idx[v]) for v in term_key): float(coeff)
        for term_key, coeff in canonical.terms.items()
        if coeff != 0
    }


def _serialize_qubo_component_legacy(
    canonical, variable_to_idx: dict
) -> dict[str, float]:
    """Serialize a component using another QUBO's variable indexing."""
    wire: dict[str, float] = {}
    for term_key, coeff in canonical.terms.items():
        if coeff == 0:
            continue
        try:
            mapped = [variable_to_idx[v] for v in term_key]
        except KeyError as exc:
            raise ValueError(
                "Penalty tuning components must use variables present in the "
                "full BinaryOptimizationProblem."
            ) from exc
        wire[",".join(str(i) for i in mapped)] = float(coeff)
    return wire


def _qubo_to_dense(canonical) -> np.ndarray:
    """Build the symmetric dense QUBO matrix from canonical polynomial terms.

    Off-diagonal coefficients are split half-and-half between ``Q[i,j]`` and
    ``Q[j,i]`` so the result is exactly symmetric. ``(i,)`` and ``(i, i)``
    terms both write to the diagonal, since ``x_i² = x_i`` for binary
    variables.
    """
    n = canonical.n_vars
    idx = canonical.variable_to_idx
    Q = np.zeros((n, n), dtype=np.float64)
    for term_key, coeff in canonical.terms.items():
        if coeff == 0:
            continue
        mapped = [idx[v] for v in term_key]
        if len(mapped) == 1:
            i = mapped[0]
            Q[i, i] += float(coeff)
        else:
            i, j = mapped
            if i == j:
                Q[i, i] += float(coeff)
            else:
                Q[i, j] += float(coeff) / 2.0
                Q[j, i] += float(coeff) / 2.0
    return Q


def _eigh_drop_noise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``eigh(matrix)`` with eigenvalues below the backward-error noise floor masked out.

    Returns ``(eigvals, V)`` where every retained ``|λ|`` exceeds
    ``max(_EIGVAL_TOL_REL, n · eps) · |λ_max|``.
    """
    eigvals, V = np.linalg.eigh(matrix)
    if not eigvals.size:
        return eigvals, V
    max_abs = float(np.abs(eigvals).max())
    if max_abs == 0.0:
        return eigvals[:0], V[:, :0]
    tol = max(_EIGVAL_TOL_REL, eigvals.size * float(np.finfo(np.float64).eps)) * max_abs
    mask = np.abs(eigvals) > tol
    return eigvals[mask], V[:, mask]


def _payload_from_eigh(
    eigvals: np.ndarray, V: np.ndarray, residual: np.ndarray, n: int
) -> dict:
    """Assemble a ``factored_v1`` payload from a (truncated) eigendecomposition.

    ``F = V · diag(√|λ|)``, ``signs = sign(λ)`` (strict ±1.0). ``F`` and
    ``residual`` are emitted as hex-encoded float64 byte arrays.
    """
    signs = np.where(eigvals >= 0.0, 1.0, -1.0).astype(np.float64)
    F = np.ascontiguousarray(V * np.sqrt(np.abs(eigvals)), dtype=np.float64)
    residual_c = np.ascontiguousarray(residual, dtype=np.float64)
    return {
        "_format": "factored_v1",
        "n": int(n),
        "k": int(eigvals.size),
        "F": F.tobytes().hex(),
        "signs": signs.tolist(),
        "diag": residual_c.tobytes().hex(),
    }


def _factored_truncated(
    eigvals: np.ndarray, V: np.ndarray, diag_orig: np.ndarray, Q: np.ndarray
) -> dict | None:
    """Truncate the off-diagonal eigendecomposition with diagonal absorption.

    Sorts eigenvalues of ``Q_off = Q − diag(Q)`` by ``|λ|`` descending, keeps
    every eigenvalue with ``|λ| ≥ _TRUNCATED_MAGNITUDE_THRESHOLD · |λ_max|``,
    and absorbs the diagonal contribution of the dropped eigencomponents into
    the residual. The diagonal of the reconstructed matrix matches ``Q``
    exactly; off-diagonal entries pick up a bounded error.

    Returns ``None`` when truncation does not apply (no eigenvalues, ``k ≥ n``
    after both magnitude and budget checks) or when the reconstruction
    relative error exceeds :data:`_TRUNCATED_REL_ERROR_MAX`.
    """
    n = Q.shape[0]
    if not eigvals.size:
        return None

    # Sort by |λ| descending so the magnitude cut and truncation both proceed
    # from the most-informative end.
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_s = eigvals[order]
    V_s = V[:, order]
    abs_s = np.abs(eigvals_s)

    lambda_max = float(abs_s[0])
    if lambda_max == 0.0:
        return None
    # Magnitude cut: keep every eigenvalue at least ε·|λ_max|.
    k_mag = int(np.sum(abs_s >= _TRUNCATED_MAGNITUDE_THRESHOLD * lambda_max))

    # Payload-budget cap. JSON cost per kept column ≈ n·16 hex chars for F
    # plus ≈5 chars for the corresponding ``signs`` entry; envelope + diag
    # are fixed costs.
    budget_for_F = _TRUNCATED_PAYLOAD_BUDGET_BYTES - n * 16 - 200
    if budget_for_F <= 0:
        return None
    k_budget = max(1, budget_for_F // (n * 16 + 5))

    k = min(k_mag, k_budget, n)
    if k >= n:
        return None  # nothing to truncate

    keep_eigvals = eigvals_s[:k]
    keep_V = V_s[:, :k]
    # Re-apply the noise-floor mask in case any kept eigenvalue is now
    # below tolerance (would emit zero-magnitude columns of F otherwise).
    # Any eigenvalues demoted here must also be absorbed into the diagonal
    # residual to preserve the diagonal-exact property.
    max_abs_kept = float(np.abs(keep_eigvals).max())
    if max_abs_kept > 0.0:
        tol = (
            max(_EIGVAL_TOL_REL, keep_eigvals.size * float(np.finfo(np.float64).eps))
            * max_abs_kept
        )
        mask = np.abs(keep_eigvals) > tol
        demoted_eigvals = keep_eigvals[~mask]
        demoted_V = keep_V[:, ~mask]
        keep_eigvals = keep_eigvals[mask]
        keep_V = keep_V[:, mask]
    else:
        demoted_eigvals = eigvals_s[:0]
        demoted_V = V_s[:, :0]

    # Diagonal absorption: drop_diag[i] = Σ_{j∈dropped} λ_j · v_{i,j}².
    # Folds both the magnitude-cut drops and any noise-floor-demoted
    # eigenpairs, so ``(F · diag(signs) · Fᵀ + diag(diag_orig + drop_diag))[i,i]``
    # matches ``Q[i,i]`` exactly — only off-diagonal entries are lossy.
    drop_eigvals = np.concatenate([eigvals_s[k:], demoted_eigvals])
    drop_V = np.concatenate([V_s[:, k:], demoted_V], axis=1)
    drop_diag = (drop_V**2) @ drop_eigvals

    residual = diag_orig + drop_diag
    payload = _payload_from_eigh(keep_eigvals, keep_V, residual, n)

    # Sanity-check reconstruction error against the original Q before
    # accepting the lossy candidate.
    F = np.frombuffer(bytes.fromhex(payload["F"]), dtype=np.float64).reshape(
        n, payload["k"]
    )
    signs = np.asarray(payload["signs"], dtype=np.float64)
    Q_recon = F @ np.diag(signs) @ F.T + np.diag(residual)
    abs_Q_max = float(np.abs(Q).max())
    err_max = float(np.abs(Q_recon - Q).max())
    rel_err = err_max if abs_Q_max == 0.0 else err_max / abs_Q_max
    if rel_err > _TRUNCATED_REL_ERROR_MAX:
        return None
    # Belt-and-suspenders against the budget formula understating reality.
    if _payload_size(payload) > _TRUNCATED_PAYLOAD_BUDGET_BYTES:
        return None
    return payload


def _serialize_qubo_factored(canonical) -> dict:
    """Encode a QUBO as ``Q = F · diag(signs) · Fᵀ + diag(residual)``.

    Up to three candidate decompositions are computed and the
    smallest-payload one is returned (tie-breaking lossless over lossy):

    A. ``residual = Q.diagonal()``, eigendecompose ``Q − diag(Q.diag())``.
       Lossless. Yields ``k = 0`` for pure-diagonal QUBOs.
    B. ``residual = 0``, eigendecompose ``Q`` itself.
       Lossless. Yields ``k = rank(Q)`` for low-rank QUBOs (e.g. ``u·uᵀ``).
    C. Truncate candidate A's eigendecomposition at the
       :data:`_TRUNCATED_MAGNITUDE_THRESHOLD` magnitude cutoff (or the
       payload-budget cap), absorbing the dropped eigencomponents' diagonal
       contribution into the residual. Lossy. Discarded if reconstruction
       error exceeds :data:`_TRUNCATED_REL_ERROR_MAX`.

    Only handles degree ≤ 2 terms — HUBO inputs must use the legacy form.
    """
    n = canonical.n_vars
    Q = _qubo_to_dense(canonical)
    diag = Q.diagonal().copy()

    # Strategy A: eigendecompose the diagonal-stripped matrix.
    eigvals_off, V_off = _eigh_drop_noise(Q - np.diag(diag))
    cand_a = _payload_from_eigh(eigvals_off, V_off, diag, n)

    # Strategy B: eigendecompose Q itself with zero residual.
    eigvals_full, V_full = _eigh_drop_noise(Q)
    cand_b = _payload_from_eigh(eigvals_full, V_full, np.zeros(n, dtype=np.float64), n)

    # Strategy C: truncated A with diagonal absorption (reuses Strategy A's eigh).
    cand_c = _factored_truncated(eigvals_off, V_off, diag, Q)

    candidates: list[tuple[dict, bool]] = [(cand_a, False), (cand_b, False)]
    if cand_c is not None:
        candidates.append((cand_c, True))

    # Sort by (payload size, lossy?) so ties favour lossless candidates.
    candidates.sort(key=lambda item: (_payload_size(item[0]), 1 if item[1] else 0))
    return candidates[0][0]


def _payload_size(payload) -> int:
    """Byte length of ``payload`` as it would appear on the JSON wire."""
    return len(json.dumps(payload, separators=(",", ":")))


def _serialize_qubo_for_wire(problem: "BinaryOptimizationProblem") -> dict:
    """Serialize a QUBO/HUBO to whichever wire format is smaller.

    Compares the JSON byte sizes of the legacy comma-key dict and the
    factored decomposition, returning the smaller. HUBO inputs (any term of
    degree > 2) skip the factored path because the format is strictly
    quadratic. QUBOs with fewer than :data:`_FACTORED_PROBE_MIN_QUBITS`
    variables skip the eigendecomposition probe — legacy always wins at
    that scale.
    """
    canonical = problem.canonical_problem
    has_hubo = any(len(k) > 2 for k in canonical.terms.keys())
    legacy = _serialize_qubo_legacy(canonical)
    if has_hubo or canonical.n_vars < _FACTORED_PROBE_MIN_QUBITS:
        return legacy

    factored = _serialize_qubo_factored(canonical)
    if _payload_size(factored) < _payload_size(legacy):
        return factored
    return legacy


def _attach_penalty_tuning_components(
    wire_options: dict | None, problem: "BinaryOptimizationProblem"
) -> dict:
    """Attach the cost/penalty split needed by composer-service tuning."""
    penalty_canonical = problem.penalty_canonical_problem
    if penalty_canonical is None:
        raise ValueError(
            "penalty_tuning=True requires BinaryOptimizationProblem(..., penalty=...)."
        )

    options = dict(wire_options or {})
    idx = problem.canonical_problem.variable_to_idx
    options["cost_qubo"] = _serialize_qubo_component_legacy(
        problem.objective_canonical_problem,
        idx,
    )
    options["penalty_qubo"] = _serialize_qubo_component_legacy(
        penalty_canonical,
        idx,
    )
    return options


_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Module-level constants used by ``_render``; pulled out to keep the
# rendering function focused on layout instead of palette bookkeeping.
_QUALITY_BAR_LEN = 40
_QUALITY_COLORS = ((75, "green"), (50, "yellow"), (25, "bright_red"))
_STRUCTURAL_SENSITIVITY_LABELS = (
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
_STRUCTURAL_SENSITIVITY_TABLE_CAP = 16

# Confidence border/label colors for the regime/certificate panel.
_CONFIDENCE_STYLES = {
    "proven": "green",
    "estimated": "yellow",
    "open": "bright_black",
}


def _threshold_pick(
    value: float, thresholds: tuple[tuple[float, str], ...], default: str
) -> str:
    """Return the first label whose threshold ``value`` meets, else ``default``."""
    return next((label for cutoff, label in thresholds if value >= cutoff), default)


def _html_to_rich(text: str) -> str:
    """Convert a small subset of HTML to ``rich`` console markup."""
    text = text.replace("<strong>", "[bold]").replace("</strong>", "[/bold]")
    return _HTML_TAG_RE.sub("", text)


# ``(attribute, format_template)`` pairs for CharacterizationResult.summary()'s
# scalar fields, grouped to preserve the original rendering order without
# forcing structurally different fields (certificate/classical_baseline/hardness/
# best_parameters, all dict-shaped) into the same table.
# "QAOA Amenability" (not "Quality Score") to match the display() gauge label
# and to avoid reading as solution quality. The approximation-ratio line is
# rendered manually in ``summary()`` (not here) since it needs its ``±ε`` band
# and regime-aware "refused" wording.
_QUALITY_SUMMARY_FIELDS = (
    (
        "quality_score",
        "  QAOA Amenability: {:.2f} / 100  (formulation fit, NOT solution quality)",
    ),
    ("concentration_ratio", "  Concentration Ratio: {:.2f}x  (vs subspace uniform)"),
)
_PENALTY_SUMMARY_FIELDS = (
    ("penalty_recommendation", "  Penalty Recommendation: λ={:.2f}"),
    ("feasibility_rate", "  Feasibility Rate: {:.1%}"),
)
_TIMESTAMP_SUMMARY_FIELDS = (
    ("created_at", "  Created: {}"),
    ("completed_at", "  Completed: {}"),
)


@dataclass
class CharacterizationResult:
    """Result container for QUBO/HUBO characterization.

    Returned by :meth:`~divi.backends.QoroService.characterize_and_validate` and
    :func:`~divi.backends.characterization.characterize_and_validate`. Displays a rich HTML report when
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

    recommendations: list[dict] = field(default_factory=list, repr=False)
    """Actionable suggestions for tuning the QUBO or QAOA setup, derived
    from the characterization report.

    Always a list — empty when no rules fire or the job didn't complete.
    Each entry is a dict with these keys:

    * ``level`` — one of ``"info"``, ``"warn"``, ``"action"``. ``action``
      recommends a concrete change; ``warn`` flags a risk; ``info`` is
      contextual.
    * ``metric`` — which report field triggered the rule
      (e.g. ``"quality_score"``, ``"feasibility_rate"``).
    * ``text`` — plain-text message, suitable for terminal/log output.
    * ``html`` — the same message with inline ``<strong>`` markup,
      consumed by the notebook ``_repr_html_`` renderer. ``text`` and
      ``html`` carry the same content; choose by output medium.
    """

    created_at: str | None = None
    """ISO timestamp when the characterization job was created."""

    completed_at: str | None = None
    """ISO timestamp when the characterization job completed."""

    html: str = field(kw_only=True, default="", repr=False, compare=False)
    """Server-rendered HTML report. Empty when the HTML endpoint was unreachable."""

    def _field(self, key: str, *fallbacks: str):
        """Return the first non-``None`` value among ``key`` and ``fallbacks``.

        Treats an explicitly-``null`` value the same as a missing key, so a
        present-but-unset field (e.g. an optional analysis that wasn't
        requested) correctly falls through to the next fallback instead of
        short-circuiting on it.
        """
        if not self.report:
            return None
        for k in (key, *fallbacks):
            val = self.report.get(k)
            if val is not None:
                return val
        return None

    @property
    def quality_score(self) -> float | None:
        """QAOA amenability score (0–100) at the best parameters found.

        Prefers the reference-dependent ``reference_concentration_score`` (how well the
        QAOA ansatz concentrates probability on the reference states at the
        swept best parameters); falls back to the structural
        :attr:`formulation_quality` when no sweep was run.

        **This is not the solution quality** — for the achievable-upper-bound
        approximation ratio, see :attr:`approximation_ratio`, and for the
        "is quantum worth it?" structural call see :attr:`certificate`.
        """
        return self._field(
            "reference_concentration_score",
            "quality_at_best",
            "quality_score",
            "formulation_quality",
        )

    @property
    def formulation_quality(self) -> float | None:
        """Structural amenability score (0–100), reference-independent.

        Scale-invariant composite of the normalized cost gap, ground-state
        degeneracy, density, and weight balance. A high score means the QUBO
        is well-formed for QAOA, not that any depth will solve it.
        """
        return self._field("formulation_quality")

    @property
    def reference_concentration_score(self) -> float | None:
        """QAOA quality (0–100) at the best swept parameters (reference-dependent)."""
        return self._field("reference_concentration_score")

    @property
    def regime(self) -> str | None:
        """Analysis regime the server used to reach its certificate/AR call.

        One of ``"exact"``, ``"structured"``, ``"estimate"``, or ``"refuse"``.
        ``"refuse"`` means the interaction light-cone is wider than the
        feasibility budget at the requested depth, so no cheap-and-correct
        assessment exists — :attr:`approximation_ratio` is ``None`` in that
        regime. See :attr:`refuse_reason` for which limit was hit. This is set
        by the light-cone width (which grows with circuit depth and
        connectivity), not by coupling density.
        """
        return self._field("regime")

    @property
    def confidence(self) -> str | None:
        """Confidence level behind :attr:`regime`: ``"proven"``, ``"estimated"``, or ``"open"``."""
        return self._field("confidence")

    @property
    def refuse_reason(self) -> str | None:
        """Why a ``"refuse"`` :attr:`regime` reported no approximation ratio.

        ``"lightcone_too_wide"`` — the depth-p interaction light-cone exceeds
        the feasibility budget a priori (a size limit, not density).
        ``"estimate_unreachable"`` — routed to the truncated estimator, but its
        ±ε tolerance was unreachable within the term budget. ``None`` outside
        the ``"refuse"`` regime. See :attr:`regime_diagnostics` for the sizes.
        """
        return self._field("refuse_reason")

    @property
    def regime_diagnostics(self) -> dict | None:
        """Light-cone sizes behind the :attr:`regime` call.

        A dict with ``max_lightcone_k``, ``n``, ``layers``, ``k_cheap``, and
        ``k_feasible`` — the numbers that determined the regime.
        """
        return self._field("regime_diagnostics")

    @property
    def certificate(self) -> dict | None:
        """Structural certificate backing :attr:`regime`/:attr:`confidence`.

        A dict with ``certified_easy``, ``no_lowdepth_advantage_expected``,
        ``uncertain`` (bools), ``easy_witnesses``, ``lowdepth_markers``
        (lists of str), and optional ``quantum_curiosity`` /
        ``structural`` sub-dicts — see :attr:`quantum_curiosity`,
        :attr:`is_psd`, :attr:`rank`.
        """
        return self._field("certificate")

    @property
    def quantum_curiosity(self) -> dict | None:
        """``certificate["quantum_curiosity"]`` — probe run when the certificate is ``uncertain``.

        A dict with ``status``, ``depth_to_escape_locality``, and ``next_step``.
        """
        cert = self.certificate
        return cert.get("quantum_curiosity") if isinstance(cert, dict) else None

    def _structural_field(self, key: str):
        """Return ``certificate["structural"][key]`` if both dicts are present."""
        cert = self.certificate
        structural = cert.get("structural") if isinstance(cert, dict) else None
        return structural.get(key) if isinstance(structural, dict) else None

    @property
    def is_psd(self) -> bool | None:
        """Whether the QUBO matrix is positive semidefinite (``certificate["structural"]["is_psd"]``)."""
        return self._structural_field("is_psd")

    @property
    def rank(self) -> int | None:
        """Rank of the QUBO matrix (``certificate["structural"]["rank"]``)."""
        return self._structural_field("rank")

    @property
    def classical_baseline(self) -> dict | None:
        """What cheap classical solvers achieve on the same QUBO.

        A dict with ``greedy_energy``, ``sa_energy``, ``best_energy``,
        ``distinct_optima``, and (for small problems) ``exact_ground_energy``.
        The reference an :attr:`approximation_ratio` needs to be meaningful.
        """
        return self._field("classical_baseline")

    @property
    def relaxation_bound(self) -> float | None:
        """Continuous relaxation bound on the optimum (e.g. LP/SDP), if computed.

        A provable lower bound on the true minimum energy, independent of any
        classical heuristic — when it's close to :attr:`classical_baseline`'s
        ``best_energy``, that baseline is already known to be near-optimal.
        """
        bl = self.classical_baseline
        return bl.get("relaxation_bound") if isinstance(bl, dict) else None

    @property
    def constraint_diagnostics(self) -> list[dict] | None:
        """Per-constraint feasibility diagnostics (violation rate, redundancy)."""
        return self._field("constraint_diagnostics")

    @property
    def penalty_lambda_safe(self) -> float | None:
        """Lucas/GKD guaranteed penalty bound (upper end of the recommended range)."""
        return self._field("penalty_lambda_safe")

    @property
    def penalty_lambda_min_feasible(self) -> float | None:
        """Empirical smallest penalty at which the optimum becomes feasible.

        Exact only for small problems; above ~15 variables it is an
        unreliable subspace-search estimate — check
        :attr:`penalty_lambda_min_feasible_estimated` and prefer
        :attr:`penalty_lambda_safe` when it is ``True``.
        """
        return self._field("penalty_lambda_min_feasible")

    @property
    def penalty_lambda_min_feasible_estimated(self) -> bool | None:
        """Whether :attr:`penalty_lambda_min_feasible` is a subspace estimate.

        ``True`` past the exact-search size cap (~15 qubits), where the value
        is advisory only; rely on :attr:`penalty_lambda_safe` there.
        """
        return self._field("penalty_lambda_min_feasible_estimated")

    def _hardness_field(self, key: str):
        """Return ``self.hardness[key]`` if the hardness dict is present."""
        return self.hardness.get(key) if isinstance(self.hardness, dict) else None

    @property
    def cost_gap(self) -> float | None:
        """Energy gap between the best and second-best assignment (cost spectrum)."""
        return self._hardness_field("cost_gap")

    @property
    def ground_state_degeneracy(self) -> int | None:
        """Number of optimal assignments (exact for small problems)."""
        return self._hardness_field("ground_state_degeneracy")

    @property
    def treewidth_estimate(self) -> int | None:
        """Upper bound on the interaction-graph treewidth (min-fill heuristic)."""
        return self._hardness_field("treewidth_estimate")

    @property
    def frustration_index(self) -> float | None:
        """Fraction of couplings unsatisfiable at the best solution."""
        return self._hardness_field("frustration_index")

    @property
    def cost_gap_normalized(self) -> float | None:
        """:attr:`cost_gap` divided by the full energy range ``E_max - E_min``.

        Scale-invariant (unlike the raw ``cost_gap``), so it's the version to
        compare across differently-scaled formulations of the same problem.
        """
        return self._hardness_field("cost_gap_normalized")

    @property
    def global_flip_symmetric(self) -> bool | None:
        """Whether flipping every bit maps the best solution to another optimum.

        When ``True``, a standard X-mixer QAOA state stays in a fixed
        global-parity eigenspace at any depth, so this degeneracy cannot be
        resolved by adding layers alone (see :attr:`ground_state_degeneracy`).
        """
        return self._hardness_field("global_flip_symmetric")

    @property
    def concentration_ratio(self) -> float | None:
        """Probability mass on reference states relative to the **subspace**
        uniform baseline ``1/2^k`` (k = simulated/variable qubits).

        ``1.0`` matches a uniform distribution *over the simulated subspace*;
        ``> 1`` means the ansatz concentrates mass *on* references; ``< 1`` means
        it concentrates *away* from them. Values near or below 1 at the
        returned parameters indicate the ansatz at this depth cannot resolve
        the reference states — increasing circuit depth (more QAOA layers) or
        running a deeper parameter sweep is the typical remedy.

        Note the baseline is the *subspace* uniform ``1/2^k``, NOT the
        full-space ``1/2^n`` used by the "× uniform" cue in the rendered
        state-probabilities table — so the two can point different directions
        on the same report (they answer different questions: concentration
        within the simulated subspace vs. against the full Hilbert space).

        Prefers the value at the best sweep parameters
        (``concentration_at_best``) when available.
        """
        return self._field("concentration_at_best", "concentration_ratio")

    @property
    def approximation_ratio(self) -> float | None:
        """Achievable upper-bound approximation ratio from the light-cone engine.

        ``r = (⟨C⟩ − C_max) / (C_min − C_max)`` ∈ [0, 1], evaluated at the
        uniform ``|+⟩`` state by the light-cone engine — an upper bound on
        what a real, cold-started QAOA run can reach at the swept depth, not
        a guarantee any live run gets there. Paired with
        :attr:`approximation_ratio_error_bound` for the ``±ε`` band around
        it. Interpret it against :attr:`classical_baseline` (an AR of 0.9
        means little if greedy already reaches the optimum).

        ``None`` in the ``"refuse"`` :attr:`regime` — the server declined to
        estimate rather than ship an unreliable number.
        """
        return self._field("approximation_ratio")

    @property
    def approximation_ratio_error_bound(self) -> float | None:
        """``±ε`` uncertainty band on :attr:`approximation_ratio`.

        ``0`` for the exact light-cone computation (``"exact"``/``"structured"``
        :attr:`regime`); positive for the truncated Pauli-propagation estimate
        used in the ``"estimate"`` regime.
        """
        return self._field("approximation_ratio_error_bound")

    @property
    def best_parameters(self) -> dict | None:
        """Best QAOA parameters found during parameter sweep (if requested)."""
        return self._field("best_parameters")

    @property
    def recommended_min_layers(self) -> int | None:
        """Recommended minimum QAOA depth p, derived from the AR-vs-depth curve.

        The smallest depth at which the (monotone) predicted approximation ratio
        reaches near-optimal or stops improving; see :attr:`recommended_layers_basis`
        for which of those fired. Falls back to a structural estimate when no
        parameter sweep was run.

        This is an *achievable-optimal* depth: it is the depth at which a
        well-tuned QAOA (optimal angles) reaches the target ratio. A real run
        with a finite-budget optimizer typically plateaus a layer or two
        shallower, so treat this as an upper guide. Warm-starting from the
        per-layer angles in :attr:`ar_vs_depth` closes much of that gap.
        """
        return self._field("recommended_min_layers")

    @property
    def recommended_layers_basis(self) -> str | None:
        """How :attr:`recommended_min_layers` was chosen.

        One of ``"threshold_reached"`` (AR hit the near-optimal threshold),
        ``"saturated"`` (deeper layers stopped helping), ``"depth_limited"`` (AR
        still climbing at the deepest probed depth), ``"structural"`` (no sweep),
        or ``"flat_spectrum"``.
        """
        return self._field("recommended_layers_basis")

    @property
    def ar_vs_depth(self) -> list[dict] | None:
        """Monotone predicted approximation ratio as a function of QAOA depth.

        Each entry has ``layers``, ``gammas``, ``betas``, ``energy``,
        ``approximation_ratio``, and ``error_bound``; the curve is
        non-decreasing in ``layers``.
        """
        return self._field("ar_vs_depth")

    def qaoa_initial_params(self, layers: int | None = None) -> np.ndarray | None:
        """Warm-start angles ready for :class:`~divi.qprog.algorithms.QAOA`'s ``run``.

        Returns the swept angles as an ``initial_params`` array of shape
        ``(1, 2 * layers)``, ordered per layer as ``[γ₀, β₀, γ₁, β₁, …]`` — the
        exact layout ``QAOA.run(initial_params=...)`` expects — so warm-starting
        a real run from a characterization is one line::

            result = characterize_and_validate(problem, service=service,
                                               options=CharacterizationOptions(parameter_sweep=True))
            p = result.recommended_min_layers
            qaoa = QAOA(problem, n_layers=p)
            qaoa.run(initial_params=result.qaoa_initial_params())

        Prefers the per-depth :attr:`ar_vs_depth` angles for the requested
        ``layers`` (default: :attr:`recommended_min_layers`, else the deepest
        available). Falls back to broadcasting the p=1 :attr:`best_parameters`
        across ``layers`` when no depth curve is available. Returns ``None`` when
        no angles were produced — the ``"refuse"`` regime, or no
        ``parameter_sweep`` — so guard the result before passing it on.

        Args:
            layers: Circuit depth to warm-start. Defaults to
                :attr:`recommended_min_layers`. Must match the ``n_layers`` of
                the ``QAOA`` you feed it into.
        """
        curve = self.ar_vs_depth
        if curve:
            target = (
                layers
                if layers is not None
                else (self.recommended_min_layers or curve[-1]["layers"])
            )
            entry = (
                next((c for c in curve if c.get("layers") == target), None) or curve[-1]
            )
            gammas = list(entry.get("gammas") or [])
            betas = list(entry.get("betas") or [])
        else:
            bp = self.best_parameters or {}
            gamma, beta = bp.get("gamma"), bp.get("beta")
            if gamma is None or beta is None:
                return None
            n = layers if layers is not None else 1
            gammas, betas = [gamma] * n, [beta] * n

        if not gammas or len(gammas) != len(betas):
            return None
        # Per-layer interleave, cost angle then mixer angle: [γ₀, β₀, γ₁, β₁, …].
        flat = [angle for g, b in zip(gammas, betas) for angle in (g, b)]
        return np.asarray([flat], dtype=float)

    @property
    def state_probabilities(self) -> list[dict] | None:
        """Per-state probability data from the characterization report."""
        return self._field("state_probabilities")

    @property
    def structural_sensitivity(self) -> list | None:
        """Per-qubit structural sensitivity analysis (if requested)."""
        return self._field("structural_sensitivity")

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

    def summary(self) -> str:
        """Return a rich text summary of the characterization result."""
        lines = [
            f"QUBO Characterization Result — Job {self.job_id[:8]}...",
            f"  Status: {self.status}",
        ]
        regime = self.regime
        if regime:
            confidence = self.confidence
            conf_suffix = f" (confidence: {confidence})" if confidence else ""
            lines.append(f"  Regime: {regime}{conf_suffix}")
        cert = self.certificate
        if isinstance(cert, dict):
            if cert.get("certified_easy"):
                witnesses = cert.get("easy_witnesses") or []
                suffix = f" — {witnesses[0]}" if witnesses else ""
                lines.append(f"  Certificate: classically easy{suffix}")
            elif cert.get("no_lowdepth_advantage_expected"):
                markers = cert.get("lowdepth_markers") or []
                suffix = f" — {markers[0]}" if markers else ""
                lines.append(
                    f"  Certificate: no low-depth quantum advantage expected{suffix}"
                )
            elif cert.get("uncertain"):
                lines.append("  Certificate: uncertain (not ruled out)")
        ar = self.approximation_ratio
        if regime == "refuse" or ar is None:
            reason = (
                "estimate tolerance unreachable within budget"
                if self.refuse_reason == "estimate_unreachable"
                else "light-cone too wide for a cheap-and-correct assessment"
            )
            lines.append(f"  Approximation Ratio: not assessed (refused — {reason})")
        else:
            ar_line = f"  Approximation Ratio (achievable upper bound): {ar:.4f}"
            err = self.approximation_ratio_error_bound
            if isinstance(err, (int, float)) and not isinstance(err, bool) and err > 0:
                ar_line += f" ± {err:.3g}"
            ar_line += " (light-cone, not a live run)"
            lines.append(ar_line)
        lines += [
            tpl.format(v)
            for attr, tpl in _QUALITY_SUMMARY_FIELDS
            if (v := getattr(self, attr)) is not None
        ]
        if isinstance(self.classical_baseline, dict):
            be = self.classical_baseline.get("best_energy")
            if isinstance(be, (int, float)):
                lines.append(f"  Classical Best Energy: {be:.4f}")
            # Surface the provable LP lower bound next to the heuristic best:
            # when they coincide, the classical answer is certifiably optimal.
            rb = self.relaxation_bound
            if isinstance(rb, (int, float)):
                lines.append(f"  Classical Relaxation Bound (provable): {rb:.4f}")
        if self.hardness:
            difficulty = self.hardness.get("difficulty", "unknown")
            lines.append(f"  Hardness: {difficulty}")
            if self.cost_gap is not None:
                # Lead with the scale-invariant normalized gap (the one meant
                # for comparison); raw gap in parentheses.
                if self.cost_gap_normalized is not None:
                    lines.append(
                        f"    Cost Gap: {self.cost_gap_normalized:.4f} normalized ({self.cost_gap:.4f} raw)"
                    )
                else:
                    lines.append(f"    Cost Gap: {self.cost_gap:.4f}")
            if self.ground_state_degeneracy is not None:
                lines.append(
                    f"    Ground-state Degeneracy: {self.ground_state_degeneracy}"
                )
            if self.treewidth_estimate is not None:
                lines.append(f"    Treewidth (<=): {self.treewidth_estimate}")
        if bp := self.best_parameters:
            lines.append(
                f"  Best Parameters: γ={bp.get('gamma', '?')}, β={bp.get('beta', '?')}"
            )
        lines += [
            tpl.format(v)
            for attr, tpl in _PENALTY_SUMMARY_FIELDS
            if (v := getattr(self, attr)) is not None
        ]
        if (
            self.penalty_lambda_min_feasible is not None
            and self.penalty_lambda_safe is not None
        ):
            lines.append(
                f"  Safe Penalty Range: λ ∈ [{self.penalty_lambda_min_feasible:.2f}, "
                f"{self.penalty_lambda_safe:.2f}]"
            )
        elif self.penalty_lambda_safe is not None:
            lines.append(f"  Safe Penalty Bound: λ ≤ {self.penalty_lambda_safe:.2f}")
        if self.constraint_diagnostics:
            lines.append(
                f"  Constraint Diagnostics: {len(self.constraint_diagnostics)} constraint(s)"
            )
        lines += [
            tpl.format(v)
            for attr, tpl in _TIMESTAMP_SUMMARY_FIELDS
            if (v := getattr(self, attr))
        ]
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


class _Ansatz(BaseModel):
    """QAOA ansatz-shape controls (:attr:`CharacterizationOptions.ansatz`)."""

    model_config = ConfigDict(extra="forbid")

    mixer: Literal["x", "xy", "I"] | None = None
    layers: int | None = None


class _Subspace(BaseModel):
    """Subspace-search controls (:attr:`CharacterizationOptions.subspace`)."""

    model_config = ConfigDict(extra="forbid")

    auto_warmstart: StrictBool = True
    solver: str | None = None
    max_variable_qubits: StrictInt | None = Field(default=None, gt=0)
    base_bitstring: str | None = None
    variable_qubits: list[Annotated[StrictInt, Field(ge=0)]] | None = None

    @field_validator("base_bitstring")
    @classmethod
    def _binary_chars_only(cls, v: str | None) -> str | None:
        if v is not None:
            invalid = set(v) - {"0", "1"}
            if invalid:
                raise ValueError(
                    f"base_bitstring contains non-binary characters: {invalid}."
                )
        return v

    @model_validator(mode="after")
    def _check_subspace(self) -> "_Subspace":
        provided = self.model_fields_set
        has_base = "base_bitstring" in provided
        has_variables = "variable_qubits" in provided
        has_auto_controls = "solver" in provided or "max_variable_qubits" in provided
        if self.auto_warmstart and (has_base or has_variables):
            raise ValueError(
                "base_bitstring and variable_qubits are manual subspace controls. "
                "Set auto_warmstart=False when providing them."
            )
        if not self.auto_warmstart and has_auto_controls:
            raise ValueError(
                "solver and max_variable_qubits only affect automatic subspace "
                "selection. Omit them when auto_warmstart=False."
            )
        if has_base != has_variables:
            raise ValueError(
                "base_bitstring and variable_qubits must be provided together for "
                "manual subspace selection."
            )
        if self.variable_qubits is not None and len(set(self.variable_qubits)) != len(
            self.variable_qubits
        ):
            raise ValueError("variable_qubits must not contain duplicate indices.")
        return self


class CharacterizationOptions(BaseModel):
    """Configuration for :func:`~divi.backends.characterization.characterize_and_validate`.

    All fields are optional; default-construct for a basic run with no
    sub-analyses. Field combinations are validated at construction time, so
    misconfiguration surfaces before any API call.

    Examples:
        >>> CharacterizationOptions(parameter_sweep=True, structural_sensitivity=True)
        >>> CharacterizationOptions(gamma=1.2, beta=0.7)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structural_sensitivity: StrictBool = False
    """Request per-qubit structural sensitivity analysis."""

    parameter_sweep: StrictBool = False
    """Request a γ/β parameter sweep.

    Mutually exclusive with fixed ``gamma`` / ``beta``.
    """

    penalty_tuning: StrictBool = False
    """Request penalty-lambda tuning.

    Requires ``BinaryOptimizationProblem(..., penalty=...)`` in the
    :func:`characterize_and_validate` call.
    """

    gamma: float | None = None
    """Fixed γ value. Mutually exclusive with ``parameter_sweep``."""

    beta: float | None = None
    """Fixed β value. Mutually exclusive with ``parameter_sweep``."""

    constraints: list | None = None
    """Constraint descriptors, each a dict ``{"type": ..., "bound": ...}``.

    Supported ``type`` values:

    * ``"max_cardinality"`` / ``"min_cardinality"`` / ``"eq_cardinality"`` —
      at most / at least / exactly ``bound`` variables set to 1.
    * ``"inequality"`` — weighted sum ``Σ wᵢ·xᵢ ≤ bound``; requires a
      ``"weights"`` dict ``{qubit_index: weight}``.
    * ``"equality"`` — weighted sum ``Σ wᵢ·xᵢ == bound``; requires
      ``"weights"``.

    Optional keys: ``"qubits"`` — the variable indices the constraint applies
    to (defaults to all; used by the cardinality types). Indices must be in
    ``[0, n_qubits)`` or the call is rejected. Example::

        constraints=[
            {"type": "max_cardinality", "bound": 3},
            {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5, 2: 7}},
        ]
    """

    ansatz: _Ansatz | None = None
    """Ansatz configuration dict (e.g. ``{"mixer": "x", "layers": 1}``).

    Supported ``mixer`` values are ``"x"``, ``"xy"``, and ``"I"`` (identity/no
    mixer, useful as a diagnostic baseline).

    Only circuit-shape controls belong here. Search-space controls such as
    ``auto_warmstart``, ``max_variable_qubits``, ``base_bitstring``, and
    ``variable_qubits`` belong in :attr:`subspace`.
    """

    subspace: _Subspace | None = None
    """Subspace-search configuration dict.

    Supported keys include ``auto_warmstart``, ``solver``,
    ``max_variable_qubits``, ``base_bitstring``, and ``variable_qubits``. These
    controls choose or bound the simulated/search subspace; they are not
    properties of the QAOA circuit ansatz itself.
    """

    n_qubits: StrictInt | None = Field(default=None, gt=0)
    """Explicit qubit count for the problem.

    Only needed to pin the dimension when a QUBO's highest-indexed variable
    has no terms (so it can't be inferred from the QUBO keys) — e.g. a
    problem on 5 variables where variable 4 is unconstrained. When ``None``,
    the qubit count is inferred from the QUBO. Must be a positive integer.
    """

    @model_validator(mode="after")
    def _check_options(self) -> "CharacterizationOptions":
        if self.parameter_sweep and (self.gamma is not None or self.beta is not None):
            raise ValueError(
                "parameter_sweep=True is mutually exclusive with fixed "
                "gamma/beta — pick one."
            )
        # Subspace checks that need n_qubits (a sibling field) live here rather
        # than on _Subspace, which cannot see the enclosing qubit count.
        if self.subspace is not None and self.n_qubits is not None:
            bitstring = self.subspace.base_bitstring
            if bitstring is not None and len(bitstring) != self.n_qubits:
                raise ValueError(
                    f"base_bitstring has {len(bitstring)} bits but "
                    f"n_qubits={self.n_qubits}."
                )
            for q in self.subspace.variable_qubits or ():
                if q >= self.n_qubits:
                    raise ValueError(
                        f"variable_qubits index {q} is out of range for "
                        f"n_qubits={self.n_qubits}."
                    )
        return self

    def _to_wire(self) -> dict | None:
        """Serialize to the wire-format options dict (or ``None`` if empty)."""
        analysis = {
            k: v
            for k, v in {
                "gamma": self.gamma,
                "beta": self.beta,
                "structural_sensitivity": self.structural_sensitivity,
                "parameter_sweep": self.parameter_sweep,
                "penalty_tuning": self.penalty_tuning,
            }.items()
            if v is not None
        }
        options = {
            k: v
            for k, v in {
                "analysis": analysis or None,
                "ansatz": (
                    self.ansatz.model_dump(exclude_none=True, exclude_unset=True)
                    if self.ansatz is not None
                    else None
                ),
                "subspace": (
                    self.subspace.model_dump(exclude_none=True, exclude_unset=True)
                    if self.subspace is not None
                    else None
                ),
                "constraints": self.constraints,
                "n_qubits": self.n_qubits,
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

    # --- Regime / certificate (decision-first: is quantum worth it here?) ---
    # Surfaced as its own panel, ahead of the quality gauge, so the single
    # most decision-relevant line isn't just a sentence buried in the header.
    regime = result.regime
    confidence = result.confidence
    cert = result.certificate
    if regime or isinstance(cert, dict):
        color = _CONFIDENCE_STYLES.get(confidence or "", "cyan")
        body_lines = []
        if regime:
            conf_suffix = f" (confidence: {confidence})" if confidence else ""
            body_lines.append(
                f"[bold {color}]Regime: {regime}{conf_suffix}[/bold {color}]"
            )
        if isinstance(cert, dict):
            if cert.get("certified_easy"):
                witnesses = cert.get("easy_witnesses") or []
                suffix = f" — {witnesses[0]}" if witnesses else ""
                body_lines.append(f"Certificate: classically easy{suffix}")
            elif cert.get("no_lowdepth_advantage_expected"):
                markers = cert.get("lowdepth_markers") or []
                suffix = f" — {markers[0]}" if markers else ""
                body_lines.append(
                    f"Certificate: no low-depth quantum advantage expected{suffix}"
                )
            elif cert.get("uncertain"):
                body_lines.append("Certificate: uncertain (not ruled out)")
                qc = result.quantum_curiosity
                next_step = qc.get("next_step") if isinstance(qc, dict) else None
                if next_step:
                    body_lines.append(f"[dim]Next step: {next_step}[/dim]")
        if regime == "refuse":
            reason = (
                "estimate tolerance unreachable within budget"
                if result.refuse_reason == "estimate_unreachable"
                else "light-cone too wide for a cheap-and-correct assessment"
            )
            body_lines.append(
                f"[bold red]Refused: {reason} — no approximation ratio "
                "given.[/bold red]"
            )
        console.print(Panel("\n".join(body_lines), border_style=color))

    # --- Quality gauge ---
    # Note: this is QAOA *amenability*, not solution quality — see
    # ``CharacterizationResult.quality_score``'s docstring. Labelled
    # accordingly so it isn't mistaken for the approximation ratio above.
    qs = result.quality_score
    if qs is not None:
        color = _threshold_pick(qs, _QUALITY_COLORS, default="red")
        filled = min(_QUALITY_BAR_LEN, int(_QUALITY_BAR_LEN * qs / 100))
        bar = (
            f"[{color}]{'█' * filled}[/{color}]"
            f"[dim]{'░' * (_QUALITY_BAR_LEN - filled)}[/dim]"
        )
        console.print(f"  QAOA Amenability: {bar} [bold]{qs:.2f}[/bold] / 100\n")

    # Pre-compute the uniform baseline; reused by the Best Parameters panel
    # (for the inline P(reference) vs uniform cue) and the State Probabilities
    # table further down.
    uniform_prob = (1.0 / (2**n_qubits)) if n_qubits is not None else None
    reference_set = set((result.report or {}).get("reference_states") or ())

    # --- Best parameters ---
    # Surfaced near the top: this is the actionable output of the sweep.
    bp = result.best_parameters
    if bp:
        gamma = bp.get("gamma")
        beta = bp.get("beta")
        # Derive ``P(reference)`` from ``state_probabilities`` so the rendered
        # number matches the rendered table further down. The server-supplied
        # ``bp["probability"]`` field has opaque semantics and does not in
        # general equal the sum of reference-state sampling probabilities.
        sp = result.state_probabilities or []
        reference_prob: float | None = None
        if reference_set and sp:
            reference_prob = sum(
                float(s.get("probability", 0))
                for s in sp
                if s.get("is_reference", s.get("state") in reference_set)
            )

        parts = []
        if gamma is not None:
            parts.append(f"[bold green]γ = {gamma:.4f}[/bold green]")
        if beta is not None:
            parts.append(f"[bold green]β = {beta:.4f}[/bold green]")
        if reference_prob is not None:
            # Inline the boost-vs-uniform so the number is self-interpretable
            # without scrolling to the State Probabilities table.
            cue = ""
            if uniform_prob:
                boost = reference_prob / uniform_prob
                # Full-space (1/2^n) baseline here -- distinct from the
                # concentration_ratio field's subspace (1/2^k) baseline.
                cue = (
                    f"  ({boost:.2f}× full-space uniform)"
                    if boost < 1.0
                    else f"  ({boost:.1f}× full-space uniform)"
                )
            parts.append(f"[dim]P(reference) = {reference_prob:.6f}{cue}[/dim]")
        console.print(
            Panel(
                "  " + "\n  ".join(parts),
                title="[green]Best Parameters[/green]",
                border_style="green",
            )
        )

    # --- Recommendations (server-supplied) ---
    # Surfaced right after Best Parameters: the most actionable interpretive
    # content the report carries. Reference tables (hardness, state probs,
    # structural_sensitivity) live below.
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

    # --- Penalty tuning ---
    pr = result.penalty_recommendation
    wt = result.is_well_tuned
    lambda_min = result.penalty_lambda_min_feasible
    lambda_safe = result.penalty_lambda_safe
    if pr is not None or wt is not None or lambda_safe is not None:
        items = [
            "[dim]λ is the constraint-penalty multiplier — too low and infeasible "
            "states can outscore the true optimum, too high and the QUBO gets "
            "harder to solve. The safe range keeps you inside both bounds.[/dim]"
        ]
        if lambda_min is not None and lambda_safe is not None:
            items.append(
                f"Safe range: λ ∈ [[bold]{lambda_min:.2f}[/bold], [bold]{lambda_safe:.2f}[/bold]]"
            )
        elif lambda_safe is not None:
            items.append(f"Safe bound: λ ≤ [bold]{lambda_safe:.2f}[/bold]")
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

    # --- Constraint diagnostics ---
    diagnostics = result.constraint_diagnostics
    if diagnostics:
        ct = Table(
            title="Constraint Diagnostics",
            caption=(
                "[dim]High violation rate → raise that constraint's penalty toward its "
                "recommended λ. Redundant constraints are already always-satisfied.[/dim]"
            ),
            border_style="red",
        )
        ct.add_column("#", justify="center")
        ct.add_column("Type")
        ct.add_column("Violation Rate", justify="right")
        ct.add_column("Redundant", justify="center")
        ct.add_column("Recommended λ", justify="right")
        for d in diagnostics:
            if not isinstance(d, dict):
                continue
            rate = d.get("violation_rate")
            rec_lambda = d.get("recommended_lambda")
            is_redundant = bool(d.get("is_redundant"))
            ct.add_row(
                str(d.get("index", "?")),
                str(d.get("type", "?")),
                f"{rate:.1%}" if isinstance(rate, (int, float)) else "—",
                "[green]✓[/green]" if is_redundant else "[dim]✗[/dim]",
                f"{rec_lambda:.2f}" if isinstance(rec_lambda, (int, float)) else "—",
            )
        console.print(ct)

    # --- State probabilities ---
    sp = result.state_probabilities
    if sp:
        st = Table(title="State Probabilities", border_style="magenta")
        st.add_column("State", style="bold")
        st.add_column("Reference?", justify="center")
        st.add_column("Probability", justify="right")
        if uniform_prob:
            st.add_column("vs Uniform", justify="right")

        for s in sp[:_STATE_TABLE_CAP]:
            state = s.get("state", "?")
            is_reference = s.get("is_reference", state in reference_set)
            marker = "[green]✓[/green]" if is_reference else "[dim]✗[/dim]"
            prob = s.get("probability", 0)
            row = [str(state), marker, f"{prob:.6f}"]
            if uniform_prob:
                boost = prob / uniform_prob
                boost_str = f"{boost:.1f}×" if boost >= 1.0 else f"{boost:.2f}×"
                row.append(f"[bold]{boost_str}[/bold]" if is_reference else boost_str)
            st.add_row(*row)

        console.print(st)
        if uniform_prob and n_qubits is not None:
            console.print(f"  [dim]Uniform: {uniform_prob:.6f} (1/{2**n_qubits})[/dim]")

    # --- Hardness analysis (reference: static QUBO structure) ---
    if result.hardness:
        ht = Table(
            title="Hardness Analysis",
            caption="[dim]Static QUBO structure metrics — does not predict QAOA quality at any specific depth.[/dim]",
            border_style="yellow",
        )
        ht.add_column("Metric", style="bold")
        ht.add_column("Value", justify="right")
        for key, value in result.hardness.items():
            label = key.replace("_", " ").title()
            fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
            ht.add_row(label, fmt)
        console.print(ht)

    # --- Structural sensitivity (reference: per-qubit flip-cost criticality) ---
    sens = result.structural_sensitivity
    if sens:
        se = Table(
            title="Structural Sensitivity (per-qubit flip-cost criticality)",
            border_style="blue",
        )
        se.add_column("Qubit", justify="center")
        se.add_column("Score", justify="right")
        se.add_column("Assessment")
        for entry in sens[:_STRUCTURAL_SENSITIVITY_TABLE_CAP]:
            val = entry.get("score", 0)
            assessment = _threshold_pick(
                val, _STRUCTURAL_SENSITIVITY_LABELS, default="[green]stable[/green]"
            )
            se.add_row(str(entry.get("qubit", "?")), f"{val:.4f}", assessment)
        console.print(se)


def _wrap_response(data: dict, service: QoroService) -> CharacterizationResult:
    # ``job_id`` and ``status`` are required fields in the server contract;
    # let ``KeyError`` surface noisily on a malformed payload rather than
    # silently fabricating defaults. Optional metadata stays as ``.get()``.
    job_id = data["job_id"]
    recs = data.get("recommendations")

    # The submit flow is synchronous today, so a completed job is expected on
    # read. Guard against a future async server returning an incomplete result
    # unnoticed: warn if the job is still pending/running when we read it.
    status = data["status"]
    if status not in ("COMPLETED", "FAILED"):
        logger.warning(
            "Characterization job %s returned status %s (expected COMPLETED). "
            "The report may be incomplete.",
            job_id,
            status,
        )

    try:
        html = service._fetch_characterization_html(job_id)
    except requests.RequestException as exc:
        logger.warning(
            "Could not fetch HTML report for job %s: %s. "
            "Returning result without rendered HTML.",
            job_id,
            exc,
        )
        html = ""

    return CharacterizationResult(
        job_id=job_id,
        status=status,
        hardness=data.get("hardness"),
        report=data.get("report"),
        recommendations=recs if recs is not None else [],
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        html=html,
    )


def characterize_and_validate(
    problem: "BinaryOptimizationProblem",
    reference_states: list[str],
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
        reference_states: Reference solution bitstrings used for
            reference-dependent diagnostics. They are not constraints, not a
            warm start, and do not have to be proven optima. Pass ``[]`` when
            you do not have reference solutions; the service may derive a
            classical reference solution for analyses that require one.
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
        >>> from divi.backends import QoroService
        >>> from divi.backends.characterization import (
        ...     CharacterizationOptions,
        ...     characterize_and_validate,
        ... )
        >>> from divi.qprog.problems import BinaryOptimizationProblem
        >>> problem = BinaryOptimizationProblem(np.array([[-1, 2], [0, -1]]))
        >>> result = characterize_and_validate(  # doctest: +SKIP
        ...     problem,
        ...     reference_states=["01", "10"],
        ...     service=QoroService(),
        ...     options=CharacterizationOptions(parameter_sweep=True),
        ... )
        >>> result.quality_score  # doctest: +SKIP
        78.5

    .. note::
        Credit cost scales with QUBO size.
    """
    options = options or CharacterizationOptions()
    wire_qubo = _serialize_qubo_for_wire(problem)
    wire_options = options._to_wire()
    if options.penalty_tuning:
        wire_options = _attach_penalty_tuning_components(wire_options, problem)
    # The factored payload encodes indices into opaque byte arrays, so the
    # qubit count must be passed alongside it for accurate credit billing.
    if isinstance(wire_qubo, dict) and wire_qubo.get("_format") == "factored_v1":
        if wire_options is None:
            wire_options = {}
        wire_options.setdefault("n_qubits", wire_qubo["n"])
    data = service.characterize_and_validate(
        qubo=wire_qubo,
        reference_states=reference_states,
        options=wire_options,
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
        >>> from divi.backends import QoroService
        >>> from divi.backends.characterization import get_characterization_result
        >>> result = get_characterization_result(  # doctest: +SKIP
        ...     "4d0550f5-ffb0-...", service=QoroService()
        ... )
        >>> result.display()   # doctest: +SKIP
        >>> result.quality_score  # doctest: +SKIP
        45.89
    """
    data = service.characterize_and_validate(job_id=job_id)
    return _wrap_response(data, service)
