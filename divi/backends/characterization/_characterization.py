# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QUBO/HUBO characterisation: serialisation, result container, and public API."""

import json
import logging
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import numpy as np
import requests
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from divi.exceptions import (
    CharacterizationFailedError,
    ExecutionCancelledError,
)

from .._job_status import InsufficientCreditsError, JobTimedOutError
from ..runners._qoro import QoroService

if TYPE_CHECKING:
    from divi.qprog.problems import BinaryOptimizationProblem

logger = logging.getLogger(__name__)

_LEVEL_STYLES = {"warning": "yellow", "action": "magenta", "info": "cyan"}


def _number(value: Any, spec: str) -> str:
    """``value`` formatted by ``spec``, or ``"—"`` when it is not a number."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return format(value, spec) if is_number else "—"


def _vector(values: Sequence[float]) -> str:
    """Angles as ``\\[a, b, …]`` to four decimals, the bracket escaped for rich."""
    return escape(f"[{', '.join(_number(v, '.4f') for v in values)}]")


# Smallest QUBO size at which the factored encoding is even probed.
# Below this, the legacy comma-key dict is always smaller on the wire,
# so the two eigendecomposition probes would be pure overhead.
_FACTORED_PROBE_MIN_QUBITS = 64

# Smallest term density at which the factored encoding can win. A sparse
# problem needs near-full rank to reconstruct, so the factored payload grows
# faster than the legacy dict and the probe cannot pay for itself.
_FACTORED_PROBE_MIN_DENSITY = 0.5

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
    """Serialise to the comma-key dict format, e.g. ``{"0": -1.0, "0,1": 2.0}``.

    Term keys are original variable *names* (which may be strings for a
    ``dimod.BinaryQuadraticModel``), so they are remapped to integer indices
    via ``variable_to_idx``. Accepts terms of any degree, so it is the only
    valid path for HUBO inputs.
    """
    return _serialize_qubo_component_legacy(canonical, canonical.variable_to_idx)


def _serialize_qubo_component_legacy(
    canonical, variable_to_idx: dict
) -> dict[str, float]:
    """Serialise a component using another QUBO's variable indexing."""
    return {
        ",".join(str(variable_to_idx[v]) for v in term_key): float(coeff)
        for term_key, coeff in canonical.terms.items()
        if coeff != 0 and term_key
    }


def _qubo_to_dense(canonical) -> np.ndarray:
    """Build the symmetric dense QUBO matrix from canonical polynomial terms.

    Off-diagonal coefficients are split half-and-half between ``Q[i,j]`` and
    ``Q[j,i]`` so the result is exactly symmetric. Linear ``(i,)`` terms write
    to the diagonal; canonical terms never repeat a variable.
    """
    n = canonical.n_vars
    idx = canonical.variable_to_idx
    Q = np.zeros((n, n), dtype=np.float64)
    for term_key, coeff in canonical.terms.items():
        if coeff == 0 or not term_key:
            continue
        mapped = [idx[v] for v in term_key]
        if len(mapped) == 1:
            i = mapped[0]
            Q[i, i] += float(coeff)
        else:
            i, j = mapped
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
        "k": eigvals.size,
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

    Returns ``None`` when truncation does not apply (no eigenvalues, or every
    eigenvalue survives both the magnitude and budget cuts) or when the
    reconstruction relative error exceeds :data:`_TRUNCATED_REL_ERROR_MAX`.
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

    # Magnitude cut: keep every eigenvalue at least ε·|λ_max|.
    k_mag = int(np.sum(abs_s >= _TRUNCATED_MAGNITUDE_THRESHOLD * abs_s[0]))

    # Payload-budget cap. JSON cost per kept column ≈ n·16 hex chars for F
    # plus ≈5 chars for the corresponding ``signs`` entry; envelope + diag
    # are fixed costs.
    budget_for_F = _TRUNCATED_PAYLOAD_BUDGET_BYTES - n * 16 - 200
    if budget_for_F <= 0:
        return None
    k_budget = max(1, budget_for_F // (n * 16 + 5))

    k = min(k_mag, k_budget)
    if k == eigvals.size:
        return None  # nothing to truncate

    # Diagonal absorption: drop_diag[i] = Σ_{j∈dropped} λ_j · v_{i,j}², so
    # ``(F · diag(signs) · Fᵀ + diag(diag_orig + drop_diag))[i,i]`` matches
    # ``Q[i,i]`` exactly — only off-diagonal entries are lossy.
    drop_diag = (V_s[:, k:] ** 2) @ eigvals_s[k:]

    residual = diag_orig + drop_diag
    payload = _payload_from_eigh(eigvals_s[:k], V_s[:, :k], residual, n)

    # Sanity-check reconstruction error against the original Q before
    # accepting the lossy candidate.
    F = np.frombuffer(bytes.fromhex(payload["F"]), dtype=np.float64).reshape(
        n, payload["k"]
    )
    signs = np.asarray(payload["signs"], dtype=np.float64)
    Q_recon = (F * signs) @ F.T + np.diag(residual)
    rel_err = float(np.abs(Q_recon - Q).max()) / float(np.abs(Q).max())
    if rel_err > _TRUNCATED_REL_ERROR_MAX:
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

    # Smallest (payload size, lossy?) so ties favour lossless candidates.
    return min(candidates, key=lambda item: (_payload_size(item[0]), item[1]))[0]


def _payload_size(payload) -> int:
    """Byte length of ``payload`` as it would appear on the JSON wire."""
    return len(json.dumps(payload, separators=(",", ":")))


def _serialize_qubo_for_wire(problem: "BinaryOptimizationProblem") -> dict:
    """Serialise a QUBO/HUBO to whichever wire format is smaller.

    Compares the JSON byte sizes of the legacy comma-key dict and the
    factored decomposition, returning the smaller. HUBO inputs (any term of
    degree > 2) skip the factored path because the format is strictly
    quadratic. QUBOs with fewer than :data:`_FACTORED_PROBE_MIN_QUBITS`
    variables skip the eigendecomposition probe — legacy always wins at
    that scale.
    """
    canonical = problem.canonical_problem
    if canonical.terms.get(()):
        warnings.warn(
            "The problem carries a constant offset, which is not transmitted. "
            "The analysis is unaffected — a constant shifts every energy "
            "equally — but the reported energies exclude it.",
            UserWarning,
            stacklevel=3,
        )
    has_hubo = any(len(k) > 2 for k in canonical.terms.keys())
    legacy = _serialize_qubo_legacy(canonical)
    n = canonical.n_vars
    if has_hubo or n < _FACTORED_PROBE_MIN_QUBITS:
        return legacy
    # The factored payload costs ~16*n*k bytes against legacy's ~30 per term,
    # and k approaches n on a sparse problem — so below this density legacy
    # always wins and the eigendecompositions are pure cost.
    if len(legacy) < _FACTORED_PROBE_MIN_DENSITY * n * n:
        return legacy

    factored = _serialize_qubo_factored(canonical)
    return factored if _payload_size(factored) < _payload_size(legacy) else legacy


def _attach_penalty_components(
    wire_options: dict | None, problem: "BinaryOptimizationProblem"
) -> dict | None:
    """Attach the problem's cost/penalty split, when it has one, for the report's penalty section."""
    penalty_canonical = problem.penalty_canonical_problem
    if penalty_canonical is None:
        return wire_options
    options = dict(wire_options or {})
    idx = problem.canonical_problem.variable_to_idx
    options["cost_qubo"] = _serialize_qubo_component_legacy(
        problem.objective_canonical_problem, idx
    )
    options["penalty_qubo"] = _serialize_qubo_component_legacy(penalty_canonical, idx)
    return options


@dataclass
class CharacterizationResult:
    """The characterisation report of a QUBO/HUBO, as Maestro's characterisation service words it.

    Returned by :meth:`~divi.backends.QoroService.characterize_and_validate` and
    :func:`~divi.backends.characterization.characterize_and_validate`. The
    report's sections are exposed as they arrive, and shortcuts read individual
    values. Evaluating the result in Jupyter shows the
    server-rendered HTML report.

    .. note::
        Credit cost scales with QUBO size.
    """

    job_id: str
    """Unique identifier for the characterisation job."""

    status: str
    """Job status (``COMPLETED``, ``FAILED``, etc.)."""

    report: dict = field(default_factory=dict, repr=False)
    """The full report, with sections ``summary``, ``problem``, ``classical``,
    ``qaoa``, ``angles``, ``references``, ``limits``, ``recommendations`` and,
    on request, ``constraints`` and ``penalty``."""

    created_at: str | None = None
    """ISO timestamp when the characterisation job was created."""

    completed_at: str | None = None
    """ISO timestamp when the characterisation job completed."""

    html: str = field(kw_only=True, default="", repr=False, compare=False)
    """Server-rendered HTML report. Empty when the HTML endpoint was unreachable."""

    def _section(self, name: str) -> dict:
        return self.report.get(name) or {}

    @property
    def summary(self) -> str:
        """One sentence on whether and how to run QAOA on this problem."""
        return self.report.get("summary") or ""

    @property
    def recommendations(self) -> list[dict]:
        """Advice, warnings and actions first, each a dict of ``level``, ``topic`` and ``text``."""
        return self.report.get("recommendations") or []

    @property
    def problem(self) -> dict:
        """The QUBO's size, terms, degree, components and light-cone widths."""
        return self._section("problem")

    @property
    def classical(self) -> dict:
        """The classical baseline's best solution, a rigorous lower bound, and the exact optimum when known."""
        return self._section("classical")

    @property
    def qaoa(self) -> dict:
        """QAOA's result per layer count (``curve``) and the recommended layer count."""
        return self._section("qaoa")

    @property
    def angles(self) -> dict:
        """Starting angles for the requested layers, with their source and circuit convention."""
        return self._section("angles")

    @property
    def references(self) -> list[dict]:
        """One entry per reference state, with its energy and, when computed, its probability."""
        return self.report.get("references") or []

    @property
    def constraints(self) -> dict:
        """Per-constraint redundancy and violation probability, and slack encodings; empty without constraints."""
        return self._section("constraints")

    @property
    def penalty(self) -> dict:
        """Penalty weight guidance; empty unless the problem has a penalty component."""
        return self._section("penalty")

    @property
    def limits(self) -> list[str]:
        """What the report leaves out at this problem's size."""
        return self.report.get("limits") or []

    @property
    def recommended_layers(self) -> int | None:
        """The recommended QAOA layer count; ``None`` when no layer count was computed."""
        return self.qaoa.get("recommended_layers")

    @property
    def gap_closed(self) -> float | None:
        """Share of the gap from random guessing to the optimum that QAOA closes at the requested layers.

        ``0`` is no better than guessing and ``1`` the optimum; ``None`` when
        the requested layer count was not computed.
        """
        at_requested = self.qaoa.get("at_requested") or {}
        return at_requested.get("gap_closed")

    @property
    def recommended_lambda(self) -> float | None:
        """The recommended penalty weight; ``None`` without a penalty component or above the exact size."""
        return self.penalty.get("recommended_lambda")

    def qaoa_initial_params(self, layers: int | None = None) -> np.ndarray | None:
        """Angles ready for :class:`~divi.qprog.algorithms.QAOA`'s ``run``.

        Returns an ``initial_params`` array of shape ``(1, 2 * layers)``,
        ordered per layer as ``[γ₀, β₀, γ₁, β₁, …]``, so warm-starting a real
        run from a characterisation is one line::

            result = characterize_and_validate(problem, ["01"], service=service)
            qaoa = QAOA(problem, n_layers=result.recommended_layers)
            qaoa.run(initial_params=result.qaoa_initial_params())

        Args:
            layers: Layer count to warm-start; defaults to
                :attr:`recommended_layers`, else the requested layers. Must
                match the ``n_layers`` of the ``QAOA`` you feed it into.

        Returns:
            The angles, or ``None`` when the report carries none.

        Raises:
            ValueError: If the report has no angles at ``layers``.
        """
        target = layers if layers is not None else self.recommended_layers
        entries = [*(self.qaoa.get("curve") or []), self.angles]
        if target is None:
            entry = self.angles
        else:
            entry = next((e for e in entries if e.get("layers") == target), None)
            if entry is None:
                available = sorted({e["layers"] for e in entries if "layers" in e})
                raise ValueError(
                    f"No angles at layers={target}; the report has them for {available}."
                )
        gammas, betas = list(entry.get("gammas") or []), list(entry.get("betas") or [])
        if not gammas or len(gammas) != len(betas):
            return None
        return np.asarray(
            [[angle for g, b in zip(gammas, betas) for angle in (g, b)]], dtype=float
        )

    def __repr__(self) -> str:
        lines = [
            f"CharacterizationResult(job_id={self.job_id!r}, status={self.status!r})"
        ]
        if self.summary:
            lines.append(f"  {self.summary}")
        lines += [
            f"  [{r.get('level', 'info')}] {r.get('text', '')}"
            for r in self.recommendations
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return the server-rendered HTML report (Jupyter)."""
        return self.html

    def display(self) -> None:
        """Print the report to the terminal; every piece of wording comes from the report itself."""
        parts: list = [f"[bold]{escape(self.summary)}[/]"] if self.summary else []
        if self.recommendations:
            parts.append(
                Panel(
                    "\n".join(
                        f"[{_LEVEL_STYLES.get(r.get('level', 'info'), 'cyan')}]•[/] {escape(r.get('text', ''))}"
                        for r in self.recommendations
                    ),
                    title="Recommendations",
                )
            )
        curve = self.qaoa.get("curve") or []
        if curve:
            confidence = _number(self.qaoa.get("shots_confidence"), ".0%")
            table = Table(title="QAOA by layer count")
            for column in (
                "Layers (p)",
                "Gap closed",
                "P(optimum)",
                f"Shots for {confidence}",
                "Method",
            ):
                table.add_column(column)
            for e in curve:
                marker = " ★" if e.get("layers") == self.recommended_layers else ""
                table.add_row(
                    f"{e.get('layers')}{marker}",
                    _number(e.get("gap_closed"), ".1%"),
                    _number(e.get("p_ground"), ".4f"),
                    _number(e.get("shots_for_confidence"), ","),
                    escape(e.get("method_label") or "—"),
                )
            parts.append(table)
            if reason := self.qaoa.get("recommended_reason"):
                parts.append(f"★ recommended: {escape(reason)}")
        if self.angles.get("gammas"):
            note = "; ".join(
                filter(
                    None,
                    (self.angles.get("source_label"), self.angles.get("convention")),
                )
            )
            parts.append(
                Panel(
                    f"γ = {_vector(self.angles['gammas'])}\nβ = {_vector(self.angles['betas'])}\n"
                    f"[dim]{escape(note)}[/]",
                    title=f"Angles (p = {self.angles.get('layers')})",
                )
            )
        if self.limits:
            parts.append(Panel(escape("\n".join(self.limits)), title="Not included"))
        Console().print(
            Panel(
                Group(*parts), title=f"QUBO Characterization Report — job {self.job_id}"
            )
        )


def _plain_integer(value: Any) -> Any:
    """Integral values, numpy's included, as ``int``; ``bool`` stays for the strict check."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return value


def _plain_list(value: Any) -> Any:
    return value.tolist() if isinstance(value, np.ndarray) else value


_Int = Annotated[StrictInt, BeforeValidator(_plain_integer)]


class _Ansatz(BaseModel):
    """QAOA ansatz-shape controls (:attr:`CharacterizationOptions.ansatz`)."""

    model_config = ConfigDict(extra="forbid")

    mixer: Literal["x", "xy", "I"] | None = None
    layers: _Int | None = Field(default=None, gt=0)


class _Constraint(BaseModel):
    """A single constraint descriptor (:attr:`CharacterizationOptions.constraints`)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "max_cardinality",
        "min_cardinality",
        "eq_cardinality",
        "inequality",
        "equality",
    ]
    bound: StrictFloat | StrictInt
    qubits: list[Annotated[_Int, Field(ge=0)]] | None = None
    weights: dict[Annotated[_Int, Field(ge=0)], StrictFloat | StrictInt] | None = None

    _plain_qubits = field_validator("qubits", mode="before")(_plain_list)

    @field_validator("weights", mode="before")
    @classmethod
    def _int_keys(cls, v):
        """Accept the string keys a JSON round-trip produces."""
        if isinstance(v, dict):
            return {
                int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k: w
                for k, w in v.items()
            }
        return v

    @model_validator(mode="after")
    def _require_weights(self) -> Self:
        if self.type in ("inequality", "equality") and not self.weights:
            raise ValueError(f"'{self.type}' requires a non-empty 'weights' mapping.")
        return self


class CharacterizationOptions(BaseModel):
    """Configuration for :func:`~divi.backends.characterization.characterize_and_validate`.

    All fields are optional and validated at construction time, so
    misconfiguration surfaces before any API call.

    Examples:
        >>> CharacterizationOptions(ansatz={"layers": 2})
        >>> CharacterizationOptions(constraints=[{"type": "eq_cardinality", "bound": 2}])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    constraints: list[_Constraint] | None = None
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
    ``[0, n_qubits)``; when :attr:`n_qubits` is set they are checked at
    construction, otherwise the server rejects out-of-range indices. Example::

        constraints=[
            {"type": "max_cardinality", "bound": 3},
            {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5, 2: 7}},
        ]
    """

    ansatz: _Ansatz | None = None
    """Ansatz configuration dict (e.g. ``{"mixer": "x", "layers": 1}``).

    ``layers`` is the requested layer count. Supported ``mixer`` values are
    ``"x"``, ``"xy"`` (Hamming-weight preserving), and ``"I"`` (no mixer).
    """

    n_qubits: StrictInt | None = Field(default=None, gt=0)
    """Explicit qubit count for the problem.

    Only needed to pin the dimension when a QUBO's highest-indexed variable
    has no terms (so it can't be inferred from the QUBO keys) — e.g. a
    problem on 5 variables where variable 4 is unconstrained. When ``None``,
    the qubit count is inferred from the QUBO. Must be a positive integer.
    """

    _plain_n_qubits = field_validator("n_qubits", mode="before")(_plain_integer)

    @model_validator(mode="after")
    def _check_constraint_indices(self) -> Self:
        if self.n_qubits is None:
            return self
        for k, c in enumerate(self.constraints or ()):
            for kind, indices in (
                ("weight", c.weights or {}),
                ("qubit", c.qubits or ()),
            ):
                for idx in indices:
                    if idx >= self.n_qubits:
                        raise ValueError(
                            f"constraints[{k}]: {kind} index {idx} is out of range for "
                            f"n_qubits={self.n_qubits}."
                        )
        return self

    def _to_wire(self) -> dict | None:
        """Serialise to the wire-format options dict (or ``None`` if empty)."""
        options = {
            k: v
            for k, v in {
                "ansatz": (
                    self.ansatz.model_dump(exclude_none=True)
                    if self.ansatz is not None
                    else None
                ),
                "constraints": (
                    [c.model_dump(exclude_none=True) for c in self.constraints]
                    if self.constraints is not None
                    else None
                ),
                "n_qubits": self.n_qubits,
            }.items()
            if v
        }
        return options or None


def _wrap_response(data: dict, service: QoroService) -> CharacterizationResult:
    # ``job_id`` and ``status`` are required fields in the server contract;
    # let ``KeyError`` surface noisily on a malformed payload rather than
    # silently fabricating defaults. Optional metadata stays as ``.get()``.
    job_id = data["job_id"]
    status = data["status"]
    if status == "FAILED":
        raise CharacterizationFailedError(
            job_id, status, "the analysis did not complete on the server."
        )
    if status == "CANCELLED":
        raise ExecutionCancelledError(
            f"Characterisation job {job_id} was cancelled before completion."
        )
    if status == "INSUFFICIENT_CREDITS":
        raise InsufficientCreditsError(job_id)
    if status == "TIMED_OUT":
        raise JobTimedOutError(job_id)
    if status != "COMPLETED":
        raise CharacterizationFailedError(
            job_id,
            status,
            "the job has not finished; re-fetch it once it reaches COMPLETED.",
        )
    if not data.get("report"):
        raise CharacterizationFailedError(
            job_id, status, "the job completed but carried no report."
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
        report=data["report"],
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        html=html,
    )


def _validate_reference_states(
    problem: "BinaryOptimizationProblem",
    reference_states: Sequence[str],
    n_qubits: int | None,
) -> None:
    """Validate the non-empty, fixed-width binary reference contract."""
    inferred_n = problem.canonical_problem.n_vars
    if n_qubits is not None and n_qubits < inferred_n:
        raise ValueError(
            f"n_qubits={n_qubits} is smaller than the problem's "
            f"{inferred_n} represented variables."
        )
    expected_n = n_qubits or inferred_n

    if len(reference_states) == 0:
        raise ValueError(
            "reference_states must contain at least one candidate solution bitstring."
        )
    for index, state in enumerate(reference_states):
        if not isinstance(state, str) or re.fullmatch(r"[01]+", state) is None:
            raise ValueError(
                f"reference_states[{index}] must be a non-empty binary bitstring."
            )
        if len(state) != expected_n:
            raise ValueError(
                f"reference_states[{index}] has {len(state)} bits; expected "
                f"{expected_n} bits."
            )


def characterize_and_validate(
    problem: "BinaryOptimizationProblem",
    reference_states: Sequence[str],
    *,
    service: QoroService,
    options: CharacterizationOptions | None = None,
) -> CharacterizationResult:
    """One-call QUBO/HUBO characterisation with rich notebook display.

    Converts the problem to wire format, submits it to the Qoro
    Characterisation Service, and returns a :class:`CharacterizationResult`
    that renders the report in Jupyter. When the problem was built with a
    penalty (``BinaryOptimizationProblem(..., penalty=...)`` or
    ``constraints=...``), its cost/penalty split is sent too and the report
    gains a ``penalty`` section.

    Args:
        problem: A :class:`~divi.qprog.problems.BinaryOptimizationProblem`.
            Wrap raw inputs (ndarray, sparse, BQM, HUBO dict, etc.) in this
            class before calling.
        reference_states: One or more candidate solution bitstrings the report
            covers. They are not constraints or a warm start and do not have
            to be proven optima. Each value must be binary and match the
            problem dimension.
        service: A :class:`~divi.backends.QoroService` instance.
        options: Optional :class:`CharacterizationOptions` with the ansatz,
            constraints and qubit count.

    Returns:
        CharacterizationResult: The report. Displaying it in Jupyter shows the
        rendered HTML report.

    Raises:
        ValueError: If reference states are empty, non-binary, or the wrong
            width, or if ``options.n_qubits`` is too small for the problem.
        requests.exceptions.HTTPError: On definite API rejections.
        ~divi.exceptions.CharacterizationSubmitError: If a job was created but
            submission, status polling (including exhausting the service's
            ``max_retries``), or result retrieval failed ambiguously. Its
            ``job_id`` identifies the existing job to re-fetch.
        ~divi.exceptions.CharacterizationFailedError: If the job failed or
            completed carrying no report.
        ~divi.exceptions.ExecutionCancelledError: If the job was cancelled.
        ~divi.backends.InsufficientCreditsError: If the account lacks the
            credits required to run the job.
        ~divi.backends.JobTimedOutError: If server-side execution timed out.

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
        ...     options=CharacterizationOptions(ansatz={"layers": 2}),
        ... )
        >>> result.summary  # doctest: +SKIP
        'An exact polynomial-time classical method solves this problem; ...'

    .. note::
        Credit cost scales with QUBO size.
    """
    options = options or CharacterizationOptions()
    reference_states = list(reference_states)
    _validate_reference_states(problem, reference_states, options.n_qubits)

    wire_qubo = _serialize_qubo_for_wire(problem)
    wire_options = _attach_penalty_components(options._to_wire(), problem)

    # Factored payloads hide their dimension in byte arrays, so send it
    # explicitly for credit billing and backend validation.
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
    """Wait for and retrieve a characterisation result by job ID.

    Polling and retrieval do not cost credits. The service waits through
    non-terminal statuses according to its polling configuration, then returns
    the completed report or raises the corresponding terminal-status error.
    Interrupting the wait does not cancel the job.

    Args:
        job_id: Identifier of a previously submitted characterisation job.
        service: A :class:`~divi.backends.QoroService` instance.

    Returns:
        CharacterizationResult: The stored structured and rendered report.

    Raises:
        ~divi.exceptions.CharacterizationFailedError: If the job failed or
            completed without a report.
        ~divi.exceptions.ExecutionCancelledError: If the job was cancelled.
        ~divi.backends.InsufficientCreditsError: If the account lacks the
            credits required to run the job.
        ~divi.backends.JobTimedOutError: If server-side execution timed out.
        ~divi.exceptions.CharacterizationSubmitError: If status polling or
            result retrieval fails ambiguously after retries.

    Examples:
        >>> from divi.backends import QoroService
        >>> from divi.backends.characterization import get_characterization_result
        >>> result = get_characterization_result(  # doctest: +SKIP
        ...     "4d0550f5-ffb0-...", service=QoroService()
        ... )
        >>> print(result)   # doctest: +SKIP
    """
    data = service.characterize_and_validate(job_id=job_id)
    return _wrap_response(data, service)
