# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Linear constraints encoded as quadratic penalties with binary slack."""

import math
import numbers
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Literal, get_args

import dimod
import numpy as np

_MAX_SLACK_BITS = 8
_SLACK_MAX = 2**_MAX_SLACK_BITS - 1
_MAX_DENOMINATOR = 10**6
_APPROX_BITS = 20
_MAX_INTEGER = 2**_APPROX_BITS
_TOL = 1e-12

Sense = Literal["==", "<=", ">="]
_SENSES: tuple[Sense, ...] = get_args(Sense)


class LinearConstraint:
    r"""A linear constraint :math:`\sum_i a_i x_i \;(\mathrm{sense})\; b` over binary variables.

    Pass it through ``constraints=`` on
    :class:`~divi.qprog.problems.BinaryOptimizationProblem`, which adds the
    squared violation in the constraint's own units as a penalty (missing the
    bound by ``v`` costs ``v**2`` times ``penalty_weight``, measured from the
    bound rounded inwards to the coefficients' common step). Equality
    constraints add no qubits; inequalities add up to eight slack qubits, and
    ones that always hold add nothing.

    Integers, decimals up to six places and simple fractions are encoded
    exactly when every value is at most ``2**20`` of their common step. Other
    values, or slack that would need more than eight bits, are rounded and a
    :class:`UserWarning` says so; if rounding would leave a constraint that
    always holds, :class:`ValueError` is raised instead.
    Feasibility checks use the original constraint.

    Args:
        coefficients: Mapping from variable label to coefficient, or a sequence
            whose position ``i`` is the coefficient of variable ``i``. Zero
            coefficients are dropped.
        sense: ``"=="``, ``"<="`` or ``">="``.
        bound: Right-hand side :math:`b`.

    Raises:
        ValueError: If ``sense`` is unknown, a value is not finite, or every
            coefficient is zero.

    Examples:
        >>> from divi.qprog.problems import LinearConstraint
        >>> LinearConstraint([1, 1, 1], "==", 2)  # choose exactly two variables
        LinearConstraint({0: 1.0, 1: 1.0, 2: 1.0}, '==', 2.0)
    """

    __slots__ = ("_coefficients", "_sense", "_bound", "_n_positions", "_magnitude")

    def __init__(
        self,
        coefficients: Mapping[Hashable, float] | Sequence[float] | np.ndarray,
        sense: Sense,
        bound: float,
    ):
        if sense not in _SENSES:
            raise ValueError(f"sense must be one of {_SENSES}, got {sense!r}.")
        if isinstance(coefficients, Mapping):
            items = coefficients.items()
            self._n_positions: int | None = None
        else:
            values = np.asarray(coefficients, dtype=float).tolist()
            items = enumerate(values)
            self._n_positions = len(values)
        coeffs: dict[Hashable, float] = {}
        for var, value in items:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Coefficient of {var!r} must be finite.")
            if value != 0.0:
                coeffs[var] = value
        if not coeffs:
            raise ValueError("A constraint needs at least one non-zero coefficient.")
        bound = float(bound)
        if not math.isfinite(bound):
            raise ValueError("bound must be finite.")
        self._coefficients = coeffs
        self._sense: Sense = sense
        self._bound = bound
        self._magnitude = max(abs(bound), math.fsum(map(abs, coeffs.values())))

    @property
    def coefficients(self) -> Mapping[Hashable, float]:
        """Non-zero coefficients keyed by variable label."""
        return MappingProxyType(self._coefficients)

    @property
    def sense(self) -> Sense:
        """``"=="``, ``"<="`` or ``">="``."""
        return self._sense

    @property
    def bound(self) -> float:
        """Right-hand side of the constraint."""
        return self._bound

    def is_satisfied(
        self, assignment: Mapping[Hashable, float], *, tol: float = _TOL
    ) -> bool:
        """Whether ``assignment`` satisfies the constraint, before rounding.

        ``tol`` is relative to ``max(|b|, Σ|a_i|)``, so the check does not
        depend on the units of the coefficients.
        """
        lhs = math.fsum(c * assignment[v] for v, c in self._coefficients.items())
        margin = tol * self._magnitude
        if self._sense == "==":
            return abs(lhs - self._bound) <= margin
        if self._sense == ">=":
            return lhs >= self._bound - margin
        return lhs <= self._bound + margin

    def __repr__(self) -> str:
        return (
            f"LinearConstraint({dict(self._coefficients)!r}, "
            f"{self._sense!r}, {self._bound!r})"
        )


@dataclass(frozen=True)
class _Slack:
    """An inequality's slack bits and the integer data that determines them."""

    terms: tuple[tuple[Hashable, int], ...]
    """``(label, coefficient)`` pairs: powers of two, then one top-up."""
    scaled_coefficients: dict[Hashable, int]
    target: int

    def bits(self, decision: Mapping[Hashable, int]) -> dict[Hashable, int]:
        """Slack bits that minimise the penalty for a decision assignment."""
        lhs = sum(c * decision[v] for v, c in self.scaled_coefficients.items())
        *powers, (top_label, top) = self.terms
        value = min(max(self.target - lhs, 0), top + 2 ** len(powers) - 1)
        use_top = value >= top
        rest = value - top if use_top else value
        bits = {label: (rest >> j) & 1 for j, (label, _) in enumerate(powers)}
        bits[top_label] = int(use_top)
        return bits


def _rational(
    constraint: LinearConstraint, tol: float
) -> tuple[dict[Hashable, Fraction], Fraction, Fraction, bool]:
    """Coefficients and bound as fractions, their common step, and whether it is exact.

    A coefficient within float noise of a fraction with denominator up to
    ``10**6`` is taken as that fraction, and so is an equality's bound within
    ``tol``. The step divides the coefficients (and an equality's bound)
    exactly when their common denominator stays within ``10**6``; otherwise
    (irrational or high-precision data) it is ``2**-20`` of the largest.
    """
    coefficients = {}
    for v, c in constraint.coefficients.items():
        f = Fraction(c).limit_denominator(_MAX_DENOMINATOR)
        coefficients[v] = f if abs(float(f) - c) <= _TOL * abs(c) else Fraction(c)
    bound = Fraction(constraint.bound)
    lattice = list(coefficients.values())
    if constraint.sense == "==":
        f = bound.limit_denominator(_MAX_DENOMINATOR)
        if abs(float(f) - constraint.bound) <= tol:
            bound = f
        lattice.append(bound)
    denominator = math.lcm(*(f.denominator for f in lattice))
    exact = denominator <= _MAX_DENOMINATOR
    if exact:
        numerator = math.gcd(*(int(f * denominator) for f in lattice))
        step = Fraction(numerator, denominator)
    else:
        step = max(map(abs, lattice)) / 2**_APPROX_BITS
    return coefficients, bound, step, exact


def _scale(
    coefficients: Mapping[Hashable, Fraction],
    bound: Fraction,
    sense: Sense,
    step: Fraction,
    tol: float,
) -> tuple[dict[Hashable, int], int]:
    """Coefficients and bound as integer multiples of ``step``.

    Inequality bounds round inwards (up for ``>=``, down for ``<=``) after
    allowing ``tol``, so an integer left-hand side meets the scaled bound
    exactly when it meets the original within that tolerance.
    """
    scaled = {v: round(c / step) for v, c in coefficients.items()}
    margin = Fraction(tol) / step
    if sense == ">=":
        return scaled, math.ceil(bound / step - margin)
    if sense == "<=":
        return scaled, math.floor(bound / step + margin)
    return scaled, round(bound / step)


class _UnreachableBoundError(ValueError):
    """No allowed assignment brings ``constraint``'s left-hand side to its bound.

    ``lo`` and ``hi`` are the extremes of the left-hand side over those
    assignments.
    """

    def __init__(self, constraint: LinearConstraint, lo: float, hi: float):
        self.constraint, self.lo, self.hi = constraint, lo, hi
        super().__init__(
            f"{constraint!r} is infeasible: over the assignments the problem "
            f"allows, its left-hand side ranges over [{lo + 0.0:.6g}, {hi + 0.0:.6g}]."
        )


def _encode_constraint(
    constraint: LinearConstraint,
    activity_bounds: Callable[[Mapping[Hashable, float]], tuple[float, float]],
) -> tuple[dimod.BinaryQuadraticModel, _Slack | None]:
    r"""Encode ``constraint`` as :math:`(\sum a_i x_i \pm \sum_j 2^j s_j - b)^2`.

    The slack is sized for the activity bounds of the left-hand side, as
    ``activity_bounds`` gives them for a coefficient mapping.

    Returns:
        The penalty in the constraint's own units, where a violation of ``v``
        costs ``v**2``, and its slack as dimod labels it, or ``None`` when it
        needs none. The penalty includes its constant, so it is zero exactly
        when the rounded constraint holds and the slack takes its matching
        value.

    Raises:
        ValueError: If no assignment the problem allows can satisfy the
            rounded constraint, or the slack range cannot fit in eight bits.
    """
    sense, b = constraint.sense, constraint.bound
    lo, hi = activity_bounds(constraint.coefficients)
    tol = _TOL * constraint._magnitude
    if (sense != "<=" and hi < b - tol) or (sense != ">=" and lo > b + tol):
        raise _UnreachableBoundError(constraint, lo, hi)
    if (sense == "<=" and hi <= b + tol) or (sense == ">=" and lo >= b - tol):
        return dimod.BinaryQuadraticModel(dimod.BINARY), None

    coefficients, bound_value, step, exact = _rational(constraint, tol)
    scaled, bound = _scale(coefficients, bound_value, sense, step, tol)
    # dimod squares these integers in float64; beyond 2**20 a unit violation
    # drowns in rounding.
    largest = max(abs(bound), *map(abs, scaled.values()))
    if largest > _MAX_INTEGER:
        step *= math.ceil(largest / _MAX_INTEGER)
        scaled, bound = _scale(coefficients, bound_value, sense, step, tol)
        exact = False
    lo_s, hi_s = map(round, activity_bounds(scaled))
    slack_range = 0
    if sense == "==":
        divisor = math.gcd(*scaled.values())
        if exact and divisor and bound % divisor:
            raise ValueError(
                f"{constraint!r} is infeasible: every left-hand side is a multiple "
                f"of {float(step * divisor):.6g}."
            )
    else:
        # Coarsen the step until the slack fits in its bits.
        multiple = 1
        while True:
            slack_range = hi_s - bound if sense == ">=" else bound - lo_s
            if slack_range <= _SLACK_MAX:
                break
            multiple = math.ceil(multiple * slack_range / _SLACK_MAX)
            scaled, bound = _scale(
                coefficients, bound_value, sense, step * multiple, tol
            )
            lo_s, hi_s = map(round, activity_bounds(scaled))
        step *= multiple
        exact = exact and multiple == 1
    r = float(step)
    bound_shift = abs(b - bound * r)
    if sense != "==":
        bound = min(max(bound, lo_s), hi_s)
        slack_range = max(slack_range, 0)
        if (sense == ">=" and lo_s >= bound) or (sense == "<=" and hi_s <= bound):
            raise ValueError(
                f"{constraint!r} needs more than {_MAX_SLACK_BITS} slack bits: "
                "rounded to fit them, it would always hold."
            )

    if not exact:
        error = max(
            abs(c - s * r)
            for c, s in zip(constraint.coefficients.values(), scaled.values())
        )
        warnings.warn(
            f"{constraint!r} is encoded with coefficients rounded to steps of "
            f"{r:.3g}, moving them by up to {error:.3g} and the bound by "
            f"{bound_shift:.3g}. The penalty may accept or reject assignments that "
            "the constraint itself does not; is_feasible still checks the original "
            "constraint.",
            UserWarning,
            stacklevel=4,
        )
    # dimod builds λ(Σ a·x + Σ c·s - target)² in integer units; λ = r² turns it
    # into the constraint's own units, so a violation of v costs v².
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    linear = [(v, c) for v, c in scaled.items() if c]
    if slack_range == 0:
        bqm.add_linear_equality_constraint(linear, r * r, -bound)
        return bqm, None
    # dimod names its slack f"slack_{label}_{j}"; keep clear of user labels.
    label = "divi"
    while any(isinstance(v, str) and v.startswith(f"slack_{label}_") for v in scaled):
        label += "_"
    lb, ub = (bound, hi_s) if sense == ">=" else (lo_s, bound)
    slack = bqm.add_linear_inequality_constraint(linear, r * r, label, lb=lb, ub=ub)
    terms = tuple((s, int(c)) for s, c in slack)
    return bqm, _Slack(terms, scaled, ub)


def _encode_constraints(
    constraints: Sequence[LinearConstraint],
    variables: set[Hashable],
    penalty_terms: Mapping[tuple, float],
    activity_bounds: Callable[[Mapping[Hashable, float]], tuple[float, float]],
) -> tuple[tuple[_Slack, ...], dict[tuple, float]]:
    """Slack of each inequality in ``constraints``, and ``penalty_terms`` plus their penalties.

    Slack variables continue the problem's integer labels after the largest
    one, or are labelled ``("slack", i, j)`` when the problem uses other
    labels.
    """
    for constraint in constraints:
        unknown = [v for v in constraint.coefficients if v not in variables]
        if unknown:
            raise ValueError(
                f"{constraint!r} refers to variables that are not in the "
                f"problem: {unknown!r}."
            )

    int_labels = [int(v) for v in variables if isinstance(v, numbers.Integral)]
    integer_labels = len(int_labels) == len(variables)
    next_int = max(int_labels, default=-1) + 1
    total = dimod.BinaryQuadraticModel(dimod.BINARY)
    slacks: list[_Slack] = []
    for index, constraint in enumerate(constraints):
        bqm, slack = _encode_constraint(constraint, activity_bounds)
        if slack is not None:
            width = len(slack.terms)
            if integer_labels:
                labels: list[Hashable] = list(range(next_int, next_int + width))
                next_int += width
            else:
                labels = [("slack", index, j) for j in range(width)]
                if variables.intersection(labels):
                    raise ValueError(
                        "Variable labels of the form ('slack', i, j) are "
                        "reserved for constraint slack."
                    )
            mapping = dict(zip((s for s, _ in slack.terms), labels))
            bqm.relabel_variables(mapping)
            relabelled = tuple((mapping[s], c) for s, c in slack.terms)
            slacks.append(replace(slack, terms=relabelled))
        total.update(bqm)

    terms: defaultdict[tuple, float] = defaultdict(float, penalty_terms)
    for v, b in total.linear.items():
        if b:
            terms[(v,)] += b
    for (u, v), b in total.quadratic.items():
        if b:
            terms[(u, v)] += b
    if total.offset:
        terms[()] += total.offset
    return tuple(slacks), dict(terms)
