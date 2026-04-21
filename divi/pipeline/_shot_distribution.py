# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shot distribution across measurement groups.

When a Hamiltonian is split into multiple measurement groups (via QWC, wire
grouping, or no grouping), each group gets its own measurement circuit. The
total shot budget can be distributed across groups uniformly, weighted by the
L1 norm of each group's coefficients, or via a user-defined callable.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

#: How to distribute a total shot budget across measurement groups.
#: See :func:`compute_shot_distribution` for the behaviour of each choice.
ShotDistStrategy = (
    Literal["uniform", "weighted", "weighted_random"]
    | Callable[[list[float], int], list[int]]
)


def compute_group_l1_norms(
    coefficients: Sequence[float],
    partition_indices: Sequence[Sequence[int]],
) -> list[float]:
    """Sum the absolute coefficient values within each measurement group.

    Args:
        coefficients: Coefficients of the original (single-term) observables,
            in the order produced by ``_extract_coeffs``.
        partition_indices: Each inner list is the set of original-observable
            indices that belong to one measurement group.

    Returns:
        One non-negative float per group; ``len(...) == len(partition_indices)``.
    """
    return [sum(abs(coefficients[i]) for i in group) for group in partition_indices]


def compute_shot_distribution(
    group_norms: Sequence[float],
    total_shots: int,
    strategy: ShotDistStrategy = "uniform",
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Distribute ``total_shots`` across measurement groups.

    The returned list always has ``len(group_norms)`` entries that sum to
    exactly ``total_shots`` (for the built-in strategies).

    Args:
        group_norms: L1 norm (or any non-negative weight) per group.
        total_shots: Total shot budget to distribute.
        strategy: How the budget is split across groups.

            * ``"uniform"`` — equal split, with the remainder distributed
              round-robin. Use when you have no prior information about
              which groups contribute most.
            * ``"weighted"`` — proportional to each group's coefficient L1
              norm, using the largest-remainder method to preserve the
              total exactly. Reduces estimator variance on Hamiltonians
              whose coefficient distribution is skewed (typical for
              chemistry).  The preferred default when using adaptive shot
              allocation.
            * ``"weighted_random"`` — multinomial sample with the same
              probabilities as ``"weighted"``.  Same expected allocation,
              but individual runs are stochastic; reproducibility requires
              seeding ``rng``.  Pick this only when you need unbiased
              estimators across independent runs (e.g. classical-shadows
              analysis) — for a single run ``"weighted"`` gives lower
              variance and better coverage of small-coefficient terms.
            * Callable
              ``(group_l1_norms, total_shots) -> per_group_shots`` for
              fully custom allocation.
        rng: Random generator used by ``"weighted_random"``. Ignored for
            other strategies. Defaults to ``np.random.default_rng()``.

    Returns:
        Per-group shot counts.

    Raises:
        ValueError: If ``total_shots`` is negative or ``group_norms`` is empty.
    """
    n_groups = len(group_norms)
    if n_groups == 0:
        raise ValueError("group_norms must contain at least one entry.")
    if total_shots < 0:
        raise ValueError(f"total_shots must be non-negative, got {total_shots}.")

    if callable(strategy):
        result = list(strategy(list(group_norms), total_shots))
        if len(result) != n_groups:
            raise ValueError(
                f"Custom shot distribution returned {len(result)} entries, "
                f"expected {n_groups}."
            )
        if any(s < 0 for s in result):
            raise ValueError("Custom shot distribution returned negative shot counts.")
        int_result = [int(s) for s in result]
        # Callable contract doesn't require integer/sum-preserving output;
        # warn when truncation drops shots so the user knows their budget
        # drifted instead of silently running with fewer shots.
        if sum(int_result) != total_shots:
            warnings.warn(
                f"Custom shot distribution returned values summing to "
                f"{sum(result):.6g} (truncated to {sum(int_result)}), which "
                f"does not equal total_shots={total_shots}. Return integer "
                f"values that sum to total_shots to avoid budget drift.",
                UserWarning,
                stacklevel=3,
            )
        return int_result

    if strategy == "uniform":
        return _uniform_distribution(n_groups, total_shots)
    if strategy == "weighted":
        return _weighted_distribution(group_norms, total_shots)
    if strategy == "weighted_random":
        return _weighted_random_distribution(group_norms, total_shots, rng)
    raise ValueError(
        f"Unknown shot distribution strategy: {strategy!r}. "
        "Expected 'uniform', 'weighted', 'weighted_random', or a callable."
    )


def _uniform_distribution(n_groups: int, total_shots: int) -> list[int]:
    """Equal split with the remainder distributed to the first ``r`` groups."""
    base, remainder = divmod(total_shots, n_groups)
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def _weighted_distribution(group_norms: Sequence[float], total_shots: int) -> list[int]:
    """Largest-remainder allocation proportional to ``group_norms``.

    If all norms are zero, falls back to a uniform distribution.
    """
    total_weight = sum(group_norms)
    if total_weight <= 0:
        return _uniform_distribution(len(group_norms), total_shots)

    exact = [total_shots * w / total_weight for w in group_norms]
    floors = [int(x) for x in exact]
    allocated = sum(floors)
    leftover = total_shots - allocated

    # Distribute leftover to groups with the largest fractional remainders.
    remainders = sorted(
        ((exact[i] - floors[i], i) for i in range(len(group_norms))),
        key=lambda t: (-t[0], t[1]),
    )
    for k in range(leftover):
        floors[remainders[k][1]] += 1
    return floors


def _weighted_random_distribution(
    group_norms: Sequence[float],
    total_shots: int,
    rng: np.random.Generator | None,
) -> list[int]:
    """Multinomial sampling with probabilities proportional to ``group_norms``.

    Falls back to a uniform distribution if all norms are zero.
    """
    total_weight = sum(group_norms)
    if total_weight <= 0:
        return _uniform_distribution(len(group_norms), total_shots)
    if total_shots == 0:
        return [0] * len(group_norms)

    if rng is None:
        rng = np.random.default_rng()
    probs = np.asarray(group_norms, dtype=np.float64) / total_weight
    return rng.multinomial(total_shots, probs).tolist()
