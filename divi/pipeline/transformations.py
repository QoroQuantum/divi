# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared reduction/grouping helpers for pipeline stages.

Stages that fan out along an axis (ham, qem, obs_group, etc.) repeatedly:
- strip that axis from child labels to get a base key,
- group child results by base key (sometimes keyed by axis index for ordering),
- then reduce (e.g. mean or a postprocess over ordered values).

This module centralizes that logic for easier DevX and consistency.
"""

from collections.abc import Callable
from typing import Any

from divi.pipeline.abc import ChildResults


def strip_axis_from_label(
    child_label: tuple[Any, ...], axis_name: str
) -> tuple[Any, ...]:
    """Remove the (axis_name, value) pair from a child label to get the base key.

    Child labels are sequences of (axis_name, value) pairs. This returns
    the same tuple with any element whose first element equals
    *axis_name* removed.

    Example::

        >>> strip_axis_from_label((('ham', 0), ('obs', 1), ('qem', 2)), 'obs')
        (('ham', 0), ('qem', 2))
    """
    return tuple(
        element
        for element in child_label
        if not (
            isinstance(element, tuple) and len(element) >= 1 and element[0] == axis_name
        )
    )


def group_by_base_key(
    results: ChildResults,
    axis_name: str,
    *,
    indexed: bool = False,
) -> dict[tuple[Any, ...], Any]:
    """Group child results by base key (label with axis stripped).

    Args:
        results: Child label -> value mapping from the pipeline.
        axis_name: Axis to strip from labels to form base_key.
        indexed: If False, values are collected into a list per base_key.
            If True, values are stored in a dict[int, value] keyed by the
            axis value (parsed as int) so they can be ordered later.

    Returns:
        - If indexed=False: ``dict[base_key, list[value]]``
        - If indexed=True: ``dict[base_key, dict[int, value]]``

    Example::

        >>> results = {(('circ', 0), ('obs', 0)): 1.5, (('circ', 0), ('obs', 1)): 2.0}
        >>> group_by_base_key(results, 'obs')
        {(('circ', 0),): [1.5, 2.0]}
        >>> group_by_base_key(results, 'obs', indexed=True)
        {(('circ', 0),): {0: 1.5, 1: 2.0}}
    """
    if not indexed:
        grouped: dict[tuple[Any, ...], list[Any]] = {}
        for child_label, child_value in results.items():
            base_key = strip_axis_from_label(child_label, axis_name)
            grouped.setdefault(base_key, []).append(child_value)
        return grouped

    grouped_indexed: dict[tuple[Any, ...], dict[int, Any]] = {}
    for child_label, child_value in results.items():
        axis_values = dict(child_label)
        axis_idx = int(axis_values[axis_name])
        base_key = strip_axis_from_label(child_label, axis_name)
        grouped_indexed.setdefault(base_key, {})[axis_idx] = child_value
    return grouped_indexed


def reduce_mean(
    grouped: dict[tuple[Any, ...], list[Any]],
) -> ChildResults:
    """Reduce grouped values by averaging (e.g. Trotter ham samples).

    Example::

        >>> reduce_mean({(('circ', 0),): [1.0, 3.0]})
        {(('circ', 0),): 2.0}
    """
    return {base_key: sum(values) / len(values) for base_key, values in grouped.items()}


def reduce_postprocess_ordered(
    grouped: dict[tuple[Any, ...], dict[int, Any]],
    postprocess_fn: (
        Callable[[list[Any]], Any] | dict[tuple[Any, ...], Callable[[list[Any]], Any]]
    ),
) -> ChildResults:
    """Reduce grouped index->value dicts by sorting by index and calling a postprocess function.

    For each base_key, values are ordered by their integer index and passed
    to the postprocess function. Use a single callable for all keys (e.g. QEM)
    or a dict mapping base_key -> callable for per-spec postprocessing
    (e.g. observable grouping).

    Example::

        >>> grouped = {(('circ', 0),): {0: 10.0, 1: 20.0}}
        >>> reduce_postprocess_ordered(grouped, sum)
        {(('circ', 0),): 30.0}
    """
    reduced: ChildResults = {}
    for base_key, values_by_index in grouped.items():
        ordered = [v for _, v in sorted(values_by_index.items())]
        fn = (
            postprocess_fn[base_key]
            if isinstance(postprocess_fn, dict)
            else postprocess_fn
        )
        reduced[base_key] = fn(ordered)
    return reduced


def reduce_merge_histograms(
    grouped: dict[tuple[Any, ...], list[dict[str, float]]],
) -> ChildResults:
    """Reduce grouped probability dicts by averaging across groups.

    Equivalent to the VQA ``_average_probabilities`` logic: for each base_key,
    collects all probability dicts, unions all bitstrings, and averages the
    probability values. Used by ``TrotterSpecStage`` in measurement pipelines
    to merge probability histograms across Hamiltonian samples.

    Example::

        >>> grouped = {(('circ', 0),): [{'00': 0.6, '11': 0.4}, {'00': 0.8, '11': 0.2}]}
        >>> reduce_merge_histograms(grouped)
        {(('circ', 0),): {'00': 0.7, '11': 0.3}}
    """
    reduced: ChildResults = {}
    for base_key, prob_dicts in grouped.items():
        if not prob_dicts:
            reduced[base_key] = {}
            continue

        all_bitstrings: set[str] = set()
        for probs in prob_dicts:
            all_bitstrings.update(probs.keys())

        n = len(prob_dicts)
        reduced[base_key] = {
            bs: sum(p.get(bs, 0.0) for p in prob_dicts) / n for bs in all_bitstrings
        }
    return reduced
