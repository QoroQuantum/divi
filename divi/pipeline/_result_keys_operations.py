# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Operations on the ``(axis, index)`` keys that tag pipeline results.

Results flow through the pipeline keyed by tuples of ``(axis_name, value)``
pairs. The helpers here parse those keys, group results by axis, reduce the
groups (mean / ordered postprocess / histogram merge), and route backend labels
back to their source branch keys. Stages that fan out along an axis (ham, qem,
obs_group, param_set, ...) and the code that consumes finished results both
build on this shared vocabulary.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from divi.pipeline.abc import BranchKey, ChildResults

#: Key injected into scoped tokens by :func:`_reduce_with_isolated_axes`
#: so that stages can identify which foreign-axis group is being reduced.
FOREIGN_KEY_ATTR = "_foreign_key"

PARAM_SET_AXIS = "param_set"


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


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


def extract_param_set_idx(key: tuple, default: int | None = None) -> int:
    """Extract the param_set index from a pipeline result KEY.

    ``key`` must be a NodeKey — a tuple of ``(axis_name, value)`` pairs, as
    found on the keys of a raw :class:`~divi.pipeline.PipelineResult` (e.g. from
    ``pipeline.run(...)``). It is NOT meant for the output of
    :meth:`~divi.qprog.QuantumProgram.evaluate`, which is already collapsed to a
    ``{param_set_idx: value}`` mapping whose int keys are the indices.

    Raises ``KeyError`` when no ``param_set`` axis is present, unless ``default``
    is given — a parameter-free body (e.g. an empty Fubini-Study prefix on the
    ``|0>`` state) carries no such axis and is the sole set, so callers pass
    ``default=0``.
    """
    if not isinstance(key, tuple):
        raise TypeError(
            f"extract_param_set_idx expects a NodeKey tuple of (axis, value) "
            f"pairs, got {type(key).__name__}. If you have evaluate() output "
            "({param_set_idx: value}), the int key is already the index — index "
            "the dict directly instead of calling this."
        )
    for axis_name, idx in key:
        if axis_name == PARAM_SET_AXIS:
            return idx
    if default is None:
        raise KeyError(
            f"No '{PARAM_SET_AXIS}' axis found in pipeline result key: {key}"
        )
    return default


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


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


def average_by_param_set(
    result: dict[tuple, Any],
    convert: Callable[[Any], np.ndarray],
) -> dict[int, np.ndarray]:
    """Average preserved pipeline results over every axis except ``param_set``."""
    grouped: dict[int, list[np.ndarray]] = {}
    for key, value in result.items():
        idx = extract_param_set_idx(key, default=0)
        grouped.setdefault(idx, []).append(convert(value))
    if not grouped:
        raise RuntimeError("Pipeline returned no results.")
    return {
        idx: np.mean(values, axis=0)
        for idx, values in sorted(grouped.items(), key=lambda item: item[0])
    }


def group_by_branch_and_param_set(
    result: dict[tuple, Any],
    convert: Callable[[Any], np.ndarray],
) -> dict[tuple, dict[int, np.ndarray]]:
    """Group preserved pipeline results by non-param axes, then ``param_set``."""
    grouped: dict[tuple, dict[int, np.ndarray]] = {}
    for key, value in result.items():
        param_idx = extract_param_set_idx(key, default=0)
        branch_key = strip_axis_from_label(key, PARAM_SET_AXIS)
        grouped.setdefault(branch_key, {})[param_idx] = convert(value)
    if not grouped:
        raise RuntimeError("Pipeline returned no results.")
    return grouped


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def reduce_mean(
    grouped: dict[tuple[Any, ...], list[Any]],
) -> ChildResults:
    """Reduce grouped values by averaging (e.g. Trotter ham samples).

    For ``EXPVALS`` results only. Each entry's values may be scalars (the
    standard case — averaged arithmetically) or per-observable lists of equal
    length emitted by a multi-observable
    :class:`~divi.pipeline.stages.MeasurementStage` postprocess (averaged
    element-wise so each observable's mean is preserved). For ``PROBS`` /
    ``COUNTS`` histograms (bitstring→probability dicts) use
    :func:`reduce_merge_histograms` instead.

    Raises:
        TypeError: If a grouped value is a dict (a ``PROBS`` / ``COUNTS``
            histogram), naming :func:`reduce_merge_histograms` as the fix.

    Example::

        >>> reduce_mean({(('circ', 0),): [1.0, 3.0]})
        {(('circ', 0),): 2.0}
        >>> reduce_mean({(('circ', 0),): [[1.0, 5.0], [3.0, 7.0]]})
        {(('circ', 0),): [2.0, 6.0]}
    """
    out: ChildResults = {}
    for base_key, values in grouped.items():
        if values and isinstance(values[0], dict):
            raise TypeError(
                "reduce_mean expects EXPVALS values (a float or a per-observable "
                "list of floats), but got a dict — that is a PROBS/COUNTS "
                "histogram. Use reduce_merge_histograms for probability/counts "
                "results."
            )
        if values and isinstance(values[0], list):
            n = len(values)
            n_obs = len(values[0])
            out[base_key] = [sum(v[i] for v in values) / n for i in range(n_obs)]
        else:
            out[base_key] = sum(values) / len(values)
    return out


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

    For ``PROBS`` / ``COUNTS`` results only. Equivalent to the VQA
    ``_average_probabilities`` logic: for each base_key, collects all
    probability dicts, unions all bitstrings, and averages the probability
    values. Used by ``TrotterSpecStage`` in measurement pipelines to merge
    probability histograms across Hamiltonian samples. For ``EXPVALS`` results
    (floats or per-observable lists) use :func:`reduce_mean` instead.

    Raises:
        TypeError: If a grouped value is not a dict (an ``EXPVALS`` float or
            list), naming :func:`reduce_mean` as the fix.

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

        if not isinstance(prob_dicts[0], dict):
            raise TypeError(
                "reduce_merge_histograms expects PROBS/COUNTS values "
                "(bitstring→probability dicts), but got "
                f"{type(prob_dicts[0]).__name__} — that is an EXPVALS value. "
                "Use reduce_mean for expectation-value results."
            )

        all_bitstrings: set[str] = set()
        for probs in prob_dicts:
            all_bitstrings.update(probs.keys())

        n = len(prob_dicts)
        reduced[base_key] = {
            bs: sum(p.get(bs, 0.0) for p in prob_dicts) / n for bs in all_bitstrings
        }
    return reduced


# ---------------------------------------------------------------------------
# Label -> key routing
# ---------------------------------------------------------------------------


def _collapse_to_parent_results(
    raw_by_label: ChildResults, lineage_by_label: dict[str, BranchKey]
) -> ChildResults:
    """Map backend labels back to structured flat axis keys.

    Example::

        >>> raw_by_label = {'circuit:0': 0.42}
        >>> lineage_by_label = {'circuit:0': (('circuit', 0),)}
        >>> _collapse_to_parent_results(raw_by_label, lineage_by_label)
        {(('circuit', 0),): 0.42}
    """
    regrouped: ChildResults = {}
    for label, value in raw_by_label.items():
        branch_key = lineage_by_label.get(label)
        if branch_key is None:
            continue
        regrouped[branch_key] = value

    return regrouped


def _find_batch_key(branch_key: tuple, batch_keys: set[tuple]) -> tuple:
    """Find the batch key whose axis labels are a subset of *branch_key*."""
    branch_axes = set(branch_key)
    for bk in batch_keys:
        if set(bk).issubset(branch_axes):
            return bk

    raise KeyError(
        f"No batch key matches branch key {branch_key}; "
        f"known batch keys: {batch_keys}"
    )
