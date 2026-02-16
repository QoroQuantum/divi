# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from itertools import product
from typing import Any

from divi.circuits import MetaCircuit
from divi.pipeline.abc import BranchKey, ChildResults


def _compile_batch(
    batch: dict[Any, MetaCircuit],
) -> tuple[dict[str, str], dict[str, BranchKey]]:
    """Lower MetaCircuits into executable QASM labels and payloads.

    Batch keys are ``NodeKey`` tuples of ``(axis_name, value)`` pairs,
    guaranteed by the stage contracts.  Each MetaCircuit's body × measurement
    Cartesian product becomes one executable circuit keyed by a flat
    ``BranchKey``.

    Example::

    Example — a MetaCircuit with 2 QEM body variants and 2 measurement
    groups produces 4 executable circuits::

        >>> # meta.circuit_body_qasms = ((('qem', 0), body0), (('qem', 1), body1))
        >>> # meta.measurement_qasms  = ((('meas', 0), mZ),   (('meas', 1), mX))
        >>> batch = {(('circuit', 0),): meta}
        >>> circuits, lineage = _compile_batch(batch)
        >>> sorted(circuits.keys())
        ['circuit:0/qem:0/meas:0', 'circuit:0/qem:0/meas:1',
         'circuit:0/qem:1/meas:0', 'circuit:0/qem:1/meas:1']
    """
    circuits: dict[str, str] = {}
    lineage_by_label: dict[str, BranchKey] = {}

    for batch_key, node in batch.items():
        if not getattr(node, "measurement_qasms", None):
            raise ValueError(
                f"MetaCircuit has no measurement_qasms for key '{batch_key}'. "
                "Run MeasurementStage before execution."
            )

        for (body_tag, body_qasm), (meas_tag, meas_qasm) in product(
            node.circuit_body_qasms, node.measurement_qasms
        ):
            branch_key: BranchKey = (*batch_key, *body_tag, *meas_tag)
            label = "/".join(f"{ax}:{val}" for ax, val in branch_key)
            circuits[label] = body_qasm + meas_qasm
            lineage_by_label[label] = branch_key

    return circuits, lineage_by_label


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
