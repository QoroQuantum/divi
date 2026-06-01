# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from divi.circuits import MetaCircuit, TemplateEntry, dag_to_qasm_body
from divi.pipeline.abc import BranchKey, ChildResults

PARAM_SET_AXIS = "param_set"


def _preamble(n_qubits: int) -> str:
    """Build the OpenQASM 2.0 header + register declarations."""
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        f"qreg q[{n_qubits}];\n"
        f"creg c[{n_qubits}];\n"
    )


def _compile_batch(
    batch: dict[Any, MetaCircuit],
) -> tuple[dict[str, str], dict[str, BranchKey]]:
    """Lower MetaCircuits into executable QASM labels and payloads.

    Each payload is assembled by concatenating three pieces:

    1. The QASM preamble (``OPENQASM 2.0``, ``qelib1.inc``, ``qreg``,
       ``creg``), computed from the DAG's qubit count.
    2. The body string — taken from ``qasm_bodies`` when the pipeline ran
       :class:`ParameterBindingStage`, otherwise serialised from the
       non-parametric DAG on the fly via :func:`dag_to_qasm_body`.
    3. The measurement QASM from ``measurement_qasms``.

    Each MetaCircuit's ``bodies × measurements`` Cartesian product becomes
    one executable circuit keyed by a flat ``BranchKey``.
    """
    circuits: dict[str, str] = {}
    lineage_by_label: dict[str, BranchKey] = {}

    for batch_key, node in batch.items():
        if not node.measurement_qasms:
            raise ValueError(
                f"MetaCircuit has no measurement_qasms for key '{batch_key}'. "
                "Run MeasurementStage before execution."
            )
        if not node.circuit_bodies:
            raise ValueError(
                f"MetaCircuit has no circuit_bodies for key '{batch_key}'."
            )

        # All body variants share qubit layout (QEM/twirl only add gates,
        # they don't alter the register).  Use the first DAG for the preamble.
        preamble = _preamble(node.n_qubits)

        # Rendered bodies take precedence when populated (binding-stage output);
        # otherwise serialise the DAGs on demand (non-parametric path).
        if node.qasm_bodies:
            body_items = node.qasm_bodies
        else:
            body_items = tuple(
                (tag, dag_to_qasm_body(dag, precision=node.precision))
                for tag, dag in node.circuit_bodies
            )

        for (body_tag, body_qasm), (meas_tag, meas_qasm) in product(
            body_items, node.measurement_qasms
        ):
            branch_key: BranchKey = (*batch_key, *body_tag, *meas_tag)
            label = "/".join(f"{ax}:{val}" for ax, val in branch_key)
            circuits[label] = preamble + body_qasm + meas_qasm
            lineage_by_label[label] = branch_key

    return circuits, lineage_by_label


def _batch_has_templates(batch: dict[Any, MetaCircuit]) -> bool:
    """True when any MetaCircuit still carries free parameters to bind.

    After :class:`~divi.pipeline.stages.ParameterBindingStage`, a body is a
    backend template iff free symbols remain: its fast path renders fully bound
    bodies and clears ``parameters``; its template path leaves the placeholders
    in and keeps ``parameters``. A non-empty ``parameters`` on any node is the
    signal for ``_default_execute_fn`` to route through
    :func:`_compile_template_batch` and submit a ``list[TemplateEntry]`` via the
    backend's template-aware path.
    """
    return any(node.parameters for node in batch.values())


def _compile_template_batch(
    batch: dict[Any, MetaCircuit],
    param_sets: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[TemplateEntry], dict[str, BranchKey]]:
    """Lower a templated MetaCircuit batch to a list of TemplateEntry payloads.

    Mirrors :func:`_compile_batch` but produces one
    :class:`~divi.circuits.TemplateEntry` per ``(body_tag, meas_tag)``
    variant, sharing the per-row ``parameter_sets`` array across all
    entries. The label of each parameter set matches the deterministic
    ``BranchKey``-derived label that :func:`_compile_batch` would emit for
    the equivalent bound circuit, so :func:`_collapse_to_parent_results`
    routes results identically regardless of which compile path ran.
    """
    param_array = np.asarray(param_sets, dtype=float)
    if param_array.ndim != 2:
        raise ValueError(
            "_compile_template_batch expects 2D param_sets; got shape "
            f"{param_array.shape}."
        )

    entries: list[TemplateEntry] = []
    lineage_by_label: dict[str, BranchKey] = {}

    for batch_key, node in batch.items():
        if not node.measurement_qasms:
            raise ValueError(
                f"MetaCircuit has no measurement_qasms for key '{batch_key}'. "
                "Run MeasurementStage before execution."
            )
        if not node.parameters or not node.qasm_bodies:
            raise ValueError(
                f"MetaCircuit for key '{batch_key}' is not a template: expected "
                "ParameterBindingStage's template path to leave free parameters "
                "and populate qasm_bodies."
            )

        preamble = _preamble(node.n_qubits)
        param_names = tuple(p.name for p in node.parameters)

        for (body_tag, body_qasm), (meas_tag, meas_qasm) in product(
            node.qasm_bodies, node.measurement_qasms
        ):
            template_qasm = preamble + body_qasm + meas_qasm
            param_set_rows: list[tuple[str, tuple[float, ...]]] = []
            for i, values in enumerate(param_array):
                param_set_tag = (PARAM_SET_AXIS, i)
                branch_key: BranchKey = (
                    *batch_key,
                    *body_tag,
                    param_set_tag,
                    *meas_tag,
                )
                label = "/".join(f"{ax}:{val}" for ax, val in branch_key)
                param_set_rows.append((label, tuple(float(v) for v in values)))
                lineage_by_label[label] = branch_key

            entries.append(
                TemplateEntry(
                    template_qasm=template_qasm,
                    parameter_names=param_names,
                    parameter_sets=tuple(param_set_rows),
                )
            )

    return entries, lineage_by_label


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
