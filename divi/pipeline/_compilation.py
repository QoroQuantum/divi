# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from itertools import product
from typing import Any

from divi.circuits import MetaCircuit
from divi.circuits._conversions import dag_to_qasm_body
from divi.pipeline.abc import BranchKey, ChildResults


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
    2. The body string — taken from ``bound_circuit_bodies`` when the
       pipeline ran :class:`ParameterBindingStage`, otherwise serialised
       from the non-parametric DAG on the fly via
       :func:`dag_to_qasm_body`.
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

        # Bound bodies take precedence when populated (parametric path);
        # otherwise serialise the DAGs on demand (non-parametric path).
        if node.bound_circuit_bodies:
            body_items = node.bound_circuit_bodies
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
