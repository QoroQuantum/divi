# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from divi.circuits import MetaCircuit, TemplateEntry, dag_to_qasm_body
from divi.pipeline._result_keys_operations import PARAM_SET_AXIS
from divi.pipeline.abc import BranchKey, ContractViolation


def _preamble(n_qubits: int) -> str:
    """Build the OpenQASM 2.0 header + register declarations."""
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        f"qreg q[{n_qubits}];\n"
        f"creg c[{n_qubits}];\n"
    )


def _branch_label(branch_key: BranchKey) -> str:
    """Flatten a branch key into the label backends and lineage maps key on."""
    return "/".join(f"{ax}:{val}" for ax, val in branch_key)


def _require_measurements(batch_key, node: MetaCircuit) -> None:
    """Reject a node that never passed through ``MeasurementStage``."""
    if not node.measurement_qasms:
        raise ValueError(
            f"MetaCircuit has no measurement_qasms for key '{batch_key}'. "
            "Run MeasurementStage before execution."
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
        _require_measurements(batch_key, node)
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
            label = _branch_label(branch_key)
            circuits[label] = preamble + body_qasm + meas_qasm
            lineage_by_label[label] = branch_key

    return circuits, lineage_by_label


def _effective_bodies(mc: MetaCircuit) -> tuple:
    """The bodies :func:`_compile_batch` would lower — rendered QASM when the
    binding stage has run, the DAGs otherwise."""
    return mc.qasm_bodies or mc.circuit_bodies or ()


def batch_lineage(batch: dict[Any, MetaCircuit]) -> dict[str, BranchKey]:
    """The ``label -> branch key`` map the batch would submit under.

    Enumerates labels without rendering circuits or binding values.

    A node that still carries parameters is bound backend-side, so its
    param-set axis is added here; a bound node already has that axis baked
    into its body tag by
    :class:`~divi.pipeline.stages.ParameterBindingStage`. Deferred nodes
    report a single ``param_set`` branch — the count lives in
    ``env.param_sets``, which describes one run rather than the pipeline's
    structure.
    """
    lineage: dict[str, BranchKey] = {}
    for batch_key, node in batch.items():
        param_axis = ((PARAM_SET_AXIS, 0),) if node.parameters else ()
        for body_tag, _ in _effective_bodies(node):
            for meas_tag, _ in node.measurement_qasms:
                branch_key: BranchKey = (
                    *batch_key,
                    *body_tag,
                    *param_axis,
                    *meas_tag,
                )
                lineage[_branch_label(branch_key)] = branch_key
    return lineage


def reject_colliding_body_tags(stage_name: str, batch: dict[Any, MetaCircuit]) -> None:
    """Raise if a stage left two circuit bodies claiming the same tag.

    :func:`_compile_batch` keys each circuit by its tag, so duplicates overwrite
    each other: the stage believes it produced N variants and one is submitted,
    while reduction reads the survivor as though nothing were missing. No
    configuration wants that, so it is a contract breach rather than a warning.

    Called after every stage's expand, which makes the first stage to collide the
    culprit — its input was checked on the previous pass.
    """
    for key, mc in batch.items():
        bodies = _effective_bodies(mc)
        distinct = len({tag for tag, _ in bodies})
        if distinct == len(bodies):
            continue
        raise ContractViolation(
            f"{stage_name} produced {len(bodies)} circuit bodies sharing "
            f"{distinct} distinct tag(s) for batch key {key!r}. Execution "
            "identifies circuits by tag, so the duplicates would collapse into one "
            "submission and the extra bodies would never run. A fan-out stage must "
            "extend each body's tag, e.g. ``(*parent_tag, (self.axis_name, i))``."
        )


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
        _require_measurements(batch_key, node)
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
                label = _branch_label(branch_key)
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
