# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qiskit.circuit import Parameter

from divi.circuits import CircuitPayload, MetaCircuit, dag_to_qasm_body
from divi.circuits._conversions import _real_float_array
from divi.circuits._qasm_template import _template_cached
from divi.pipeline._result_keys_operations import (
    OBS_GROUP_AXIS,
    PARAM_SET_AXIS,
    _find_batch_key,
)
from divi.pipeline.abc import BranchKey, ContractViolation


def _build_preamble(n_qubits: int) -> str:
    """Build the OpenQASM 2.0 header + register declarations."""
    return (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        f"qreg q[{n_qubits}];\n"
        f"creg c[{n_qubits}];\n"
    )


def _format_branch_label(branch_key: BranchKey) -> str:
    """Flatten a branch key into the label backends and lineage maps key on."""
    return "/".join(f"{ax}:{val}" for ax, val in branch_key)


def _require_measurements(batch_key, node: MetaCircuit) -> None:
    """Reject a node that never passed through ``MeasurementStage``."""
    if not node.measurement_qasms:
        raise ValueError(
            f"MetaCircuit has no measurement_qasms for key '{batch_key}'. "
            "Run MeasurementStage before execution."
        )


def _effective_bodies(mc: MetaCircuit) -> tuple:
    """The bodies :func:`_compile_batch` would lower — rendered QASM when
    the binding stage has run, the DAGs otherwise."""
    return mc.qasm_bodies or mc.circuit_bodies or ()


def reject_colliding_body_tags(
    stage_name: str, batch: Mapping[Any, MetaCircuit]
) -> None:
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


def _row_keys(
    batch_key, body_tag, meas_tag, node: MetaCircuit, n_param_sets: int
) -> list[tuple[tuple, int | None]]:
    """The ``(branch_key, param_index)`` rows one body/measurement pair emits."""
    if node.parameters:
        return [
            ((*batch_key, *body_tag, (PARAM_SET_AXIS, i), *meas_tag), i)
            for i in range(n_param_sets)
        ]
    # Bound bodies already carry the param-set axis in body_tag.
    return [((*batch_key, *body_tag, *meas_tag), None)]


def _has_param_group_allocation(
    node: MetaCircuit, body_tag: tuple, meas_tag: tuple
) -> bool:
    """Whether a bound parameter/group branch received estimator samples."""
    allocation = node.param_group_shots
    if allocation is None:
        return True

    param_idx = next(
        (value for axis, value in body_tag if axis == PARAM_SET_AXIS), None
    )
    group_idx = next(
        (value for axis, value in meas_tag if axis == OBS_GROUP_AXIS), None
    )
    if param_idx is None or group_idx is None:
        raise ValueError(
            "Per-parameter-set group shots require bound param_set body tags and "
            "obs_group measurement tags."
        )
    return allocation.get(param_idx, {}).get(group_idx, 0) > 0


def batch_lineage(
    batch: Mapping[Any, MetaCircuit], n_param_sets: int = 1
) -> dict[str, BranchKey]:
    """The ``label -> branch key`` map the batch would submit under.

    Enumerates labels without rendering circuits or binding values, for
    callers that need the expansion shape rather than an executable payload.

    Deferred-binding nodes report ``n_param_sets`` ``param_set`` branches: the
    count belongs to a run (``env.param_sets``), not to the pipeline.
    """
    return {
        _format_branch_label(branch_key): branch_key
        for batch_key, node in batch.items()
        for body_tag, _ in _effective_bodies(node)
        for meas_tag, _ in node.measurement_qasms
        for branch_key, _ in _row_keys(
            batch_key, body_tag, meas_tag, node, n_param_sets
        )
    }


def _batch_has_free_parameters(batch: Mapping[Any, MetaCircuit]) -> bool:
    """True when any MetaCircuit still carries free parameters to bind.

    After :class:`~divi.pipeline.stages.ParameterBindingStage`, free symbols
    survive only on the payload path: the bound paths render concrete bodies and
    clear ``parameters``. A non-empty ``parameters`` on any node is the signal
    for ``_default_execute_fn`` to route through :func:`_compile_batch`
    and submit payloads rather than bound circuits.
    """
    return any(node.parameters for node in batch.values())


def _select_present_columns(parameters: Sequence[Parameter], body: str) -> list[int]:
    """Return the indices of the ``parameters`` that ``body`` references."""
    if not parameters:
        return []
    names = tuple(parameter.name for parameter in parameters)
    return sorted(set(_template_cached(body, names).slot_indices))


def _insert_unmeasured_rows(
    raw: Mapping[BranchKey, Any],
    batch: Mapping[Any, MetaCircuit],
    n_param_sets: int,
) -> dict[BranchKey, Any]:
    """Return ``raw`` plus a ``0.0`` row per branch of each unmeasured node.

    Every node's rows keep ``batch`` order.
    """
    rows_by_node: dict[Any, dict[BranchKey, Any]] = {
        key: (
            dict.fromkeys(batch_lineage({key: node}, n_param_sets).values(), 0.0)
            if node.is_unmeasured
            else {}
        )
        for key, node in batch.items()
    }
    for branch_key, value in raw.items():
        rows_by_node[_find_batch_key(branch_key, batch.keys())][branch_key] = value
    return {key: value for rows in rows_by_node.values() for key, value in rows.items()}


def _compile_batch(
    batch: Mapping[Any, MetaCircuit],
    param_sets: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[CircuitPayload], dict[str, BranchKey]]:
    """Lower a MetaCircuit batch to executable QASM payloads.

    Emits one :class:`~divi.circuits.CircuitPayload` per ``(body_tag, meas_tag)``
    variant: the rendered template text with measurement appended.  Bound
    batches fall out as single-row payloads.
    """
    param_array = _real_float_array(param_sets, source="param_sets")
    if param_array.ndim != 2:
        raise ValueError(
            f"_compile_batch expects 2D param_sets; got shape {param_array.shape}."
        )

    payloads: list[CircuitPayload] = []
    lineage_by_label: dict[str, BranchKey] = {}

    for batch_key, node in batch.items():
        _require_measurements(batch_key, node)

        # Rendered bodies take precedence when the binding stage has run;
        # otherwise serialize the DAGs on demand.
        bodies = node.qasm_bodies or tuple(
            (tag, dag_to_qasm_body(dag, precision=node.precision))
            for tag, dag in node.circuit_bodies
        )

        for body_tag, body in bodies:
            columns = _select_present_columns(node.parameters, body)
            parameters = tuple(node.parameters[i] for i in columns)
            present_values = param_array[:, columns]
            for meas_tag, meas_qasm in node.measurement_qasms:
                if not _has_param_group_allocation(node, body_tag, meas_tag):
                    continue
                param_set_rows: list[tuple[str, tuple[float, ...]]] = []
                for branch_key, i in _row_keys(
                    batch_key, body_tag, meas_tag, node, len(param_array)
                ):
                    label = _format_branch_label(branch_key)
                    values = () if i is None else tuple(present_values[i].tolist())
                    param_set_rows.append((label, values))
                    lineage_by_label[label] = branch_key

                payloads.append(
                    CircuitPayload(
                        circuit=_build_preamble(node.n_qubits) + body + meas_qasm,
                        parameters=parameters,
                        parameter_sets=tuple(param_set_rows),
                    )
                )

    return payloads, lineage_by_label
