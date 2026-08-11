# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation: lower MetaCircuit batches to executable QASM payloads."""

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from divi.circuits import MetaCircuit, dag_to_qasm_body
from divi.circuits._payloads import CircuitPayload
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


def _effective_bodies(mc: MetaCircuit) -> tuple:
    """The bodies :func:`_compile_batch` would lower — rendered QASM when
    the binding stage has run, the DAGs otherwise."""
    return mc.qasm_bodies or mc.circuit_bodies or ()


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


def batch_lineage(batch: dict[Any, MetaCircuit]) -> dict[str, BranchKey]:
    """The ``label -> branch key`` map the batch would submit under.

    Enumerates labels without rendering circuits or binding values, for
    callers that need the expansion shape rather than an executable payload.

    Deferred-binding nodes report a single ``param_set`` branch: the count
    lives in ``env.param_sets``, which describes one run rather than the
    pipeline's structure.
    """
    return {
        _branch_label(branch_key): branch_key
        for batch_key, node in batch.items()
        for body_tag, _ in _effective_bodies(node)
        for meas_tag, _ in node.measurement_qasms
        for branch_key, _ in _row_keys(batch_key, body_tag, meas_tag, node, 1)
    }


def _batch_has_free_parameters(batch: dict[Any, MetaCircuit]) -> bool:
    """True when any MetaCircuit still carries free parameters to bind.

    After :class:`~divi.pipeline.stages.ParameterBindingStage`, free symbols
    survive only on the payload path: the bound paths render concrete bodies and
    clear ``parameters``. A non-empty ``parameters`` on any node is the signal
    for ``_default_execute_fn`` to route through :func:`_compile_batch`
    and submit payloads rather than bound circuits.
    """
    return any(node.parameters for node in batch.values())


def _compile_batch(
    batch: dict[Any, MetaCircuit],
    param_sets: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[CircuitPayload], dict[str, BranchKey]]:
    """Lower a MetaCircuit batch to executable QASM payloads.

    Emits one :class:`~divi.circuits.CircuitPayload` per ``(body_tag, meas_tag)``
    variant: the rendered template text with measurement appended.  Bound
    batches fall out as single-row payloads.
    """
    param_array = np.asarray(param_sets, dtype=float)
    if param_array.ndim != 2:
        raise ValueError(
            f"_compile_batch expects 2D param_sets; got shape {param_array.shape}."
        )

    payloads: list[CircuitPayload] = []
    lineage_by_label: dict[str, BranchKey] = {}

    for batch_key, node in batch.items():
        _require_measurements(batch_key, node)

        if not node.circuit_bodies:
            raise ValueError(
                f"MetaCircuit has no circuit_bodies for key '{batch_key}'."
            )
        # Rendered bodies take precedence when the binding stage has run;
        # otherwise serialize the DAGs on demand.
        bodies = node.qasm_bodies or tuple(
            (tag, dag_to_qasm_body(dag, precision=node.precision))
            for tag, dag in node.circuit_bodies
        )

        for (body_tag, body), (meas_tag, meas_qasm) in product(
            bodies, node.measurement_qasms
        ):
            param_set_rows: list[tuple[str, tuple[float, ...]]] = []
            for branch_key, i in _row_keys(
                batch_key, body_tag, meas_tag, node, len(param_array)
            ):
                label = _branch_label(branch_key)
                values = () if i is None else tuple(float(v) for v in param_array[i])
                param_set_rows.append((label, values))
                lineage_by_label[label] = branch_key

            payloads.append(
                CircuitPayload(
                    circuit=_preamble(node.n_qubits) + body + meas_qasm,
                    parameters=node.parameters,
                    parameter_sets=tuple(param_set_rows),
                )
            )

    return payloads, lineage_by_label
