# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage that applies Pauli twirling to DAG circuit bodies.

Pauli twirling inserts random Pauli gates around each two-qubit Clifford
gate (CNOT, CZ) so that coherent errors are converted into stochastic
Pauli noise.  The ideal circuit is unchanged up to a measurement-invariant
global phase; only the noise channel is affected.

During *expand*, each DAG body is replaced by ``n_twirls`` randomized
copies.  During *reduce*, the expectation values from all copies are
averaged to produce a single result per original circuit.

Two output paths:

* **Parametric fast path** — when
  :class:`~divi.pipeline.stages.ParameterBindingStage` runs upstream *and*
  no downstream stage reads body DAGs.  Bodies in the same upstream group
  share an identical gate sequence; we parametrise the rotation angles,
  twirl the parametric DAG once per ``twirl_idx``, and render the resulting
  QASM template against each variant's concrete angles — populating
  ``bound_circuit_bodies`` directly so that :func:`_compile_batch` skips
  its per-variant ``dag_to_qasm_body`` pass.

* **Structural path** — used whenever the parametric path's preconditions
  aren't met.  Bodies are grouped by structural tag, sampled
  twirl-label-index vectors are deduplicated and reused across variants,
  and each twirled body is produced via ``deepcopy`` + targeted
  ``substitute_node_with_dag`` on the cx/cz subset using a precomputed
  twirl plan (positions + gate names).  This keeps the hot loop in C-level
  node substitution while avoiding redundant Python dispatch in each
  twirl application.
"""

import copy
import random
from typing import Any

from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit

from divi.circuits import MetaCircuit, build_template, dag_to_qasm_body, render_template
from divi.circuits._qem_passes import (
    _TWIRL_DAG_TABLES,
    _TWO_QUBIT_PAULI_LABELS,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key, strip_axis_from_label

TWIRL_AXIS = "twirl"

# Must match ``ParameterBindingStage.PARAM_SET_AXIS``.  Duplicated here to
# avoid an import cycle (ParameterBinding -> QEM -> PauliTwirl -> ParameterBinding).
_PARAM_SET_AXIS = "param_set"
_TWIRL_LABEL_INDICES = tuple(range(len(_TWO_QUBIT_PAULI_LABELS)))
_TWIRL_DAG_ARRAY_TABLES = {
    gate_name: tuple(sub_table[label] for label in _TWO_QUBIT_PAULI_LABELS)
    for gate_name, sub_table in _TWIRL_DAG_TABLES.items()
}


def _format_param(value: float, precision: int) -> str:
    """Format a numeric parameter for QASM insertion (mirrors ParameterBindingStage)."""
    s = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return "0" if s in {"-0", ""} else s


def _parametrise_rotations(
    dag: DAGCircuit,
) -> tuple[DAGCircuit, tuple[str, ...]]:
    """Return a clone of *dag* with each rotation gate's concrete params
    replaced by fresh :class:`~qiskit.circuit.Parameter`s, plus the
    parameter names in topological-rotation order.

    "Rotation gate" = any op whose ``params`` is non-empty.  After
    ParameterBindingStage's slow path every such param is a concrete
    float; Pauli/Clifford gates carry no params and survive unchanged.
    """
    new_dag = dag.copy_empty_like()
    names: list[str] = []
    for node in dag.topological_op_nodes():
        op = node.op
        if op.params:
            new_params = []
            for _ in op.params:
                p = Parameter(f"__tw_{len(names)}")
                names.append(p.name)
                new_params.append(p)
            new_op = op.copy()
            new_op.params = new_params
            new_dag.apply_operation_back(new_op, node.qargs, node.cargs)
        else:
            new_dag.apply_operation_back(op, node.qargs, node.cargs)
    return new_dag, tuple(names)


def _extract_rotation_values(dag: DAGCircuit) -> tuple[float, ...]:
    """Pull every rotation gate's concrete params off *dag* in topological order."""
    values: list[float] = []
    for node in dag.topological_op_nodes():
        for p in node.op.params:
            values.append(float(p))
    return tuple(values)


def _discover_twirl_plan(
    dag: DAGCircuit,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return ``(positions, gate_names)`` for twirl-eligible nodes in topological order."""
    positions: list[int] = []
    gate_names: list[str] = []
    for i, node in enumerate(dag.op_nodes()):
        name = node.op.name
        if name in _TWIRL_DAG_ARRAY_TABLES:
            positions.append(i)
            gate_names.append(name)
    return tuple(positions), tuple(gate_names)


def _apply_twirl_substitute(
    dag: DAGCircuit,
    label_indices: list[int],
    twirl_positions: tuple[int, ...] | None = None,
    twirl_gate_names: tuple[str, ...] | None = None,
) -> DAGCircuit:
    """Produce a twirled copy of *dag* via ``deepcopy`` + targeted
    ``substitute_node_with_dag`` on each ``cx`` / ``cz``.

    *label_indices* is one sampled index per twirl-eligible gate (aligned to
    :data:`_TWO_QUBIT_PAULI_LABELS` order), in topological order.

    The DAG-level ``deepcopy`` is the Rust-optimised clone path for a
    Rust-backed DAGCircuit; a ``copy_empty_like`` + per-node
    ``apply_operation_back`` rebuild turns out to be strictly slower
    here because the per-call Python→Rust overhead on the 80 % of
    untouched nodes dwarfs what ``deepcopy`` handles in a single
    batched pass.
    """
    if twirl_positions is None or twirl_gate_names is None:
        discovered_positions, discovered_gate_names = _discover_twirl_plan(dag)
        if twirl_positions is None:
            twirl_positions = discovered_positions
        if twirl_gate_names is None:
            twirl_gate_names = discovered_gate_names

    dag_copy = copy.deepcopy(dag)
    op_nodes = list(dag_copy.op_nodes())
    for pos, gate_name, label_idx in zip(
        twirl_positions, twirl_gate_names, label_indices
    ):
        dag_copy.substitute_node_with_dag(
            op_nodes[pos], _TWIRL_DAG_ARRAY_TABLES[gate_name][label_idx]
        )
    return dag_copy


class PauliTwirlStage(BundleStage):
    """Fan out each DAG body into Pauli-twirled copies and average on reduce.

    Args:
        n_twirls: Number of randomized copies per circuit body.
        seed: Optional seed for deterministic twirl sampling (useful in tests).
    """

    @property
    def axis_name(self) -> str | None:
        return TWIRL_AXIS

    @property
    def stateful(self) -> bool:
        return False

    def __init__(self, n_twirls: int = 100, seed: int | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._n_twirls = n_twirls
        self._seed = seed
        # Conservative default — an un-validated stage uses structural path.
        self._fast_path = False

    def validate(self, before: tuple[Stage, ...], after: tuple[Stage, ...]) -> None:
        # Parametric fast path preconditions: (a) nothing downstream
        # reads DAGs, and (b) ParameterBindingStage is upstream so the
        # K topologically-identical variants share a gate sequence.
        no_dag_consumer_after = not any(
            getattr(s, "consumes_dag_bodies", True) for s in after
        )
        param_set_upstream = any(
            getattr(s, "axis_name", None) == _PARAM_SET_AXIS for s in before
        )
        self._fast_path = no_dag_consumer_after and param_set_upstream

    def _rng_for_twirl(self, twirl_idx: int) -> random.Random:
        if self._seed is not None:
            return random.Random(self._seed + twirl_idx)
        return random.Random()

    def _sample_labels(self, twirl_idx: int, n_positions: int) -> list[int]:
        rng = self._rng_for_twirl(twirl_idx)
        return rng.choices(_TWIRL_LABEL_INDICES, k=n_positions)

    def _sample_unique_labels(
        self, n_positions: int
    ) -> tuple[list[list[int]], list[int]]:
        """Sample label-index vectors for all twirls with deduplication.

        Returns:
            Tuple of:
            - ``unique_labels``: distinct label-index vectors.
            - ``twirl_to_unique``: for each ``twirl_idx``, index into
              ``unique_labels``.
        """
        unique_labels: list[list[int]] = []
        twirl_to_unique: list[int] = []
        seen: dict[tuple[int, ...], int] = {}

        for twirl_idx in range(self._n_twirls):
            labels = self._sample_labels(twirl_idx, n_positions)
            key = tuple(labels)
            unique_idx = seen.get(key)
            if unique_idx is None:
                unique_idx = len(unique_labels)
                seen[key] = unique_idx
                unique_labels.append(labels)
            twirl_to_unique.append(unique_idx)

        return unique_labels, twirl_to_unique

    def _apply_twirl_on_reference(
        self,
        dag: DAGCircuit,
        label_indices: list[int],
        twirl_positions: tuple[int, ...] | None = None,
        twirl_gate_names: tuple[str, ...] | None = None,
    ) -> DAGCircuit:
        """One-shot twirl used inside the parametric fast path. Same algo
        as :func:`_apply_twirl_substitute`; kept as a thin wrapper so the
        parametric path can pass in the labels it already sampled."""
        return _apply_twirl_substitute(
            dag,
            label_indices,
            twirl_positions=twirl_positions,
            twirl_gate_names=twirl_gate_names,
        )

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: MetaCircuitBatch = {}

        for parent_key, meta in batch.items():
            if self._fast_path:
                out[parent_key] = self._expand_fast(meta)
            else:
                out[parent_key] = self._expand_structural(meta)

        return ExpansionResult(batch=out), None

    def _group_by_topology(
        self, meta: MetaCircuit
    ) -> tuple[list[tuple], dict[tuple, list[tuple[tuple, DAGCircuit]]]]:
        """Partition ``meta.circuit_bodies`` by tag with stable group ordering."""
        groups: dict[tuple, list[tuple[tuple, DAGCircuit]]] = {}
        order: list[tuple] = []
        for tag, dag in meta.circuit_bodies:
            key = tag
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((tag, dag))
        return order, groups

    def _expand_structural(self, meta: MetaCircuit) -> MetaCircuit:
        """Group bodies by topology, sample labels once per
        ``(group, twirl_idx)``, and twirl each variant via
        ``deepcopy`` + substitution with precomputed twirl plans.
        """
        order, groups = self._group_by_topology(meta)

        updated_bodies: list[tuple] = []
        for group_key in order:
            variants = groups[group_key]
            _, ref_dag = variants[0]

            ref_twirl_positions, _ = _discover_twirl_plan(ref_dag)
            n_twirl_gates = len(ref_twirl_positions)

            unique_labels, twirl_to_unique = self._sample_unique_labels(n_twirl_gates)

            for variant_tag, variant_dag in variants:
                variant_twirl_positions, variant_twirl_gate_names = (
                    _discover_twirl_plan(variant_dag)
                )
                twirled_by_unique = [
                    _apply_twirl_substitute(
                        variant_dag,
                        labels,
                        twirl_positions=variant_twirl_positions,
                        twirl_gate_names=variant_twirl_gate_names,
                    )
                    for labels in unique_labels
                ]
                for twirl_idx, unique_idx in enumerate(twirl_to_unique):
                    twirled = twirled_by_unique[unique_idx]
                    twirl_tag = (*variant_tag, (self.axis_name, twirl_idx))
                    updated_bodies.append((twirl_tag, twirled))

        return meta.set_circuit_bodies(tuple(updated_bodies))

    def _expand_fast(self, meta: MetaCircuit) -> MetaCircuit:
        """Parametric fast path — emits rendered QASM strings into
        ``bound_circuit_bodies``, bypassing ``dag_to_qasm_body`` for the
        K param_set variants of each upstream body.

        Only fires when ParameterBindingStage is upstream (so variants
        share a gate sequence) and no downstream stage consumes DAGs.
        """
        precision = meta.precision

        # Group bodies by topology — tag with the param_set axis stripped.
        topology_groups: dict[tuple, list[tuple[tuple, DAGCircuit]]] = {}
        group_order: list[tuple] = []
        for tag, dag in meta.circuit_bodies:
            key = strip_axis_from_label(tag, _PARAM_SET_AXIS)
            if key not in topology_groups:
                topology_groups[key] = []
                group_order.append(key)
            topology_groups[key].append((tag, dag))

        bound_bodies: list[tuple[tuple, str]] = []

        for topology_key in group_order:
            variants = topology_groups[topology_key]
            _, ref_dag = variants[0]

            parametric_dag, param_names = _parametrise_rotations(ref_dag)
            twirl_positions, twirl_gate_names = _discover_twirl_plan(parametric_dag)
            n_twirl_gates = len(twirl_positions)

            # One template per unique label vector, shared across all K variants.
            unique_labels, twirl_to_unique = self._sample_unique_labels(n_twirl_gates)
            templates_per_unique: list = []
            for labels in unique_labels:
                twirled_parametric = self._apply_twirl_on_reference(
                    parametric_dag,
                    labels,
                    twirl_positions=twirl_positions,
                    twirl_gate_names=twirl_gate_names,
                )
                twirled_qasm = dag_to_qasm_body(twirled_parametric, precision=precision)
                templates_per_unique.append(build_template(twirled_qasm, param_names))

            # Render each variant × each twirl from the cached templates.
            for variant_tag, variant_dag in variants:
                values = _extract_rotation_values(variant_dag)
                if len(values) != len(param_names):
                    raise RuntimeError(
                        f"PauliTwirlStage topology cache: reference and variant "
                        f"disagree on rotation count ({len(param_names)} vs "
                        f"{len(values)}). Tag: {variant_tag!r}."
                    )
                formatted = tuple(_format_param(v, precision) for v in values)
                for twirl_idx, unique_idx in enumerate(twirl_to_unique):
                    template = templates_per_unique[unique_idx]
                    bound_tag = (*variant_tag, (self.axis_name, twirl_idx))
                    bound_bodies.append(
                        (bound_tag, render_template(template, formatted))
                    )

        return meta.set_bound_bodies(tuple(bound_bodies))

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        return {
            "n_twirls": self._n_twirls,
            "fast_path": self._fast_path,
        }

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        reduced: ChildResults = {}
        for base_key, values in grouped.items():
            if isinstance(values[0], dict):
                # Per-obs expval dicts — average each observable independently.
                obs_keys = values[0].keys()
                reduced[base_key] = {
                    k: sum(v[k] for v in values) / len(values) for k in obs_keys
                }
            else:
                reduced[base_key] = sum(values) / len(values)
        return reduced
