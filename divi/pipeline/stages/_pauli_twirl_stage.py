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
  ``qasm_bodies`` directly so that :func:`_compile_batch` skips its
  per-variant ``dag_to_qasm_body`` pass.

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

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, CZGate, IGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford, Pauli
from qiskit.transpiler.basepasses import TransformationPass

from divi.circuits import MetaCircuit, build_template, dag_to_qasm_body, render_template
from divi.circuits._conversions import _format_bound_param
from divi.pipeline._result_keys_operations import (
    PARAM_SET_AXIS,
    group_by_base_key,
    strip_axis_from_label,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageOutput,
    StageToken,
)

TWIRL_AXIS = "twirl"
_SINGLE_QUBIT_PAULI = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}
_PAULI_CHARS = ("I", "X", "Y", "Z")
_TWO_QUBIT_PAULI_LABELS = tuple(p1 + p0 for p1 in _PAULI_CHARS for p0 in _PAULI_CHARS)
_TWIRL_LABEL_INDICES = tuple(range(len(_TWO_QUBIT_PAULI_LABELS)))


def _strip_sign(label: str) -> str:
    """Remove any global-phase prefix from a Pauli label."""
    for prefix in ("-i", "+i", "-", "+", "i"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _build_twirl_table(gate) -> dict[str, str]:
    """Return ``{pre_label: post_label}`` for every 2-qubit Pauli pre-op."""
    cliff = Clifford(gate)
    return {
        label: _strip_sign(Pauli(label).evolve(cliff).to_label())
        for label in _TWO_QUBIT_PAULI_LABELS
    }


def _build_twirl_sub_dag(gate, pre_label: str, post_label: str) -> DAGCircuit:
    """Build a pre-Pauli · gate · post-Pauli sub-DAG for substitution."""
    qc = QuantumCircuit(2)
    p1, p0 = pre_label[0], pre_label[1]
    if p0 != "I":
        qc.append(_SINGLE_QUBIT_PAULI[p0], [0])
    if p1 != "I":
        qc.append(_SINGLE_QUBIT_PAULI[p1], [1])
    qc.append(gate, [0, 1])
    p1, p0 = post_label[0], post_label[1]
    if p0 != "I":
        qc.append(_SINGLE_QUBIT_PAULI[p0], [0])
    if p1 != "I":
        qc.append(_SINGLE_QUBIT_PAULI[p1], [1])
    return circuit_to_dag(qc)


def _precompute_twirl_dags(gate, twirl_table: dict[str, str]) -> dict[str, DAGCircuit]:
    """Pre-build the substitution sub-DAG for every possible pre-label."""
    return {
        pre: _build_twirl_sub_dag(gate, pre, twirl_table[pre])
        for pre in _TWO_QUBIT_PAULI_LABELS
    }


_CX_TWIRL_TABLE = _build_twirl_table(CXGate())
_CZ_TWIRL_TABLE = _build_twirl_table(CZGate())
_CX_TWIRL_DAGS = _precompute_twirl_dags(CXGate(), _CX_TWIRL_TABLE)
_CZ_TWIRL_DAGS = _precompute_twirl_dags(CZGate(), _CZ_TWIRL_TABLE)
_TWIRL_DAG_TABLES = {"cx": _CX_TWIRL_DAGS, "cz": _CZ_TWIRL_DAGS}
_TWIRL_DAG_ARRAY_TABLES = {
    gate_name: tuple(sub_table[label] for label in _TWO_QUBIT_PAULI_LABELS)
    for gate_name, sub_table in _TWIRL_DAG_TABLES.items()
}


def _twirl_tag(variant_tag: tuple, axis_name: str, twirl_idx: int) -> tuple:
    """Compose the twirl axis suffix onto a variant tag.

    Shared by both real expand paths and :meth:`PauliTwirlStage.dry_expand` —
    the same labelling ensures ``_compile_batch`` and the dry-run counter
    produce consistent keys regardless of which path built the batch.
    """
    return (*variant_tag, (axis_name, twirl_idx))


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
    """Return ``(positions, gate_names)`` for twirl-eligible nodes in op_nodes order."""
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


class PauliTwirlPass(TransformationPass):
    """Insert random Pauli gates around each 2-qubit Clifford in a DAG."""

    def __init__(self, rng: random.Random | None = None):
        super().__init__()
        self._rng = rng or random.Random()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        twirl_specs = [
            (node, _TWIRL_DAG_TABLES[node.op.name])
            for node in dag.op_nodes()
            if node.op.name in _TWIRL_DAG_TABLES
        ]
        if not twirl_specs:
            return dag
        return self._apply(dag, twirl_specs)

    def _apply(
        self,
        dag: DAGCircuit,
        twirl_specs: list,
    ) -> DAGCircuit:
        labels = self._rng.choices(_TWO_QUBIT_PAULI_LABELS, k=len(twirl_specs))
        for (node, sub_table), pre_label in zip(twirl_specs, labels):
            dag.substitute_node_with_dag(node, sub_table[pre_label])
        return dag


class PauliTwirlStage(BundleStage):
    """Fan out each DAG body into Pauli-twirled copies and average on reduce.

    Args:
        n_twirls: Number of randomized copies per circuit body.
        seed: Optional seed for deterministic twirl sampling (useful in tests).
    """

    @property
    def axis_name(self) -> str:
        return TWIRL_AXIS

    @property
    def volatile(self) -> bool:
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
            getattr(s, "axis_name", None) == PARAM_SET_AXIS for s in before
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
    ) -> StageOutput[MetaCircuitBatch]:
        out: MetaCircuitBatch = {}

        for parent_key, meta in batch.items():
            if self._fast_path:
                out[parent_key] = self._expand_fast(meta)
            else:
                out[parent_key] = self._expand_structural(meta)

        return StageOutput(batch=out)

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Analytic path: emit ``n_bodies × n_twirls`` shape-correct placeholders.

        Skips label sampling, topology grouping, deep-copying, and QASM
        rendering entirely — twirling is a purely multiplicative fan-out, so
        the circuit count is exact from ``n_twirls`` alone. Matches the output
        slot (``qasm_bodies`` vs ``circuit_bodies``) of the real path
        so downstream dry stages and ``_compile_batch``-aware counters see the
        right shape.
        """
        out: MetaCircuitBatch = {}
        for parent_key, meta in batch.items():
            out[parent_key] = self._expand_dry(meta)
        return StageOutput(batch=out)

    def _expand_dry(self, meta: MetaCircuit) -> MetaCircuit:
        """Produce placeholder twirled bodies matching the real path's shape."""
        if self._fast_path:
            # Fast path emits pre-rendered QASM strings into qasm_bodies.
            placeholders = tuple(
                (_twirl_tag(variant_tag, self.axis_name, twirl_idx), "")
                for variant_tag, _ in meta.circuit_bodies
                for twirl_idx in range(self._n_twirls)
            )
            return meta.set_qasm_bodies(placeholders)

        # Structural path replaces circuit_bodies with twirled DAGs.  The
        # source DAG is shared across placeholders — nothing in the dry
        # forward pass mutates it.
        placeholders = tuple(
            (_twirl_tag(variant_tag, self.axis_name, twirl_idx), variant_dag)
            for variant_tag, variant_dag in meta.circuit_bodies
            for twirl_idx in range(self._n_twirls)
        )
        return meta.set_circuit_bodies(placeholders)

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
                    updated_bodies.append(
                        (_twirl_tag(variant_tag, self.axis_name, twirl_idx), twirled)
                    )

        return meta.set_circuit_bodies(tuple(updated_bodies))

    def _expand_fast(self, meta: MetaCircuit) -> MetaCircuit:
        """Parametric fast path — emits rendered QASM strings into
        ``qasm_bodies``, bypassing ``dag_to_qasm_body`` for the
        K param_set variants of each upstream body.

        Only fires when ParameterBindingStage is upstream (so variants
        share a gate sequence) and no downstream stage consumes DAGs.
        """
        precision = meta.precision

        # Group bodies by topology — tag with the param_set axis stripped.
        topology_groups: dict[tuple, list[tuple[tuple, DAGCircuit]]] = {}
        group_order: list[tuple] = []
        for tag, dag in meta.circuit_bodies:
            key = strip_axis_from_label(tag, PARAM_SET_AXIS)
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
                formatted = tuple(_format_bound_param(v, precision) for v in values)
                for twirl_idx, unique_idx in enumerate(twirl_to_unique):
                    template = templates_per_unique[unique_idx]
                    bound_bodies.append(
                        (
                            _twirl_tag(variant_tag, self.axis_name, twirl_idx),
                            render_template(template, formatted),
                        )
                    )

        return meta.set_qasm_bodies(tuple(bound_bodies))

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
            elif isinstance(values[0], list):
                # Per-observable list[float] from MeasurementStage —
                # average element-wise.
                n = len(values)
                n_obs = len(values[0])
                reduced[base_key] = [
                    sum(v[i] for v in values) / n for i in range(n_obs)
                ]
            else:
                reduced[base_key] = sum(values) / len(values)
        return reduced
