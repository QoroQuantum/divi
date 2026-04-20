# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QEM transpiler passes: circuit-level transformations for error mitigation.

``GlobalFoldPass``
    ZNE-style global unitary folding (``U → U U† U …``).

``PauliTwirlPass``
    Random Pauli insertion around 2-qubit Cliffords (CX, CZ).
"""

# TODO: Add LocalFoldPass — per-gate folding (G → G G† G) with support for
#       fractional scale factors via partial-layer folding.  This would
#       complement GlobalFoldPass for fine-grained noise scaling on circuits
#       where global folding is too coarse.

import copy
import random

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, IGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford, Pauli
from qiskit.transpiler.basepasses import TransformationPass

__all__ = ["GlobalFoldPass", "PauliTwirlPass"]


# ---------------------------------------------------------------------------
# Pauli twirl tables — precomputed once per module load.
# ---------------------------------------------------------------------------
_SINGLE_QUBIT_PAULI = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}
_PAULI_CHARS = ("I", "X", "Y", "Z")
_TWO_QUBIT_PAULI_LABELS = tuple(
    p1 + p0 for p1 in _PAULI_CHARS for p0 in _PAULI_CHARS
)  # little-endian: label[0]=q1, label[1]=q0


def _strip_sign(label: str) -> str:
    """Remove any leading ``-`` / ``i`` / ``-i`` phase from a Pauli label.

    Pauli twirling preserves the ideal unitary up to a *global* phase
    (``C·P·C†`` can have a ``±`` / ``±i`` prefix).  Global phases do not
    affect measurement outcomes, so we drop them to emit plain gate
    sequences.
    """
    for prefix in ("-i", "+i", "-", "+", "i"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _build_twirl_table(gate) -> dict[str, str]:
    """Return ``{pre_label: post_label}`` for every 2-qubit Pauli pre-operator.

    For a Clifford ``C`` and input Pauli ``P``, the identity
    ``C · P · C† · C = C · P`` means that applying the output Pauli
    ``P' = C P C†`` *after* ``C`` and ``P`` *before* preserves the gate's
    action up to a global phase.  This is the Pauli twirl.
    """
    cliff = Clifford(gate)
    return {
        label: _strip_sign(Pauli(label).evolve(cliff).to_label())
        for label in _TWO_QUBIT_PAULI_LABELS
    }


_CX_TWIRL_TABLE = _build_twirl_table(CXGate())
_CZ_TWIRL_TABLE = _build_twirl_table(CZGate())


def _build_twirl_sub_dag(gate, pre_label: str, post_label: str) -> DAGCircuit:
    """Build a pre-Pauli · gate · post-Pauli sub-DAG for node substitution."""
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
    """Pre-build the sub-DAG for every possible pre-label."""
    return {
        pre: _build_twirl_sub_dag(gate, pre, twirl_table[pre])
        for pre in _TWO_QUBIT_PAULI_LABELS
    }


_CX_TWIRL_DAGS = _precompute_twirl_dags(CXGate(), _CX_TWIRL_TABLE)
_CZ_TWIRL_DAGS = _precompute_twirl_dags(CZGate(), _CZ_TWIRL_TABLE)
_TWIRL_DAG_TABLES = {"cx": _CX_TWIRL_DAGS, "cz": _CZ_TWIRL_DAGS}


# ---------------------------------------------------------------------------
# GlobalFoldPass
# ---------------------------------------------------------------------------
class GlobalFoldPass(TransformationPass):
    """ZNE-style global unitary folding.

    For an odd integer scale factor ``s = 1 + 2k``, the returned DAG is
    ``U · (U† · U)^k`` — i.e. ``k`` forward-and-back cycles appended after
    the original circuit.  Even and fractional scales are rejected
    explicitly; partial folding (fractional) is not yet supported and
    currently unused by the protocol layer.

    Args:
        scale_factor: Odd integer ≥ 1.  ``1.0`` is a pass-through.

    Raises:
        ValueError: If ``scale_factor`` is not an odd integer ≥ 1.
    """

    def __init__(self, scale_factor: float):
        super().__init__()
        if scale_factor < 1.0:
            raise ValueError(
                f"GlobalFoldPass: scale_factor must be >= 1, got {scale_factor}."
            )
        if scale_factor != float(int(scale_factor)) or int(scale_factor) % 2 == 0:
            raise ValueError(
                f"GlobalFoldPass: scale_factor must be an odd integer "
                f"(1, 3, 5, …), got {scale_factor}. Fractional folding is "
                f"not yet implemented."
            )
        self.scale_factor = int(scale_factor)

    @staticmethod
    def _inverse_dag(dag: DAGCircuit) -> DAGCircuit:
        """Build U† by reversing topological order and inverting each gate."""
        inv = dag.copy_empty_like()
        for node in reversed(list(dag.topological_op_nodes())):
            inv.apply_operation_back(node.op.inverse(), node.qargs, node.cargs)
        return inv

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if self.scale_factor == 1:
            return dag

        inv = self._inverse_dag(dag)
        k = (self.scale_factor - 1) // 2
        out = copy.deepcopy(dag)
        for _ in range(k):
            out.compose(inv, inplace=True)
            out.compose(dag, inplace=True)
        return out


# ---------------------------------------------------------------------------
# PauliTwirlPass
# ---------------------------------------------------------------------------
class PauliTwirlPass(TransformationPass):
    """Insert random Pauli gates around each 2-qubit Clifford in the DAG.

    For each ``cx`` / ``cz`` node, samples a 2-qubit Pauli uniformly at
    random, looks up the corresponding post-Pauli from the gate's twirl
    table, and substitutes the node with a sub-DAG containing
    ``pre_pauli · gate · post_pauli``.  The ideal unitary is preserved up
    to a measurement-invariant global phase.

    Each ``run(dag)`` call produces *one* random variant — sample the pass
    ``n_twirls`` times (with a fresh ``PassManager`` per sample) to
    generate the randomised ensemble.

    Args:
        rng: Optional :class:`random.Random` for reproducible sampling.
            When ``None``, uses the global random module (default).
    """

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
        """Apply Pauli twirling to pre-identified ``(node, sub_table)`` pairs.

        Called by :meth:`run` (which discovers nodes itself) or directly
        by :class:`~divi.pipeline.stages.PauliTwirlStage` with pre-cached
        node→sub-table mappings to skip repeated node filtering and
        gate-name lookups.
        """
        labels = self._rng.choices(_TWO_QUBIT_PAULI_LABELS, k=len(twirl_specs))
        for (node, sub_table), pre_label in zip(twirl_specs, labels):
            dag.substitute_node_with_dag(node, sub_table[pre_label])
        return dag
