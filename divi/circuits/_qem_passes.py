# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QEM transpiler passes: circuit-level transformations for error mitigation.

``GlobalFoldPass``
    Global unitary folding (``U · (U†·U)^k · L†·L``).  Deterministic —
    same ``(circuit, scale_factor)`` always yields the same result.

``LocalFoldPass``
    Per-gate folding (``G · (G†·G)^k``) with fractional scale-factor
    support via partial-layer folding.  Default selection is random at
    fractional scales; pass ``rng`` or ``selection="from_left"`` /
    ``"from_right"`` for deterministic output.

``PauliTwirlPass``
    Random Pauli insertion around 2-qubit Cliffords (CX, CZ).

All three are Qiskit :class:`~qiskit.transpiler.basepasses.TransformationPass`
subclasses and mutate their input DAG in place.  Classes live in this
private module but are re-exported from the public :mod:`divi.circuits.qem`;
prefer the public import path::

    from qiskit.transpiler import PassManager
    from divi.circuits.qem import GlobalFoldPass

    folded = PassManager([GlobalFoldPass(3.0)]).run(circuit)

For Zero-Noise Extrapolation, prefer the higher-level :func:`global_fold`
/ :func:`local_fold` helpers exposed from :mod:`divi.circuits.qem` — they
also compute the *effective* scale factor realised by the pass, which
the :class:`~divi.circuits.qem.ZNE` protocol forwards to the extrapolator.
"""

import random
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, IGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford, Pauli
from qiskit.transpiler.basepasses import TransformationPass

__all__ = ["GlobalFoldPass", "LocalFoldPass", "PauliTwirlPass"]


_NON_UNITARY_OP_NAMES = frozenset(("measure", "reset", "barrier"))


def _count_foldable_gates(
    dag: DAGCircuit,
    exclude_names: frozenset[str] = frozenset(),
    exclude_arities: frozenset[int] = frozenset(),
) -> int:
    """Count unitary gates eligible for folding (respecting exclude filters)."""
    return sum(
        1
        for node in dag.op_nodes()
        if node.op.name not in _NON_UNITARY_OP_NAMES
        and node.op.name not in exclude_names
        and len(node.qargs) not in exclude_arities
    )


def _compute_fold_plan(d: int, scale_factor: float) -> tuple[int, int]:
    """Return ``(k, n)`` — base folds applied to every gate and the number
    of gates receiving one extra fold, for a pool of ``d`` foldable gates
    at a given ``scale_factor``.  See class docstrings for the arithmetic.

    Python's :func:`round` uses banker's rounding (round-half-to-even),
    which is asymmetric relative to the user's request: for ``d=2``,
    ``s=1.5`` yields ``n=0`` (rounds down) while ``s=2.5`` yields ``n=2``
    (rounds up).  The effective scale factor is therefore not always
    equal to the requested ``scale_factor`` — callers that care (e.g.
    ZNE extrapolation) should consult ``_compute_effective_scale``.
    """
    if d == 0 or scale_factor == 1.0:
        return 0, 0
    k = int((scale_factor - 1) // 2)
    remainder = scale_factor - (1 + 2 * k)
    n = max(0, min(d, round(remainder * d / 2)))
    return k, n


def _compute_effective_scale(d: int, scale_factor: float) -> float:
    """Effective scale factor actually realised by folding ``d`` gates.

    Because the achievable scales form a discrete grid of granularity
    ``2/d``, a requested non-integer ``scale_factor`` may snap to a
    different value.  Returns ``1.0`` for ``d == 0`` (nothing to fold).
    """
    if d == 0:
        return 1.0
    k, n = _compute_fold_plan(d, scale_factor)
    return 1.0 + 2 * k + 2 * n / d


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
    """Global unitary folding with fractional scale-factor support.

    For a target scale factor ``s`` on a circuit of ``d`` unitary gates::

        k         = (s - 1) // 2             # full forward-and-back folds
        remainder = s - (1 + 2k)
        n         = round(remainder · d / 2) # gates folded at the tail

    The returned DAG is ``U · (U† · U)^k · L† · L`` where ``L`` is the
    sub-circuit of the last ``n`` unitary gates of ``U``.  Post-transform
    gate count is ``d · (1 + 2k) + 2n`` and effective scale factor is
    ``1 + 2k + 2n/d``.  Non-unitary instructions (``measure``, ``reset``,
    ``barrier``) are ignored when counting ``d`` and selecting the tail.

    Example — for a 4-gate circuit::

        requested s | k | n | size | effective s
        ----------------------------------------
        1.5         | 0 | 1 |  6   | 1.5
        2.0         | 0 | 2 |  8   | 2.0
        3.0         | 1 | 0 | 12   | 3.0
        3.5         | 1 | 1 | 14   | 3.5

    The achievable scales form a grid of granularity ``2/d`` — for small
    ``d`` the effective scale may differ from the request.  Use
    :meth:`effective_scale` to query the realised value.

    Mirrors Mitiq's ``fold_global`` behavior.  Deterministic: the same
    ``(circuit, scale_factor)`` always produces the same folded circuit.

    Note: this pass folds the full unitary and has no per-gate exclude
    mechanism — use :class:`LocalFoldPass` with ``exclude={"cx", ...}``
    if you need to protect specific gates from folding.

    Args:
        scale_factor: Real number ≥ 1.  ``1.0`` is a pass-through.

    Raises:
        ValueError: If ``scale_factor`` < 1.
    """

    def __init__(self, scale_factor: float):
        super().__init__()
        if scale_factor < 1.0:
            raise ValueError(
                f"GlobalFoldPass: scale_factor must be >= 1, got {scale_factor}."
            )
        self.scale_factor = float(scale_factor)

    def effective_scale(self, dag: DAGCircuit) -> float:
        """Scale factor actually realised on ``dag`` (may differ from the
        requested value when the gate count is too small for the fractional
        part to round cleanly — see ``_compute_effective_scale``)."""
        return _compute_effective_scale(_count_foldable_gates(dag), self.scale_factor)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Fold ``dag`` in place and return the mutated DAG.

        The input is consumed — callers that need the original should
        deepcopy before invoking the pass.  Matches the standard Qiskit
        :class:`~qiskit.transpiler.basepasses.TransformationPass` contract.

        Returns:
            The same ``dag`` object, mutated to contain the folded circuit.
        """
        if self.scale_factor == 1.0:
            return dag

        # Snapshot every op *before* mutating — forward appends will
        # otherwise leak into iteration and double-fold.
        all_ops = [
            (node.op, node.qargs, node.cargs) for node in dag.topological_op_nodes()
        ]
        unitary_ops = [
            entry for entry in all_ops if entry[0].name not in _NON_UNITARY_OP_NAMES
        ]
        d = len(unitary_ops)
        k, n = _compute_fold_plan(d, self.scale_factor)
        if k == 0 and n == 0:
            return dag

        # U† = reverse topological order of the unitary ops, each inverted.
        inv_ops = [
            (op.inverse(), qargs, cargs) for op, qargs, cargs in reversed(unitary_ops)
        ]

        for _ in range(k):
            for op, qargs, cargs in inv_ops:
                dag.apply_operation_back(op, qargs, cargs)
            for op, qargs, cargs in all_ops:
                dag.apply_operation_back(op, qargs, cargs)

        if n > 0:
            tail = unitary_ops[-n:]
            tail_inv = [
                (op.inverse(), qargs, cargs) for op, qargs, cargs in reversed(tail)
            ]
            for op, qargs, cargs in tail_inv:
                dag.apply_operation_back(op, qargs, cargs)
            for op, qargs, cargs in tail:
                dag.apply_operation_back(op, qargs, cargs)

        return dag


# ---------------------------------------------------------------------------
# LocalFoldPass
# ---------------------------------------------------------------------------
class LocalFoldPass(TransformationPass):
    """Per-gate folding with fractional scale-factor support.

    Each unitary gate ``G`` is replaced by ``G · (G† · G)^k``.  For a
    target scale factor ``s`` on a circuit with ``d`` unitary gates::

        k         = (s - 1) // 2             # base folds applied to every gate
        remainder = s - (1 + 2k)
        n         = round(remainder · d / 2) # gates receiving one extra fold

    ``n`` gates are then selected for the extra fold according to
    ``selection``, yielding a post-transform gate count of
    ``d · (1 + 2k) + 2n`` and an effective scale factor of
    ``1 + 2k + 2n/d``.  Non-unitary instructions (``measure``, ``reset``,
    ``barrier``) are skipped.  Mirrors Mitiq's ``fold_gates_from_left`` /
    ``_from_right`` / ``_at_random`` selection strategies.

    Excluded gates (see ``exclude``) are removed from the candidate pool
    *before* the ``k`` / ``n`` arithmetic.  A gate is excluded if its op
    name matches any entry in ``exclude`` **or** its arity matches one
    of the shorthands (``"single"``, ``"double"``, ``"triple"``).

    The requested ``scale_factor`` therefore applies to the foldable
    subset only: excluded gates appear once in the output and the
    circuit-wide noise scale is effectively lower than requested.  This
    matches Mitiq's ``fold_all`` ``exclude`` semantics.

    The achievable scales form a grid of granularity ``2/d`` — for small
    ``d`` the effective scale may differ from the request.  Use
    :meth:`effective_scale` to query the realised value.  Example, for a
    4-gate circuit::

        requested s | k | n | size | effective s
        ----------------------------------------
        1.5         | 0 | 1 |  6   | 1.5
        2.0         | 0 | 2 |  8   | 2.0
        3.0         | 1 | 0 | 12   | 3.0
        3.5         | 1 | 1 | 14   | 3.5

    Complements :class:`GlobalFoldPass` for fine-grained noise scaling on
    deep circuits where global folding is too coarse.  Unlike
    :class:`GlobalFoldPass`, this pass introduces randomness at
    fractional scales (which ``n`` gates receive the extra fold) — pass
    a fixed ``rng`` or use ``selection="from_left"`` / ``"from_right"``
    for deterministic output.

    Args:
        scale_factor: Real number ≥ 1.  ``1.0`` is a pass-through.
        selection: Which ``n`` gates receive the extra fold:
            ``"random"`` (default) — uniformly sampled without replacement;
            ``"from_left"`` — the first ``n`` gates in topological order;
            ``"from_right"`` — the last ``n`` gates in topological order.
        exclude: Optional set of op names (``"h"``, ``"cx"``, …) and/or
            arity shorthands (``"single"``, ``"double"``, ``"triple"``)
            to skip during folding.  Unknown names match nothing (harmless).
        rng: Optional :class:`random.Random` for reproducible selection
            when ``selection="random"``.  Ignored otherwise.  When
            ``None``, uses a fresh :class:`random.Random`.

    Raises:
        ValueError: If ``scale_factor`` < 1 or ``selection`` is unknown.
    """

    _VALID_SELECTIONS = ("random", "from_left", "from_right")
    _ARITY_SHORTHANDS = {"single": 1, "double": 2, "triple": 3}

    def __init__(
        self,
        scale_factor: float,
        selection: Literal["random", "from_left", "from_right"] = "random",
        exclude: set[str] | None = None,
        rng: random.Random | None = None,
    ):
        super().__init__()
        if scale_factor < 1.0:
            raise ValueError(
                f"LocalFoldPass: scale_factor must be >= 1, got {scale_factor}."
            )
        if selection not in self._VALID_SELECTIONS:
            raise ValueError(
                f"LocalFoldPass: selection must be one of "
                f"{self._VALID_SELECTIONS}, got {selection!r}."
            )
        self.scale_factor = float(scale_factor)
        self.selection = selection
        self._rng = rng or random.Random()

        exclude = set(exclude) if exclude else set()
        self._exclude_arities = frozenset(
            self._ARITY_SHORTHANDS[e] for e in exclude if e in self._ARITY_SHORTHANDS
        )
        self._exclude_names = frozenset(
            e for e in exclude if e not in self._ARITY_SHORTHANDS
        )

    def _pick_extra_indices(self, d: int, n: int) -> set[int]:
        """Choose the ``n`` gate indices (in topological order) that receive
        one extra fold, according to ``self.selection``."""
        if n <= 0:
            return set()
        if self.selection == "from_left":
            return set(range(n))
        if self.selection == "from_right":
            return set(range(d - n, d))
        return set(self._rng.sample(range(d), n))

    @staticmethod
    def _folded_sub_dag(node, num_folds: int) -> DAGCircuit:
        """Build ``G · (G† · G)^num_folds`` as a sub-DAG for node substitution."""
        n_qubits = len(node.qargs)
        qc = QuantumCircuit(n_qubits)
        qargs = list(range(n_qubits))
        qc.append(node.op, qargs)
        inv_op = node.op.inverse()
        for _ in range(num_folds):
            qc.append(inv_op, qargs)
            qc.append(node.op, qargs)
        return circuit_to_dag(qc)

    def _is_foldable(self, node) -> bool:
        if node.op.name in _NON_UNITARY_OP_NAMES:
            return False
        if node.op.name in self._exclude_names:
            return False
        if len(node.qargs) in self._exclude_arities:
            return False
        return True

    def effective_scale(self, dag: DAGCircuit) -> float:
        """Scale factor actually realised on ``dag`` (may differ from the
        requested value when the foldable pool is too small for the
        fractional part to round cleanly)."""
        d = _count_foldable_gates(dag, self._exclude_names, self._exclude_arities)
        return _compute_effective_scale(d, self.scale_factor)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Fold ``dag`` in place and return the mutated DAG.

        The input is consumed — callers that need the original should
        deepcopy before invoking the pass.  Matches the standard Qiskit
        :class:`~qiskit.transpiler.basepasses.TransformationPass` contract.

        Returns:
            The same ``dag`` object, mutated to contain the folded circuit.
        """
        if self.scale_factor == 1.0:
            return dag

        op_nodes = [node for node in dag.op_nodes() if self._is_foldable(node)]
        d = len(op_nodes)
        k, n = _compute_fold_plan(d, self.scale_factor)
        if k == 0 and n == 0:
            return dag

        extra = self._pick_extra_indices(d, n)

        for i, node in enumerate(op_nodes):
            num_folds = k + 1 if i in extra else k
            if num_folds == 0:
                continue
            dag.substitute_node_with_dag(node, self._folded_sub_dag(node, num_folds))
        return dag


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
