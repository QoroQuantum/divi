# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

# qiskit re-exports Parameter/ParameterExpression as runtime assignments
# (``Parameter = qiskit._accelerate.circuit.Parameter``) — pylance reads
# those as value bindings rather than classes, so any annotation that
# references them trips reportInvalidTypeForm.  Silence file-wide.
# pyright: reportInvalidTypeForm=false

"""Quantum Enhanced Pauli Propagation (QuEPP) error mitigation protocol.

.. automodapi note: only ``QuEPP`` is public; internal helpers are excluded via ``__all__``.

Implements the hybrid classical-quantum error mitigation scheme from
Majumder et al. (arXiv:2603.14485).  QuEPP decomposes a quantum circuit
into alternating Clifford layers and non-Clifford Pauli rotations via
Clifford Perturbation Theory (CPT).  Low-order Pauli paths are simulated
classically; the residual is corrected using noisy quantum execution with
an empirical rescaling factor.

The decomposition uses the paper's Heisenberg-picture back-propagation
with weights ``cos(θ)``/``sin(θ)`` and ``R_P(π/2)`` Clifford replacements.
The observable is tracked as a Pauli string; gates that commute with the
current observable contribute weight 1 (no branching), which can
dramatically reduce path count.
"""

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import (
    CXGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SXdgGate,
    SXGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits.qem import QEMContext, QEMProtocol

__all__ = ["QuEPP"]

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

# Pauli index: 0=I, 1=X, 2=Y, 3=Z  (stim convention)
_PAULI_I, _PAULI_X, _PAULI_Y, _PAULI_Z = 0, 1, 2, 3

# Generator index for each rotation axis
_GENERATOR = {"x": _PAULI_X, "y": _PAULI_Y, "z": _PAULI_Z}

# R_P(π/2) conjugation rules: maps (axis, input_pauli) → (output_pauli, sign_flip)
# R_X(π/2): X→X, Y→-Z, Z→Y
# R_Y(π/2): X→Z, Y→Y, Z→-X
# R_Z(π/2): X→-Y, Y→X, Z→Z
_RP_CONJUGATION: dict[str, dict[int, tuple[int, int]]] = {
    "x": {_PAULI_X: (_PAULI_X, 1), _PAULI_Y: (_PAULI_Z, -1), _PAULI_Z: (_PAULI_Y, 1)},
    "y": {_PAULI_X: (_PAULI_Z, 1), _PAULI_Y: (_PAULI_Y, 1), _PAULI_Z: (_PAULI_X, -1)},
    "z": {_PAULI_X: (_PAULI_Y, -1), _PAULI_Y: (_PAULI_X, 1), _PAULI_Z: (_PAULI_Z, 1)},
}

# Qiskit Clifford rotations for angle normalization (axis, n_mod_4) → gate or None.
#
# Using the specialised gate classes (``SXGate``, ``SGate``) — rather than
# ``RXGate(π/2)`` / ``RZGate(π/2)`` — ensures the n=1 entries do NOT match
# the ``rx`` / ``ry`` / ``rz`` discriminator in :func:`_is_pauli_rotation`.
# The Y axis has no specialised gate in Qiskit's standard library; we keep
# ``RYGate(±π/2)`` and rely on ``_normalize_circuit``'s symbolic-angle
# passthrough to avoid re-recognising it as a rotation.
_CLIFFORD_POWER_QISKIT: dict[tuple[str, int], Any] = {
    ("x", 0): None,  # Identity
    ("x", 1): SXGate(),
    ("x", 2): XGate(),
    ("x", 3): SXdgGate(),
    ("y", 0): None,
    ("y", 1): RYGate(np.pi / 2),
    ("y", 2): YGate(),
    ("y", 3): RYGate(-np.pi / 2),
    ("z", 0): None,
    ("z", 1): SGate(),
    ("z", 2): ZGate(),
    ("z", 3): SdgGate(),
}

# Qiskit gate for sin-branch replacement: R_P(π/2) as a Clifford — the n=1
# slice of :data:`_CLIFFORD_POWER_QISKIT`, materialised separately for hot-loop
# lookups keyed only by axis.
_CLIFFORD_ROTATION_QISKIT = {
    axis: _CLIFFORD_POWER_QISKIT[(axis, 1)] for axis in ("x", "y", "z")
}

# Rotation gate classes by axis (for detecting rotation type and constructing new ones)
_ROTATION_CLASS = {"x": RXGate, "y": RYGate, "z": RZGate}

# Clifford gate name mapping Qiskit → stim (for _qiskit_to_stim helper)
_QISKIT_CLIFFORD_TO_STIM: dict[str, str] = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S_DAG",
    "sx": "SQRT_X",
    "sxdg": "SQRT_X_DAG",
    "cx": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
    "id": "I",
    "i": "I",
}

# Clifford rotation → stim gate: (axis, n_mod_4) → stim gate name (or None for I)
_CLIFFORD_ROTATION_TO_STIM: dict[tuple[str, int], str | None] = {
    ("x", 0): None,
    ("x", 1): "SQRT_X",
    ("x", 2): "X",
    ("x", 3): "SQRT_X_DAG",
    ("y", 0): None,
    ("y", 1): "SQRT_Y",
    ("y", 2): "Y",
    ("y", 3): "SQRT_Y_DAG",
    ("z", 0): None,
    ("z", 1): "S",
    ("z", 2): "Z",
    ("z", 3): "S_DAG",
}


# ---------------------------------------------------------------------------
# Angle-type helpers
# ---------------------------------------------------------------------------


def _coerce_angle(angle) -> float | ParameterExpression:
    """Return a concrete float or keep as :class:`ParameterExpression`."""
    if isinstance(angle, (int, float, np.floating, np.integer)):
        return float(angle)
    if isinstance(angle, ParameterExpression):
        return angle
    raise TypeError(f"Unsupported angle type for QuEPP: {type(angle).__name__}")


def _is_parametric(x) -> bool:
    """True when *x* is a symbolic :class:`ParameterExpression`, not a plain number."""
    return isinstance(x, ParameterExpression)


def _qiskit_clifford_to_stim(qc_or_dag: QuantumCircuit | DAGCircuit) -> stim.Circuit:
    """Convert a Qiskit circuit of Clifford-only gates to a stim.Circuit.

    Accepts a :class:`QuantumCircuit` or a :class:`DAGCircuit`.

    Supports the Clifford gate names in :data:`_QISKIT_CLIFFORD_TO_STIM`
    plus Clifford-valued rotations (``rx``/``ry``/``rz`` with angle
    ``n·π/2``).  Non-Clifford instructions raise ``ValueError``.

    The output always has the circuit's qubit count — qubits untouched by
    any gate are padded with an explicit no-op ``I`` so downstream
    tableau construction produces a register of the expected width.
    """
    # Hot path: called ~1M times over large DAGs.  Dispatch once on the
    # container type to normalise into an (op, qargs) iterator + qubit
    # register, then tight-loop without per-node generator frame overhead.
    clifford_map = _QISKIT_CLIFFORD_TO_STIM
    rotation_map = _CLIFFORD_ROTATION_TO_STIM
    rotation_names = ("rx", "ry", "rz")
    pi_over_2 = np.pi / 2

    if isinstance(qc_or_dag, DAGCircuit):
        n_qubits = qc_or_dag.num_qubits()
        qubits = qc_or_dag.qubits
        op_stream = ((node.op, node.qargs) for node in qc_or_dag.topological_op_nodes())
    else:
        n_qubits = qc_or_dag.num_qubits
        qubits = qc_or_dag.qubits
        op_stream = ((inst.operation, inst.qubits) for inst in qc_or_dag.data)

    qubit_idx = {q: i for i, q in enumerate(qubits)}

    sc = stim.Circuit()
    if n_qubits > 0:
        sc.append("I", list(range(n_qubits)), ())

    for op, qargs in op_stream:
        name = op.name
        stim_name = clifford_map.get(name)
        if stim_name is not None:
            sc.append(stim_name, [qubit_idx[q] for q in qargs], ())
            continue
        if name in rotation_names:
            (angle,) = op.params
            if isinstance(angle, ParameterExpression):
                raise ValueError(
                    f"Cannot convert parametric {name} gate to stim — "
                    f"expected a Clifford-valued angle."
                )
            angle_f = float(angle)
            n = int(round(angle_f / pi_over_2))
            if abs(angle_f - n * pi_over_2) > 1e-10:
                raise ValueError(
                    f"Non-Clifford angle for {name}: {angle_f}; "
                    f"expected a multiple of π/2."
                )
            stim_rot = rotation_map[(name[1], n % 4)]
            if stim_rot is not None:
                sc.append(stim_rot, [qubit_idx[q] for q in qargs], ())
            continue
        raise ValueError(
            f"Gate {name!r} is not recognised as Clifford by QuEPP's stim "
            f"converter."
        )
    return sc


# ---------------------------------------------------------------------------
# Observable conversion
# ---------------------------------------------------------------------------


def _obs_to_stim_terms(
    observable: SparsePauliOp,
    n_qubits: int,
) -> list[tuple[float, stim.PauliString]]:
    """Convert a :class:`SparsePauliOp` to weighted stim PauliStrings.

    Qiskit's label layout is little-endian (qubit 0 on the right); stim
    reads strings in big-endian (qubit 0 on the left), so we reverse each
    label to align coordinate conventions before constructing
    :class:`stim.PauliString`.
    """
    terms: list[tuple[float, stim.PauliString]] = []
    for label, coeff in zip(observable.paulis.to_labels(), observable.coeffs):
        # label has length `observable.num_qubits`; pad/align to n_qubits.
        qiskit_label = label
        if len(qiskit_label) < n_qubits:
            qiskit_label = "I" * (n_qubits - len(qiskit_label)) + qiskit_label
        big_endian = qiskit_label[::-1]
        terms.append((float(np.real(coeff)), stim.PauliString("+" + big_endian)))
    return terms


# ---------------------------------------------------------------------------
# CPT decomposition data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RotationGate:
    """A single Pauli rotation gate in the circuit."""

    inst_idx: int  # position in qc.data
    qubit_idx: int  # index of the target qubit in the circuit's qubit list
    axis: str  # "x", "y", or "z"
    angle: float | ParameterExpression  # rotation angle θ (in radians)


@dataclass(frozen=True)
class _PauliPath:
    """One term in the CPT expansion."""

    branches: tuple[int, ...]  # 0=cos/skip, 1=sin per rotation gate
    weight: float | ParameterExpression
    order: int  # number of sine branches taken


@dataclass(frozen=True)
class _PreprocResult:
    """Bundle of quantities shared across :meth:`QuEPP._preprocess` and
    :meth:`QuEPP._select_paths`."""

    working: QuantumCircuit
    n_qubits: int
    rotations: list["_RotationGate"]
    tableaus: list[stim.Tableau]
    obs_terms: list[tuple[float, stim.PauliString]]
    symbolic: bool


# ---------------------------------------------------------------------------
# Circuit parsing
# ---------------------------------------------------------------------------

_HALF_PI = np.pi / 2


def _normalize_angle(theta: float) -> tuple[int, float]:
    """Decompose θ = n·(π/2) + θ' with |θ'| ≤ π/4.

    Returns ``(n, theta_prime)``.
    """
    n = int(round(theta / _HALF_PI))
    theta_prime = theta - n * _HALF_PI
    return n, theta_prime


def _rotation_instr_data(op) -> tuple[str, float | ParameterExpression] | None:
    """If *op* is any ``rx`` / ``ry`` / ``rz`` gate, return ``(axis, angle)``.

    Used by :func:`_normalize_circuit` — accepts Clifford-valued angles so
    the normalisation step can factor them out.
    """
    name = op.name
    if name not in ("rx", "ry", "rz"):
        return None
    (raw_angle,) = op.params
    return name[1], _coerce_angle(raw_angle)


def _is_pauli_rotation(op) -> tuple[str, float | ParameterExpression] | None:
    """If *op* is a **non-Clifford** ``rx`` / ``ry`` / ``rz`` rotation,
    return ``(axis, angle)``.

    Concrete rotations whose angle is a multiple of π/2 are Clifford and
    are treated here as *not* being Pauli rotations: ``_normalize_circuit``
    uses :data:`_CLIFFORD_POWER_QISKIT` to emit them (``SXGate``/``SGate``
    for the X/Z axes; ``RYGate(n·π/2)`` for the Y axis since Qiskit lacks
    an ``SYGate``), and we want :func:`_extract_rotation_gates` to skip
    those Y-axis ``RYGate(π/2)`` passthroughs on subsequent passes.

    Symbolic angles always count as rotations — their magnitudes are
    unknown at circuit-build time.
    """
    rot = _rotation_instr_data(op)
    if rot is None:
        return None
    axis, angle = rot
    if isinstance(angle, float):
        n = round(angle / (np.pi / 2))
        if abs(angle - n * np.pi / 2) < 1e-10:
            return None
    return axis, angle


def _decompose_controlled_rotations(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose controlled Pauli rotations into CNOT + single-qubit form.

    Handles ``CRX(θ)`` / ``CRY(θ)`` / ``CRZ(θ)`` with exactly one control
    qubit:

    * ``CRY(θ) → RY(θ/2), CNOT, RY(-θ/2), CNOT``
    * ``CRZ(θ) → RZ(θ/2), CNOT, RZ(-θ/2), CNOT``
    * ``CRX(θ) → H, RZ(θ/2), CNOT, RZ(-θ/2), CNOT, H``

    Other gates pass through unchanged.
    """
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("crx", "cry", "crz"):
            ctrl, target = inst.qubits
            (theta_raw,) = inst.operation.params
            theta = _coerce_angle(theta_raw)
            axis = name[-1]  # crx → "x"
            if axis == "x":
                out.append(HGate(), [target])
                out.append(_ROTATION_CLASS["z"](theta / 2), [target])
                out.append(CXGate(), [ctrl, target])
                out.append(_ROTATION_CLASS["z"](-theta / 2), [target])
                out.append(CXGate(), [ctrl, target])
                out.append(HGate(), [target])
            else:
                out.append(_ROTATION_CLASS[axis](theta / 2), [target])
                out.append(CXGate(), [ctrl, target])
                out.append(_ROTATION_CLASS[axis](-theta / 2), [target])
                out.append(CXGate(), [ctrl, target])
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _normalize_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Factor out Clifford components from rotations so residual |θ| ≤ π/4.

    For each ``R_P(θ)`` with |θ| > π/4, decomposes into
    ``R_P(n·π/2) · R_P(θ')`` where ``R_P(n·π/2)`` is Clifford and
    |θ'| ≤ π/4.  The Clifford part becomes a standard Qiskit gate
    (``XGate``, ``SGate``, ``SXGate``, etc.) so it is NOT detected as a
    Pauli rotation by :func:`_is_pauli_rotation`.

    Symbolic angles are left untouched — their Clifford components cannot
    be factored without concrete values.
    """
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        # Use the permissive rotation detector here — Clifford-angle
        # rotations SHOULD be normalised (factored into Clifford gates).
        rot = _rotation_instr_data(inst.operation)
        if rot is None:
            out.append(inst.operation, inst.qubits, inst.clbits)
            continue

        axis, angle = rot
        if _is_parametric(angle):
            # Symbolic — leave as-is.
            out.append(inst.operation, inst.qubits, inst.clbits)
            continue

        n, theta_prime = _normalize_angle(float(angle))
        n_mod = n % 4

        # Clifford part first, then the residual rotation on the same qubit.
        cliff_gate = _CLIFFORD_POWER_QISKIT[(axis, n_mod)]
        if cliff_gate is not None:
            out.append(cliff_gate, inst.qubits)
        if abs(theta_prime) > 1e-15:
            out.append(_ROTATION_CLASS[axis](theta_prime), inst.qubits)
    return out


def _has_symbolic_angles(qc: QuantumCircuit) -> bool:
    """Return True if any ``rx`` / ``ry`` / ``rz`` gate carries a symbolic angle."""
    for inst in qc.data:
        if inst.operation.name not in ("rx", "ry", "rz"):
            continue
        (raw,) = inst.operation.params
        if isinstance(raw, ParameterExpression):
            return True
    return False


def _extract_rotation_gates(qc: QuantumCircuit) -> list[_RotationGate]:
    """Identify all single-qubit Pauli rotations in *qc* (flat order)."""
    q_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    gates: list[_RotationGate] = []
    for i, inst in enumerate(qc.data):
        rot = _is_pauli_rotation(inst.operation)
        if rot is None:
            continue
        axis, angle = rot
        (qubit,) = inst.qubits
        gates.append(
            _RotationGate(
                inst_idx=i,
                qubit_idx=q_to_idx[qubit],
                axis=axis,
                angle=angle,
            )
        )
    return gates


def _build_clifford_tableaus(
    qc: QuantumCircuit,
    rotations: list[_RotationGate],
) -> list[stim.Tableau]:
    """Build stim Tableaus for the Clifford layers between rotations.

    Returns ``K + 1`` tableaus where ``K = len(rotations)``:

    * ``tableaus[0]``: Clifford ops before the first rotation
    * ``tableaus[i]``: Clifford ops between rotation i-1 and rotation i
    * ``tableaus[K]``: Clifford ops after the last rotation
    """
    n_qubits = qc.num_qubits
    rotation_idxs = {r.inst_idx for r in rotations}

    segments: list[list] = [[] for _ in range(len(rotations) + 1)]
    seg = 0
    for i, inst in enumerate(qc.data):
        if i in rotation_idxs:
            seg += 1
        else:
            segments[seg].append(inst)

    tableaus: list[stim.Tableau] = []
    for seg_insts in segments:
        if not seg_insts:
            tableaus.append(stim.Tableau(n_qubits))
        else:
            seg_qc = QuantumCircuit(*qc.qregs)
            for inst in seg_insts:
                seg_qc.append(inst.operation, inst.qubits, inst.clbits)
            tableaus.append(stim.Tableau.from_circuit(_qiskit_clifford_to_stim(seg_qc)))
    return tableaus


# ---------------------------------------------------------------------------
# Heisenberg-picture Pauli back-propagation (stim-native, unchanged)
# ---------------------------------------------------------------------------


def _is_diagonal(pauli_string: stim.PauliString) -> bool:
    """Check if a Pauli string is diagonal (only I and Z, no X or Y).

    Uses stim's numpy view: a Pauli is I/Z iff its X-bit is unset.  One
    C-level ``.any()`` beats a 64+ iteration Python loop over stim's
    indexing API on large strings.
    """
    xs, _ = pauli_string.to_numpy()
    return not xs.any()


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------


def _merge_paths_by_branch(paths: Iterable[_PauliPath]) -> list[_PauliPath]:
    """Collapse paths with identical branch choices by summing their weights.

    Paths with equal ``branches`` have equal ``order`` by construction
    (order = popcount of the branch tuple), so the merged order keeps
    the first-observed value.
    """
    merged: dict[tuple[int, ...], _PauliPath] = {}
    for p in paths:
        if p.branches in merged:
            old = merged[p.branches]
            merged[p.branches] = _PauliPath(
                branches=p.branches,
                weight=old.weight + p.weight,
                order=old.order,
            )
        else:
            merged[p.branches] = p
    return list(merged.values())


def _enumerate_paths_dfs(
    rotations: list[_RotationGate],
    tableaus: list[stim.Tableau],
    observable_terms: list[tuple[float, stim.PauliString]],
    max_order: int,
    coefficient_threshold: float = 0.0,
) -> list[_PauliPath]:
    """Enumerate Pauli paths via Heisenberg-picture DFS."""
    K = len(rotations)
    if K == 0:
        return [_PauliPath(branches=(), weight=1.0, order=0)]

    # Precompute per-rotation metadata once.  The DFS inner loop accessed
    # rot.axis, rot.qubit_idx, rot.angle and called _is_parametric on
    # every iteration — all constant per rotation across millions of iters.
    inv_tableaus = [t.inverse() for t in tableaus]
    rot_meta = [
        (
            _GENERATOR[rot.axis],
            rot.qubit_idx,
            rot.axis,
            rot.angle,
            _is_parametric(rot.angle),
        )
        for rot in rotations
    ]

    all_paths: list[_PauliPath] = []

    for obs_coeff, obs_pauli in observable_terms:
        initial_pauli = inv_tableaus[K](obs_pauli)
        # Branches stored as an int bitmask to avoid list-concat copies
        # at every push.  Bit ``i`` corresponds to rotation K-1-i (i.e.
        # bit 0 is the last decision made = rotation 0's branch).
        stack: list[tuple[int, stim.PauliString, int, float, int]] = [
            (K - 1, initial_pauli, 0, obs_coeff, 0)
        ]

        # Inner ``while idx >= 0`` walks commute/cos branches in local
        # variables without touching the stack.  Only the *sin* branch of
        # a non-commuting rotation ever gets pushed.  This collapses what
        # was a ~29M push/pop workload (almost entirely commute-step
        # round-trips) down to the number of actually-kept sin decisions.
        while stack:
            idx, pauli, branches_bits, weight, order = stack.pop()

            while idx >= 0:
                gen, qubit_idx, axis, angle, symbolic = rot_meta[idx]
                inv_tab = inv_tableaus[idx]
                branches_bits <<= 1
                p = pauli[qubit_idx]

                if p == _PAULI_I or p == gen:
                    # Commute: bit stays 0 (default after shift), walk on.
                    pauli = inv_tab(pauli)
                    idx -= 1
                    continue

                if symbolic:
                    cos_w = weight * angle.cos()
                    sin_w = weight * angle.sin()
                else:
                    cos_w = weight * np.cos(angle)
                    sin_w = weight * np.sin(angle)

                sin_order = order + 1
                sin_kept = sin_order <= max_order and (
                    symbolic or abs(sin_w) >= coefficient_threshold
                )
                cos_kept = symbolic or abs(cos_w) >= coefficient_threshold

                if sin_kept:
                    # Inline R_P(π/2)† · pauli · R_P(π/2) on qubit_idx
                    # (p != I guaranteed here by the earlier commute check).
                    new_p, sign = _RP_CONJUGATION[axis][p]
                    sin_pauli = stim.PauliString(pauli)
                    sin_pauli[qubit_idx] = new_p
                    if sign < 0:
                        sin_pauli *= -1
                    prop_sin = inv_tab(sin_pauli)
                    stack.append(
                        (idx - 1, prop_sin, branches_bits | 1, sin_w, sin_order)
                    )

                if cos_kept:
                    # Continue cos branch inline — bit stays 0.
                    pauli = inv_tab(pauli)
                    weight = cos_w
                    idx -= 1
                    continue

                # Neither branch survives threshold — abandon this path.
                idx = -2
                break

            if idx == -1 and _is_diagonal(pauli):
                all_paths.append(
                    _PauliPath(
                        branches=tuple((branches_bits >> i) & 1 for i in range(K)),
                        weight=weight,
                        order=order,
                    )
                )

    return _merge_paths_by_branch(all_paths)


def _all_cos_path_weight(
    rotations: list[_RotationGate],
    inv_tableaus: list[stim.Tableau],
    observable_terms: list[tuple[float, stim.PauliString]],
) -> float:
    """Weight of the all-zero-branch (all-cos) CPT path.

    Walks each observable term backward through *rotations*, always taking
    the cos branch at non-commuting rotations (and passing through
    commuting ones with no factor).  A term contributes
    ``obs_coeff * Π cos(θ_i)`` when the fully back-propagated Pauli is
    diagonal, and zero otherwise.  Mirrors the ``branches=(0,)*K`` path
    of the exhaustive DFS exactly, and is used as a deterministic fallback
    when every Monte Carlo sample is discarded.
    """
    K = len(rotations)
    total = 0.0
    for obs_coeff, obs_pauli in observable_terms:
        pauli = inv_tableaus[K](obs_pauli)
        weight = obs_coeff
        for idx in range(K - 1, -1, -1):
            rot = rotations[idx]
            p = pauli[rot.qubit_idx]
            if p != _PAULI_I and p != _GENERATOR[rot.axis]:
                weight *= np.cos(rot.angle)
            pauli = inv_tableaus[idx](pauli)
        if _is_diagonal(pauli):
            total += weight
    return total


def _sample_paths_montecarlo(
    rotations: list[_RotationGate],
    tableaus: list[stim.Tableau],
    observable_terms: list[tuple[float, stim.PauliString]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[_PauliPath]:
    """Sample Pauli paths via Monte Carlo."""
    K = len(rotations)
    if K == 0:
        return [_PauliPath(branches=(), weight=1.0, order=0)]

    # Precompute tableau inverses once (same optimisation as _enumerate_paths_dfs).
    inv_tableaus = [t.inverse() for t in tableaus]

    samples: list[_PauliPath] = []

    abs_coeffs = np.array([abs(c) for c, _ in observable_terms])
    coeff_sum = abs_coeffs.sum()
    term_probs = abs_coeffs / coeff_sum

    for _ in range(n_samples):
        if len(observable_terms) == 1:
            obs_coeff, obs_pauli = observable_terms[0]
        else:
            term_idx = rng.choice(len(observable_terms), p=term_probs)
            obs_coeff, obs_pauli = observable_terms[term_idx]

        is_weight = np.sign(obs_coeff) * coeff_sum

        pauli = inv_tableaus[K](obs_pauli)
        branches: list[int] = []

        for idx in range(K - 1, -1, -1):
            rot = rotations[idx]
            axis = rot.axis
            gen = _GENERATOR[axis]
            qubit_idx = rot.qubit_idx
            inv_tab = inv_tableaus[idx]

            # Inline commute-at-qubit check (same hot-loop inlining as DFS).
            p = pauli[qubit_idx]
            if p == _PAULI_I or p == gen:
                branches.append(0)
                pauli = inv_tab(pauli)
            else:
                cos_val = np.cos(rot.angle)
                sin_val = np.sin(rot.angle)
                cos_p = abs(cos_val)
                sin_p = abs(sin_val)
                normalizer = cos_p + sin_p
                if rng.random() < cos_p / normalizer:
                    branches.append(0)
                    is_weight *= np.sign(cos_val) * normalizer
                    pauli = inv_tab(pauli)
                else:
                    branches.append(1)
                    is_weight *= np.sign(sin_val) * normalizer
                    # Inline R_P(π/2)† · pauli · R_P(π/2) on qubit_idx
                    # (non-commuting ⇒ p != I).
                    new_p, sign = _RP_CONJUGATION[axis][p]
                    pauli = stim.PauliString(pauli)
                    pauli[qubit_idx] = new_p
                    if sign < 0:
                        pauli *= -1
                    pauli = inv_tab(pauli)

        branches.reverse()

        if not _is_diagonal(pauli):
            continue

        samples.append(
            _PauliPath(
                branches=tuple(branches),
                weight=is_weight / n_samples,
                order=sum(branches),
            )
        )

    if not samples:
        fallback_weight = _all_cos_path_weight(
            rotations, inv_tableaus, observable_terms
        )
        if fallback_weight == 0.0:
            warnings.warn(
                f"QuEPP Monte Carlo: all {n_samples} samples produced "
                f"non-diagonal Pauli strings, and the deterministic "
                f"all-cos fallback path has zero diagonal contribution.  "
                f"Returning an empty path list (zero estimate).  Consider "
                f"increasing n_samples or using exhaustive enumeration.",
                stacklevel=2,
            )
            return []
        warnings.warn(
            f"QuEPP Monte Carlo: all {n_samples} samples produced "
            f"non-diagonal Pauli strings.  Falling back to the "
            f"deterministic all-cos path with weight "
            f"{fallback_weight:.4e}.  Consider increasing n_samples or "
            f"using exhaustive enumeration.",
            stacklevel=2,
        )
        return [_PauliPath(branches=(0,) * K, weight=fallback_weight, order=0)]
    return _merge_paths_by_branch(samples)


# ---------------------------------------------------------------------------
# Circuit building from branch choices
# ---------------------------------------------------------------------------


def _build_path_dag(
    base_dag: DAGCircuit,
    rotation_positions: list[tuple[int, _RotationGate]],
    branches: tuple[int, ...],
) -> DAGCircuit:
    """Build a Clifford DAG by replacing rotations according to branch choices.

    * branch 0: rotation is removed (identity / skip).
    * branch 1: rotation replaced with ``R_P(π/2)`` Clifford.

    Uses ``copy_empty_like`` + ``apply_operation_back`` (8x faster than
    ``deepcopy``) to build the path DAG in a single pass, skipping or
    replacing rotation nodes inline.
    """
    replacements: dict[int, Any] = {}
    for (topo_idx, rot), branch in zip(rotation_positions, branches):
        replacements[topo_idx] = (
            None if branch == 0 else _CLIFFORD_ROTATION_QISKIT[rot.axis]
        )

    dag = base_dag.copy_empty_like()
    for i, node in enumerate(base_dag.topological_op_nodes()):
        if i in replacements:
            gate = replacements[i]
            if gate is not None:
                dag.apply_operation_back(gate, node.qargs, node.cargs)
        else:
            dag.apply_operation_back(node.op, node.qargs, node.cargs)

    return dag


# ---------------------------------------------------------------------------
# Classical simulation of the Clifford ensemble
# ---------------------------------------------------------------------------


def _simulate_clifford_ensemble(
    circuits: Sequence[QuantumCircuit | DAGCircuit],
    observable: SparsePauliOp,
    n_qubits: int,
) -> np.ndarray:
    """Compute exact expectation values for Clifford circuits via stim."""
    terms = _obs_to_stim_terms(observable, n_qubits)
    values = np.empty(len(circuits), dtype=float)
    for i, qc_or_dag in enumerate(circuits):
        sc = _qiskit_clifford_to_stim(qc_or_dag)
        sim = stim.TableauSimulator()
        sim.do_circuit(sc)
        values[i] = sum(
            coeff * sim.peek_observable_expectation(ps) for coeff, ps in terms
        )
    return values


# ---------------------------------------------------------------------------
# QuEPP protocol
# ---------------------------------------------------------------------------


class QuEPP(QEMProtocol):
    """Quantum Enhanced Pauli Propagation (QuEPP) error mitigation.

    Decomposes the target circuit into Clifford + non-Clifford parts via
    Clifford Perturbation Theory (CPT), classically simulates the low-order
    Pauli paths, and uses noisy quantum execution to correct the truncation
    bias through an empirical rescaling factor η.

    Args:
        truncation_order: Maximum number of sine branches in the CPT
            expansion (K_T).  Higher values reduce bias at the cost of
            more auxiliary circuits.  Ignored when ``sampling="montecarlo"``.
        coefficient_threshold: Prune paths whose absolute weight falls
            below this value during DFS enumeration.  Only used with
            ``sampling="exhaustive"``.
        sampling: Path selection strategy — ``"montecarlo"`` (default) or
            ``"exhaustive"``.
        n_samples: Number of Monte Carlo path samples (default 200).
        seed: RNG seed for Monte Carlo reproducibility.
        n_twirls: Number of Pauli twirling samples.  When non-zero, the
            pipeline builder appends a ``PauliTwirlStage``.  Default ``10``.
        bind_before_mitigation: When ``False`` (default), QuEPP runs on the
            parametric circuit and produces symbolically-weighted paths
            (using Qiskit :class:`~qiskit.circuit.ParameterExpression`
            arithmetic) that are later bound by the parameter-binding
            stage.  Set to ``True`` to bind parameters first and mitigate
            per-parameter-set.
    """

    def __init__(
        self,
        truncation_order: int = 2,
        coefficient_threshold: float | None = None,
        sampling: str = "montecarlo",
        n_samples: int = 200,
        seed: int | None = None,
        n_twirls: int = 10,
        bind_before_mitigation: bool = False,
    ) -> None:
        if truncation_order < 0:
            raise ValueError("truncation_order must be non-negative.")
        if sampling not in ("exhaustive", "montecarlo"):
            raise ValueError(
                f"sampling must be 'exhaustive' or 'montecarlo', got {sampling!r}"
            )
        if sampling == "montecarlo" and (n_samples is None or n_samples < 1):
            raise ValueError(
                "n_samples must be a positive integer for montecarlo sampling."
            )
        self._K_T = truncation_order
        self._coeff_threshold = (
            0.0 if coefficient_threshold is None else coefficient_threshold
        )
        self._sampling = sampling
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self.n_twirls = n_twirls
        self.bind_before_mitigation = bind_before_mitigation

    @property
    def name(self) -> str:
        return "quepp"

    def expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Build path DAGs and per-observable classical context.

        Preprocessing and Clifford tableau construction run once; path
        enumeration and classical simulation run per observable.  Paths
        sharing a ``branches`` tuple produce the same Clifford circuit
        and are deduped across observables.
        """
        observables = self._validate_observable_tuple(observable)

        # ----- Observable-independent preprocessing ---------------------- #
        target_qc = dag_to_circuit(dag)
        n_qubits = target_qc.num_qubits
        decomposed = _decompose_controlled_rotations(target_qc)
        symbolic = _has_symbolic_angles(decomposed)
        working = _normalize_circuit(decomposed)
        rotations = _extract_rotation_gates(working)
        tableaus = _build_clifford_tableaus(working, rotations)

        # ----- Per-observable path enumeration --------------------------- #
        obs_paths_list: list[list[_PauliPath]] = []
        for obs in observables:
            prep = _PreprocResult(
                working=working,
                n_qubits=n_qubits,
                rotations=rotations,
                tableaus=tableaus,
                obs_terms=_obs_to_stim_terms(obs, n_qubits),
                symbolic=symbolic,
            )
            obs_paths_list.append(self._select_paths(prep))
        self._warn_on_truncation_ratio(len(rotations))
        self._warn_no_diagonal_paths(
            len(rotations), [len(paths) for paths in obs_paths_list]
        )

        # ----- Path-DAG dedup across observables ------------------------- #
        working_dag = circuit_to_dag(working)
        rotation_positions = [(rot.inst_idx, rot) for rot in rotations]
        branch_to_dag_idx: dict[tuple[int, ...], int] = {}
        merged_path_dags: list[DAGCircuit] = []
        for paths in obs_paths_list:
            for p in paths:
                if p.branches not in branch_to_dag_idx:
                    branch_to_dag_idx[p.branches] = len(merged_path_dags)
                    merged_path_dags.append(
                        _build_path_dag(working_dag, rotation_positions, p.branches)
                    )

        merged_dags = (dag,) + tuple(merged_path_dags)

        # ----- Per-observable per-path classical sim + weights ----------- #
        all_params: set[Parameter] | None = None
        if symbolic:
            all_params = set().union(
                *(
                    rot.angle.parameters
                    for rot in rotations
                    if _is_parametric(rot.angle)
                )
            )

        per_obs: list[dict] = []
        for obs, paths in zip(observables, obs_paths_list):
            # +1 offset because merged_dags[0] is the shared target.
            path_positions = [branch_to_dag_idx[p.branches] + 1 for p in paths]
            dag_indices = [0] + path_positions

            obs_path_dags = [
                merged_path_dags[branch_to_dag_idx[p.branches]] for p in paths
            ]
            classical_values = _simulate_clifford_ensemble(obs_path_dags, obs, n_qubits)
            weights = [p.weight for p in paths]
            per_obs.append(
                {
                    "classical_values": classical_values,
                    "weights": (
                        np.array(weights, dtype=object)
                        if symbolic
                        else np.array(weights)
                    ),
                    "n_paths": len(paths),
                    "dag_indices": dag_indices,
                }
            )

        context: QEMContext = {
            "per_obs": per_obs,
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": len(rotations),
            "n_paths": len(merged_path_dags),
        }
        if symbolic:
            context["symbolic"] = True
            context["weight_symbols"] = sorted(
                all_params if all_params is not None else [],
                key=lambda p: p.name,
            )
        return merged_dags, context

    def dry_expand(
        self,
        dag: DAGCircuit,
        observable: tuple[SparsePauliOp, ...] | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Analytic path: emit ``1 + n_paths`` placeholder DAGs, no Clifford sim.

        Runs the same preprocessing + per-observable path enumeration as
        :meth:`expand` to get an accurate fan-out factor, then skips the
        per-path DAG surgery and Clifford ensemble simulation — the two
        dominant costs — and reuses the input ``dag`` as a placeholder for
        every emitted slot.  ``n_paths`` reflects the deduplicated count
        across all observables.
        """
        observables = self._validate_observable_tuple(observable)

        target_qc = dag_to_circuit(dag)
        n_qubits = target_qc.num_qubits
        decomposed = _decompose_controlled_rotations(target_qc)
        symbolic = _has_symbolic_angles(decomposed)
        working = _normalize_circuit(decomposed)
        rotations = _extract_rotation_gates(working)
        tableaus = _build_clifford_tableaus(working, rotations)

        unique_branches: set[tuple[int, ...]] = set()
        n_paths_per_obs: list[int] = []
        for obs in observables:
            prep = _PreprocResult(
                working=working,
                n_qubits=n_qubits,
                rotations=rotations,
                tableaus=tableaus,
                obs_terms=_obs_to_stim_terms(obs, n_qubits),
                symbolic=symbolic,
            )
            paths = self._select_paths(prep)
            n_paths_per_obs.append(len(paths))
            for p in paths:
                unique_branches.add(p.branches)
        self._warn_on_truncation_ratio(len(rotations))
        self._warn_no_diagonal_paths(len(rotations), n_paths_per_obs)

        n_paths = len(unique_branches)
        all_dags = (dag,) * (1 + n_paths)
        context: QEMContext = {
            "per_obs": None,
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": len(rotations),
            "n_paths": n_paths,
        }
        if symbolic:
            context["symbolic"] = True
            context["weight_symbols"] = []
        return all_dags, context

    @staticmethod
    def _validate_observable_tuple(
        observable: SparsePauliOp | tuple[SparsePauliOp, ...] | None,
    ) -> tuple[SparsePauliOp, ...]:
        if observable is None:
            raise ValueError(
                "QuEPP requires at least one observable (SparsePauliOp) for "
                "classical Clifford simulation."
            )
        if isinstance(observable, SparsePauliOp):
            observable = (observable,)
        if not observable:
            raise ValueError(
                "QuEPP requires at least one observable (SparsePauliOp) for "
                "classical Clifford simulation."
            )
        for obs in observable:
            if not isinstance(obs, SparsePauliOp):
                raise TypeError(
                    f"QuEPP.expand observable tuple entries must be "
                    f"SparsePauliOp; got {type(obs).__name__}."
                )
        return observable

    @staticmethod
    def _preprocess(dag: DAGCircuit, observable: SparsePauliOp) -> "_PreprocResult":
        """Decompose, normalize, extract rotations + tableaus + obs terms."""
        target_qc = dag_to_circuit(dag)
        n_qubits = target_qc.num_qubits
        decomposed = _decompose_controlled_rotations(target_qc)
        symbolic = _has_symbolic_angles(decomposed)
        working = _normalize_circuit(decomposed)
        rotations = _extract_rotation_gates(working)
        tableaus = _build_clifford_tableaus(working, rotations)
        obs_terms = _obs_to_stim_terms(observable, n_qubits)
        return _PreprocResult(
            working=working,
            n_qubits=n_qubits,
            rotations=rotations,
            tableaus=tableaus,
            obs_terms=obs_terms,
            symbolic=symbolic,
        )

    def _select_paths(self, prep: "_PreprocResult") -> list[_PauliPath]:
        """Choose sampling strategy and enumerate / sample the Pauli paths."""
        symbolic = prep.symbolic
        if self._sampling == "montecarlo" and symbolic:
            warnings.warn(
                "QuEPP: Monte Carlo sampling requires concrete angles. "
                "Falling back to exhaustive enumeration for symbolic circuit.",
                stacklevel=3,
            )

        coeff_threshold = self._coeff_threshold
        if symbolic and coeff_threshold > 0:
            warnings.warn(
                "QuEPP: coefficient_threshold pruning disabled for symbolic "
                "circuit (angle magnitudes unknown).",
                stacklevel=3,
            )
            coeff_threshold = 0.0

        if self._sampling == "montecarlo" and not symbolic:
            return _sample_paths_montecarlo(
                prep.rotations,
                prep.tableaus,
                prep.obs_terms,
                self._n_samples,
                self._rng,
            )
        return _enumerate_paths_dfs(
            prep.rotations,
            prep.tableaus,
            prep.obs_terms,
            max_order=self._K_T,
            coefficient_threshold=coeff_threshold,
        )

    def _warn_on_truncation_ratio(self, n_rotations: int) -> None:
        if n_rotations > 0 and self._K_T / n_rotations > 0.33:
            warnings.warn(
                f"QuEPP: truncation order K={self._K_T} replaces a large "
                f"fraction of the {n_rotations} non-Clifford rotations "
                f"({self._K_T / n_rotations:.0%}). Mitigation quality may "
                f"degrade on shallow circuits — consider reducing "
                f"truncation_order or using a deeper circuit.",
                stacklevel=3,
            )

    def _warn_no_diagonal_paths(
        self, n_rotations: int, n_paths_per_obs: list[int]
    ) -> None:
        """Warn when one or more observables produce zero diagonal Pauli paths.

        The Heisenberg-DFS enumerator only accepts paths whose final
        back-propagated Pauli is diagonal (composed of ``I`` and ``Z``);
        when the circuit's Clifford structure routes an observable into
        a strictly non-diagonal basis (e.g. a final ``H`` rotates ``Z``
        to ``X``), no path survives and ``classical_values`` is empty.
        :meth:`reduce` then takes the ``eta is None`` branch and returns
        the raw noisy target unchanged — so mitigation is a no-op even
        though it was configured.  Surface this so the user does not
        silently consume noisy results believing they were mitigated.

        Emits a single warning listing every offending observable index,
        so a multi-observable job with 50 failing observables produces
        one diagnostic, not 50.
        """
        # All-Clifford circuits trivially have no rotations; mitigation
        # is structurally a no-op and warning would be noise.
        if n_rotations == 0:
            return
        zero_indices = [i for i, n in enumerate(n_paths_per_obs) if n == 0]
        if not zero_indices:
            return
        warnings.warn(
            f"QuEPP: observable(s) at index/indices {zero_indices} produced "
            f"zero diagonal Pauli paths (truncation_order={self._K_T}, "
            f"{n_rotations} non-Clifford rotation(s)). The Heisenberg "
            f"back-propagation terminates in a non-diagonal basis, so "
            f"mitigation will be a no-op for these observables and the "
            f"raw noisy expectation will be returned. Consider rebasing "
            f"the observable or restructuring the circuit's final "
            f"Clifford layer.",
            stacklevel=3,
        )

    @staticmethod
    def compute_eta(
        classical_values: np.ndarray,
        ensemble_noisy: np.ndarray,
        min_eta: float = 0.1,
    ) -> float | None:
        """Compute the rescaling factor η from noisy/ideal ratios."""
        valid = np.abs(classical_values) > 1e-12
        if not np.any(valid):
            return None
        eta = float(np.median(ensemble_noisy[valid] / classical_values[valid]))
        return eta if eta > min_eta else None

    @staticmethod
    def evaluate_symbolic_weights(
        per_obs_entry: dict,
        symbols: Sequence[Parameter],
        param_values: np.ndarray,
    ) -> None:
        """Substitute concrete parameter values into symbolic weight expressions.

        Operates on a single ``per_obs`` slot.  *symbols* are the Qiskit
        :class:`~qiskit.circuit.Parameter` objects referenced by the
        weight expressions; *param_values* are their numeric values in
        the same positional order.
        """
        if "per_obs" in per_obs_entry:
            raise TypeError(
                "evaluate_symbolic_weights expects a per_obs entry "
                "(context['per_obs'][i]), not the full QuEPP context."
            )
        binding = {p: float(v) for p, v in zip(symbols, param_values)}
        per_obs_entry["weights"] = np.array(
            [
                (
                    float(w.bind({p: binding[p] for p in w.parameters}))
                    if isinstance(w, ParameterExpression)
                    else float(w)
                )
                for w in per_obs_entry["weights"]
            ]
        )

    def reduce(
        self,
        quantum_results: Sequence[float],
        context: QEMContext,
    ) -> list[float]:
        """Combine quantum results with per-observable classical context(s)
        into a list of mitigated expectation values.

        The context's ``per_obs`` list carries one entry per observable;
        each entry has its own ``dag_indices`` slicing
        ``quantum_results``, its own ``classical_values`` and ``weights``,
        and produces one mitigated value via the QuEPP η formula.
        """
        if context.get("symbolic"):
            raise ValueError(
                "QuEPP weights are still symbolic — parameter values were "
                "never substituted. Add ParameterBindingStage to the "
                "pipeline or use QuEPP(bind_before_mitigation=True)."
            )
        per_obs: list[dict] | None = context.get("per_obs")
        if per_obs is None:
            raise RuntimeError(
                "QuEPP.reduce: context has no per_obs entries (was this a "
                "dry-run context?)."
            )
        target_idx = context["target_idx"]
        ensemble_start = context["ensemble_start"]
        n_rotations = context["n_rotations"]

        out: list[float] = []
        for obs_idx, entry in enumerate(per_obs):
            indices = entry["dag_indices"]
            obs_results: list[float] = []
            for i in indices:
                row = quantum_results[i]
                if isinstance(row, (list, tuple)):
                    obs_results.append(float(row[obs_idx]))
                else:
                    obs_results.append(float(row))

            target_noisy = obs_results[target_idx]
            ensemble_noisy = np.array(obs_results[ensemble_start:], dtype=float)
            classical_values = entry["classical_values"]
            weights = entry["weights"]

            if n_rotations == 0:
                out.append(float(weights @ classical_values))
                continue

            classical_est = float(weights @ classical_values)
            noisy_est = float(weights @ ensemble_noisy)

            eta = self.compute_eta(classical_values, ensemble_noisy)
            if eta is None:
                valid = np.abs(classical_values) > 1e-12
                if np.any(valid):
                    entry["_signal_destroyed"] = True
                out.append(float(target_noisy))
                continue

            out.append(classical_est + (target_noisy - noisy_est) / eta)

        return out

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        destroyed = 0
        for ctx in contexts:
            per_obs = ctx.get("per_obs") or []
            destroyed += sum(1 for entry in per_obs if entry.get("_signal_destroyed"))
        if destroyed:
            warnings.warn(
                "QuEPP: signal destroyed — η fell below the safety threshold "
                "and mitigation fell back to the raw noisy value. "
                "Consider increasing shots or reducing noise.",
                stacklevel=3,
            )
