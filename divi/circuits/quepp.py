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

import copy
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral
from typing import Any, Literal

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter, ParameterExpression
from qiskit.circuit.library import (
    CXGate,
    HGate,
    IGate,
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

from divi.circuits._types import ResultFormat
from divi.circuits.qem import OBSERVABLE_OVERRIDE, QEMContext, QEMProtocol
from divi.hamiltonians._term_ops import _split_identity_terms

__all__ = ["QuEPP", "SymbolicAngleWarning"]

# stim's first PauliString.to_numpy() deadlocks when made from several threads at
# once (ensembles expand QuEPP programs concurrently); make it here, single-threaded.
stim.PauliString(1).to_numpy()

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

# Pauli index: 0=I, 1=X, 2=Y, 3=Z  (stim convention)
_PAULI_I, _PAULI_X, _PAULI_Y, _PAULI_Z = 0, 1, 2, 3

# Generator index for each rotation axis
_GENERATOR = {"x": _PAULI_X, "y": _PAULI_Y, "z": _PAULI_Z}

# R_P(π/2) conjugation rules, up to sign: maps (axis, input_pauli) → output_pauli
# R_X(π/2): X→X, Y→Z, Z→Y
# R_Y(π/2): X→Z, Y→Y, Z→X
# R_Z(π/2): X→Y, Y→X, Z→Z
_RP_CONJUGATION: dict[str, dict[int, int]] = {
    "x": {_PAULI_X: _PAULI_X, _PAULI_Y: _PAULI_Z, _PAULI_Z: _PAULI_Y},
    "y": {_PAULI_X: _PAULI_Z, _PAULI_Y: _PAULI_Y, _PAULI_Z: _PAULI_X},
    "z": {_PAULI_X: _PAULI_Y, _PAULI_Y: _PAULI_X, _PAULI_Z: _PAULI_Z},
}

# (X bit, Z bit) of each rotation axis's generator
_GENERATOR_BITS = {"x": (True, False), "y": (True, True), "z": (False, True)}

# Qiskit Clifford rotations for angle normalisation (axis, n_mod_4) → gate or None.
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

# Clifford gate name mapping Qiskit → stim
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


def _stim_gate_name(op) -> str | None:
    """The stim gate for a Clifford Qiskit operation, or ``None`` for a no-op.

    Supports the Clifford gate names in :data:`_QISKIT_CLIFFORD_TO_STIM`
    plus Clifford-valued rotations (``rx``/``ry``/``rz`` with angle
    ``n·π/2``).  Non-Clifford instructions raise ``ValueError``.
    """
    name = op.name
    stim_name = _QISKIT_CLIFFORD_TO_STIM.get(name)
    if stim_name is not None:
        return stim_name
    if name == "barrier":
        return None
    if name in ("rx", "ry", "rz"):
        (angle,) = op.params
        if isinstance(angle, ParameterExpression):
            raise ValueError(
                f"Cannot convert parametric {name} gate to stim — "
                f"expected a Clifford-valued angle."
            )
        angle_f = float(angle)
        n = int(round(angle_f / _HALF_PI))
        if abs(angle_f - n * _HALF_PI) > 1e-10:
            raise ValueError(
                f"Non-Clifford angle for {name}: {angle_f}; expected a multiple of π/2."
            )
        return _CLIFFORD_ROTATION_TO_STIM[(name[1], n % 4)]
    raise ValueError(
        f"Gate {name!r} is not recognised as Clifford by QuEPP's stim converter."
    )


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
        qiskit_label = label.rjust(n_qubits, "I")
        big_endian = qiskit_label[::-1]
        terms.append((float(np.real(coeff)), stim.PauliString("+" + big_endian)))
    return terms


#: ``(T - N)`` is divided by η, so a small η amplifies the shot noise that
#: lives there. Report above this factor even when η clears ``min_eta``.
_ETA_AMPLIFICATION_LIMIT = 5.0


@dataclass
class _ObservableCPT:
    """CPT bookkeeping for one requested observable.

    ``weights``, ``classical_values`` and ``entry_slots`` run in parallel
    over ``(Pauli term, path)`` pairs. ``dag_indices`` holds the shared
    target's slot followed by one path slot per pair, so it is one longer.
    :meth:`__post_init__` enforces both invariants: a misalignment here
    would otherwise surface as a plausible-looking wrong expectation value
    rather than an error.
    """

    weights: np.ndarray
    classical_values: np.ndarray
    dag_indices: list[int]
    entry_slots: list[int]
    target_slots: list[int]
    constant: float = 0.0
    """Sum of the identity terms' coefficients, added exactly after mitigation."""
    eta_rejection: str | None = None
    """Why η was unusable, from :meth:`QuEPP.compute_eta`; ``None`` if it was."""
    eta_amplifying: float | None = None
    """``1/η`` when it exceeds :data:`_ETA_AMPLIFICATION_LIMIT`."""

    def __post_init__(self) -> None:
        n_entries = len(self.weights)
        if len(self.classical_values) != n_entries or (
            len(self.entry_slots) != n_entries
        ):
            raise ValueError(
                f"_ObservableCPT: weights ({n_entries}), classical_values "
                f"({len(self.classical_values)}) and entry_slots "
                f"({len(self.entry_slots)}) must run in parallel over "
                f"(term, path) pairs."
            )
        if len(self.dag_indices) != n_entries + 1:
            raise ValueError(
                f"_ObservableCPT: dag_indices has {len(self.dag_indices)} "
                f"entries; expected {n_entries + 1} (the target slot plus one "
                f"per (term, path) pair)."
            )


def _read_slots(results: np.ndarray, dag_idx: Any, slots: Any) -> np.ndarray:
    """Read declared observables' values out of per-circuit result rows.

    Rows are per-observable sequences whenever the measurement stage
    measured more than one observable (a 2-D ``results``); a bare scalar
    per row means there was only one, so every slot reads it. Indexing
    anything else out of range still raises, so a mismatched slot fails
    loudly rather than reading a plausible wrong value.
    """
    if results.ndim == 1:
        return results[dag_idx]
    return results[dag_idx, slots]


def _split_into_single_terms(
    observables: Sequence[SparsePauliOp],
) -> tuple[tuple[SparsePauliOp, ...], list[list[int]]]:
    """Split each observable into one single-term observable per Pauli.

    Coefficients are kept on the split terms. Grouping and shot allocation
    build their union by deduplicating Paulis and summing absolute
    coefficients, so declaring ``(c_1 P_1, ..., c_n P_n)`` yields the same
    union — and hence the same measurement circuits and shot split — as
    declaring ``sum_i c_i P_i``.

    Returns the flat tuple of single-term observables and, per input
    observable, the flat indices of its terms.
    """
    declared: list[SparsePauliOp] = []
    slots_per_obs: list[list[int]] = []
    for obs in observables:
        start = len(declared)
        declared.extend(obs[term_idx] for term_idx in range(len(obs.paulis)))
        slots_per_obs.append(list(range(start, len(declared))))
    return tuple(declared), slots_per_obs


class SymbolicAngleWarning(UserWarning):
    """A QuEPP option needing concrete rotation angles was disabled because the
    circuit still carries unbound parameters.

    Its own category, so a caller that expects symbolic angles can silence exactly
    this and nothing else — a message-text filter stops suppressing the moment the
    wording changes, and does so silently.
    """


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
    """One term in the CPT expansion, for one Pauli term of the observable.

    ``weight`` is the bare trigonometric product — the observable
    coefficient is *not* folded in, so it stays dimensionless and the
    measured Pauli it pairs with is ``term_idx``'s, not the whole
    observable's.
    """

    branches: tuple[int, ...]  # 0=cos/skip, 1=sin per rotation gate
    weight: float | ParameterExpression
    order: int
    term_idx: int = 0
    """Which Pauli term of the observable this path back-propagates from."""


@dataclass(frozen=True)
class _PreprocResult:
    """Bundle of quantities shared across :meth:`QuEPP._preprocess` and
    :meth:`QuEPP._select_paths`."""

    working: QuantumCircuit
    rotations: list["_RotationGate"]
    segments: list[stim.Circuit]
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
    n = round(theta / _HALF_PI)
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
    """Decompose two-qubit Pauli rotations into CNOT + single-qubit form.

    Handles ``CRX(θ)`` / ``CRY(θ)`` / ``CRZ(θ)`` with exactly one control
    qubit, and ``RZZ(θ)`` / ``RXX(θ)``:

    * ``CRY(θ) → RY(θ/2), CNOT, RY(-θ/2), CNOT``
    * ``CRZ(θ) → RZ(θ/2), CNOT, RZ(-θ/2), CNOT``
    * ``CRX(θ) → H, RZ(θ/2), CNOT, RZ(-θ/2), CNOT, H``
    * ``RZZ(θ) → CNOT, RZ(θ), CNOT``
    * ``RXX(θ) → H⊗H, CNOT, RZ(θ), CNOT, H⊗H``

    Other gates pass through unchanged.
    """
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for inst in qc.data:
        name = inst.operation.name
        if name in ("rzz", "rxx"):
            a, b = inst.qubits
            (theta_raw,) = inst.operation.params
            basis_change = (a, b) if name == "rxx" else ()
            for qubit in basis_change:
                out.append(HGate(), [qubit])
            out.append(CXGate(), [a, b])
            out.append(_ROTATION_CLASS["z"](_coerce_angle(theta_raw)), [b])
            out.append(CXGate(), [a, b])
            for qubit in basis_change:
                out.append(HGate(), [qubit])
        elif name in ("crx", "cry", "crz"):
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


def _build_clifford_segments(
    qc: QuantumCircuit,
    rotations: Sequence[_RotationGate],
) -> list[stim.Circuit]:
    """Build stim circuits for the Clifford layers between rotations.

    Returns ``K + 1`` segments where ``K = len(rotations)``:

    * ``segments[0]``: Clifford ops before the first rotation
    * ``segments[i]``: Clifford ops between rotation i-1 and rotation i
    * ``segments[K]``: Clifford ops after the last rotation
    """
    qubit_str = {q: str(i) for i, q in enumerate(qc.qubits)}
    rotation_idxs = {r.inst_idx for r in rotations}

    lines: list[list[str]] = [[] for _ in range(len(rotations) + 1)]
    seg = 0
    for i, inst in enumerate(qc.data):
        if i in rotation_idxs:
            seg += 1
            continue
        stim_name = _stim_gate_name(inst.operation)
        if stim_name is not None:
            lines[seg].append(
                f"{stim_name} {' '.join(qubit_str[q] for q in inst.qubits)}"
            )
    return [stim.Circuit("\n".join(seg_lines)) for seg_lines in lines]


# ---------------------------------------------------------------------------
# Heisenberg-picture Pauli back-propagation
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

    Merging is keyed on ``(term_idx, branches)`` — paths from different
    Pauli terms pair with different measured Paulis even when their branch
    tuples match.

    Paths with equal ``branches`` have equal ``order`` by construction
    (order = popcount of the branch tuple), so the merged order keeps
    the first-observed value.
    """
    merged: dict[tuple[int, tuple[int, ...]], _PauliPath] = {}
    for p in paths:
        key = (p.term_idx, p.branches)
        if key in merged:
            old = merged[key]
            merged[key] = replace(old, weight=old.weight + p.weight)
        else:
            merged[key] = p
    return list(merged.values())


def _enumerate_paths_dfs(
    rotations: Sequence[_RotationGate],
    segments: Sequence[stim.Circuit],
    observable_terms: Sequence[tuple[float, stim.PauliString]],
    max_order: int,
    coefficient_threshold: float = 0.0,
) -> list[_PauliPath]:
    """Enumerate Pauli paths via Heisenberg-picture DFS, per observable term.

    Each term of *observable_terms* is back-propagated independently and
    its paths are tagged with that term's index. Weights are seeded at
    ``1.0``, so ``coefficient_threshold`` prunes on the dimensionless
    trigonometric product and is therefore invariant to rescaling the
    observable.
    """
    K = len(rotations)
    if K == 0:
        return [
            _PauliPath(branches=(), weight=1.0, order=0, term_idx=term_idx)
            for term_idx in range(len(observable_terms))
        ]

    # Precompute per-rotation metadata once.  The DFS inner loop accessed
    # rot.axis, rot.qubit_idx, rot.angle and called _is_parametric on
    # every iteration — all constant per rotation across millions of iters.
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

    for term_idx, (_obs_coeff, obs_pauli) in enumerate(observable_terms):
        initial_pauli = obs_pauli.before(segments[K])
        # Branches stored as an int bitmask to avoid list-concat copies
        # at every push.  Bit ``i`` corresponds to rotation K-1-i (i.e.
        # bit 0 is the last decision made = rotation 0's branch).
        stack: list[tuple[int, stim.PauliString, int, float, int]] = [
            (K - 1, initial_pauli, 0, 1.0, 0)
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
                segment = segments[idx]
                branches_bits <<= 1
                p = pauli[qubit_idx]

                if p == _PAULI_I or p == gen:
                    # Commute: bit stays 0 (default after shift), walk on.
                    pauli = pauli.before(segment)
                    idx -= 1
                    continue

                if isinstance(angle, ParameterExpression):
                    cos_w = weight * angle.cos()
                    sin_w = weight * angle.sin()
                else:
                    cos_w = weight * np.cos(angle)
                    sin_w = weight * np.sin(angle)

                # A weight already carrying a symbolic factor has no magnitude to prune on.
                unprunable = symbolic or isinstance(weight, ParameterExpression)
                sin_order = order + 1
                sin_kept = sin_order <= max_order and (
                    unprunable or abs(sin_w) >= coefficient_threshold
                )
                cos_kept = unprunable or abs(cos_w) >= coefficient_threshold

                if sin_kept:
                    # Inline R_P(π/2)† · pauli · R_P(π/2) on qubit_idx
                    # (p != I guaranteed here by the earlier commute check).
                    sin_pauli = stim.PauliString(pauli)
                    sin_pauli[qubit_idx] = _RP_CONJUGATION[axis][p]
                    prop_sin = sin_pauli.before(segment)
                    stack.append(
                        (idx - 1, prop_sin, branches_bits | 1, sin_w, sin_order)
                    )

                if cos_kept:
                    # Continue cos branch inline — bit stays 0.
                    pauli = pauli.before(segment)
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
                        term_idx=term_idx,
                    )
                )

    return _merge_paths_by_branch(all_paths)


def _all_cos_paths(
    rotations: Sequence[_RotationGate],
    segments: Sequence[stim.Circuit],
    observable_terms: Sequence[tuple[float, stim.PauliString]],
) -> list[_PauliPath]:
    """The all-zero-branch (all-cos) CPT path, one per observable term.

    Walks each observable term backward through *rotations*, always taking
    the cos branch at non-commuting rotations (and passing through
    commuting ones with no factor).  A term contributes a path of weight
    ``Π cos(θ_i)`` when the fully back-propagated Pauli is diagonal, and
    none otherwise.  Mirrors the ``branches=(0,)*K`` paths of the
    exhaustive DFS exactly — coefficient-free, and tagged per term — and is
    used as a deterministic fallback when every Monte Carlo sample is
    discarded.
    """
    K = len(rotations)
    paths: list[_PauliPath] = []
    for term_idx, (_obs_coeff, obs_pauli) in enumerate(observable_terms):
        pauli = obs_pauli.before(segments[K])
        weight = 1.0
        for idx in range(K - 1, -1, -1):
            rot = rotations[idx]
            p = pauli[rot.qubit_idx]
            if p != _PAULI_I and p != _GENERATOR[rot.axis]:
                weight *= np.cos(rot.angle)
            pauli = pauli.before(segments[idx])
        if _is_diagonal(pauli):
            paths.append(
                _PauliPath(
                    branches=(0,) * K,
                    weight=weight,
                    order=0,
                    term_idx=term_idx,
                )
            )
    return paths


# Conjugation of a Pauli's (X, Z) bit rows, in place on ``(n_qubits, n_samples)``
# arrays; signs are dropped, so a gate and its inverse act identically.


def _swap_xz(q: int, x: np.ndarray, z: np.ndarray) -> None:
    new_x = np.array(z[q])
    z[q] = x[q]
    x[q] = new_x


def _x_xor_z(q: int, x: np.ndarray, z: np.ndarray) -> None:
    x[q] ^= z[q]


def _z_xor_x(q: int, x: np.ndarray, z: np.ndarray) -> None:
    z[q] ^= x[q]


def _cx_bits(a: int, b: int, x: np.ndarray, z: np.ndarray) -> None:
    """CX with control *a* and target *b*."""
    x[b] ^= x[a]
    z[a] ^= z[b]


def _cz_bits(a: int, b: int, x: np.ndarray, z: np.ndarray) -> None:
    z[a] ^= x[b]
    z[b] ^= x[a]


def _swap_bits(a: int, b: int, x: np.ndarray, z: np.ndarray) -> None:
    x[[a, b]] = x[[b, a]]
    z[[a, b]] = z[[b, a]]


_BitOp = Callable[[np.ndarray, np.ndarray], None]

# Bit action of each stim gate a Clifford segment can hold; ``None`` leaves bits as is.
_ONE_QUBIT_BIT_ACTION: dict[
    str, Callable[[int, np.ndarray, np.ndarray], None] | None
] = {
    "I": None,
    "X": None,
    "Y": None,
    "Z": None,
    "H": _swap_xz,
    "SQRT_Y": _swap_xz,
    "SQRT_Y_DAG": _swap_xz,
    "S": _z_xor_x,
    "S_DAG": _z_xor_x,
    "SQRT_X": _x_xor_z,
    "SQRT_X_DAG": _x_xor_z,
}
_TWO_QUBIT_BIT_ACTION: dict[str, Callable[[int, int, np.ndarray, np.ndarray], None]] = {
    "CX": _cx_bits,
    "CZ": _cz_bits,
    "SWAP": _swap_bits,
}


def _pauli_bit_ops(segment: stim.Circuit) -> list[_BitOp]:
    """Steps that move Pauli bits back through *segment*, last gate first.

    Applying them in order matches :meth:`stim.PauliString.before`.
    """
    ops: list[_BitOp] = []
    for inst in reversed(segment):
        if not isinstance(inst, stim.CircuitInstruction):
            raise ValueError("Clifford segments must not contain REPEAT blocks.")
        qubits = [t.value for t in inst.targets_copy()]
        if inst.name in _TWO_QUBIT_BIT_ACTION:
            two_qubit = _TWO_QUBIT_BIT_ACTION[inst.name]
            pairs = list(zip(qubits[::2], qubits[1::2]))
            ops.extend(partial(two_qubit, a, b) for a, b in reversed(pairs))
        elif inst.name in _ONE_QUBIT_BIT_ACTION:
            one_qubit = _ONE_QUBIT_BIT_ACTION[inst.name]
            if one_qubit is not None:
                ops.extend(partial(one_qubit, q) for q in reversed(qubits))
        else:
            raise ValueError(f"No Pauli bit action for stim gate {inst.name!r}.")
    return ops


def _apply_pauli_bit_ops(ops: Sequence[_BitOp], x: np.ndarray, z: np.ndarray) -> None:
    """Run *ops* in order on the ``(n_qubits, n_samples)`` bit arrays."""
    for op in ops:
        op(x, z)


def _quarter_turn_bits(
    axis: str, q: int, x: np.ndarray, z: np.ndarray, mask: np.ndarray
) -> None:
    """Conjugate qubit *q*'s bits by ``R_P(π/2)``, on the *mask* samples only."""
    turned_x, turned_z = np.array(x[q : q + 1]), np.array(z[q : q + 1])
    stim_name = _CLIFFORD_ROTATION_TO_STIM[(axis, 1)]
    action = None if stim_name is None else _ONE_QUBIT_BIT_ACTION[stim_name]
    if action is None:
        raise ValueError(f"R_{axis.upper()}(π/2) has no Pauli bit action.")
    action(0, turned_x, turned_z)
    x[q] = np.where(mask, turned_x[0], x[q])
    z[q] = np.where(mask, turned_z[0], z[q])


def _sample_paths_montecarlo(
    rotations: Sequence[_RotationGate],
    segments: Sequence[stim.Circuit],
    observable_terms: Sequence[tuple[float, stim.PauliString]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[_PauliPath]:
    """Sample Pauli paths via Monte Carlo, tagged per observable term.

    A term is drawn with probability ``|c_i| / Σ|c|`` and the sample's
    importance weight divides that probability back out, so the returned
    weights are coefficient-free: the coefficient is applied once, later,
    on the measured value the weight pairs with.  Dividing by ``|c_i|`` is
    safe because a term with ``c_i == 0`` is never drawn.

    The estimate is unbiased except when no sample ends diagonal: the
    all-cos paths are returned then, so with probability ``q**n_samples``
    (``q`` the chance a sample ends off-diagonal) the estimate is the
    all-cos value rather than zero.
    """
    K = len(rotations)
    if K == 0:
        return [
            _PauliPath(branches=(), weight=1.0, order=0, term_idx=term_idx)
            for term_idx in range(len(observable_terms))
        ]

    abs_coeffs = np.array([abs(c) for c, _ in observable_terms])
    coeff_sum = abs_coeffs.sum()
    if coeff_sum == 0.0:
        warnings.warn(
            f"QuEPP Monte Carlo: every coefficient of a "
            f"{len(observable_terms)}-term observable is zero, so it has no "
            f"paths to sample and mitigation is a no-op. Check whether the "
            f"observable was built as intended (a coefficient that optimises "
            f"to zero reaches this too).",
            stacklevel=2,
        )
        return []
    term_probs = abs_coeffs / coeff_sum
    _warn_on_term_starvation(term_probs, n_samples)

    # All samples propagate together as (n_qubits, n_samples) X/Z bit arrays.
    # Signs are dropped: a sample only needs to know whether it ends diagonal.
    if len(observable_terms) == 1:
        term_idx = np.zeros(n_samples, dtype=np.intp)
    else:
        term_idx = rng.choice(len(observable_terms), size=n_samples, p=term_probs)
    term_x, term_z = zip(*(pauli.to_numpy() for _, pauli in observable_terms))
    x = np.ascontiguousarray(np.stack(term_x)[term_idx].T)
    z = np.ascontiguousarray(np.stack(term_z)[term_idx].T)
    # 1 / P(term) — the coefficient itself stays out of the weight.
    is_weight = 1.0 / term_probs[term_idx]
    # Rotation idx's branch is bit 7 - idx % 8 of byte idx // 8 (np.unpackbits order).
    branch_bits = np.zeros(((K + 7) // 8, n_samples), dtype=np.uint8)

    _apply_pauli_bit_ops(_pauli_bit_ops(segments[K]), x, z)
    for idx in range(K - 1, -1, -1):
        rot = rotations[idx]
        q = rot.qubit_idx
        gen_x, gen_z = _GENERATOR_BITS[rot.axis]
        commutes = (~x[q] & ~z[q]) | ((x[q] == gen_x) & (z[q] == gen_z))

        cos_val = np.cos(rot.angle)
        sin_val = np.sin(rot.angle)
        normalizer = abs(cos_val) + abs(sin_val)
        take_sin = ~commutes & (rng.random(n_samples) >= abs(cos_val) / normalizer)
        branch_factor = normalizer * np.where(
            take_sin, np.sign(sin_val), np.sign(cos_val)
        )
        is_weight = np.where(commutes, is_weight, is_weight * branch_factor)
        branch_bits[idx >> 3] |= take_sin.astype(np.uint8) << (7 - (idx & 7))

        _quarter_turn_bits(rot.axis, q, x, z, take_sin)
        _apply_pauli_bit_ops(_pauli_bit_ops(segments[idx]), x, z)

    merged: dict[tuple[int, bytes], float] = {}
    for j in np.flatnonzero(~x.any(axis=0)):
        key = (int(term_idx[j]), branch_bits[:, j].tobytes())
        merged[key] = merged.get(key, 0.0) + is_weight[j] / n_samples
    samples = []
    for (t_idx, packed), weight in merged.items():
        branches = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=K)
        samples.append(
            _PauliPath(
                branches=tuple(branches.tolist()),
                weight=float(weight),
                order=int(branches.sum()),
                term_idx=t_idx,
            )
        )

    if not samples:
        fallback = _all_cos_paths(rotations, segments, observable_terms)
        if not fallback:
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
            f"deterministic all-cos path of {len(fallback)} observable "
            f"term(s).  Consider increasing n_samples or using exhaustive "
            f"enumeration.",
            stacklevel=2,
        )
        return fallback
    return samples


def _warn_on_term_starvation(
    term_probs: np.ndarray, n_samples: int, min_expected: int = 20
) -> None:
    """Warn when a Pauli term is expected to receive too few Monte Carlo samples.

    Terms are drawn proportionally to ``|c_i|``, so a Hamiltonian whose
    coefficients span orders of magnitude starves its small terms: their
    contribution is estimated from a handful of samples, or none at all.
    """
    if term_probs.size < 2:
        return
    expected = term_probs.min() * n_samples
    if expected >= min_expected:
        return
    warnings.warn(
        f"QuEPP Monte Carlo: with n_samples={n_samples} across "
        f"{term_probs.size} Pauli terms, the smallest-coefficient term "
        f"expects only {expected:.1f} samples (terms are drawn in proportion "
        f"to |coefficient|). Its contribution will be poorly estimated, and "
        f"it may draw none at all. Raise n_samples to at least "
        f"{int(np.ceil(min_expected / term_probs.min()))} or use "
        f"sampling='exhaustive'.",
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Circuit building from branch choices
# ---------------------------------------------------------------------------


def _build_all_cos_dag(
    working: QuantumCircuit,
    rotations: Sequence[_RotationGate],
) -> tuple[DAGCircuit, list[tuple[int, Gate]]]:
    """The all-cos path DAG and, per rotation, ``(node id, R_P(π/2) gate)``.

    Every rotation becomes the identity gate, keeping the path circuit's
    gate count and depth equal to the target's so it accumulates
    comparable noise.  Rotations are located by ``inst_idx``, their
    position in ``working.data``, so the sites follow ``rotations``' order.
    """
    rotation_at = {rot.inst_idx: rot for rot in rotations}
    sites: list[tuple[int, Gate]] = []

    dag = circuit_to_dag(working.copy_empty_like())
    for i, inst in enumerate(working.data):
        rot = rotation_at.get(i)
        if rot is None:
            dag.apply_operation_back(inst.operation, inst.qubits, inst.clbits)
            continue
        node = dag.apply_operation_back(IGate(), inst.qubits, inst.clbits)
        # DAGCircuit.node looks a node up by this id, which deepcopy preserves.
        sites.append((node._node_id, _CLIFFORD_ROTATION_QISKIT[rot.axis]))
    return dag, sites


def _build_path_dag(
    all_cos_dag: DAGCircuit,
    sites: Sequence[tuple[int, Gate]],
    branches: Sequence[int],
) -> DAGCircuit:
    """A copy of *all_cos_dag* with the sin gate at each branch-1 site.

    *all_cos_dag* and *sites* come from :func:`_build_all_cos_dag`.
    """
    dag = copy.deepcopy(all_cos_dag)
    for (node_id, sin_gate), branch in zip(sites, branches):
        if branch:
            dag.substitute_node(dag.node(node_id), sin_gate)
    return dag


# ---------------------------------------------------------------------------
# Classical simulation of the Clifford ensemble
# ---------------------------------------------------------------------------


def _simulate_clifford_ensemble(
    segments: Sequence[stim.Circuit],
    rotations: Sequence[_RotationGate],
    entry_branches: Sequence[tuple[int, ...]],
    entry_terms: Sequence[tuple[float, stim.PauliString]],
) -> np.ndarray:
    """Exact ``coeff * <P>`` per entry on its Clifford path circuit, via stim.

    ``entry_branches[j]`` and ``entry_terms[j]`` describe the same entry:
    the branch choices selecting the path circuit — the Clifford
    ``segments`` with ``R_P(π/2)`` at each branch-1 rotation — and the single Pauli term
    (with its coefficient) to evaluate on it.  Distinct terms sharing a
    path are common, so the tableau simulation is cached per branch tuple
    and only the cheap ``peek`` repeats.
    """
    sin_layers = [
        stim.Circuit(f"{_CLIFFORD_ROTATION_TO_STIM[(rot.axis, 1)]} {rot.qubit_idx}")
        for rot in rotations
    ]
    values = np.empty(len(entry_branches), dtype=float)
    sims: dict[tuple[int, ...], stim.TableauSimulator] = {}
    for i, branches in enumerate(entry_branches):
        sim = sims.get(branches)
        if sim is None:
            sim = stim.TableauSimulator()
            sim.do_circuit(segments[0])
            for branch, sin_layer, segment in zip(branches, sin_layers, segments[1:]):
                if branch:
                    sim.do_circuit(sin_layer)
                sim.do_circuit(segment)
            sims[branches] = sim
        coeff, ps = entry_terms[i]
        values[i] = coeff * sim.peek_observable_expectation(ps)
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
            more auxiliary circuits.  Governs every enumerated path set: always
            with ``sampling="exhaustive"``, and on symbolic circuits otherwise
            (see ``sampling``).
        coefficient_threshold: Prune paths whose absolute weight falls
            below this value during DFS enumeration (disabled on symbolic
            circuits, whose angle magnitudes are unknown).
        sampling: Path selection strategy.  ``"auto"`` (default) draws
            ``n_samples`` random paths on a concrete (parameter-bound) circuit
            and enumerates paths up to ``truncation_order`` on a symbolic one —
            which is what variational programs present at mitigation time,
            before parameter binding, so the symbolic circuit is expanded once
            for every parameter set.  ``"montecarlo"`` always samples, warning
            and falling back to enumeration on a symbolic circuit.
            ``"exhaustive"`` always enumerates, binding parameters before
            mitigation so it can see concrete angles.
        n_samples: Number of Monte Carlo path samples (default 200); applies only
            when a concrete circuit is sampled.  If no sample ends on a
            diagonal Pauli, QuEPP warns and uses the all-cos path instead of a
            zero estimate.  That keeps a small budget stable but biases it
            towards the all-cos value; it matters only when nearly every
            sampled path ends off-diagonal.
        seed: Seed for Monte Carlo path sampling. When set, every expansion of
            a circuit draws the same paths; when ``None``, paths come from the
            generator the pipeline passes, which in a program is seeded by the
            program's ``seed``.
        n_twirls: Number of Pauli twirling samples.  When non-zero, the
            pipeline builder appends a ``PauliTwirlStage``.  Default ``10``.
    """

    def __init__(
        self,
        truncation_order: int = 2,
        coefficient_threshold: float | None = None,
        sampling: Literal["auto", "montecarlo", "exhaustive"] = "auto",
        n_samples: int = 200,
        seed: int | None = None,
        n_twirls: int = 10,
    ) -> None:
        if not isinstance(truncation_order, Integral) or isinstance(
            truncation_order, bool
        ):
            raise ValueError("truncation_order must be an integer.")
        if truncation_order < 0:
            raise ValueError("truncation_order must be non-negative.")
        if sampling not in ("auto", "exhaustive", "montecarlo"):
            raise ValueError(
                "sampling must be 'auto', 'exhaustive' or 'montecarlo', "
                f"got {sampling!r}"
            )
        if sampling != "exhaustive":
            if (
                not isinstance(n_samples, Integral)
                or isinstance(n_samples, bool)
                or n_samples < 1
            ):
                raise ValueError(
                    "n_samples must be a positive integer for montecarlo sampling."
                )
            n_samples = int(n_samples)
        if (
            not isinstance(n_twirls, Integral)
            or isinstance(n_twirls, bool)
            or n_twirls < 0
        ):
            raise ValueError("n_twirls must be a non-negative integer.")
        self._K_T = int(truncation_order)
        self._coeff_threshold = (
            0.0 if coefficient_threshold is None else coefficient_threshold
        )
        self._sampling = sampling
        self._n_samples = n_samples
        self._seed = seed
        self.n_twirls = int(n_twirls)
        self.requires_bound_params = sampling == "exhaustive"

    @property
    def name(self) -> str:
        return "quepp"

    def applies_to(self, result_format: ResultFormat) -> bool:
        # QuEPP reduces to expectation values; raw sampling has none.
        return result_format is ResultFormat.EXPVALS

    def _path_rng(self, rng: np.random.Generator | None) -> np.random.Generator:
        """Return a fresh ``seed`` generator if set, else ``rng``, else an unseeded one."""
        if self._seed is not None:
            return np.random.default_rng(self._seed)
        return rng if rng is not None else np.random.default_rng()

    def expand(
        self,
        dag: DAGCircuit,
        observable: Sequence[SparsePauliOp] | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[tuple[DAGCircuit, ...], QEMContext]:
        """Build path DAGs and per-observable classical context.

        Preprocessing and Clifford segment construction run once; path
        enumeration and classical simulation run per ``(observable term,
        path)`` pair.  Paths sharing a ``branches`` tuple produce the same
        Clifford circuit and are deduped across terms and observables — the
        circuit depends only on the branch choices — while each pair keeps
        its own weight and measured Pauli.

        Declares an ``OBSERVABLE_OVERRIDE`` of single-term observables so the
        measurement stage reports each Pauli term separately; the noisy side
        of the estimator needs the same term resolution as the classical side.
        """
        observables = self._validate_observable_tuple(observable)
        terms_per_obs, identities = zip(
            *(_split_identity_terms(obs) for obs in observables)
        )

        preps = self._preprocess(dag, terms_per_obs)
        working, rotations, symbolic = (
            preps[0].working,
            preps[0].rotations,
            preps[0].symbolic,
        )

        # ----- Per-observable path enumeration --------------------------- #
        obs_terms_list = [prep.obs_terms for prep in preps]
        path_rng = self._path_rng(rng)
        obs_paths_list: list[list[_PauliPath]] = [
            self._select_paths(prep, path_rng) if prep.obs_terms else []
            for prep in preps
        ]
        self._warn_on_truncation_ratio(len(rotations))
        self._warn_no_diagonal_paths(
            len(rotations),
            {
                obs_idx: len(paths)
                for obs_idx, (prep, paths) in enumerate(zip(preps, obs_paths_list))
                if prep.obs_terms
            },
        )

        # ----- Path-DAG dedup across observables ------------------------- #
        working_dag = circuit_to_dag(working)
        unique_branches = dict.fromkeys(
            p.branches for paths in obs_paths_list for p in paths
        )
        branch_to_dag_idx: dict[tuple[int, ...], int] = {
            branches: idx for idx, branches in enumerate(unique_branches)
        }
        all_cos_dag, sites = _build_all_cos_dag(working, rotations)
        merged_path_dags: list[DAGCircuit] = [
            _build_path_dag(all_cos_dag, sites, branches)
            for branches in unique_branches
        ]

        # The target runs as the same gate sequence its paths are built from,
        # so both see the same noise (and the same twirl sites).
        merged_dags = (working_dag,) + tuple(merged_path_dags)

        # ----- Single-term observables to measure ------------------------- #
        # One declared observable per (requested observable, Pauli term),
        # carrying the original coefficient so the union that drives grouping
        # and shot allocation is identical to declaring the whole observable.
        declared, slots_per_obs = _split_into_single_terms(terms_per_obs)

        # ----- Per (term, path) classical sim + weights ------------------- #
        per_obs: list[_ObservableCPT] = []
        for obs_idx, paths in enumerate(obs_paths_list):
            obs_terms = obs_terms_list[obs_idx]
            obs_slots = slots_per_obs[obs_idx]

            # +1 offset because merged_dags[0] is the shared target.
            path_positions = [branch_to_dag_idx[p.branches] + 1 for p in paths]
            # Pair each entry's weight with *its own* term's Pauli; the
            # coefficient rides on the value, applied exactly once.
            entry_terms = [obs_terms[p.term_idx] for p in paths]
            weights = [p.weight for p in paths]
            per_obs.append(
                _ObservableCPT(
                    weights=(
                        np.array(weights, dtype=object)
                        if symbolic
                        else np.array(weights)
                    ),
                    classical_values=_simulate_clifford_ensemble(
                        preps[0].segments,
                        rotations,
                        [p.branches for p in paths],
                        entry_terms,
                    ),
                    dag_indices=[0] + path_positions,
                    entry_slots=[obs_slots[p.term_idx] for p in paths],
                    target_slots=obs_slots,
                    constant=math.fsum(identities[obs_idx].coeffs.real),
                )
            )

        context: QEMContext = {
            "per_obs": per_obs,
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": len(rotations),
            "n_paths": len(merged_path_dags),
            # Identity-only observables are declared whole and run no circuit.
            OBSERVABLE_OVERRIDE: declared or observables,
        }
        if symbolic:
            all_params: set[Parameter] = set().union(
                *(
                    rot.angle.parameters
                    for rot in rotations
                    if isinstance(rot.angle, ParameterExpression)
                )
            )
            context["symbolic"] = True
            context["weight_symbols"] = sorted(all_params, key=lambda p: p.name)
        return merged_dags, context

    def dry_expand(
        self,
        dag: DAGCircuit,
        observable: Sequence[SparsePauliOp] | None = None,
        *,
        rng: np.random.Generator | None = None,
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
        terms_per_obs = [_split_identity_terms(obs)[0] for obs in observables]

        preps = self._preprocess(dag, terms_per_obs)
        rotations, symbolic = preps[0].rotations, preps[0].symbolic

        # Without a generator, preview from a fixed one so the count is stable.
        path_rng = self._path_rng(rng if rng is not None else np.random.default_rng(0))
        sampled_paths = self._sampling != "exhaustive" and not symbolic

        unique_branches: set[tuple[int, ...]] = set()
        n_paths_by_obs: dict[int, int] = {}
        for obs_idx, prep in enumerate(preps):
            if not prep.obs_terms:
                continue
            # The run itself reports any symbolic fallback; keep the preview quiet.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SymbolicAngleWarning)
                paths = self._select_paths(prep, path_rng)
            n_paths_by_obs[obs_idx] = len(paths)
            for p in paths:
                unique_branches.add(p.branches)
        # The measurement stage groups whatever ``expand`` would declare, so
        # the preview must declare it too or its fan-out understates the real
        # one.
        declared, _ = _split_into_single_terms(terms_per_obs)
        self._warn_on_truncation_ratio(len(rotations))
        self._warn_no_diagonal_paths(len(rotations), n_paths_by_obs)

        n_paths = len(unique_branches)
        all_dags = (dag,) * (1 + n_paths)
        context: QEMContext = {
            "n_rotations": len(rotations),
            "n_paths": n_paths,
            # Persisted for introspect(); the dry path skips per_obs
            # construction, so the observable count needs its own slot.
            "n_observables": len(observables),
            OBSERVABLE_OVERRIDE: declared or observables,
        }
        if sampled_paths:
            context["sampled_paths"] = True
        if symbolic:
            context["symbolic"] = True
        return all_dags, context

    @staticmethod
    def _validate_observable_tuple(
        observable: SparsePauliOp | Sequence[SparsePauliOp] | None,
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
        return tuple(observable)

    @staticmethod
    def _preprocess(
        dag: DAGCircuit, observables: Sequence[SparsePauliOp]
    ) -> list["_PreprocResult"]:
        """Decompose, normalise, extract rotations + segments once; obs terms per observable."""
        target_qc = dag_to_circuit(dag)
        n_qubits = target_qc.num_qubits
        decomposed = _decompose_controlled_rotations(target_qc)
        symbolic = _has_symbolic_angles(decomposed)
        working = _normalize_circuit(decomposed)
        rotations = _extract_rotation_gates(working)
        segments = _build_clifford_segments(working, rotations)
        return [
            _PreprocResult(
                working=working,
                rotations=rotations,
                segments=segments,
                obs_terms=_obs_to_stim_terms(obs, n_qubits),
                symbolic=symbolic,
            )
            for obs in observables
        ]

    def _select_paths(
        self, prep: "_PreprocResult", rng: np.random.Generator
    ) -> list[_PauliPath]:
        """Choose sampling strategy and enumerate / sample the Pauli paths,
        drawing any sampled ones from ``rng``."""
        symbolic = prep.symbolic
        if self._sampling == "montecarlo" and symbolic:
            warnings.warn(
                "QuEPP: Monte Carlo sampling requires concrete angles. "
                "Falling back to exhaustive enumeration for symbolic circuit.",
                SymbolicAngleWarning,
                stacklevel=3,
            )

        coeff_threshold = self._coeff_threshold
        if symbolic and coeff_threshold > 0:
            warnings.warn(
                "QuEPP: coefficient_threshold pruning disabled for symbolic "
                "circuit (angle magnitudes unknown).",
                SymbolicAngleWarning,
                stacklevel=3,
            )
            coeff_threshold = 0.0

        if self._sampling != "exhaustive" and not symbolic:
            return _sample_paths_montecarlo(
                prep.rotations,
                prep.segments,
                prep.obs_terms,
                self._n_samples,
                rng,
            )
        return _enumerate_paths_dfs(
            prep.rotations,
            prep.segments,
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
        self, n_rotations: int, n_paths_by_obs: Mapping[int, int]
    ) -> None:
        """Warn when one or more observables produce zero diagonal Pauli paths.

        ``n_paths_by_obs`` maps each observable with non-identity terms to its
        path count. The Heisenberg-DFS enumerator only accepts paths whose final
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
        zero_indices = [i for i, n in n_paths_by_obs.items() if n == 0]
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
    ) -> tuple[float, None] | tuple[None, str]:
        """Compute the rescaling factor η from noisy/ideal ratios.

        Returns ``(eta, None)`` when η is usable, otherwise ``(None,
        reason)`` where *reason* is ``"no_signal"`` (no path carries a
        measurable classical value), ``"negative"`` (the noisy ensemble has
        the opposite sign to the exact one, which rescaling cannot repair),
        or ``"below_floor"`` (η at or under *min_eta*, where ``1/η`` would
        amplify noise rather than suppress it).
        """
        valid = np.abs(classical_values) > 1e-12
        if not np.any(valid):
            return None, "no_signal"
        eta = float(np.median(ensemble_noisy[valid] / classical_values[valid]))
        if eta < 0.0:
            return None, "negative"
        if eta <= min_eta:
            return None, "below_floor"
        return eta, None

    @staticmethod
    def evaluate_symbolic_weights(
        per_obs_entry: "_ObservableCPT",
        symbols: Sequence[Parameter],
        param_values: Sequence[float] | np.ndarray,
    ) -> None:
        """Substitute concrete parameter values into symbolic weight expressions.

        Operates on a single ``per_obs`` slot.  *symbols* are the Qiskit
        :class:`~qiskit.circuit.Parameter` objects referenced by the
        weight expressions; *param_values* are their numeric values in
        the same positional order.
        """
        if not isinstance(per_obs_entry, _ObservableCPT):
            raise TypeError(
                f"evaluate_symbolic_weights expects a per_obs entry "
                f"(context['per_obs'][i]); got {type(per_obs_entry).__name__}."
            )
        binding = {p: float(v) for p, v in zip(symbols, param_values)}
        per_obs_entry.weights = np.array(
            [
                (
                    float(w.bind({p: binding[p] for p in w.parameters}))
                    if isinstance(w, ParameterExpression)
                    else float(w)
                )
                for w in per_obs_entry.weights
            ]
        )

    def reduce(
        self,
        quantum_results: Sequence[float],
        context: QEMContext,
    ) -> list[float]:
        """Combine quantum results with per-observable classical context(s)
        into a list of mitigated expectation values.

        ``quantum_results`` rows are indexed by the single-term observables
        :meth:`expand` declared, so each entry reads its own term's slot via
        ``entry_slots`` while the noisy target is summed back over the
        observable's own terms (``target_slots``). The context's ``per_obs``
        list still carries one entry per *requested* observable, and this
        still returns one value per requested observable. Identity terms are
        added exactly, after mitigation.
        """
        if context.get("symbolic"):
            raise ValueError(
                "QuEPP weights are still symbolic — parameter values were "
                "never substituted. Add ParameterBindingStage to the "
                "pipeline or use QuEPP(sampling='exhaustive') to bind "
                "parameters before mitigation."
            )
        per_obs: list[_ObservableCPT] | None = context.get("per_obs")
        if per_obs is None:
            raise RuntimeError(
                "QuEPP.reduce: context has no per_obs entries (was this a "
                "dry-run context?)."
            )
        target_idx = context["target_idx"]
        ensemble_start = context["ensemble_start"]
        n_rotations = context["n_rotations"]

        results = np.asarray(quantum_results, dtype=float)

        return [
            self._mitigate_entry(
                entry, results, target_idx, ensemble_start, n_rotations
            )
            + entry.constant
            for entry in per_obs
        ]

    def _mitigate_entry(
        self,
        entry: _ObservableCPT,
        results: np.ndarray,
        target_idx: int,
        ensemble_start: int,
        n_rotations: int,
    ) -> float:
        """Return the mitigated value of ``entry``'s non-identity terms."""
        # Cached traces reuse entries, so clear the previous evaluation's flags.
        entry.eta_rejection = None
        entry.eta_amplifying = None
        if not entry.target_slots:
            return 0.0
        dag_indices = entry.dag_indices
        target_slots = np.asarray(entry.target_slots, dtype=np.intp)
        target_rows = np.full(len(target_slots), dag_indices[target_idx])
        target_noisy = sum(_read_slots(results, target_rows, target_slots).tolist())
        ensemble_noisy = _read_slots(
            results,
            np.asarray(dag_indices[ensemble_start:], dtype=np.intp),
            np.asarray(entry.entry_slots, dtype=np.intp),
        )
        classical_est = float(entry.weights @ entry.classical_values)
        if n_rotations == 0:
            return classical_est

        eta, rejection = self.compute_eta(entry.classical_values, ensemble_noisy)
        if eta is None:
            entry.eta_rejection = rejection
            return float(target_noisy)

        if 1.0 / eta > _ETA_AMPLIFICATION_LIMIT:
            entry.eta_amplifying = float(1.0 / eta)
        noisy_est = float(entry.weights @ ensemble_noisy)
        return classical_est + (target_noisy - noisy_est) / eta

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        reasons: set[str] = set()
        amplified = False
        for ctx in contexts:
            for entry in ctx.get("per_obs") or []:
                if entry.eta_rejection is not None:
                    reasons.add(entry.eta_rejection)
                if entry.eta_amplifying is not None:
                    amplified = True

        # Messages carry no per-call counts on purpose: post_reduce runs once
        # per evaluation, and Python deduplicates warnings by message text, so
        # an interpolated count that drifts between optimizer iterations would
        # defeat the dedup and emit one warning per iteration.
        if "no_signal" in reasons:
            warnings.warn(
                "QuEPP: an observable had no Pauli path with a non-negligible "
                "classical expectation value, so η is undefined and the raw "
                "noisy value was returned unmitigated. Check that the "
                "observable's coefficients are not all negligible, and that "
                "the circuit's final Clifford layer leaves the "
                "back-propagated Pauli diagonal. If you also saw the "
                "zero-diagonal-paths warning, this is that same cause "
                "surfacing at reduction time.",
                stacklevel=3,
            )
        if "below_floor" in reasons:
            warnings.warn(
                "QuEPP: signal destroyed — η fell below the safety threshold "
                "and mitigation fell back to the raw noisy value. "
                "Consider increasing shots or reducing noise.",
                stacklevel=3,
            )
        if "negative" in reasons:
            warnings.warn(
                "QuEPP: an observable produced a negative η — the noisy "
                "Clifford ensemble came back with the opposite sign to the "
                "exact one, which rescaling cannot repair, so the raw noisy "
                "value was returned. Raise the shot count first: a sign flip "
                "on a weak signal is often statistical. If it persists, the "
                "noise is past this protocol's usable range — use ZNE "
                "instead.",
                stacklevel=3,
            )
        if amplified:
            warnings.warn(
                f"QuEPP: η was small enough to amplify the noisy residual "
                f"(T - N) by more than {_ETA_AMPLIFICATION_LIMIT:.0f}x. That "
                f"residual is where the shot noise lives, so the mitigated "
                f"value is correspondingly noisier than the unmitigated one. "
                f"Increase shots, or reduce noise so η moves closer to 1.",
                stacklevel=3,
            )
