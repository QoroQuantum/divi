# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum Enhanced Pauli Propagation (QuEPP) error mitigation protocol.

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cirq
import numpy as np
import stim
import stimcirq
from cirq.circuits.circuit import Circuit
from pennylane.exceptions import TermsUndefinedError as _TermsUndefinedError

from divi.circuits.qem import QEMContext, QEMProtocol

# ---------------------------------------------------------------------------
# Clifford simulation via stim
# ---------------------------------------------------------------------------


def _obs_to_stim_terms(
    observable: Any,
    n_qubits: int,
) -> list[tuple[float, stim.PauliString]]:
    """Convert a PennyLane observable to weighted stim PauliStrings."""
    try:
        coeffs, ops = observable.terms()
    except (AttributeError, TypeError, _TermsUndefinedError):
        coeffs, ops = [1.0], [observable]

    terms = []
    for coeff, op in zip(coeffs, ops):
        chars = ["I"] * n_qubits
        if op.pauli_rep is not None:
            for pw, pw_coeff in op.pauli_rep.items():
                for w, p in pw.items():
                    chars[w] = p
                coeff = float(np.real(coeff * pw_coeff))
        terms.append((float(np.real(coeff)), stim.PauliString("+" + "".join(chars))))
    return terms


def _simulate_clifford_ensemble(
    circuits: Sequence[Circuit],
    observable: Any,
    n_qubits: int,
) -> np.ndarray:
    """Compute exact expectation values for Clifford circuits via stim."""
    terms = _obs_to_stim_terms(observable, n_qubits)
    values = np.empty(len(circuits), dtype=float)
    for i, c in enumerate(circuits):
        sc = stimcirq.cirq_circuit_to_stim_circuit(c)
        sim = stim.TableauSimulator()
        sim.do_circuit(sc)
        values[i] = sum(
            coeff * sim.peek_observable_expectation(ps) for coeff, ps in terms
        )
    return values


# ---------------------------------------------------------------------------
# CPT decomposition data structures
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

# Cirq gate for sin-branch replacement: R_P(π/2) as PowGate (stim-compatible)
_CLIFFORD_ROTATION = {
    "x": lambda q: (cirq.X**0.5)(q),
    "y": lambda q: (cirq.Y**0.5)(q),
    "z": lambda q: cirq.S(q),
}

# Clifford power gates for angle normalization: R_P(n·π/2) as PowGate
_CLIFFORD_POWER = {
    "x": cirq.X,
    "y": cirq.Y,
    "z": cirq.Z,
}

# Rotation constructors keyed by axis
_ROTATION_CTOR = {
    "x": cirq.rx,
    "y": cirq.ry,
    "z": cirq.rz,
}


@dataclass(frozen=True)
class _RotationGate:
    """A single Pauli rotation gate in the circuit."""

    moment_idx: int
    op_idx: int
    qubit: cirq.Qid
    qubit_idx: int  # index in the sorted qubit list
    axis: str  # "x", "y", or "z"
    angle: float  # rotation angle θ (in radians)


@dataclass(frozen=True)
class _PauliPath:
    """One term in the CPT expansion."""

    branches: tuple[int, ...]  # 0=cos/skip, 1=sin per rotation gate
    weight: float
    order: int  # number of sine branches taken


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


def _normalize_circuit(circuit: Circuit) -> Circuit:
    """Factor out Clifford components from rotations so residual |θ| ≤ π/4.

    For each R_P(θ) with |θ| > π/4, decomposes into R_P(n·π/2) · R_P(θ')
    where R_P(n·π/2) is Clifford and |θ'| ≤ π/4.  Clifford parts use
    PowGate representation (``X**0.5``, ``Y**0.5``, ``S``, etc.) so they
    are NOT detected as Pauli rotations by ``_is_pauli_rotation``.
    """
    new_moments: list[cirq.Moment] = []
    for moment in circuit:
        clifford_ops: list[cirq.Operation] = []
        main_ops: list[cirq.Operation] = []
        needs_split = False

        for op in moment.operations:
            if len(op.qubits) != 1:
                main_ops.append(op)
                continue
            rot = _is_pauli_rotation(op.gate)
            if rot is None:
                main_ops.append(op)
                continue

            axis, angle = rot
            n, theta_prime = _normalize_angle(angle)
            n_mod = n % 4
            q = op.qubits[0]

            # If nothing changes, keep original
            if n_mod == 0 and abs(angle - theta_prime) < 1e-15:
                main_ops.append(op)
                continue

            # Clifford part: R_P(n·π/2) as PowGate (invisible to rotation detection)
            if n_mod != 0:
                clifford_ops.append((_CLIFFORD_POWER[axis] ** (n_mod * 0.5))(q))
                needs_split = True

            # Residual rotation with |θ'| ≤ π/4
            if abs(theta_prime) > 1e-15:
                main_ops.append(_ROTATION_CTOR[axis](theta_prime)(q))

        # Clifford parts must precede residual rotations on the same qubit
        if needs_split and clifford_ops:
            new_moments.append(cirq.Moment(clifford_ops))
            if main_ops:
                new_moments.append(cirq.Moment(main_ops))
        else:
            all_ops = main_ops + clifford_ops
            if all_ops:
                new_moments.append(cirq.Moment(all_ops))

    result = Circuit(new_moments)
    # Ensure all original qubits are present
    missing = circuit.all_qubits() - result.all_qubits()
    if missing:
        result = Circuit(
            [cirq.Moment(cirq.I(q) for q in missing)] + list(result.moments)
        )
    return result


def _is_pauli_rotation(gate: cirq.Gate) -> tuple[str, float] | None:
    """Return (axis, angle) if *gate* is an explicit Rx/Ry/Rz rotation, else None.

    Uses ``_rads`` (set by Cirq's Rx/Ry/Rz constructors) to preserve the
    exact input radians.  The public ``exponent`` property divides by π,
    which introduces floating-point drift on the round-trip.
    """
    if isinstance(gate, cirq.Rx):
        return ("x", gate._rads)
    if isinstance(gate, cirq.Ry):
        return ("y", gate._rads)
    if isinstance(gate, cirq.Rz):
        return ("z", gate._rads)
    return None


def _extract_rotation_gates(
    circuit: Circuit, qubit_order: list[cirq.Qid]
) -> list[_RotationGate]:
    """Identify all single-qubit Pauli rotations in *circuit*."""
    q_to_idx = {q: i for i, q in enumerate(qubit_order)}
    gates: list[_RotationGate] = []
    for m_idx, moment in enumerate(circuit):
        for o_idx, op in enumerate(moment.operations):
            if len(op.qubits) != 1:
                continue
            rot = _is_pauli_rotation(op.gate)
            if rot is not None:
                axis, angle = rot
                gates.append(
                    _RotationGate(
                        moment_idx=m_idx,
                        op_idx=o_idx,
                        qubit=op.qubits[0],
                        qubit_idx=q_to_idx[op.qubits[0]],
                        axis=axis,
                        angle=angle,
                    )
                )
    return gates


def _build_clifford_tableaus(
    circuit: Circuit,
    rotations: list[_RotationGate],
    qubit_order: list[cirq.Qid],
) -> list[stim.Tableau]:
    """Build stim Tableaus for the Clifford layers between rotations.

    Returns K+1 tableaus where K = len(rotations):
    - tableaus[0]: Clifford ops before the first rotation
    - tableaus[i]: Clifford ops between rotation i-1 and rotation i
    - tableaus[K]: Clifford ops after the last rotation
    """
    n_qubits = len(qubit_order)
    rotation_positions = {(r.moment_idx, r.op_idx) for r in rotations}

    # Split circuit ops into K+1 segments
    segments: list[list[cirq.Operation]] = [[] for _ in range(len(rotations) + 1)]
    rot_idx = 0
    for m_idx, moment in enumerate(circuit):
        for o_idx, op in enumerate(moment.operations):
            if (m_idx, o_idx) in rotation_positions:
                rot_idx += 1
            else:
                # Clamp to last segment if past all rotations
                seg = min(rot_idx, len(rotations))
                segments[seg].append(op)

    q_to_idx = {q: i for i, q in enumerate(qubit_order)}
    tableaus = []
    for seg_ops in segments:
        if not seg_ops:
            tableaus.append(stim.Tableau(n_qubits))
        else:
            # Ensure all qubits are present so the tableau is n_qubits wide
            seg_circuit = cirq.Circuit([cirq.I(q) for q in qubit_order], seg_ops)
            stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(
                seg_circuit, qubit_to_index_dict=q_to_idx
            )
            tableaus.append(stim.Tableau.from_circuit(stim_circuit))
    return tableaus


# ---------------------------------------------------------------------------
# Heisenberg-picture Pauli back-propagation
# ---------------------------------------------------------------------------


def _commutes_at_qubit(
    pauli_string: stim.PauliString, qubit_idx: int, generator: int
) -> bool:
    """Check if the Pauli on *qubit_idx* commutes with *generator*."""
    p = pauli_string[qubit_idx]
    if p == _PAULI_I or p == generator:
        return True  # I commutes with everything; same Pauli commutes
    return False  # different non-I Paulis anti-commute


def _apply_rp_conjugation(
    pauli_string: stim.PauliString, qubit_idx: int, axis: str
) -> stim.PauliString:
    """Apply R_P(π/2)† · pauli_string · R_P(π/2) on *qubit_idx*."""
    p = pauli_string[qubit_idx]
    if p == _PAULI_I:
        return stim.PauliString(pauli_string)  # I is invariant
    new_p, sign = _RP_CONJUGATION[axis][p]
    result = stim.PauliString(pauli_string)
    result[qubit_idx] = new_p
    if sign < 0:
        result *= -1
    return result


def _is_diagonal(pauli_string: stim.PauliString) -> bool:
    """Check if a Pauli string is diagonal (only I and Z, no X or Y)."""
    for i in range(len(pauli_string)):
        p = pauli_string[i]
        if p == _PAULI_X or p == _PAULI_Y:
            return False
    return True


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------


def _enumerate_paths_dfs(
    rotations: list[_RotationGate],
    tableaus: list[stim.Tableau],
    observable_terms: list[tuple[float, stim.PauliString]],
    max_order: int,
    coefficient_threshold: float = 0.0,
) -> list[_PauliPath]:
    """Enumerate Pauli paths via Heisenberg-picture DFS.

    Back-propagates each observable term through the circuit from last
    rotation to first, branching at anti-commuting gates.
    """
    K = len(rotations)
    if K == 0:
        return [_PauliPath(branches=(), weight=1.0, order=0)]

    all_paths: list[_PauliPath] = []

    for obs_coeff, obs_pauli in observable_terms:
        # Start from the output: propagate backward through the last Clifford layer
        initial_pauli = tableaus[K].inverse()(obs_pauli)

        # DFS stack: (rotation_idx, pauli_string, branches_so_far, weight, order)
        # Process from rotation K-1 down to 0
        stack: list[tuple[int, stim.PauliString, list[int], float, int]] = [
            (K - 1, initial_pauli, [], obs_coeff, 0)
        ]

        while stack:
            idx, pauli, branches, weight, order = stack.pop()

            if idx < 0:
                # Pauli has already been propagated through all layers
                if _is_diagonal(pauli):
                    all_paths.append(
                        _PauliPath(
                            branches=tuple(branches),
                            weight=weight,
                            order=order,
                        )
                    )
                continue

            rot = rotations[idx]
            gen = _GENERATOR[rot.axis]

            if _commutes_at_qubit(pauli, rot.qubit_idx, gen):
                # s=0: gate is transparent, weight unchanged, no branching
                # Propagate through Clifford layer before this rotation
                prop_pauli = tableaus[idx].inverse()(pauli)
                stack.append((idx - 1, prop_pauli, [0] + branches, weight, order))
            else:
                # s=1: anti-commutes — branch with cos(θ)/sin(θ)

                # Cos branch: gate → identity, pauli unchanged
                cos_w = weight * np.cos(rot.angle)
                if abs(cos_w) >= coefficient_threshold:
                    prop_cos = tableaus[idx].inverse()(pauli)
                    stack.append((idx - 1, prop_cos, [0] + branches, cos_w, order))

                # Sin branch: gate → R_P(π/2), pauli transforms
                sin_w = weight * np.sin(rot.angle)
                sin_order = order + 1
                if sin_order <= max_order and abs(sin_w) >= coefficient_threshold:
                    sin_pauli = _apply_rp_conjugation(pauli, rot.qubit_idx, rot.axis)
                    prop_sin = tableaus[idx].inverse()(sin_pauli)
                    stack.append((idx - 1, prop_sin, [1] + branches, sin_w, sin_order))

    # Merge paths with identical branch vectors (from different observable terms)
    merged: dict[tuple[int, ...], _PauliPath] = {}
    for p in all_paths:
        if p.branches in merged:
            old = merged[p.branches]
            merged[p.branches] = _PauliPath(
                branches=p.branches,
                weight=old.weight + p.weight,
                order=p.order,
            )
        else:
            merged[p.branches] = p

    return list(merged.values())


def _sample_paths_montecarlo(
    rotations: list[_RotationGate],
    tableaus: list[stim.Tableau],
    observable_terms: list[tuple[float, stim.PauliString]],
    n_samples: int,
    rng: np.random.Generator,
) -> list[_PauliPath]:
    """Sample Pauli paths via Monte Carlo.

    At each anti-commuting gate, the cos branch is taken with probability
    |cos(θ)|/(|cos(θ)|+|sin(θ)|) and the sin branch otherwise.
    """
    K = len(rotations)
    if K == 0:
        return [_PauliPath(branches=(), weight=1.0, order=0)]

    seen: dict[tuple[int, ...], _PauliPath] = {}

    # Pre-compute for importance-sampling correction
    abs_coeffs = np.array([abs(c) for c, _ in observable_terms])
    coeff_sum = abs_coeffs.sum()
    if len(observable_terms) > 1:
        term_probs = abs_coeffs / coeff_sum

    for _ in range(n_samples):
        # Pick a random observable term proportional to |coeff|
        if len(observable_terms) == 1:
            obs_coeff, obs_pauli = observable_terms[0]
        else:
            term_idx = rng.choice(len(observable_terms), p=term_probs)
            obs_coeff, obs_pauli = observable_terms[term_idx]

        # IS correction for term selection: coeff / sampling_probability
        is_weight = np.sign(obs_coeff) * coeff_sum

        pauli = tableaus[K].inverse()(obs_pauli)
        branches: list[int] = []

        for idx in range(K - 1, -1, -1):
            rot = rotations[idx]
            gen = _GENERATOR[rot.axis]

            if _commutes_at_qubit(pauli, rot.qubit_idx, gen):
                branches.append(0)
                pauli = tableaus[idx].inverse()(pauli)
            else:
                cos_val = np.cos(rot.angle)
                sin_val = np.sin(rot.angle)
                cos_p = abs(cos_val)
                sin_p = abs(sin_val)
                normalizer = cos_p + sin_p
                if rng.random() < cos_p / normalizer:
                    branches.append(0)
                    is_weight *= np.sign(cos_val) * normalizer
                    pauli = tableaus[idx].inverse()(pauli)
                else:
                    branches.append(1)
                    is_weight *= np.sign(sin_val) * normalizer
                    pauli = _apply_rp_conjugation(pauli, rot.qubit_idx, rot.axis)
                    pauli = tableaus[idx].inverse()(pauli)

        branches.reverse()

        if not _is_diagonal(pauli):
            continue

        # NOTE: the back-propagated Pauli sign is NOT multiplied in here
        # because the stim simulation (or quantum measurement) already
        # captures it.  Including it would double-count the ±1 factor.
        sample_weight = is_weight / n_samples

        key = tuple(branches)
        if key in seen:
            seen[key] = _PauliPath(
                branches=key,
                weight=seen[key].weight + sample_weight,
                order=seen[key].order,
            )
        else:
            seen[key] = _PauliPath(
                branches=key, weight=sample_weight, order=sum(branches)
            )

    return (
        list(seen.values())
        if seen
        else [_PauliPath(branches=(0,) * K, weight=1.0, order=0)]
    )


# ---------------------------------------------------------------------------
# Circuit building from branch choices
# ---------------------------------------------------------------------------


def _build_path_circuit(
    circuit: Circuit,
    rotations: list[_RotationGate],
    branches: tuple[int, ...],
) -> Circuit:
    """Build a Clifford circuit by replacing rotations according to branch choices.

    - branch 0 at a commuting gate: identity (gate removed)
    - branch 0 at an anti-commuting gate (cos): identity (gate removed)
    - branch 1 (sin): R_P(π/2) Clifford replacement
    """
    replacements: dict[tuple[int, int], cirq.Operation | None] = {}
    for rot, branch in zip(rotations, branches):
        key = (rot.moment_idx, rot.op_idx)
        if branch == 0:
            replacements[key] = None  # identity
        else:
            replacements[key] = _CLIFFORD_ROTATION[rot.axis](rot.qubit)

    all_qubits = circuit.all_qubits()
    new_moments: list[cirq.Moment] = []
    for m_idx, moment in enumerate(circuit):
        new_ops = []
        for o_idx, op in enumerate(moment.operations):
            key = (m_idx, o_idx)
            if key in replacements:
                if replacements[key] is not None:
                    new_ops.append(replacements[key])
            else:
                new_ops.append(op)
        if new_ops:
            new_moments.append(cirq.Moment(new_ops))

    result = Circuit(new_moments)
    missing = all_qubits - result.all_qubits()
    if missing:
        new_moments.insert(0, cirq.Moment(cirq.I(q) for q in missing))
        result = Circuit(new_moments)

    return result


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
            below this value during DFS enumeration.  Provides early
            termination so subtrees with negligible contribution are
            never explored.  Only used with ``sampling="exhaustive"``.
        sampling: Path selection strategy.

            * ``"montecarlo"`` *(default)* — draw *n_samples* random paths
              by sampling branches at each anti-commuting gate.  Fixed
              budget regardless of circuit size.
            * ``"exhaustive"`` — DFS enumeration of all paths up to
              *truncation_order*, pruned by *coefficient_threshold*.
              Deterministic; cost grows as O(n^K_T).
        n_samples: Number of Monte Carlo path samples (default 200).
            Required when ``sampling="montecarlo"``.
        seed: RNG seed for Monte Carlo reproducibility.
        eta_mode: Documents the intended rescaling strategy when the
            observable is a multi-term Hamiltonian.

            * ``"per_group"`` *(default)* — η is computed independently
              for each measurement group (Pauli term).
            * ``"global"`` — a single η is used for the full Hamiltonian.
              This happens automatically when the backend supports native
              expectation values (``ham_ops`` path).

            .. note::
               The QuEPP paper (arXiv:2603.14485) defines η for a single
               observable and does not discuss multi-term Hamiltonians.
               This parameter is a divi extension.

        n_twirls: Number of Pauli twirling samples.  When non-zero, the
            pipeline builder appends a ``PauliTwirlStage`` that generates
            *n_twirls* randomised copies of each circuit before submission.
            Set to ``0`` to disable twirling.  Default ``10``.

    Example::

        # Default (Monte Carlo, 200 samples)
        QuEPP(truncation_order=2)

        # More samples for higher accuracy
        QuEPP(n_samples=500, seed=42)

        # Exhaustive for small circuits (deterministic)
        QuEPP(sampling="exhaustive", truncation_order=2)

        # Exhaustive with aggressive pruning (medium circuits)
        QuEPP(sampling="exhaustive", truncation_order=3, coefficient_threshold=0.01)
    """

    def __init__(
        self,
        truncation_order: int = 2,
        coefficient_threshold: float | None = None,
        sampling: str = "montecarlo",
        n_samples: int = 200,
        seed: int | None = None,
        eta_mode: str = "per_group",
        n_twirls: int = 10,
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
        if eta_mode not in ("per_group", "global"):
            raise ValueError(
                f"eta_mode must be 'per_group' or 'global', got {eta_mode!r}"
            )
        self._K_T = truncation_order
        self._coeff_threshold = (
            0.0 if coefficient_threshold is None else coefficient_threshold
        )
        self._sampling = sampling
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self._eta_mode = eta_mode
        self.n_twirls = n_twirls

    @property
    def name(self) -> str:
        return "quepp"

    def expand(
        self,
        cirq_circuit: Circuit,
        observable: Any | None = None,
    ) -> tuple[tuple[Circuit, ...], QEMContext]:
        if observable is None:
            raise ValueError(
                "QuEPP requires an observable for classical Clifford simulation."
            )

        qubits = sorted(cirq_circuit.all_qubits())
        n_qubits = len(qubits)

        # Normalize rotations so residual |θ| ≤ π/4 (paper Sec. II–III).
        # The normalized circuit is used for CPT decomposition; the
        # *original* circuit is the target sent to the quantum backend.
        normalized = _normalize_circuit(cirq_circuit)

        rotations = _extract_rotation_gates(normalized, qubits)
        tableaus = _build_clifford_tableaus(normalized, rotations, qubits)
        obs_terms = _obs_to_stim_terms(observable, n_qubits)

        # Select Pauli paths
        if self._sampling == "montecarlo":
            paths = _sample_paths_montecarlo(
                rotations, tableaus, obs_terms, self._n_samples, self._rng
            )
        else:
            paths = _enumerate_paths_dfs(
                rotations,
                tableaus,
                obs_terms,
                max_order=self._K_T,
                coefficient_threshold=self._coeff_threshold,
            )

        # Build Clifford circuits from the *normalized* circuit
        path_circuits = [
            _build_path_circuit(normalized, rotations, p.branches) for p in paths
        ]
        classical_values = _simulate_clifford_ensemble(
            path_circuits, observable, n_qubits
        )
        weights = np.array([p.weight for p in paths])

        all_circuits = (cirq_circuit,) + tuple(path_circuits)

        context = {
            "classical_values": classical_values,
            "weights": weights,
            "target_idx": 0,
            "ensemble_start": 1,
            "n_rotations": len(rotations),
            "n_paths": len(paths),
        }
        return all_circuits, context

    @staticmethod
    def compute_eta(
        classical_values: np.ndarray,
        ensemble_noisy: np.ndarray,
        min_eta: float = 0.1,
    ) -> float | None:
        """Compute the rescaling factor η from noisy/ideal ratios.

        Returns ``None`` when η is below *min_eta* — the noise has
        destroyed the signal and the ``1/η`` correction would amplify
        noise rather than correct it.
        """
        valid = np.abs(classical_values) > 1e-12
        if not np.any(valid):
            return None
        eta = float(np.median(ensemble_noisy[valid] / classical_values[valid]))
        return eta if eta > min_eta else None

    def reduce(
        self,
        quantum_results: Sequence[float],
        context: QEMContext,
    ) -> float:
        d = context
        target_noisy = quantum_results[d["target_idx"]]
        ensemble_noisy = np.array(quantum_results[d["ensemble_start"] :], dtype=float)
        classical_values = d["classical_values"]
        weights = d["weights"]

        if d["n_rotations"] == 0:
            return float(weights @ classical_values)

        classical_est = float(weights @ classical_values)
        noisy_est = float(weights @ ensemble_noisy)

        eta = self.compute_eta(classical_values, ensemble_noisy)
        if eta is None:
            valid = np.abs(classical_values) > 1e-12
            if np.any(valid):
                context["_signal_destroyed"] = True
            return float(target_noisy)

        return classical_est + (target_noisy - noisy_est) / eta

    def post_reduce(self, contexts: Sequence[QEMContext]) -> None:
        destroyed = sum(1 for c in contexts if c.get("_signal_destroyed"))
        if destroyed:
            warnings.warn(
                f"QuEPP: signal destroyed for {destroyed}/{len(contexts)} "
                f"observable group(s) — mitigation fell back to noisy values. "
                f"Consider increasing shots or reducing noise.",
                stacklevel=3,
            )
