# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""StateDRIFT term-sampling probabilities for QDrift (arXiv:2610.31515)."""

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Self, cast
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliList, SparsePauliOp

from divi.hamiltonians._term_ops import _require_qiskit_num_qubits

if TYPE_CHECKING:
    from divi.qprog.initial_states import InitialState

# Least share τ of the vanilla (|c_j| / λ) distribution mixed into every term.
_VANILLA_WEIGHT = 0.2
# Mixing exponents of the α scan, relative to the γ/β² scale of the problem.
_ALPHA_EXPONENTS = np.linspace(-4.0, 4.0, 161)
# Largest rotation angle |c_j|·dt/q_j a distribution may give a term (θ_max, Eq. F10).
_MAX_ROTATION_ANGLE = 0.5
_ANTICOMMUTATION_CHUNK = 4096
_PAIR_BUDGET = 1 << 18
# Commutator terms below this fraction of the largest one are rounding residue.
_RELATIVE_ATOL = 1e-12


def _anticommuting_products(
    left: SparsePauliOp, right: SparsePauliOp
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], SparsePauliOp | None]:
    """Anticommuting term pairs ``(a, b)`` and their products ``left_a · right_b``.

    The products keep one term per pair, coefficients included; ``None`` when
    no pair anticommutes.
    """
    left_x = left.paulis.x.astype(np.int64)
    left_z = left.paulis.z.astype(np.int64)
    right_x = right.paulis.x.astype(np.int64)
    right_z = right.paulis.z.astype(np.int64)
    rows, cols = [], []
    for start in range(0, right.size, _ANTICOMMUTATION_CHUNK):
        stop = start + _ANTICOMMUTATION_CHUNK
        parity = (left_x @ right_z[start:stop].T + left_z @ right_x[start:stop].T) & 1
        a, b = np.nonzero(parity)
        rows.append(a)
        cols.append(b + start)
    a = np.concatenate(rows) if rows else np.zeros(0, dtype=np.intp)
    b = np.concatenate(cols) if cols else np.zeros(0, dtype=np.intp)
    if a.size == 0:
        return a, b, None
    products = SparsePauliOp(
        cast(PauliList, left.paulis[a]).dot(cast(PauliList, right.paulis[b])),
        left.coeffs[a] * right.coeffs[b],
    )
    return a, b, products


def _commutator(left: SparsePauliOp, right: SparsePauliOp) -> SparsePauliOp | None:
    """``[left, right]``, or ``None`` when every term pair commutes."""
    _, _, products = _anticommuting_products(left, right)
    if products is None:
        return None
    doubled = 2 * products
    return doubled.simplify(atol=_RELATIVE_ATOL * np.abs(doubled.coeffs).max())


def _expectation_of_commutator(
    left: SparsePauliOp,
    right: SparsePauliOp,
    initial_state: "InitialState",
    n_qubits: int,
) -> float:
    """``<[left, right]>`` in ``initial_state``, built and evaluated a block of
    ``left``'s terms at a time so that at most about ``_PAIR_BUDGET`` products
    are held at once."""
    block = max(1, _PAIR_BUDGET // right.size)
    total = 0.0
    for start in range(0, left.size, block):
        part = _commutator(left[start : start + block], right)
        if part is not None:
            values = initial_state.pauli_expectations(part.paulis, n_qubits)
            total += float((part.coeffs * values).sum().real)
    return total


@dataclass(frozen=True, eq=False)
class StateDriftDensities:
    """How ``<O>`` in ``initial_state`` responds to each term of ``hamiltonian``.

    Independent of time and step count, so one instance serves every time
    point of a trajectory. Per term ``P_j`` of ``terms`` (``hamiltonian``
    simplified), the shift is ``<P_j O P_j> - <O>`` and the slope
    ``-i<[O, P_j]>``; ``curvature`` is ``<[H, [O, H]]>``, the second time
    derivative of ``<O(t)>`` at 0.
    """

    hamiltonian: SparsePauliOp
    observable: SparsePauliOp
    initial_state: "InitialState"
    terms: SparsePauliOp
    conjugation_shifts: npt.NDArray[np.floating]
    rotation_slopes: npt.NDArray[np.floating]
    curvature: float

    @classmethod
    def compute(
        cls,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
    ) -> Self:
        """Evaluate the densities from Pauli expectation values in ``initial_state``."""
        terms = hamiltonian.simplify()
        simple_observable = observable.simplify()
        n_qubits = _require_qiskit_num_qubits(terms.num_qubits)
        obs_idx, term_idx, slope_products = _anticommuting_products(
            simple_observable, SparsePauliOp(terms.paulis)
        )
        first = _commutator(simple_observable, terms)

        parts = [simple_observable]
        if slope_products is not None:
            parts.append(slope_products)
        stacked = SparsePauliOp.sum(parts)
        values = initial_state.pauli_expectations(stacked.paulis, n_qubits)
        obs_values = values[: simple_observable.size]

        shifts = -2 * np.bincount(
            term_idx,
            weights=simple_observable.coeffs.real[obs_idx] * obs_values[obs_idx],
            minlength=terms.size,
        )
        slopes = np.zeros(terms.size)
        if slope_products is not None:
            slope_terms = (stacked.coeffs * values)[simple_observable.size :]
            slopes = np.bincount(
                term_idx, weights=(-2j * slope_terms).real, minlength=terms.size
            )
        curvature = (
            0.0
            if first is None
            else _expectation_of_commutator(terms, first, initial_state, n_qubits)
        )
        return cls(
            hamiltonian, observable, initial_state, terms, shifts, slopes, curvature
        )


# Hamiltonian, observable and the circuit that prepares the initial state.
_Evolution = tuple[SparsePauliOp, SparsePauliOp, QuantumCircuit]


def _same_operator(a: SparsePauliOp, b: SparsePauliOp) -> bool:
    """The same Pauli strings with exactly the same coefficients, in order."""
    return a.paulis.to_labels() == b.paulis.to_labels() and np.array_equal(
        a.coeffs, b.coeffs
    )


def _same_evolution(a: _Evolution, b: _Evolution) -> bool:
    return _same_operator(a[0], b[0]) and _same_operator(a[1], b[1]) and a[2] == b[2]


class StateDriftCache:
    """Thread-safe store of :class:`StateDriftDensities`, shared by every copy
    of the strategy that owns it so each evolution is evaluated once. Entries
    are kept for as long as the strategy or any of its copies exists."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: list[tuple[_Evolution, Future[StateDriftDensities]]] = []

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    def __copy__(self) -> Self:
        return self

    def __reduce__(self):
        return (type(self), ())

    def densities(
        self,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
    ) -> StateDriftDensities:
        """The stored densities of this evolution, computing them on first use.

        The computation runs outside the lock, so other evolutions are not
        held up; a thread asking for an evolution already being computed
        waits for that result instead of computing it again.
        """
        evolution = (
            hamiltonian,
            observable,
            initial_state.build(
                range(_require_qiskit_num_qubits(hamiltonian.num_qubits))
            ),
        )
        with self._lock:
            pending = next(
                (
                    future
                    for stored, future in self._entries
                    if _same_evolution(stored, evolution)
                ),
                None,
            )
            if pending is not None:
                owned = None
            else:
                pending = Future()
                owned = (evolution, pending)
                self._entries.append(owned)
        if owned is None:
            return pending.result()
        try:
            pending.set_result(
                StateDriftDensities.compute(hamiltonian, observable, initial_state)
            )
        except BaseException as error:
            with self._lock:
                self._entries.remove(owned)
            pending.set_exception(error)
            raise
        return pending.result()


def _mean_squared_error(
    probs: npt.NDArray[np.floating],
    densities: StateDriftDensities,
    *,
    dt: float,
    n_samples: int,
    n_trajectories: int,
) -> float:
    """Single-step bias² + variance / M of the sampled estimator to O(dt⁴)
    (Eqs. 16, 17 and 31 without the cross-step terms)."""
    coeffs = densities.terms.coeffs.real
    shifts = densities.conjugation_shifts
    slopes = densities.rotation_slopes
    ratio = coeffs / probs
    weight = coeffs * ratio
    leading_bias = np.sum(weight * shifts)
    mean_slope = np.sum(coeffs * slopes)
    var_2 = np.sum(weight * slopes**2) - mean_slope**2
    var_3 = 2 * (np.sum(weight * ratio * slopes * shifts) - mean_slope * leading_bias)
    var_4 = np.sum(weight * ratio**2 * shifts**2) - leading_bias**2
    var_4 -= (4 / 3) * (
        np.sum(weight * ratio**2 * slopes**2)
        - mean_slope * np.sum(weight * ratio * slopes)
    )
    bias = n_samples * (leading_bias - densities.curvature / 2) * dt**2
    variance = n_samples * (var_2 * dt**2 + var_3 * dt**3 + var_4 * dt**4)
    return float(bias**2 + variance / n_trajectories)


def _mix_with_vanilla(
    spread: npt.NDArray[np.floating],
    vanilla: npt.NDArray[np.floating],
    floors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating] | None:
    """``τ·vanilla + (1 − τ)·spread`` at the smallest ``τ ≥ 0.2`` that lifts
    every probability to its floor, or ``None`` when ``τ = 1`` does not."""
    gain = vanilla - spread
    shortfall = floors - spread
    short = shortfall > 0
    if np.any(short & (gain <= 0)):
        return None
    tau = max(_VANILLA_WEIGHT, np.max(shortfall[short] / gain[short], initial=0.0))
    if tau > 1:
        return None
    return tau * vanilla + (1 - tau) * spread


def state_drift_distribution(
    densities: StateDriftDensities,
    *,
    time: float,
    n_samples: int,
    n_trajectories: int,
) -> dict[str, float]:
    """StateDRIFT probabilities for every term of ``densities.terms``.

    ``q_j ∝ |c_j| sqrt(|γ_j| + α β_j²)`` (Eq. 29), mixed with the vanilla
    distribution at weight τ (Eq. 23), with α picked from a scan by the
    estimated mean-squared error of ``n_trajectories`` averaged runs of
    ``n_samples`` sampled rotations each. τ is the smallest value in
    ``[0.2, 1]`` that keeps every rotation angle ``|c_j| dt / q_j`` within 0.5
    (Eq. F10).
    """
    abs_coeffs = np.abs(densities.terms.coeffs.real)
    vanilla = abs_coeffs / abs_coeffs.sum()
    dt = time / n_samples
    floors = abs_coeffs * abs(dt) / _MAX_ROTATION_ANGLE

    abs_shifts = np.abs(densities.conjugation_shifts)
    squared_slopes = densities.rotation_slopes**2
    shift_scale = np.sum(abs_coeffs * abs_shifts)
    slope_scale = np.sum(abs_coeffs * squared_slopes)
    alpha_scale = (
        shift_scale / slope_scale if shift_scale > 0 and slope_scale > 0 else 1.0
    )
    best_probs, best_error = None, np.inf
    informative = False
    for alpha in (0.0, *(alpha_scale * 10.0**_ALPHA_EXPONENTS)):
        weights = abs_coeffs * np.sqrt(abs_shifts + alpha * squared_slopes)
        if weights.sum() == 0:
            continue
        informative = True
        probs = _mix_with_vanilla(weights / weights.sum(), vanilla, floors)
        if probs is None:
            continue
        error = _mean_squared_error(
            probs,
            densities,
            dt=dt,
            n_samples=n_samples,
            n_trajectories=n_trajectories,
        )
        if error < best_error:
            best_probs, best_error = probs, error

    if best_probs is None:
        reason = (
            f"no distribution keeps every rotation angle |c_j|·dt/q_j within "
            f"{_MAX_ROTATION_ANGLE} (dt = time / (n_steps · sampling_budget)); "
            "increase n_steps or sampling_budget"
            if informative
            else "no Hamiltonian term changes the observable in the initial state"
        )
        warn(
            f"StateDRIFT falls back to weighted sampling: {reason}.",
            UserWarning,
            stacklevel=2,
        )
        best_probs = vanilla
    best_probs = best_probs / best_probs.sum()
    return dict(zip(densities.terms.paulis.to_labels(), best_probs.tolist()))
