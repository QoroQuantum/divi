# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""How the bias and variance of QDrift depend on its sampling distribution
(StateDRIFT, arXiv:2610.31515), from Pauli expectation values of the initial
state."""

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import _require_qiskit_num_qubits
from divi.hamiltonians.trotterization import _pauli_bits as _bits

if TYPE_CHECKING:
    from divi.qprog.initial_states import InitialState


@dataclass(frozen=True)
class _Ladder:
    """``O``, ``L_H O = i[H, O]`` and what is read off them, ``L_H`` acting a
    term pair at a time.

    ``first`` holds ``L_H O`` term by term, one per anticommuting pair
    ``(P_j, O_m)``: term ``t`` comes from ``hamiltonian_term[t]`` and
    ``observable_term[t]``, and ``[P_j, o_m O_m] = pair_factors[t] · S_t``.
    """

    hamiltonian: _bits.PackedPaulis
    observable: _bits.PackedPaulis
    observable_values: npt.NDArray[np.floating]
    hamiltonian_term: npt.NDArray[np.intp]
    observable_term: npt.NDArray[np.intp]
    pair_factors: npt.NDArray[np.complexfloating]
    first: _bits.PackedPaulis
    first_values: npt.NDArray[np.floating]
    commutators_with_first: npt.NDArray[np.complexfloating]
    """``<[H, S_t]>`` for every term ``S_t`` of ``first``."""
    shifts: npt.NDArray[np.floating]
    slopes: npt.NDArray[np.floating]
    outer_curvatures: npt.NDArray[np.floating]
    slope_shifts: npt.NDArray[np.floating]

    @classmethod
    def climb(
        cls,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        source: _bits.Values,
    ) -> Self:
        h = _bits.pack(hamiltonian)
        o = _bits.pack(observable)
        n_terms = h.size
        observable_values = source.values(o.x, o.z)

        pairs = list(_bits.commutators(h, o))
        j = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros(0, np.intp)
        m = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros(0, np.intp)
        pair_factors = (
            np.concatenate([p[4] * o.coeffs[p[1]] for p in pairs])
            if pairs
            else np.zeros(0, complex)
        )
        first = _bits.PackedPaulis(
            np.concatenate([p[2] for p in pairs]) if pairs else h.x[:0],
            np.concatenate([p[3] for p in pairs]) if pairs else h.z[:0],
            1j * h.coeffs[j] * pair_factors,
        )
        first_values = source.values(first.x, first.z)

        # γ_j = <Δ_j O>, with Δ_j(O_m) = -2 O_m when P_j anticommutes with O_m.
        shifts = -2 * _bits.accumulate(j, o.coeffs[m] * observable_values[m], n_terms)
        # β_j = -i<[O, P_j]>.
        slopes = 1j * _bits.accumulate(j, pair_factors * first_values, n_terms)

        commutators_with_first = np.zeros(first.size, dtype=complex)
        with_first = np.zeros(n_terms, dtype=complex)
        for a, b, values in _bits.commutator_values(h, first, source):
            with_first += _bits.accumulate(a, first.coeffs[b] * values, n_terms)
            commutators_with_first += _bits.accumulate(
                b, h.coeffs[a] * values, first.size
            )

        return cls(
            h,
            o,
            observable_values,
            j,
            m,
            pair_factors,
            first,
            first_values,
            commutators_with_first,
            shifts.real,
            slopes.real,
            # φ_{j,H} = <[P_j, [O, H]]>, and [O, H] = i L_H O.
            (1j * with_first).real,
            _bits.shift_sums(h, first, first.coeffs * first_values).real,
        )


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
        simplified = observable.simplify()
        ladder = _bits.evaluate(
            initial_state,
            _require_qiskit_num_qubits(terms.num_qubits),
            lambda source: _Ladder.climb(terms, simplified, source),
        )
        return cls._from_ladder(hamiltonian, observable, initial_state, terms, ladder)

    @classmethod
    def _from_ladder(
        cls,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
        terms: SparsePauliOp,
        ladder: _Ladder,
    ) -> Self:
        return cls(
            hamiltonian,
            observable,
            initial_state,
            terms,
            ladder.shifts,
            ladder.slopes,
            # φ_HH = <[H, [O, H]]> = Σ_j c_j φ_{j,H}.
            float(np.dot(terms.coeffs.real, ladder.outer_curvatures)),
        )


@dataclass(frozen=True, eq=False)
class StateDriftExpansion:
    """The q-independent inputs of the bias ``B_N`` and variance ``V_N`` to
    ``dt⁴`` (Eqs. C13, C52, C63, C94), from which :meth:`truncated_bias` and
    :meth:`truncated_variance` evaluate them for any distribution. Per term
    ``P_j`` of ``densities.terms``, with ``L_H X = i[H, X]`` and
    ``Δ_j X = P_j X P_j - X``:
    """

    densities: StateDriftDensities
    inner_curvatures: npt.NDArray[np.floating]
    """``φ_{H,j} = <[H, [O, P_j]]>``."""
    outer_curvatures: npt.NDArray[np.floating]
    """``φ_{j,H} = <[P_j, [O, H]]>``."""
    shift_slopes: npt.NDArray[np.floating]
    """``β_H(Δ_j O) = <L_H Δ_j O>``."""
    shift_curvatures: npt.NDArray[np.floating]
    """``φ_{H,H}(Δ_j O) = <L_H² Δ_j O>``."""
    slope_shifts: npt.NDArray[np.floating]
    """``γ_j(L_H O) = <Δ_j L_H O>``."""
    slope_shift_slopes: npt.NDArray[np.floating]
    """``β_H(Δ_j L_H O) = <L_H Δ_j L_H O>``."""
    curvature_shifts: npt.NDArray[np.floating]
    """``γ_j(L_H² O) = <Δ_j L_H² O>``."""
    curvature_slopes: npt.NDArray[np.floating]
    """``β_j(L_H² O) = -i<[L_H² O, P_j]>``."""
    rotated_slope_slopes: npt.NDArray[np.floating]
    """``β_H(i[P_j, L_H O])``."""
    rotated_curvatures: npt.NDArray[np.floating]
    """``φ_{H,H}(i[P_j, O])``."""
    third_derivative: float
    """``<L_H³ O>``, the third time derivative of ``<O(t)>`` at 0."""
    fourth_derivative: float
    """``<L_H⁴ O>``, the fourth time derivative of ``<O(t)>`` at 0."""
    pair_terms: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    """``(j, m)`` for every term ``P_j`` anticommuting with observable term
    ``O_m``; pair ``t`` makes ``[P_j, o_m O_m] = f_t S_t``."""
    observable_weights: npt.NDArray[np.floating]
    """``o_m <O_m>`` per observable term."""
    pair_slopes: npt.NDArray[np.floating]
    """``i f_t <S_t>`` per pair, summing to ``β_j`` over the pairs of ``P_j``."""
    pair_slope_anticommutations: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    """``(k, t)`` for every term ``P_k`` anticommuting with a string ``S_t``
    whose pair slope is non-zero."""
    nested_curvatures: tuple[
        npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]
    ]
    """``(a, b, φ_{a,b})`` for every non-zero ``φ_{a,b} = <[P_a, [O, P_b]]>``."""

    @classmethod
    def compute(
        cls,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
    ) -> Self:
        """Evaluate the inputs from Pauli expectation values in ``initial_state``."""
        terms = hamiltonian.simplify()
        return _bits.evaluate(
            initial_state,
            _require_qiskit_num_qubits(terms.num_qubits),
            lambda source: cls._expand(
                hamiltonian, observable, initial_state, terms, source
            ),
        )

    @classmethod
    def _expand(
        cls,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
        terms: SparsePauliOp,
        source: _bits.Values,
    ) -> Self:
        ladder = _Ladder.climb(terms, observable.simplify(), source)
        h, o, first = ladder.hamiltonian, ladder.observable, ladder.first
        j, m, f = ladder.hamiltonian_term, ladder.observable_term, ladder.pair_factors
        n_terms = h.size

        # <L_H S_t> per term of L_H O, and its sums per observable term:
        # o_m <L_H O_m> and o_m <L_H² O_m>.
        first_slopes = 1j * ladder.commutators_with_first
        observable_slopes = _bits.accumulate(
            m, first.coeffs * ladder.first_values, o.size
        )
        observable_curvatures = _bits.accumulate(m, first.coeffs * first_slopes, o.size)
        # [O, P_b] = -Σ_{t from b} f_t S_t, so φ_{a,b} = -Σ f_t <[P_a, S_t]>,
        # and φ_{H,b} sums it over a.
        inner = -_bits.accumulate(j, f * ladder.commutators_with_first, n_terms)
        nested = [
            (a, j[b], -f[b] * values)
            for a, b, values in _bits.commutator_values(h, first, source)
        ]
        pair_slopes = (1j * f * ladder.first_values).real
        sloped = np.flatnonzero(pair_slopes)
        sloped_strings = _bits.PackedPaulis(
            first.x[sloped], first.z[sloped], first.coeffs[sloped]
        )
        anticommutations = list(_bits.anticommuting_pairs(h, sloped_strings))

        # L_H² O = i[H, L_H O], merged across term pairs.
        parts = [
            _bits.combine(x, z, 1j * h.coeffs[a] * first.coeffs[b] * factor)
            for a, b, x, z, factor in _bits.commutators(h, first)
        ]
        joined = _bits.concatenate(parts, h.x.shape[1])
        second = _bits.combine(joined.x, joined.z, joined.coeffs)
        second_values = source.values(second.x, second.z)
        # β_j(L_H² O) = i<[P_j, L_H² O]>; <L_H³ O> = Σ_j c_j β_j(L_H² O).
        curvature_slopes = np.zeros(n_terms, dtype=complex)
        for a, b, values in _bits.commutator_values(h, second, source):
            curvature_slopes += _bits.accumulate(
                a, 1j * second.coeffs[b] * values, n_terms
            )
        # <L_H² X> = -<[H, [H, X]]>.
        _, second_curvatures = _bits.double_commutator_sums(h, second, source)
        first_rotated, first_curvatures = _bits.double_commutator_sums(h, first, source)

        return cls(
            StateDriftDensities._from_ladder(
                hamiltonian, observable, initial_state, terms, ladder
            ),
            inner.real,
            ladder.outer_curvatures,
            (-2 * _bits.accumulate(j, observable_slopes[m], n_terms)).real,
            (-2 * _bits.accumulate(j, observable_curvatures[m], n_terms)).real,
            ladder.slope_shifts,
            _bits.shift_sums(h, first, first.coeffs * first_slopes).real,
            _bits.shift_sums(h, second, second.coeffs * second_values).real,
            curvature_slopes.real,
            # β_H(i[P_j, L_H O]) = -<[H, [P_j, L_H O]]>.
            (-first_rotated).real,
            # φ_HH(i[P_j, O]) = i Σ_{t from j} f_t φ_HH(S_t).
            _bits.accumulate(j, -1j * f * first_curvatures, n_terms).real,
            float(np.dot(h.coeffs, curvature_slopes).real),
            float(-np.dot(second.coeffs, second_curvatures).real),
            (j, m),
            (o.coeffs * ladder.observable_values).real,
            pair_slopes,
            (
                np.concatenate([a for a, _ in anticommutations]),
                sloped[np.concatenate([b for _, b in anticommutations])],
            )
            if anticommutations
            else (np.zeros(0, np.intp), np.zeros(0, np.intp)),
            _merge_pairs(nested, n_terms),
        )

    def truncated_bias(
        self, probs: npt.NDArray[np.floating], *, dt: float, n_samples: int
    ) -> tuple[float, float, float]:
        """``B_N`` truncated at ``dt²``, ``dt³`` and ``dt⁴`` (Eq. C52) for
        ``n_samples`` rotations drawn from ``probs`` (ordered as
        ``densities.terms``)."""
        values, _ = self._evaluate(self._bias_orders(dt, n_samples), probs, False)
        return values

    def truncated_variance(
        self, probs: npt.NDArray[np.floating], *, dt: float, n_samples: int
    ) -> tuple[float, float, float]:
        """``V_N`` of one sampled run truncated at ``dt²``, ``dt³`` and
        ``dt⁴`` (Eq. C94) for ``n_samples`` rotations drawn from ``probs``
        (ordered as ``densities.terms``)."""
        values, _ = self._evaluate(self._variance_orders(dt, n_samples), probs, False)
        return values

    def bias_with_gradient(
        self, probs: npt.NDArray[np.floating], *, dt: float, n_samples: int
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """``B_N`` to ``dt⁴`` and its gradient with respect to ``probs``."""
        values, gradient = self._evaluate(self._bias_orders(dt, n_samples), probs, True)
        return values[-1], gradient

    def variance_with_gradient(
        self, probs: npt.NDArray[np.floating], *, dt: float, n_samples: int
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """``V_N`` to ``dt⁴`` and its gradient with respect to ``probs``."""
        values, gradient = self._evaluate(
            self._variance_orders(dt, n_samples), probs, True
        )
        return values[-1], gradient

    def _bias_orders(self, dt: float, n: int) -> list["_Order"]:
        """The ``dt²``, ``dt³`` and ``dt⁴`` groups of Eq. (C52)."""
        d = self.densities
        gamma, third, fourth = (
            d.conjugation_shifts,
            self.third_derivative,
            self.fourth_derivative,
        )
        pairs = n * (n - 1) / 2
        nested = n * (n - 1) * (2 * n - 1) / 12
        triples = n * (n - 1) * (n - 2) / 6
        return [
            _Order({2: n * gamma}, {}, -n * d.curvature / 2, dt**2),
            _Order(
                {
                    3: -2 / 3 * n * d.rotation_slopes,
                    2: pairs * (self.shift_slopes + self.slope_shifts),
                },
                {},
                -n * third / 6 - pairs * third,
                dt**3,
            ),
            _Order(
                {
                    4: -n / 3 * gamma,
                    3: -2 / 3 * pairs * (self.inner_curvatures + self.outer_curvatures),
                    2: (nested - pairs / 2)
                    * (self.shift_curvatures + self.curvature_shifts)
                    + triples * self.slope_shift_slopes,
                },
                {"pair": pairs},
                fourth * (-n / 24 - pairs / 3 - nested - triples / 2 + pairs / 4),
                dt**4,
            ),
        ]

    def _variance_orders(self, dt: float, n: int) -> list["_Order"]:
        """The ``dt²``, ``dt³`` and ``dt⁴`` groups of Eq. (C94)."""
        d = self.densities
        c = d.terms.coeffs.real
        gamma, beta, phi_hh = d.conjugation_shifts, d.rotation_slopes, d.curvature
        inner, outer = self.inner_curvatures, self.outer_curvatures
        beta_h = float(np.dot(c, beta))
        pairs = n * (n - 1)
        triples = n * (n - 1) * (n - 2)
        slope_terms = self.slope_shifts + self.shift_slopes
        return [
            _Order({2: n * beta**2}, {}, -n * beta_h**2, dt**2),
            _Order(
                {
                    3: 2 * n * beta * gamma,
                    2: -2 * n * beta_h * gamma + pairs * beta * (inner + outer),
                },
                {},
                -2 * pairs * beta_h * phi_hh,
                dt**3,
            ),
            _Order(
                {
                    4: n * (gamma**2 - 4 / 3 * beta**2),
                    3: 4 / 3 * n * beta_h * beta
                    + pairs * (gamma * (inner + outer) + beta * slope_terms),
                    2: pairs * (-2 * beta_h * slope_terms - 2 * phi_hh * gamma)
                    + triples
                    / 3
                    * (
                        inner**2
                        + outer**2
                        + inner * outer
                        + beta
                        * (
                            self.curvature_slopes
                            + self.rotated_slope_slopes
                            + self.rotated_curvatures
                        )
                    ),
                },
                {"gamma_squared": -n, "nested": pairs / 2, "beta_cross": pairs},
                -pairs * phi_hh**2 / 2
                - triples * (beta_h * self.third_derivative + phi_hh**2),
                dt**4,
            ),
        ]

    def _evaluate(
        self,
        orders: list["_Order"],
        probs: npt.NDArray[np.floating],
        with_gradient: bool,
    ) -> tuple[tuple[float, float, float], npt.NDArray[np.floating]]:
        """The cumulative truncations of ``orders``, and the gradient of the
        last with respect to ``probs`` when asked for."""
        c = self.densities.terms.coeffs.real
        w = c**2 / probs
        totals = []
        running = 0.0
        gradient = np.zeros(len(c))
        for order in orders:
            value = order.constant
            for power, coefficients in order.linear.items():
                value += np.dot(c**power / probs ** (power - 1), coefficients)
                if with_gradient:
                    gradient -= (
                        order.scale
                        * (power - 1)
                        * c**power
                        / probs**power
                        * coefficients
                    )
            for name, weight in order.nonlinear.items():
                piece, piece_gradient = self._nonlinear(name, w, with_gradient)
                value += weight * piece
                if with_gradient:
                    gradient -= order.scale * weight * piece_gradient * c**2 / probs**2
            running += order.scale * value
            totals.append(float(running))
        return (totals[0], totals[1], totals[2]), gradient

    def _nonlinear(
        self, name: str, w: npt.NDArray[np.floating], with_gradient: bool
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """A term of ``B_N`` or ``V_N`` that is not linear in ``w_j = c_j² / q_j``,
        and its gradient with respect to ``w``."""
        gamma = self.densities.conjugation_shifts
        beta = self.densities.rotation_slopes
        j, m = self.pair_terms
        if name == "gamma_squared":
            b_gamma = np.dot(w, gamma)
            return b_gamma**2, 2 * b_gamma * gamma
        if name == "nested":
            a, b, phi = self.nested_curvatures
            squares = phi**2
            gradient = (
                np.bincount(a, w[b] * squares, len(w))
                + np.bincount(b, w[a] * squares, len(w))
                if with_gradient
                else np.zeros(0)
            )
            return float(np.dot(w[a] * w[b], squares)), gradient
        # Both remaining terms go through the pair sums A_m = Σ_{(j, m)} w_j.
        weights = self.observable_weights
        pair_sums = np.bincount(m, w[j], len(weights))
        if name == "pair":
            # Σ_{j,k} w_j w_k γ_{j,k} = 4 Σ_m o_m <O_m> A_m².
            gradient = (
                np.bincount(j, 8 * weights[m] * pair_sums[m], len(w))
                if with_gradient
                else np.zeros(0)
            )
            return float(4 * np.dot(weights, pair_sums**2)), gradient
        # Σ_j w_j β_j (β_j(Δ_q O) + γ_q(i[P_j, O])) = -2 Σ_t s_t (A_{m_t} + K_t),
        # with s_t = w_{j_t} β_{j_t} p_t and K_t = Σ_{k anticommuting with S_t} w_k.
        k, t = self.pair_slope_anticommutations
        slopes = self.pair_slopes
        crossings = np.bincount(t, w[k], len(slopes))
        s = w[j] * beta[j] * slopes
        value = -2 * float(np.dot(s, pair_sums[m] + crossings))
        if not with_gradient:
            return value, np.zeros(0)
        gradient = -2 * (
            np.bincount(j, beta[j] * slopes * (pair_sums[m] + crossings), len(w))
            + np.bincount(j, np.bincount(m, s, len(weights))[m], len(w))
            + np.bincount(k, s[t], len(w))
        )
        return value, gradient


@dataclass(frozen=True)
class _Order:
    """One ``dt^p`` group of ``B_N`` or ``V_N``: ``scale`` times
    ``Σ_p Σ_j (c_j^p / q_j^{p-1}) linear[p][j]`` plus the named non-linear
    terms with their weights, plus ``constant``."""

    linear: dict[int, npt.NDArray[np.floating]]
    nonlinear: dict[str, float]
    constant: float
    scale: float


def _merge_pairs(
    entries: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]], size: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """Sum the values of repeated ``(a, b)`` and drop those that cancel."""
    a = np.concatenate([e[0] for e in entries] or [np.zeros(0, np.intp)])
    if a.size == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0)
    b = np.concatenate([e[1] for e in entries])
    values = np.concatenate([e[2] for e in entries]).real
    pairs, inverse = np.unique(a * size + b, return_inverse=True)
    summed = np.bincount(inverse, values, len(pairs))
    keep = np.abs(summed) > 1e-12 * np.abs(summed).max()
    return pairs[keep] // size, pairs[keep] % size, summed[keep]


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
    """Thread-safe store of :class:`StateDriftExpansion`, shared by every copy
    of the strategy that owns it so each evolution is evaluated once. Entries
    are kept for as long as the strategy or any of its copies exists."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: list[tuple[_Evolution, Future[StateDriftExpansion]]] = []

    def __deepcopy__(self, memo: dict) -> Self:
        return self

    def __copy__(self) -> Self:
        return self

    def __reduce__(self):
        return (type(self), ())

    def expansion(
        self,
        hamiltonian: SparsePauliOp,
        observable: SparsePauliOp,
        initial_state: "InitialState",
    ) -> StateDriftExpansion:
        """The stored expansion of this evolution, computing it on first use.

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
                StateDriftExpansion.compute(hamiltonian, observable, initial_state)
            )
        except BaseException as error:
            with self._lock:
                self._entries.remove(owned)
            pending.set_exception(error)
            raise
        return pending.result()
