# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""ADAPT-VQE: a VQE whose ansatz grows one pool operator at a time.

Grimsley et al., *Nat. Commun.* **10**, 3007 (2019), arXiv:1812.11173. Pools of
qubit excitations (Yordanov et al., arXiv:2011.10540), Pauli words (Tang et al.,
arXiv:1911.10205) and coupled exchange operators with their circuits (Ramôa et
al., arXiv:2407.08696); disjoint-support growth after TETRIS-ADAPT-VQE
(Anastasiou et al., arXiv:2209.10562); inverse-Hessian recycling after Ramôa et
al.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import (
    combinations,
    combinations_with_replacement,
    permutations,
    product,
)
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeVar
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, commutator
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import norm

from divi.pipeline import CircuitPreprocessor, ResultFormat
from divi.pipeline._grouping import _compute_measurement_groups
from divi.pipeline._shot_distribution import _weighted_distribution
from divi.qprog._early_stopping import EarlyStopping, StopReason
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.reporting._events import ProgressEvent, TerminalStatus

from ._ansatze import (
    _DOUBLE_EXCITATION_TERMS,
    _ODD_Y_ROTATIONS,
    ElectronCounts,
    ReferenceStateAnsatz,
    _emit_double_excitation,
    _emit_odd_y_evolution,
    _emit_pauli_rotation,
    _emit_single_excitation,
    _enumerate_excitations,
    _pair_support,
)
from ._vqe import VQE

__all__ = [
    "AdaptAnsatz",
    "AdaptVQE",
    "CEOPool",
    "CoupledExchange",
    "OperatorPool",
    "PauliPool",
    "PauliRotation",
    "PoolOperator",
    "QubitExcitation",
    "QubitExcitationPool",
]


class PoolOperator(ABC):
    """A pool operator ``exp(-i sum_j theta_j G_j)`` with commuting generators."""

    @property
    @abstractmethod
    def qubits(self) -> tuple[int, ...]:
        """The qubits the operator acts on."""

    @property
    def n_params(self) -> int:
        """Number of parameters, one per generator."""
        return 1

    @property
    def frequencies(self) -> tuple[tuple[float, int], ...]:
        """Each parameter's ``(omega, order)``: ``{1/2, 1}`` for generator
        eigenvalues ``{0, +-1/2}``."""
        return ((0.5, 2),) * self.n_params

    @abstractmethod
    def build_generators(self, n_qubits: int) -> tuple[SparsePauliOp, ...]:
        """The Hermitian ``G_j``, one per parameter, on ``n_qubits`` qubits."""

    @abstractmethod
    def emit(self, qc: QuantumCircuit, params: Sequence) -> None:
        """Append ``exp(-i sum_j params[j] G_j)`` to ``qc``."""


@dataclass(frozen=True)
class QubitExcitation(PoolOperator):
    """Givens rotation exchanging the occupations of ``source`` and ``target``.

    A fermionic excitation without its Jordan-Wigner parity string.
    """

    source: tuple[int, ...]
    target: tuple[int, ...]

    def __post_init__(self):
        if len(self.source) not in (1, 2) or len(self.target) != len(self.source):
            raise ValueError(
                "A qubit excitation moves one or two qubits onto as many; got "
                f"source={self.source}, target={self.target}."
            )

    @property
    def qubits(self) -> tuple[int, ...]:
        return self.source + self.target

    def build_generators(self, n_qubits: int) -> tuple[SparsePauliOp, ...]:
        if len(self.source) == 1:
            wires = [*self.target, *self.source]
            terms = [("XY", wires, 0.25), ("YX", wires, -0.25)]
        else:
            wires = [*reversed(self.target), *reversed(self.source)]
            terms = [
                (letters, wires, sign / 16)
                for letters, sign in _DOUBLE_EXCITATION_TERMS
            ]
        return (SparsePauliOp.from_sparse_list(terms, n_qubits),)

    def emit(self, qc: QuantumCircuit, params: Sequence) -> None:
        (theta,) = params
        if len(self.source) == 1:
            _emit_single_excitation(qc, theta, *self.qubits)
        else:
            _emit_double_excitation(qc, theta, *self.qubits)


@dataclass(frozen=True)
class PauliRotation(PoolOperator):
    """``exp(-i theta/2 P)`` for the Pauli word ``letters`` on ``wires``."""

    wires: tuple[int, ...]
    letters: str

    @property
    def qubits(self) -> tuple[int, ...]:
        return self.wires

    @property
    def frequencies(self) -> tuple[tuple[float, int], ...]:
        """``(1, 1)``: the generator's eigenvalues are ``+-1/2``."""
        return ((1.0, 1),)

    def build_generators(self, n_qubits: int) -> tuple[SparsePauliOp, ...]:
        return (
            SparsePauliOp.from_sparse_list(
                [(self.letters, list(self.wires), 0.5)], n_qubits
            ),
        )

    def emit(self, qc: QuantumCircuit, params: Sequence) -> None:
        (theta,) = params
        _emit_pauli_rotation(
            qc, theta, dict(zip(self.wires, self.letters, strict=True))
        )


#: Generator of :func:`_build_coupled_sum` per unit angle, letter ``k`` on wire ``k``.
_COUPLED_SUM_TERMS = {"XYXX": 0.25, "XYYY": 0.25, "YXXX": -0.25, "YXYY": -0.25}


def _read_words(generator: SparsePauliOp, qubits: Sequence[int]) -> dict[str, float]:
    """``generator``'s real coefficient per word on ``qubits``, letter ``k`` on
    ``qubits[k]``."""
    return {
        "".join(label[::-1][q] for q in qubits): coeff.real
        for label, coeff in generator.to_list()
    }


def _match_coupled_sum(
    words: Mapping[str, float],
) -> tuple[tuple[int, ...], float] | None:
    """A wiring ``perm`` (wire ``w`` on qubit ``perm[w]``) and scale ``r`` with
    ``words`` equal to ``r`` times the rewired :data:`_COUPLED_SUM_TERMS`."""
    for perm in permutations(range(4)):
        rewired = {
            "".join(word[perm.index(q)] for q in range(4)): coeff
            for word, coeff in _COUPLED_SUM_TERMS.items()
        }
        if rewired.keys() != words.keys():
            continue
        word, coeff = next(iter(rewired.items()))
        scale = words[word] / coeff
        if all(np.isclose(words[w], scale * c) for w, c in rewired.items()):
            return perm, scale
    return None


def _build_coupled_sum(perm: Sequence[int], scale: float, theta) -> QuantumCircuit:
    """``exp(-i scale theta G)`` for the rewired :data:`_COUPLED_SUM_TERMS`
    (Ramôa et al., arXiv:2407.08696, Fig. 9)."""
    q = list(perm)
    t = scale * theta
    qc = QuantumCircuit(4)
    qc.cx(q[0], q[1])
    qc.cx(q[2], q[3])
    qc.cx(q[0], q[2])
    qc.ry(-t / 2, q[0])
    qc.h(q[1])
    qc.h(q[3])
    qc.cx(q[0], q[3])
    qc.ry(t / 2, q[0])
    qc.cx(q[0], q[1])
    qc.ry(-t / 2, q[0])
    qc.cx(q[0], q[3])
    qc.ry(t / 2, q[0])
    qc.ry(-np.pi / 2, q[1])
    qc.h(q[3])
    qc.cx(q[0], q[2])
    qc.s(q[0])
    qc.sdg(q[1])
    qc.cx(q[0], q[1])
    qc.cx(q[2], q[3])
    qc.sdg(q[1])
    return qc


@dataclass(frozen=True)
class CoupledExchange(PoolOperator):
    """Double qubit excitations on one set of four qubits, applied as one
    exponential.

    With ``signs``, one parameter scales ``sum_k signs[k] G_k`` (circuit of Fig.
    9 of Ramôa et al.); without, each excitation takes its own parameter (their
    Fig. 4).
    """

    excitations: tuple[QubitExcitation, ...]
    signs: tuple[int, ...] | None = None

    def __post_init__(self):
        if any(len(e.source) != 2 for e in self.excitations) or len(self.qubits) != 4:
            raise ValueError(
                "A coupled exchange couples double excitations on one set of four "
                "qubits."
            )
        pairings = {
            frozenset((frozenset(e.source), frozenset(e.target)))
            for e in self.excitations
        }
        if len(pairings) != len(self.excitations):
            raise ValueError(
                "A coupled exchange takes each pairing of its four qubits at most once."
            )

    @property
    def qubits(self) -> tuple[int, ...]:
        return tuple(sorted(set().union(*(e.qubits for e in self.excitations))))

    @property
    def n_params(self) -> int:
        return 1 if self.signs is not None else len(self.excitations)

    def build_generators(self, n_qubits: int) -> tuple[SparsePauliOp, ...]:
        parts = [e.build_generators(n_qubits)[0] for e in self.excitations]
        if self.signs is None:
            return tuple(parts)
        return (
            SparsePauliOp.sum(
                [sign * part for sign, part in zip(self.signs, parts, strict=True)]
            ).simplify(),
        )

    @cached_property
    def _template(self) -> tuple[QuantumCircuit, ParameterVector]:
        """The evolution on the support's four local qubits, built once."""
        generators = self.build_generators(max(self.qubits) + 1)
        thetas = ParameterVector("theta", len(generators))
        words = [_read_words(generator, self.qubits) for generator in generators]
        if self.signs is not None and (match := _match_coupled_sum(words[0])):
            return _build_coupled_sum(*match, thetas[0]), thetas
        qc = QuantumCircuit(4)
        _emit_odd_y_evolution(
            qc,
            {
                word: sum(theta * w.get(word, 0.0) for theta, w in zip(thetas, words))
                for word, _ in _ODD_Y_ROTATIONS
            },
            range(4),
        )
        return qc, thetas

    def emit(self, qc: QuantumCircuit, params: Sequence) -> None:
        circuit, thetas = self._template
        qc.compose(
            circuit.assign_parameters(dict(zip(thetas, params, strict=True))),
            qubits=list(self.qubits),
            inplace=True,
        )


class OperatorPool(ABC):
    """The operators ADAPT-VQE screens each step and grows its ansatz from."""

    @abstractmethod
    def build_operators(
        self, n_qubits: int, occupied: Sequence[int]
    ) -> tuple[PoolOperator, ...]:
        """Every operator whose energy gradient is measured, each with one
        parameter, for the reference occupying ``occupied``."""

    def mark_candidates(
        self, operators: Sequence[PoolOperator]
    ) -> npt.NDArray[np.bool_]:
        """Which of ``operators`` may be appended; all by default. The result
        must depend on ``operators`` alone."""
        return np.ones(len(operators), dtype=bool)

    def resolve(
        self,
        index: int,
        operators: Sequence[PoolOperator],
        active: npt.NDArray[np.bool_],
    ) -> PoolOperator:
        """The operator to append when candidate ``index`` is chosen, given
        whether each operator's gradient counts as nonzero (``active``)."""
        return operators[index]


def _build_qubit_excitations(
    n_qubits: int, occupied: Sequence[int], generalised: bool
) -> list[QubitExcitation]:
    singles, doubles = _enumerate_excitations(occupied, n_qubits, generalised)
    return [QubitExcitation((i,), (a,)) for i, a in singles] + [
        QubitExcitation((i, j), (a, b)) for i, j, a, b in doubles
    ]


@dataclass(frozen=True)
class QubitExcitationPool(OperatorPool):
    """Spin-conserving single and double qubit excitations (QEB-ADAPT-VQE).

    Args:
        generalised: Exchange any qubits rather than taking occupied qubits of
            the reference to virtual ones.
    """

    generalised: bool = False

    def build_operators(
        self, n_qubits: int, occupied: Sequence[int]
    ) -> tuple[PoolOperator, ...]:
        return tuple(_build_qubit_excitations(n_qubits, occupied, self.generalised))


@dataclass(frozen=True)
class PauliPool(OperatorPool):
    """Each Pauli word of the qubit excitations as its own rotation
    (qubit-ADAPT-VQE): every word has an odd number of ``Y``.

    Args:
        generalised: Take the words of generalised excitations.
    """

    generalised: bool = False

    def build_operators(
        self, n_qubits: int, occupied: Sequence[int]
    ) -> tuple[PoolOperator, ...]:
        words = dict.fromkeys(
            (tuple(qubits), letters)
            for excitation in _build_qubit_excitations(
                n_qubits, occupied, self.generalised
            )
            for generator in excitation.build_generators(n_qubits)
            for letters, qubits, _ in generator.to_sparse_list()
        )
        return tuple(PauliRotation(qubits, letters) for qubits, letters in words)


@dataclass(frozen=True)
class CEOPool(OperatorPool):
    """Coupled exchange operators (CEO-ADAPT-VQE).

    Each double's four qubits carry every spin-conserving qubit excitation that
    exchanges two of them with the other two. The candidates are the singles and
    the sums and differences of two excitations on one support, each under a
    single parameter. A chosen sum or difference becomes one parameter per
    excitation on its support whose gradient counts as nonzero, when more than
    one does.

    Args:
        generalised: Take supports and singles from generalised excitations.
    """

    generalised: bool = False

    def build_operators(
        self, n_qubits: int, occupied: Sequence[int]
    ) -> tuple[PoolOperator, ...]:
        singles, doubles = _enumerate_excitations(occupied, n_qubits, self.generalised)
        coupled: list[PoolOperator] = []
        excitations: list[PoolOperator] = []
        for support in dict.fromkeys(tuple(sorted(double)) for double in doubles):
            on_support = [
                QubitExcitation((i, j), (a, b)) for i, j, a, b in _pair_support(support)
            ]
            excitations.extend(on_support)
            coupled.extend(
                CoupledExchange((first, second), (1, sign))
                for first, second in combinations(on_support, 2)
                for sign in (1, -1)
            )
        return (
            *(QubitExcitation((i,), (a,)) for i, a in singles),
            *coupled,
            *excitations,
        )

    def mark_candidates(
        self, operators: Sequence[PoolOperator]
    ) -> npt.NDArray[np.bool_]:
        return np.array(
            [
                not (isinstance(op, QubitExcitation) and len(op.source) == 2)
                for op in operators
            ],
            dtype=bool,
        )

    def resolve(
        self,
        index: int,
        operators: Sequence[PoolOperator],
        active: npt.NDArray[np.bool_],
    ) -> PoolOperator:
        chosen = operators[index]
        if not isinstance(chosen, CoupledExchange):
            return chosen
        support = set(chosen.qubits)
        excitations = tuple(
            op
            for op, is_active in zip(operators, active, strict=True)
            if is_active
            and isinstance(op, QubitExcitation)
            and set(op.qubits) == support
        )
        return chosen if len(excitations) <= 1 else CoupledExchange(excitations)


#: Shots each measurement group gets at least when screening or measuring a
#: landscape on a sampling backend.
_MIN_GROUP_SHOTS = 100

#: Gaussian draws of the landscape coefficients behind a gain's standard error.
_N_GAIN_DRAWS = 200

#: Grid points per angle when searching a landscape, by number of parameters.
_SEARCH_POINTS = {1: 2001, 2: 81, 3: 25}

#: Grid points per angle when re-minimising each draw, by number of parameters.
_DRAW_POINTS = {1: 401, 2: 41, 3: 17}

#: Half-width and points per angle of the extra grid around the lowest angles
#: when re-minimising each draw.
_LOCAL_SPAN, _LOCAL_POINTS = 0.2, 9

_Chosen = TypeVar("_Chosen")


def _distribute_with_floor(group_norms: list[float], total_shots: int) -> list[int]:
    """At least :data:`_MIN_GROUP_SHOTS` per group, the rest by group L1 norm."""
    floor = min(_MIN_GROUP_SHOTS, total_shots // len(group_norms))
    rest = _weighted_distribution(group_norms, total_shots - floor * len(group_norms))
    return [floor + shots for shots in rest]


def _detect_cycle(picks: Sequence[Hashable]) -> bool:
    """Whether ``picks`` ends with a block of picks repeated back to back."""
    return any(
        picks[-size:] == picks[-2 * size : -size]
        for size in range(1, len(picks) // 2 + 1)
    )


def _add_disjoint(
    chosen: list[_Chosen],
    indices: Iterable[int],
    operators: Sequence[PoolOperator],
    resolve: Callable[[int], _Chosen],
    operator_of: Callable[[_Chosen], PoolOperator],
) -> list[_Chosen]:
    """Append, in order, each resolved candidate whose qubits are disjoint from
    every operator already chosen."""
    used = set().union(*(operator_of(item).qubits for item in chosen))
    for index in indices:
        if not used.isdisjoint(operators[index].qubits):
            continue
        candidate = resolve(index)
        if used.isdisjoint(operator_of(candidate).qubits):
            chosen.append(candidate)
            used.update(operator_of(candidate).qubits)
    return chosen


def _is_zero(op: SparsePauliOp) -> bool:
    return bool(np.allclose(op.simplify(atol=1e-12).coeffs, 0.0, atol=1e-10))


def _build_landscape(
    operator: PoolOperator, hamiltonian: SparsePauliOp, n_qubits: int
) -> tuple[tuple[SparsePauliOp, ...], tuple[tuple[int, int], ...]]:
    """Observables ``Q_q`` and amplitude pairs ``(m, n)`` such that appending
    ``operator`` at angles ``theta`` gives energy ``sum_q a_m a_n <Q_q>``, the
    expectations taken without it.

    With ``B_k = 2 G_k``, ``B_k^3 = B_k`` and ``B_k B_l = 0`` make the evolution
    ``I + sum_k [(cos(theta_k/2) - 1) B_k^2 - i sin(theta_k/2) B_k]``; amplitude
    ``0`` is ``1``, ``2k + 1`` is ``cos(theta_k/2) - 1`` and ``2k + 2`` is
    ``sin(theta_k/2)``.

    Raises:
        ValueError: If the generators break either identity.
    """
    halves = [(2 * g).simplify() for g in operator.build_generators(n_qubits)]
    for k, half in enumerate(halves):
        if not _is_zero(half @ half @ half - half) or not all(
            _is_zero(half @ other) for other in halves[k + 1 :]
        ):
            raise ValueError(
                f"{operator!r} needs generators G with (2G)^3 = 2G that annihilate "
                "each other pairwise."
            )
    terms = [SparsePauliOp("I" * n_qubits)]
    for half in halves:
        terms += [(half @ half).simplify(), half]

    def is_rotation(index: int) -> bool:
        return index > 0 and index % 2 == 0

    observables, pairs = [], []
    for m, n in combinations_with_replacement(range(len(terms)), 2):
        left = terms[m] @ hamiltonian @ terms[n]
        right = terms[n] @ hamiltonian @ terms[m]
        if m == n:
            term = left
        elif is_rotation(m) == is_rotation(n):
            term = left + right
        elif is_rotation(n):
            term = 1j * (right - left)
        else:
            term = 1j * (left - right)
        term = term.simplify(atol=1e-12)
        if not _is_zero(term):
            observables.append(SparsePauliOp(term.paulis, term.coeffs.real))
            pairs.append((m, n))
    return tuple(observables), tuple(pairs)


def _evaluate_landscape(
    pairs: Sequence[tuple[int, int]],
    values: npt.NDArray[np.float64],
    thetas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Energies at each row of ``thetas``; ``values`` holds one set of landscape
    expectations, or one per row of a leading axis."""
    thetas = np.atleast_2d(thetas)
    amplitudes = np.ones((len(thetas), 1 + 2 * thetas.shape[1]))
    amplitudes[:, 1::2] = np.cos(thetas / 2) - 1
    amplitudes[:, 2::2] = np.sin(thetas / 2)
    first, second = np.asarray(pairs).T
    return np.asarray(values) @ (amplitudes[:, first] * amplitudes[:, second]).T


def _find_landscape_angles(
    pairs: Sequence[tuple[int, int]],
    values: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    n_params: int,
    z: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Angles, predicted energy gain and its standard error.

    Of the grid angles whose energy lies within ``z`` standard errors of the
    lowest grid energy, starts from the one nearest zero and descends to its
    minimum.
    """

    def build_grid(points: int) -> npt.NDArray[np.float64]:
        axis = np.linspace(-2 * np.pi, 2 * np.pi, points, endpoint=False)
        return np.array(list(product(axis, repeat=n_params)))

    def evaluate_energy(angles: npt.NDArray[np.float64]) -> float:
        return float(_evaluate_landscape(pairs, values, angles)[0])

    def descend(start: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        bounds = [(-2 * np.pi, 2 * np.pi)] * n_params
        return minimize(evaluate_energy, start, method="L-BFGS-B", bounds=bounds).x

    origin = np.zeros((1, n_params))
    grid = build_grid(_SEARCH_POINTS[n_params])
    energies = _evaluate_landscape(pairs, values, grid)
    lowest = descend(grid[np.argmin(energies)])

    draws = rng.multivariate_normal(
        values, covariance, size=_N_GAIN_DRAWS, method="eigh"
    )
    offsets = np.linspace(-_LOCAL_SPAN, _LOCAL_SPAN, _LOCAL_POINTS)
    local = lowest + np.array(list(product(offsets, repeat=n_params)))
    probes = np.vstack([build_grid(_DRAW_POINTS[n_params]), local])
    draw_gains = _evaluate_landscape(pairs, draws, origin)[:, 0] - np.min(
        _evaluate_landscape(pairs, draws, probes), axis=1
    )
    gain_se = float(np.std(draw_gains, ddof=1))

    near = np.flatnonzero(energies <= energies.min() + z * gain_se)
    start = grid[near[np.argmin(np.linalg.norm(grid[near], axis=1))]]
    angles = descend(start)
    return angles, evaluate_energy(origin) - evaluate_energy(angles), gain_se


class AdaptAnsatz(ReferenceStateAnsatz):
    """The reference determinant followed by ``operators`` in order, repeated
    once per layer.

    The first ``len(fixed_params)`` parameters are bound to ``fixed_params``; the
    rest are trainable. Fixed parameters need a single layer.
    """

    starts_at_reference_state = True

    def __init__(
        self,
        operators: Sequence[PoolOperator] = (),
        fixed_params: Sequence[float] | npt.NDArray[np.float64] = (),
    ):
        self.operators = tuple(operators)
        self.fixed_params = tuple(float(value) for value in fixed_params)

    def n_params_per_layer(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> int:
        """Parameters of every operator, less the fixed ones."""
        return sum(op.n_params for op in self.operators) - len(self.fixed_params)

    def parameter_frequencies(
        self, n_qubits: int, reference: ElectronCounts | None = None
    ) -> Sequence[tuple[float, int] | None]:
        """Each trainable parameter's frequencies, from its operator."""
        frequencies = [f for op in self.operators for f in op.frequencies]
        return frequencies[len(self.fixed_params) :]

    def build(
        self,
        params,
        n_qubits: int,
        n_layers: int,
        reference: ElectronCounts | None = None,
    ) -> QuantumCircuit:
        if self.fixed_params and n_layers != 1:
            raise ValueError(
                f"An AdaptAnsatz with fixed parameters takes one layer; got {n_layers}."
            )
        values = [*self.fixed_params, *np.asarray(params, dtype=object).reshape(-1)]
        qc = self._build_reference_state(n_qubits, self._require_reference(reference))
        offset = 0
        for _ in range(n_layers):
            for op in self.operators:
                op.emit(qc, values[offset : offset + op.n_params])
                offset += op.n_params
        return qc


class AdaptVQE(VQE):
    """ADAPT-VQE: grows the ansatz from a pool, one step at a time.

    Each step measures the energy gradient ``<i[G, H]>`` of every pool operator
    at the current state in one batched evaluation and appends a candidate. With
    ``growth="tetris"`` a step also appends, in order of gradient magnitude,
    further candidates whose qubits are disjoint from those already added that
    step. With ``reoptimize="all"`` every parameter is then optimised again; with
    ``"new"`` only that step's, the earlier ones held fixed, and the run stops
    when its choices start to repeat.

    On a backend that computes expectation values, the candidate with the largest
    gradient magnitude is appended at zero angle, and further TETRIS candidates
    need a gradient of ``admission_threshold``. A
    :class:`~divi.qprog.optimizers.ScipyOptimizer` using BFGS under ``"all"``
    starts each step from the previous step's inverse Hessian, padded with the
    identity for the new parameters. The run stops when no candidate's gradient
    magnitude reaches ``gradient_threshold`` or after ``max_steps`` steps. Small
    gradients do not bound the energy error (Grimsley et al., arXiv:2204.07179).

    On a sampling backend, gradients come with their shot-noise errors and count
    only when they clear a family-wise significance test at ``significance``.
    Without one, the step measures again with twice the shots, up to
    ``max_shot_scale`` times the backend's per-group budget, and ends the run
    when that does not suffice. The ``shortlist_size`` largest significant
    candidates have their energy, as a function of the new angles, measured
    exactly as a trigonometric series; the one predicted to lower the energy
    most is appended at its best angles, and a coupled exchange takes one
    parameter per excitation only when that lowers the energy significantly
    further. Screening and landscape measurements split their shots by group
    weight with a floor per group, whatever ``shot_distribution`` says. The
    optimiser then runs at the reached shot scale, and step energies are
    measured at ``max_shot_scale``. ``max_steps`` is the run's budget.

    An ``early_stopping`` passed to the constructor applies across growth steps
    on either backend. ``stop_reason`` says which rule ended the run.
    """

    _supports_fixed_param_scans = False

    def __init__(
        self,
        problem,
        *,
        pool: OperatorPool | None = None,
        growth: Literal["single", "tetris"] = "single",
        reoptimize: Literal["all", "new"] = "all",
        gradient_threshold: float = 1e-3,
        admission_threshold: float = 1e-5,
        max_steps: int = 50,
        max_iterations_per_step: int = 100,
        significance: float = 0.05,
        shortlist_size: int = 5,
        max_shot_scale: int = 16,
        **kwargs,
    ):
        """Initialise ADAPT-VQE.

        Args:
            problem: A :class:`~divi.qprog.problems.HamiltonianProblem` with
                electron counts, such as a
                :class:`~divi.qprog.problems.MolecularProblem`.
            pool: The operator pool. Defaults to :class:`~divi.qprog.algorithms.adapt.QubitExcitationPool`.
            growth: ``"single"`` appends one candidate per step, ``"tetris"`` one
                per disjoint set of qubits.
            reoptimize: ``"all"`` or ``"new"`` parameters each step.
            gradient_threshold: On an expectation-value backend, the run
                converges once no candidate's gradient magnitude reaches this.
            admission_threshold: On an expectation-value backend, the gradient
                magnitude a further candidate needs to join a TETRIS step, and an
                excitation needs to take its own parameter in a
                :class:`~divi.qprog.algorithms.adapt.CEOPool` operator.
            max_steps: Maximum number of growth steps; on a sampling backend,
                the run's budget.
            max_iterations_per_step: Optimiser iterations per step.
            significance: On a sampling backend, the chance that any candidate
                counts as having a gradient when none has, over every shot
                doubling of a step.
            shortlist_size: On a sampling backend, the number of significant
                candidates whose energy landscape is measured each step.
            max_shot_scale: On a sampling backend, the largest multiple of the
                backend's shots per measurement group a step may reach; a power
                of two. Step energies and the final energy are measured at it.
            **kwargs: Remaining :class:`VQE` arguments (``backend``,
                ``optimizer``, ...). ``early_stopping`` applies across growth
                steps, not within each step's optimisation. On a sampling
                backend it compares measured energies, so it can stop while each
                step gains less than their noise.

        Raises:
            TypeError: If ``ansatz``, ``n_layers`` or ``max_iterations`` is
                passed.
            ValueError: If an option is out of range, or a sampling backend gets
                ``shot_distribution="weighted_random"``, a shadow grouping or an
                error-mitigation protocol.
        """
        for fixed, replacement in (
            ("ansatz", "pool"),
            ("n_layers", "max_steps"),
            ("max_iterations", "max_iterations_per_step"),
        ):
            if fixed in kwargs:
                raise TypeError(
                    f"AdaptVQE grows its own ansatz; pass {replacement} instead of "
                    f"{fixed}."
                )
        if growth not in ("single", "tetris"):
            raise ValueError(f"growth must be 'single' or 'tetris'; got {growth!r}.")
        if reoptimize not in ("all", "new"):
            raise ValueError(f"reoptimize must be 'all' or 'new'; got {reoptimize!r}.")
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1; got {max_steps}.")
        if not 0 < significance < 1:
            raise ValueError(
                f"significance must lie between 0 and 1; got {significance}."
            )
        if shortlist_size < 1:
            raise ValueError(
                f"shortlist_size must be at least 1; got {shortlist_size}."
            )
        if max_shot_scale < 1 or max_shot_scale & (max_shot_scale - 1):
            raise ValueError(
                f"max_shot_scale must be a power of two; got {max_shot_scale}."
            )
        self.pool = QubitExcitationPool() if pool is None else pool
        self.growth = growth
        self.reoptimize = reoptimize
        self.gradient_threshold = gradient_threshold
        self.admission_threshold = admission_threshold
        self.max_steps = max_steps
        self.significance = significance
        self.shortlist_size = shortlist_size
        self.max_shot_scale = max_shot_scale
        self._growth_stopping: EarlyStopping | None = kwargs.pop("early_stopping", None)
        self._step_history: list[dict[str, Any]] = []
        self._operators: tuple[PoolOperator, ...] = ()
        self._shot_scale = 1
        self._landscapes: dict[
            PoolOperator,
            tuple[tuple[tuple[int, int], ...], CircuitPreprocessor, int],
        ] = {}
        super().__init__(
            problem,
            ansatz=AdaptAnsatz(),
            max_iterations=max_iterations_per_step,
            **kwargs,
        )
        if self._is_sampling:
            if self._shot_distribution == "weighted_random":
                raise ValueError(
                    "AdaptVQE on a sampling backend needs fixed shots per "
                    "measurement group; shot_distribution='weighted_random' draws "
                    "them at random."
                )
            if self._grouping_strategy not in ("qwc", "default", "wires", None):
                raise ValueError(
                    "AdaptVQE on a sampling backend measures by commuting groups; "
                    f"grouping_strategy={self._grouping_strategy!r} is not "
                    "supported."
                )
            if self._build_mitigation_stages(ResultFormat.EXPVALS):
                raise ValueError(
                    "AdaptVQE on a sampling backend takes its uncertainties from "
                    "the measurement counts, which error mitigation does not "
                    "carry; pass no qem_protocol."
                )

    @property
    def step_history(self) -> list[dict[str, Any]]:
        """The growth steps, each a dict of the ``operators`` appended, the
        ``max_gradient`` that chose them, and the ``energy``, ``n_params`` and
        ``n_iterations`` after optimising; on a sampling backend also the
        predicted ``gains`` of the operators and the ``shot_scale`` reached."""
        return list(self._step_history)

    @property
    def operators(self) -> tuple[PoolOperator, ...]:
        """The operators of the grown ansatz, in circuit order."""
        return self._operators

    @property
    def _is_sampling(self) -> bool:
        return not self.backend.supports_expval

    def save_state(self, checkpoint_config: CheckpointConfig) -> NoReturn:
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("AdaptVQE does not support checkpointing.")

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str, *args, **kwargs) -> NoReturn:
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("AdaptVQE does not support checkpointing.")

    def _set_operators(
        self,
        operators: tuple[PoolOperator, ...],
        fixed_params: npt.NDArray[np.float64] | tuple[()] = (),
    ) -> None:
        """Rebuild the ansatz from ``operators``, the first parameters fixed."""
        self._operators = operators
        self.ansatz = AdaptAnsatz(operators, fixed_params)
        self._drop_built_circuits()

    @cached_property
    def _pool_operators(self) -> tuple[PoolOperator, ...]:
        ansatz = AdaptAnsatz()
        occupied = ansatz._occupied_qubits(
            self.n_qubits, ansatz._require_reference(self._ansatz_reference())
        )
        operators = self.pool.build_operators(self.n_qubits, occupied)
        if any(op.n_params != 1 for op in operators):
            raise ValueError(
                "Pool operators whose gradient is measured need one parameter each."
            )
        return operators

    @cached_property
    def _candidates(self) -> npt.NDArray[np.bool_]:
        """Which pool operators may be appended."""
        return self.pool.mark_candidates(self._pool_operators)

    @cached_property
    def _screening_observables(self) -> tuple[SparsePauliOp, ...]:
        """The Hamiltonian, then ``i[G, H]`` for each pool operator."""
        hamiltonian = self.cost_hamiltonian

        def build_gradient_observable(op: PoolOperator) -> SparsePauliOp:
            (generator,) = op.build_generators(self.n_qubits)
            gradient = (1j * commutator(generator, hamiltonian)).simplify()
            return SparsePauliOp(gradient.paulis, gradient.coeffs.real)

        return (
            hamiltonian,
            *(build_gradient_observable(op) for op in self._pool_operators),
        )

    @cached_property
    def _screening(self) -> CircuitPreprocessor:
        """Measures :attr:`_screening_observables`."""
        return self._build_tuple_preprocessor(
            self._screening_observables, "adapt_screening"
        )

    def _build_tuple_preprocessor(
        self, observables: tuple[SparsePauliOp, ...], cache_key: str
    ) -> CircuitPreprocessor:
        """Measures ``observables`` together. On a sampling backend each group
        gets at least :data:`_MIN_GROUP_SHOTS` and the rest by group weight,
        whatever ``shot_distribution`` says."""
        stage = (
            self._make_measurement_stage(shot_distribution=_distribute_with_floor)
            if self._is_sampling
            else self._make_measurement_stage()
        )
        return CircuitPreprocessor(
            cache_key,
            preprocess=lambda meta: replace(
                meta.set_observable(observables, validated=True), _was_multi_obs=True
            ),
            terminal_stage=stage,
            cache_key=cache_key,
        )

    def _count_groups(self, observables: tuple[SparsePauliOp, ...]) -> int:
        return len(
            _compute_measurement_groups(
                observables, self._grouping_strategy, self.n_qubits
            )[0]
        )

    def _build_pipeline_env(self, **overrides):
        """Runs at the reached shot scale unless the caller sets a budget."""
        if (
            self._shot_scale > 1
            and not {
                "shots_override",
                "estimator_samples",
            }
            & overrides.keys()
        ):
            overrides["shots_override"] = self.backend.shots * self._shot_scale
        return super()._build_pipeline_env(**overrides)

    def _measure_gradients(self, params) -> tuple[float, npt.NDArray[np.float64]]:
        """The energy and every pool operator's gradient at ``params``."""
        (values,) = self.evaluate(np.atleast_2d(params), self._screening).values()
        values = np.asarray(values, dtype=np.float64)
        return float(values[0]), values[1:]

    def _measure_with_covariance(
        self, preprocessor: CircuitPreprocessor, params, shots: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Values and covariance of a preprocessor's observables at ``params``,
        measured with ``shots`` in total."""
        values, covariances = self.evaluate(
            np.atleast_2d(params), preprocessor, shots=shots, return_variance=True
        )
        return np.asarray(values[0], dtype=np.float64), np.atleast_2d(covariances[0])

    def _find_max_gradient(self, gradients: npt.NDArray[np.float64]) -> float:
        """The largest candidate gradient magnitude, ``0`` with no candidates."""
        return float(np.max(np.abs(gradients[self._candidates]), initial=0.0))

    def _choose(self, gradients: npt.NDArray[np.float64]) -> list[PoolOperator]:
        """The operators this step appends, best first."""
        operators = self._pool_operators
        magnitudes = np.where(self._candidates, np.abs(gradients), -1.0)
        order = np.argsort(-magnitudes, kind="stable")
        active = np.abs(gradients) >= self.admission_threshold

        def resolve(index: int) -> PoolOperator:
            return self.pool.resolve(int(index), operators, active)

        chosen = [resolve(order[0])]
        if self.growth == "tetris":
            admitted = (
                int(i) for i in order[1:] if magnitudes[i] >= self.admission_threshold
            )
            _add_disjoint(chosen, admitted, operators, resolve, lambda op: op)
        return chosen

    def _build_warm_start(self, n_previous: int, n_new: int) -> Mapping[str, Any]:
        """The last step's inverse Hessian padded for ``n_new`` parameters, when
        BFGS recycles it."""
        if self.reoptimize != "all" or not (
            isinstance(self.optimizer, ScipyOptimizer)
            and self.optimizer.method == ScipyMethod.BFGS
        ):
            return {}
        previous = getattr(self.optimize_result, "hess_inv", None)
        if previous is None or np.shape(previous) != (n_previous, n_previous):
            return {}
        previous = (previous + previous.T) / 2
        try:
            np.linalg.cholesky(previous)
        except np.linalg.LinAlgError:
            return {}
        return {"initial_inverse_hessian": block_diag(previous, np.eye(n_new))}

    def run(
        self,
        initial_params=None,
        perform_final_computation=True,
        checkpoint_config=None,
        **kwargs,
    ):
        """Grow and optimise the ansatz until a stopping rule fires.

        Args:
            initial_params: Not supported; each step starts from the previous
                optimum.
            perform_final_computation: Whether to sample the final state.
            checkpoint_config: Not supported.
            **kwargs: Any keyword raises ``TypeError``.

        Returns:
            AdaptVQE: ``self``.

        Raises:
            ValueError: If ``initial_params`` or a checkpoint directory is given.
        """
        if initial_params is not None:
            raise ValueError(
                "AdaptVQE starts from the reference state; initial_params is not "
                "supported."
            )
        if checkpoint_config is not None and checkpoint_config.checkpoint_dir:
            raise ValueError("AdaptVQE does not support checkpointing.")
        self._reject_unclaimed_run_kwargs(kwargs)
        if self._stop_reason is not None:
            warn(
                "This AdaptVQE has already run, so run() neither grows nor "
                "finalises it. Build a new program to run again.",
                UserWarning,
                stacklevel=2,
            )
            return self

        with self._ensure_progress_session(label="ADAPT-VQE", total=None):
            if self._is_sampling:
                self._grow_sampled()
            else:
                self._grow()
            if self.operators:
                self._finalize(perform_final_computation)
            elif not self._step_history:
                warn(
                    "No pool gradient at the reference state counts as nonzero, "
                    "so no operator was appended.",
                    UserWarning,
                    stacklevel=2,
                )
            self._progress_emitter(
                ProgressEvent.finish(
                    self._progress_key,
                    TerminalStatus.SUCCESS,
                    detail="Finished successfully!",
                )
            )
        return self

    def _grow(self) -> None:
        """The growth loop of :meth:`run` on an expectation-value backend."""
        stopping = self._growth_stopping
        if stopping is not None:
            stopping.reset()
        params = np.zeros(0)
        _, gradients = self._measure_gradients(params)
        picks: list[PoolOperator] = []
        reason = StopReason.MAX_STEPS
        for step in range(1, self.max_steps + 1):
            max_gradient = self._find_max_gradient(gradients)
            if max_gradient < self.gradient_threshold:
                reason = StopReason.CONVERGED
                break
            added = self._choose(gradients)
            if self.reoptimize == "new":
                picks.append(added[0])
                if _detect_cycle(picks):
                    reason = StopReason.CYCLE
                    break
            self._progress_emitter(
                ProgressEvent.show(self._progress_key, f"Step {step}/{self.max_steps}")
            )
            n_new = sum(op.n_params for op in added)
            fixed = params if self.reoptimize == "new" else np.zeros(0)
            params = self._optimise_step(
                fixed,
                np.concatenate([params[len(fixed) :], np.zeros(n_new)]),
                added,
            )
            self._step_history.append(
                {
                    "operators": tuple(added),
                    "max_gradient": max_gradient,
                    "energy": self._best_loss,
                    "n_params": len(params),
                    "n_iterations": self.current_iteration,
                }
            )
            _, gradients = self._measure_gradients(params)
            if stopping is not None and (
                stopped := stopping.check(self._step_history[-1]["energy"])
            ):
                reason = stopped
                break
        else:
            if self._find_max_gradient(gradients) < self.gradient_threshold:
                reason = StopReason.CONVERGED

        if reason is StopReason.MAX_STEPS:
            warn(
                f"AdaptVQE reached max_steps={self.max_steps} before its "
                "gradients fell below gradient_threshold.",
                UserWarning,
                stacklevel=3,
            )
        self._settle(params, reason)

    def _build_gate(self) -> float:
        """Standard errors a gradient must clear: Šidák's family-wise bound over
        the candidates and the shot doublings of a step at ``significance``."""
        n_looks = self.max_shot_scale.bit_length()
        n_candidates = max(int(self._candidates.sum()), 1)
        per_test = 1 - (1 - self.significance / n_looks) ** (1 / n_candidates)
        return float(norm.isf(per_test / 2))

    def _grow_sampled(self) -> None:
        """The growth loop of :meth:`run` on a sampling backend."""
        gate = self._build_gate()
        stopping = self._growth_stopping
        if stopping is not None:
            stopping.reset()
        params = np.zeros(0)
        picks: list[PoolOperator] = []
        reason = StopReason.MAX_STEPS
        for step in range(1, self.max_steps + 1):
            self._progress_emitter(
                ProgressEvent.show(self._progress_key, f"Step {step}/{self.max_steps}")
            )
            screened = self._screen(params, gate)
            if screened is None:
                reason = StopReason.RESOLUTION_LIMITED
                break
            gradients, significant = screened
            order = [
                int(i)
                for i in np.argsort(-np.abs(gradients), kind="stable")
                if significant[i] and self._candidates[i]
            ]
            added = self._choose_sampled(order, significant, params)
            if self.reoptimize == "new":
                picks.append(added[0][0])
                if _detect_cycle(picks):
                    reason = StopReason.CYCLE
                    break
            params, n_iterations = self._extend(params, added)
            energy = self._measure_energy(params)
            self._step_history.append(
                {
                    "operators": tuple(op for op, _, _ in added),
                    "max_gradient": self._find_max_gradient(gradients),
                    "gains": tuple(gain for _, _, gain in added),
                    "shot_scale": self._shot_scale,
                    "energy": energy,
                    "n_params": len(params),
                    "n_iterations": n_iterations,
                }
            )
            if stopping is not None and (stopped := stopping.check(energy)):
                reason = stopped
                break

        if reason is StopReason.MAX_STEPS:
            warn(
                f"AdaptVQE reached max_steps={self.max_steps} while a pool "
                "gradient was still significant.",
                UserWarning,
                stacklevel=3,
            )
        self._settle(params, reason)

    def _has_run_optimization(self) -> bool:
        """Whether growth has run, including a run that appended no operator."""
        return self._stop_reason is not None or super()._has_run_optimization()

    def _settle(self, params: npt.NDArray[np.float64], reason: StopReason) -> None:
        """Record ``params`` as the result, with the energy measured there, and
        why growth stopped."""
        self._best_params = params
        self._final_params = params.copy()
        self._best_loss = self._measure_energy(params)
        self._stop_reason = reason

    def _measure_energy(self, params: npt.NDArray[np.float64]) -> float:
        """The cost at ``params``, at ``max_shot_scale`` on a sampling backend."""
        shots = self.backend.shots * self.max_shot_scale if self._is_sampling else None
        (energy,) = self._evaluate_cost_param_sets(
            np.atleast_2d(params), shots=shots
        ).values()
        return energy

    def _screen(
        self, params: npt.NDArray[np.float64], gate: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]] | None:
        """Every gradient and whether it clears ``gate`` standard errors,
        measured afresh with twice the shots until a candidate does; ``None`` if
        none does at :attr:`max_shot_scale`."""
        n_groups = self._count_groups(self._screening_observables)
        while True:
            mean, covariance = self._measure_with_covariance(
                self._screening,
                params,
                n_groups * self.backend.shots * self._shot_scale,
            )
            gradients = mean[1:]
            errors = np.sqrt(np.clip(np.diag(covariance)[1:], 0.0, None))
            significant = np.abs(gradients) > gate * errors
            if (significant & self._candidates).any():
                return gradients, significant
            if self._shot_scale >= self.max_shot_scale:
                return None
            self._shot_scale *= 2

    def _judge(
        self, operator: PoolOperator, params: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float, float]:
        """``operator``'s best angles, predicted gain and its standard error."""
        if operator not in self._landscapes:
            observables, pairs = _build_landscape(
                operator, self.cost_hamiltonian, self.n_qubits
            )
            self._landscapes[operator] = (
                pairs,
                self._build_tuple_preprocessor(
                    observables, f"adapt_landscape_{len(self._landscapes)}"
                ),
                self._count_groups(observables),
            )
        pairs, preprocessor, n_groups = self._landscapes[operator]
        values, covariance = self._measure_with_covariance(
            preprocessor, params, n_groups * self.backend.shots * self._shot_scale
        )
        return _find_landscape_angles(
            pairs,
            values,
            covariance,
            operator.n_params,
            float(norm.isf(self.significance / 2)),
            self._rng,
        )

    def _choose_sampled(
        self,
        order: Sequence[int],
        significant: npt.NDArray[np.bool_],
        params: npt.NDArray[np.float64],
    ) -> list[tuple[PoolOperator, npt.NDArray[np.float64], float]]:
        """The operators this step appends with their angles and predicted gains,
        best first."""
        operators = self._pool_operators
        judged = {
            index: self._judge(operators[index], params)
            for index in order[: self.shortlist_size]
        }

        def resolve(index: int) -> tuple[PoolOperator, npt.NDArray[np.float64], float]:
            if index not in judged:
                judged[index] = self._judge(operators[index], params)
            angles, gain, error = judged[index]
            resolved = self.pool.resolve(index, operators, significant)
            if resolved == operators[index]:
                return resolved, angles, gain
            wider_angles, wider_gain, wider_error = self._judge(resolved, params)
            if wider_gain - gain > norm.isf(self.significance) * np.hypot(
                error, wider_error
            ):
                return resolved, wider_angles, wider_gain
            return operators[index], angles, gain

        best = max(judged, key=lambda index: judged[index][1])
        chosen = [resolve(best)]
        if self.growth == "tetris":
            others = (index for index in order if index != best)
            _add_disjoint(chosen, others, operators, resolve, lambda item: item[0])
        return chosen

    def _extend(
        self,
        params: npt.NDArray[np.float64],
        added: Sequence[tuple[PoolOperator, npt.NDArray[np.float64], float]],
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Append ``added`` at their angles, or turn the last operator further when
        it is the only one chosen again, and optimise; returns the parameters and
        the optimiser's iterations."""
        if len(added) == 1 and self.operators and self.operators[-1] == added[0][0]:
            params = params.copy()
            params[-len(added[0][1]) :] += added[0][1]
            new_operators: tuple[PoolOperator, ...] = ()
            n_trainable = len(added[0][1])
        else:
            new_operators = tuple(op for op, _, _ in added)
            params = np.concatenate([params, *(angles for _, angles, _ in added)])
            n_trainable = sum(op.n_params for op in new_operators)
        n_fixed = len(params) - n_trainable if self.reoptimize == "new" else 0
        params = self._optimise_step(params[:n_fixed], params[n_fixed:], new_operators)
        return params, self.current_iteration

    def _optimise_step(
        self,
        fixed: npt.NDArray[np.float64],
        trainable: npt.NDArray[np.float64],
        added: Sequence[PoolOperator],
    ) -> npt.NDArray[np.float64]:
        """Append ``added`` and optimise ``trainable`` from its values with the
        ``fixed`` parameters bound first; returns every parameter."""
        operators = (*self.operators, *added)
        n_new = sum(op.n_params for op in added)
        warm_start = self._build_warm_start(len(trainable) - n_new, n_new)
        self._set_operators(operators, fixed_params=fixed)
        self._reset_optimization_state()
        self._optimize(trainable, CheckpointConfig(), warm_start)
        trained = np.concatenate(
            [fixed, np.asarray(self._best_params, dtype=np.float64)]
        )
        if len(fixed):
            self._set_operators(operators)
        return trained
