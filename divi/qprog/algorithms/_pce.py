# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, is_dataclass, replace
from functools import cached_property, lru_cache
from typing import Any, Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.circuit.library import CXGate, RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    BinaryPolynomialProblem,
    CompiledBinaryPolynomial,
    compile_problem,
)
from divi.hamiltonians._polynomial import (
    _binary_polynomial_gradient,
    _compute_hard_cvar_energy_jit,
    _evaluate_binary_polynomial,
)
from divi.pipeline import CircuitPreprocessor, ResultFormat, cost_preprocessor
from divi.pipeline.stages import MeasurementStage
from divi.qprog.algorithms._ansatze import GenericLayerAnsatz
from divi.qprog.algorithms._numba_kernels import _popcount_parity_jit
from divi.qprog.mixins import (
    LayeredAnsatzMixin,
    SolutionEntry,
    SolutionSamplingMixin,
)
from divi.qprog.mixins._solution_sampling import _SAMPLE_HINT
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


def _fast_popcount_parity(arr_input: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
    """
    Vectorized calculation of (popcount % 2) for an array of integers.
    Uses a Numba JIT kernel with XOR-fold bit manipulation.
    """
    # copy=False skips the copy on the hot path when already uint64.
    return _popcount_parity_jit(arr_input.astype(np.uint64, copy=False))


_LIMB_BITS = 64
_LIMB_MASK = (1 << _LIMB_BITS) - 1


def _pack_masks(mask_ints: Sequence[int], n_qubits: int) -> npt.NDArray[np.uint64]:
    """Pack per-variable bitmask integers into a uint64 representation.

    For ``n_qubits <= 64`` this is a 1-D ``(n_vars,)`` array (unchanged from the
    original encoding). For wider registers it is a 2-D ``(n_vars, L)`` array of
    little-endian 64-bit limbs, where limb ``j`` holds bits ``[64j, 64j+64)``.
    """
    if n_qubits <= _LIMB_BITS:
        return np.array(mask_ints, dtype=np.uint64)
    n_limbs = (n_qubits + _LIMB_BITS - 1) // _LIMB_BITS
    return np.array(
        [
            [(m >> (_LIMB_BITS * limb)) & _LIMB_MASK for limb in range(n_limbs)]
            for m in mask_ints
        ],
        dtype=np.uint64,
    )


def _states_to_limbs(
    state_strings: Sequence[str], n_limbs: int
) -> npt.NDArray[np.uint64]:
    """Unpack big-endian bitstrings into ``(n_states, n_limbs)`` little-endian limbs.

    Raises:
        ValueError: If the states are wider than the masks' limbs hold, or if
            they are not all the same width.
    """
    n_states = len(state_strings)
    n_bits = len(state_strings[0])
    if n_bits > n_limbs * _LIMB_BITS:
        raise ValueError(
            f"{n_bits}-bit states do not fit in {n_limbs} 64-bit limb(s); "
            "the masks were packed for a narrower register."
        )

    offender = next((s for s in state_strings if len(s) != n_bits), None)
    if offender is not None:
        raise ValueError(
            f"expected {n_bits}-bit states, got one of width {len(offender)}"
        )

    chars = np.frombuffer("".join(state_strings).encode("ascii"), dtype=np.uint8)
    if np.any((chars < ord("0")) | (chars > ord("1"))):
        raise ValueError("state_strings contains a non-binary bitstring")
    padded = np.zeros((n_states, n_limbs * _LIMB_BITS), dtype=np.uint8)
    padded[:, -n_bits:] = (chars - np.uint8(ord("0"))).reshape(n_states, n_bits)

    # packbits is MSB-first, so word 0 is the most significant limb.
    words = np.packbits(padded, axis=1).view(">u8").astype(np.uint64)
    return np.ascontiguousarray(words[:, ::-1])


def _decode_parities(
    state_strings: Sequence[str], variable_masks_u64: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint8]:
    """Decode bitstring parities using the precomputed variable masks.

    Handles both the 1-D uint64 masks (``n_qubits <= 64``) and the 2-D limb
    masks (wider registers); the limb path XORs the per-limb overlaps, since the
    parity of a XOR is the XOR of the parities.
    """
    if not state_strings:
        return np.zeros((variable_masks_u64.shape[0], 0), dtype=np.uint8)

    if variable_masks_u64.ndim == 1:
        states = _states_to_limbs(state_strings, 1)[:, 0]
        overlaps = variable_masks_u64[:, None] & states[None, :]
        return _fast_popcount_parity(overlaps)

    states = _states_to_limbs(state_strings, variable_masks_u64.shape[1])
    overlaps = np.bitwise_xor.reduce(
        variable_masks_u64[:, None, :] & states[None, :, :], axis=-1
    )
    return _fast_popcount_parity(overlaps)


def _soft_relaxation(
    u: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``x = (1 + tanh(u)) / 2`` without cancellation for negative ``u``, and
    ``exp(-2|u|)``."""
    decay = np.exp(-2.0 * np.abs(u))
    return np.where(u >= 0, 1.0, decay) / (1.0 + decay), decay


class PCEObjective:
    """Base class for what :class:`PCE` minimises: an
    :class:`ExpectationObjective` or a :class:`SampleObjective`."""


class ExpectationObjective(PCEObjective, ABC):
    """An objective computed from each variable's expectation value ``z_i``.

    ``z_i`` is the expectation of the variable's Pauli string: its Z-parity
    for :class:`DenseEncoding` and :class:`PolyEncoding`, its X, Y or Z
    correlator for :class:`MultiBasisEncoding`.
    """

    @abstractmethod
    def score(
        self,
        z: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> float:
        """Return the objective for the expectations ``z`` (shape ``(n_vars,)``).

        Args:
            z: Each variable's expectation value, in ``[-1, 1]``.
            problem: The canonical binary polynomial being minimised.
            compiled: ``problem`` compiled by
                :func:`~divi.hamiltonians.compile_problem`.
        """

    @abstractmethod
    def gradient(
        self,
        z: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> npt.NDArray[np.float64]:
        """Return the derivative of :meth:`score` with respect to each ``z_i``."""


class SampleObjective(PCEObjective, ABC):
    """An objective computed from sampled bitstrings.

    Needs a sampling backend and an encoding whose bitstrings decode to
    assignments (:class:`DenseEncoding` or :class:`PolyEncoding`).
    """

    @abstractmethod
    def score(
        self,
        assignments: npt.NDArray[np.uint8],
        counts: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> float:
        """Return the objective for one shot histogram.

        Args:
            assignments: ``(n_vars, n_states)`` 0/1 assignment decoded from
                each sampled state.
            counts: Shot count of each state; sums to a positive total.
            problem: The canonical binary polynomial being minimised.
            compiled: ``problem`` compiled by
                :func:`~divi.hamiltonians.compile_problem`.
        """


@dataclass(frozen=True)
class SoftObjective(ExpectationObjective):
    """PCE objective on a smooth relaxation of the variables' expectations.

    Each variable becomes ``x_i = (1 + tanh(alpha * z_i)) / 2`` and the
    problem's objective is evaluated at these continuous values.

    Args:
        alpha: Sharpness of the relaxation, a positive finite number. Larger
            values push the variables towards 0 and 1 at the cost of a flatter
            landscape.

    Raises:
        ValueError: If ``alpha`` is not a positive finite number.
    """

    alpha: float = 2.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and self.alpha > 0):
            raise ValueError(
                f"alpha must be a positive finite number, got {self.alpha}"
            )

    def score(
        self,
        z: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> float:
        """Evaluate the problem at the relaxed variables."""
        x_soft, _ = _soft_relaxation(self.alpha * z)
        return float(_evaluate_binary_polynomial(x_soft, problem, _compiled=compiled))

    def gradient(
        self,
        z: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> npt.NDArray[np.float64]:
        """Chain the problem's gradient through the tanh relaxation.

        Warns when ``tanh(alpha * z)`` rounds to ``±1`` for every variable: the
        loss is then flat to machine precision and gradient steps stall.
        """
        u = self.alpha * z
        if np.all(np.abs(np.tanh(u)) == 1.0):
            warn(
                f"SoftObjective(alpha={self.alpha:g}) has saturated: tanh(alpha * z) "
                "is ±1 for every variable, so the loss is flat and its gradient "
                "vanishes. Lower alpha or restart from other parameters.",
                UserWarning,
            )
        x_soft, decay = _soft_relaxation(u)
        d_energy = _binary_polynomial_gradient(x_soft, compiled)
        sech_squared = 4.0 * decay / (1.0 + decay) ** 2
        return d_energy * 0.5 * self.alpha * sech_squared


@dataclass(frozen=True)
class CVaRObjective(SampleObjective):
    """PCE objective on the lowest-energy tail of the sampled bitstrings.

    Each shot is decoded to a binary assignment and scored; the objective is the
    mean energy of the lowest ``fraction`` of shots.

    Args:
        fraction: Share of the shots averaged, in ``(0, 1]``.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``.
    """

    fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    def score(
        self,
        assignments: npt.NDArray[np.uint8],
        counts: npt.NDArray[np.float64],
        problem: BinaryPolynomialProblem,
        compiled: CompiledBinaryPolynomial,
    ) -> float:
        """Mean energy of the lowest ``fraction`` of the histogram's shots."""
        return float(
            _compute_hard_cvar_energy_jit(
                np.ascontiguousarray(assignments, dtype=np.float64),
                np.ascontiguousarray(counts, dtype=np.float64),
                float(counts.sum()),
                float(self.fraction),
                *compiled,
            )
        )


@dataclass(frozen=True, kw_only=True)
class PCEEncoding(ABC):
    """Base class for how :class:`PCE` maps variables onto qubits.

    Args:
        n_qubits: Override of the minimum qubit count. Larger values warn,
            since the extra qubits encode nothing.
    """

    n_qubits: int | None = None

    @abstractmethod
    def min_qubits(self, n_vars: int) -> int:
        """Fewest qubits that hold ``n_vars`` variables."""

    @abstractmethod
    def strings(self, n_vars: int, n_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
        """Each variable's Pauli string on ``n_qubits`` qubits.

        Each string is ``(letters, qubits)``, e.g. ``("ZZ", (0, 3))`` for
        :math:`Z_0 Z_3`.
        """

    @abstractmethod
    def decode(
        self, states: Sequence[str], n_vars: int, n_qubits: int
    ) -> npt.NDArray[np.uint8]:
        """``(n_states, n_vars)`` assignments of the stored solution states."""

    def default_objective(self, n_qubits: int) -> PCEObjective:
        """The objective :class:`PCE` uses when none is given."""
        return SoftObjective()

    def layout(self, n_vars: int) -> tuple[int, list[tuple[str, tuple[int, ...]]]]:
        """Return the qubit count and each variable's Pauli string.

        Raises:
            ValueError: If ``n_qubits`` is below :meth:`min_qubits`.
        """
        min_qubits = self.min_qubits(n_vars)
        if self.n_qubits is not None and self.n_qubits < min_qubits:
            raise ValueError(
                f"{type(self).__name__} needs n_qubits >= {min_qubits} for "
                f"{n_vars} variables, got n_qubits={self.n_qubits}."
            )
        if self.n_qubits is not None and self.n_qubits > min_qubits:
            warn(
                "n_qubits exceeds the minimum required; extra qubits increase circuit "
                "size and can add noise without representing more variables.",
                UserWarning,
            )
        n_qubits = min_qubits if self.n_qubits is None else self.n_qubits
        return n_qubits, self.strings(n_vars, n_qubits)


class ParityEncoding(PCEEncoding):
    """Base class for encodings whose variables are Z-parities of the qubits.

    Each variable is the parity of a subset of qubits, so a sampled bitstring
    decodes to an assignment.
    """

    def _masks(self, n_vars: int, n_qubits: int) -> npt.NDArray[np.uint64]:
        """Each variable's qubit subset as a bitmask over a big-endian bitstring,
        packed by :func:`_pack_masks`."""
        return _parity_masks(self, n_vars, n_qubits)

    def decode(
        self, states: Sequence[str], n_vars: int, n_qubits: int
    ) -> npt.NDArray[np.uint8]:
        """Each variable is 1 where its qubits' parity is even."""
        return (1 - _decode_parities(states, self._masks(n_vars, n_qubits))).T


@lru_cache(maxsize=64)
def _parity_masks(
    encoding: ParityEncoding, n_vars: int, n_qubits: int
) -> npt.NDArray[np.uint64]:
    masks = _pack_masks(
        [
            sum(1 << (n_qubits - 1 - q) for q in qubits)
            for _, qubits in encoding.strings(n_vars, n_qubits)
        ],
        n_qubits,
    )
    masks.setflags(write=False)
    return masks


def _bits_to_qubits(mask: int, n_qubits: int) -> tuple[str, tuple[int, ...]]:
    """The Z-parity string of ``mask``, whose most significant bit is qubit 0."""
    qubits = tuple(sorted(n_qubits - 1 - j for j in range(n_qubits) if (mask >> j) & 1))
    return "Z" * len(qubits), qubits


@dataclass(frozen=True)
class DenseEncoding(ParityEncoding):
    """Variable ``i`` is the parity of the qubits set in the binary form of
    ``i + 1``, read with qubit 0 as the most significant bit:
    :math:`\\lceil\\log_2(N + 1)\\rceil` qubits for ``N`` variables."""

    def min_qubits(self, n_vars: int) -> int:
        return n_vars.bit_length()

    def strings(self, n_vars: int, n_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
        return [_bits_to_qubits(i + 1, n_qubits) for i in range(n_vars)]


@dataclass(frozen=True)
class PolyEncoding(ParityEncoding):
    """Each variable is the parity of one or two qubits: the smallest ``q``
    with :math:`q(q + 1)/2 \\geq N` qubits for ``N`` variables."""

    def min_qubits(self, n_vars: int) -> int:
        q = (math.isqrt(8 * n_vars + 1) - 1) // 2
        return q if q * (q + 1) // 2 >= n_vars else q + 1

    def strings(self, n_vars: int, n_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
        last = n_qubits - 1
        singles = (("Z", (last - i,)) for i in range(n_qubits))
        pairs = (
            ("ZZ", (last - j, last - i))
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
        return list(itertools.islice(itertools.chain(singles, pairs), n_vars))


@dataclass(frozen=True)
class MultiBasisEncoding(PCEEncoding):
    """Each variable is a ``order``-body correlator of one Pauli type.

    The strings are :math:`X^{\\otimes k}`, :math:`Y^{\\otimes k}` and
    :math:`Z^{\\otimes k}` on every ``k``-subset of the qubits, so ``q`` qubits
    hold :math:`3\\binom{q}{k}` variables. Variables take the X strings first,
    then Y, then Z, each in lexicographic order of qubit subsets. A variable is
    read as the sign of its expectation value, an exact zero reading as 0, so
    only :class:`ExpectationObjective` objectives apply. The sign-read
    assignment is stored in ``best_probs`` as one outcome of probability 1.0;
    it is not a measured distribution.

    Without an explicit objective, :class:`PCE` uses
    ``SoftObjective(alpha=1.5 * q ** (k // 2))``, the scaling of Sciorilli et
    al. (2025).

    Args:
        order: Number of qubits ``k`` in each string, at least 1.

    Raises:
        ValueError: If ``order`` is below 1.
    """

    order: int = 2

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"order must be at least 1, got {self.order}")

    def min_qubits(self, n_vars: int) -> int:
        return next(
            q
            for q in itertools.count(self.order)
            if 3 * math.comb(q, self.order) >= n_vars
        )

    def strings(self, n_vars: int, n_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
        strings: Iterator[tuple[str, tuple[int, ...]]] = (
            (letter * self.order, qubits)
            for letter in "XYZ"
            for qubits in itertools.combinations(range(n_qubits), self.order)
        )
        return list(itertools.islice(strings, n_vars))

    def decode(
        self, states: Sequence[str], n_vars: int, n_qubits: int
    ) -> npt.NDArray[np.uint8]:
        """The stored states already are the sign-read assignments."""
        return np.array(
            [[int(bit) for bit in state] for state in states], dtype=np.uint8
        ).reshape(len(states), n_vars)

    def default_objective(self, n_qubits: int) -> PCEObjective:
        return SoftObjective(alpha=1.5 * n_qubits ** (self.order // 2))


def _params_key(params: npt.NDArray[np.float64]) -> bytes:
    """A hashable key for one parameter vector; ``-0.0`` and ``0.0`` match."""
    return (np.asarray(params, dtype=np.float64).reshape(-1) + 0.0).tobytes()


class PCE(SolutionSamplingMixin, LayeredAnsatzMixin, VariationalQuantumAlgorithm):
    """
    Generalised Pauli Correlation Encoding (PCE) VQE.

    Encodes an N-variable QUBO/HUBO into fewer qubits by mapping each variable
    to a Pauli string (see :class:`PCEEncoding`). The cost pipeline measures
    every variable's expectation value and the objective (see
    :class:`PCEObjective`) turns them into the loss: a smooth relaxation of the
    problem (:class:`SoftObjective`) or the CVaR of sampled energies
    (:class:`CVaRObjective`).
    """

    def __init__(
        self,
        problem: BinaryOptimizationProblem,
        encoding: PCEEncoding | None = None,
        objective: PCEObjective | None = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Args:
            problem (BinaryOptimizationProblem): The binary problem to minimise.
                Its combined objective + penalty QUBO/HUBO is used. Wrap a raw
                QUBO/HUBO with ``BinaryOptimizationProblem(...)``.
            encoding (PCEEncoding | None): How variables map onto qubits.
                ``None`` uses ``DenseEncoding()``.
            objective (PCEObjective | None): What the optimiser minimises.
                ``None`` uses the encoding's default: ``SoftObjective()``, or
                for :class:`MultiBasisEncoding` a ``SoftObjective`` with
                ``alpha`` scaled to the qubit count.
            max_iterations (int): Maximum number of optimisation iterations.
                Defaults to 10.
            **kwargs: ``ansatz``, ``n_layers``, ``ansatz_kwargs`` and the
                :class:`~divi.qprog.VariationalQuantumAlgorithm` arguments
                (e.g. ``backend``, ``optimizer``). ``ansatz`` defaults to a
                hardware-efficient, entangling
                ``GenericLayerAnsatz([RYGate, RZGate], entangler=CXGate)``.

        Raises:
            TypeError: If ``problem``, ``encoding`` or ``objective`` has the
                wrong type, or a :class:`SampleObjective` is paired with an
                encoding whose bitstrings do not decode to assignments.
            ValueError: If ``qem_protocol`` is passed.
        """
        if not isinstance(problem, BinaryOptimizationProblem):
            raise TypeError(
                f"PCE requires a BinaryOptimizationProblem, got {type(problem).__name__}. "
                "Wrap your QUBO/HUBO: PCE(BinaryOptimizationProblem(qubo))."
            )
        encoding = DenseEncoding() if encoding is None else encoding
        if not isinstance(encoding, PCEEncoding):
            raise TypeError(
                f"encoding must be a PCEEncoding, got {type(encoding).__name__}."
            )
        if kwargs.get("qem_protocol") is not None:
            raise ValueError("PCE does not currently support qem_protocol.")

        self.problem = problem
        self.encoding = encoding
        self._polynomial: BinaryPolynomialProblem = problem.canonical_problem
        self.n_vars = self._polynomial.n_vars
        self._compiled_problem = compile_problem(self._polynomial)

        self.n_qubits, strings = encoding.layout(self.n_vars)
        objective = (
            encoding.default_objective(self.n_qubits)
            if objective is None
            else objective
        )
        if not isinstance(objective, ExpectationObjective | SampleObjective):
            raise TypeError(
                "objective must be an ExpectationObjective or SampleObjective, got "
                f"{type(objective).__name__}."
            )
        if isinstance(objective, SampleObjective) and not isinstance(
            encoding, ParityEncoding
        ):
            raise TypeError(
                f"{type(objective).__name__} scores sampled bitstrings, which "
                f"{type(encoding).__name__} cannot decode. Use an "
                "ExpectationObjective such as SoftObjective."
            )
        self.objective: ExpectationObjective | SampleObjective = objective
        self._cost_expectations: dict[bytes, npt.NDArray[np.float64]] = {}
        self._variable_observables = tuple(
            SparsePauliOp.from_sparse_list(
                [(letters, list(qubits), 1.0)], num_qubits=self.n_qubits
            )
            for letters, qubits in strings
        )

        kwargs.setdefault(
            "ansatz", GenericLayerAnsatz([RYGate, RZGate], entangler=CXGate)
        )
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.cost_hamiltonian = sum(
            self._variable_observables[1:], self._variable_observables[0]
        )
        self.loss_constant = 0.0

    @cached_property
    def _counts_cost_preprocessor(self) -> CircuitPreprocessor:
        # The variables' Z-parities commute qubit-wise, so "qwc" measures them
        # in one circuit whose full-width counts decode shot by shot.
        return replace(
            cost_preprocessor(),
            result_format=ResultFormat.COUNTS,
            terminal_stage=MeasurementStage(
                grouping_strategy="qwc", result_format_override=ResultFormat.COUNTS
            ),
        )

    def cost_preprocessor(self) -> CircuitPreprocessor:
        """Measure every variable's expectation value, or the raw counts for a
        :class:`SampleObjective`; :meth:`_evaluate_cost_param_sets` turns
        either into the loss."""
        if isinstance(self.objective, SampleObjective):
            return self._counts_cost_preprocessor
        return super().cost_preprocessor()

    def _evaluate_cost_param_sets(
        self,
        param_sets: np.ndarray,
        *,
        shots: int | None = None,
        estimator_samples: int | Sequence[int] | None = None,
        collect_variance: bool = False,
        **kwargs,
    ) -> dict[int, float]:
        """Score every parameter set with the objective.

        Raises:
            ValueError: If shot-noise variances are requested (PCE's loss is a
                non-linear function of the measurements), or a
                :class:`SampleObjective` meets an expectation-value backend.
        """
        if collect_variance:
            raise ValueError(
                "PCE does not provide shot-noise variances of its loss, so "
                "shot-adaptive optimizers (RosalinOptimizer, QUIVEROptimizer "
                "with adaptive shots) cannot drive it."
            )
        objective = self.objective
        if isinstance(objective, SampleObjective):
            if self.backend.supports_expval:
                raise ValueError(
                    f"PCE with {type(objective).__name__} requires shot "
                    "histograms and cannot use expectation-value backends. Use a "
                    "sampling backend: MaestroSimulator(force_sampling=True), "
                    "QiskitSimulator(force_sampling=True), or QoroService with "
                    "JobConfig(force_sampling=True)."
                )
            histograms = self.evaluate(
                np.atleast_2d(param_sets),
                self.cost_preprocessor(),
                shots=shots,
                estimator_samples=estimator_samples,
            )
            return {
                idx: self._score_histogram(objective, histogram)
                for idx, histogram in histograms.items()
            }
        param_sets = np.atleast_2d(np.asarray(param_sets, dtype=np.float64))
        expectations = self._variable_expectations(
            param_sets, shots=shots, estimator_samples=estimator_samples
        )
        self._cost_expectations = {
            _params_key(row): z for row, z in zip(param_sets, expectations)
        }
        return {
            idx: objective.score(z, self._polynomial, self._compiled_problem)
            for idx, z in enumerate(expectations)
        }

    def _score_histogram(
        self, objective: SampleObjective, histogram: dict[str, int]
    ) -> float:
        """Score one cost-circuit histogram with ``objective``.

        Raises:
            ValueError: If the histogram is empty or its bitstrings do not span
                the full register.
        """
        if not histogram or sum(histogram.values()) <= 0:
            raise ValueError(
                "Backend returned an empty PCE histogram; the energy is undefined "
                "without shots. This indicates a backend fault, so check the "
                "job's results."
            )
        states = list(histogram)
        if any(len(state) != self.n_qubits for state in states):
            raise ValueError(
                f"Backend returned PCE bitstrings narrower or wider than the "
                f"{self.n_qubits}-qubit register; every bitstring must report all "
                "qubits for the variables' parities to decode."
            )
        counts = np.array(list(histogram.values()), dtype=np.float64)
        return objective.score(
            self._decode_states(states).T,
            counts,
            self._polynomial,
            self._compiled_problem,
        )

    def _variable_expectations(
        self, param_sets: npt.NDArray[np.float64], **kwargs
    ) -> npt.NDArray[np.float64]:
        """Each variable's expectation value, shape ``(n_sets, n_vars)``."""
        result = self.evaluate(
            np.atleast_2d(param_sets), super().cost_preprocessor(), **kwargs
        )
        return np.array(
            [np.atleast_1d(result[idx]) for idx in range(len(result))],
            dtype=np.float64,
        )

    def _evaluate_gradient_at(
        self, params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The parameter-shift Jacobian of the variables' expectations, chained
        through the objective's derivative.

        Exact on a backend that computes expectation values. On a sampling
        backend the objective's derivative is taken at noisy expectations, so
        the gradient carries a bias that shrinks with the shot count.

        Raises:
            ValueError: If the objective scores samples, which have no gradient.
        """
        objective = self._differentiable_objective()
        shifts, weights = self._grad_shift_rule
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        expectations = self._variable_expectations(np.vstack([params, shifts + params]))
        jacobian = weights @ expectations[1:]
        return jacobian @ objective.gradient(
            expectations[0], self._polynomial, self._compiled_problem
        )

    def _loss_gradient_by_observable(
        self, params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The objective's slope in each variable's expectation at ``params``.

        Reuses the expectations of the latest cost evaluation at ``params``,
        which QNG runs just before asking for the gradient, and measures them
        otherwise.

        Raises:
            ValueError: If the objective scores samples, which have no gradient.
        """
        objective = self._differentiable_objective()
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        z = self._cost_expectations.get(_params_key(params))
        if z is None:
            z = self._variable_expectations(np.atleast_2d(params))[0]
        return objective.gradient(z, self._polynomial, self._compiled_problem)

    def _differentiable_objective(self) -> ExpectationObjective:
        """The objective, which must have a gradient.

        Raises:
            ValueError: If the objective scores samples, which have no gradient.
        """
        if not isinstance(self.objective, ExpectationObjective):
            raise ValueError(
                f"{type(self.objective).__name__} scores sampled bitstrings and "
                "has no gradient; use a gradient-free optimizer."
            )
        return self.objective

    def _run_solution_measurement_for(
        self, param_sets: npt.NDArray[np.float64], *, backend=None
    ) -> None:
        """Sample parity encodings; read other encodings by expectation sign.

        A multi-basis encoding's strings do not commute, so no measured
        bitstring decodes to an assignment. Its solution is
        ``x_i = [<P_i> > 0]``, stored as a certain outcome per parameter set.
        """
        if isinstance(self.encoding, ParityEncoding):
            super()._run_solution_measurement_for(param_sets, backend=backend)
            return
        expectations = self._variable_expectations(param_sets, backend=backend)
        self._results["best_probs"] = {
            idx: {"".join("1" if value > 0 else "0" for value in z): 1.0}
            for idx, z in enumerate(expectations)
        }

    def _decode_states(self, states: Sequence[str]) -> npt.NDArray[np.uint8]:
        """``(n_states, n_vars)`` assignments of measured or stored states."""
        return self.encoding.decode(states, self.n_vars, self.n_qubits)

    def _create_cost_circuit(self) -> MetaCircuit:
        """The layered ansatz, measuring every variable's Pauli string."""
        return self._cost_meta_from_ansatz(self._variable_observables)

    @property
    def _serialized_cost_hamiltonian_fingerprint(self) -> str:
        """Identifies the problem, encoding and objective, so a checkpoint
        resumes only the PCE that wrote it."""
        term_indices, term_offsets, coeffs, constant = self._compiled_problem
        objective = (
            repr(self.objective)
            if is_dataclass(self.objective)
            else type(self.objective).__qualname__
        )
        payload = (
            repr(self.encoding),
            objective,
            self.n_qubits,
            [repr(var) for var in self._polynomial.variable_order],
            term_indices.tolist(),
            term_offsets.tolist(),
            np.round(coeffs, 6).tolist(),
            round(float(constant), 6),
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def get_top_solutions(
        self,
        n: int = 10,
        *,
        min_prob: float = 0.0,
        include_decoded: bool = False,
        sort_by: Literal["prob", "energy"] = "prob",
        feasibility: Literal["ignore", "filter", "repair"] = "ignore",
    ) -> list[SolutionEntry]:
        """Get the top-N solutions with decoded QUBO variable assignments.

        This method overrides the base implementation to decode the stored
        distribution into QUBO variable assignments: a parity encoding's encoded
        qubit states are decoded through their parities, and a multi-basis
        encoding's single sign-decoded assignment is returned as is.

        Args:
            n (int): Maximum number of solutions to return. Must be non-negative;
                ``0`` returns every available solution. If n exceeds the
                number of available solutions (after filtering), returns all
                available solutions. Defaults to 10.
            min_prob (float): Minimum probability threshold for including solutions.
                Only solutions with probability >= min_prob will be included.
                Must be in range [0.0, 1.0]. Defaults to 0.0 (no filtering).
            include_decoded (bool): Whether to populate the `decoded` field of
                each SolutionEntry with the numpy array representation of the
                QUBO solution. If False, the decoded field will be None.
                Defaults to False.
            sort_by: Sort order for the returned solutions.
                ``"prob"`` (default): descending by probability.
                ``"energy"``: ascending by the penalised QUBO energy, which
                also fills each ``SolutionEntry``'s ``energy``.
            feasibility: ``"ignore"`` (default), ``"filter"`` or ``"repair"``,
                as in :meth:`QAOA.get_top_solutions
                <divi.qprog.algorithms.QAOA.get_top_solutions>`. ``"filter"``
                and ``"repair"`` rank by the problem's ``compute_energy``
                whatever ``sort_by`` is, and need the ``"native"`` Hamiltonian
                builder.

        Returns:
            list[SolutionEntry]: List of solution entries sorted according to
                ``sort_by``, with decoded bitstring for deterministic tie-breaking.
                The `bitstring` field contains the decoded QUBO solution as a
                binary string (e.g., "01011" for 5 variables), not the encoded
                qubit state.

        Raises:
            RuntimeError: If probability distribution is not available because
                optimisation has not been run or final computation was not performed.
            ValueError: If min_prob is not in range [0.0, 1.0], n is negative,
                or sort_by is not one of ``"prob"`` or ``"energy"``.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError(f"min_prob must be in range [0.0, 1.0], got {min_prob}")
        if sort_by not in ("prob", "energy"):
            raise ValueError(f"sort_by must be 'prob' or 'energy', got '{sort_by}'")
        if feasibility not in ("ignore", "filter", "repair"):
            raise ValueError(
                "feasibility must be 'ignore', 'filter' or 'repair', "
                f"got '{feasibility}'"
            )
        if feasibility != "ignore" and self.problem.hamiltonian_builder != "native":
            raise ValueError(
                "feasibility='filter' and 'repair' need a problem built with "
                "hamiltonian_builder='native'."
            )

        limit = n or None

        probs_dict = self._single_distribution()

        filtered = [(bs, prob) for bs, prob in probs_dict.items() if prob >= min_prob]

        decoded_qubo_solutions = self._decode_states([bs for bs, _ in filtered])

        if feasibility != "ignore":
            decoded_bitstrings = (
                "".join(str(int(x)) for x in solution)
                for solution in decoded_qubo_solutions
            )
            return self.problem._rank_feasible(
                zip(decoded_bitstrings, (prob for _, prob in filtered)),
                feasibility,
                self.problem.decode_fn if include_decoded else None,
            )[:limit]

        compute_energy = sort_by == "energy"
        energies = (
            np.asarray(
                _evaluate_binary_polynomial(
                    decoded_qubo_solutions.T.astype(float),
                    self._polynomial,
                    _compiled=self._compiled_problem,
                )
            ).tolist()
            if compute_energy
            else [None] * len(filtered)
        )
        digits = np.ascontiguousarray(decoded_qubo_solutions.astype(np.uint8) + 48)
        result = []
        for (_, prob), decoded_solution, row, energy in zip(
            filtered, decoded_qubo_solutions, digits, energies
        ):
            result.append(
                SolutionEntry(
                    bitstring=row.tobytes().decode("ascii"),
                    prob=prob,
                    decoded=(
                        self._decode_assignment(decoded_solution.astype(np.int32))
                        if include_decoded
                        else None
                    ),
                    energy=energy,
                )
            )

        if sort_by == "energy":
            result.sort(key=lambda e: (e.energy, e.bitstring))
        else:
            result.sort(key=lambda e: (-e.prob, e.bitstring))

        return result[:limit]

    def _decode_assignment(self, vector: npt.NDArray[np.integer]) -> Any:
        """Decode a canonical-order assignment.

        Uses the problem's ``decode_fn`` with the ``"native"`` Hamiltonian
        builder. Otherwise returns the raw assignment, as a dict keyed by
        variable name when the variable order is not ``0..n_vars-1``.
        """
        if self.problem.hamiltonian_builder == "native":
            return self.problem.decode_fn("".join(str(int(x)) for x in vector))
        vo = self._polynomial.variable_order
        if vo != tuple(range(self.n_vars)):
            return dict(zip(vo, vector))
        return vector

    @property
    def solution(self) -> npt.NDArray[np.integer] | dict:
        """Return the most-probable decoded assignment from the final measurement.

        .. note::

            This returns the assignment corresponding to the **highest-probability**
            encoded bitstring, which may not be the lowest-energy solution.
            For energy-ranked solutions, use
            :meth:`get_top_solutions(sort_by="energy") <get_top_solutions>` instead,
            or ``feasibility="repair"`` when the problem has constraints.

        Returns:
            For QUBO problems, a binary 0/1 NumPy array. For HUBO problems
            with non-integer variable names, a dictionary mapping variable
            names to binary values.

        Raises:
            RuntimeError: If nothing has been sampled yet.
        """
        if not self._results.get("best_probs"):
            raise RuntimeError(f"PCE.solution is not available yet. {_SAMPLE_HINT}")

        if isinstance(self.encoding, ParityEncoding):
            warn(
                "PCE.solution returns the decoded assignment of the most-probable "
                "encoded bitstring. Because PCE operates in a compressed qubit "
                "space (fewer qubits than variables), the most-probable "
                "encoded state does not necessarily decode to the lowest-energy "
                "QUBO solution. Use get_top_solutions(sort_by='energy') for "
                "energy-ranked results, or feasibility='repair' when the problem "
                "has constraints.",
                stacklevel=2,
            )
        probs = self._single_distribution()
        state = max(probs, key=probs.__getitem__)
        return self._decode_assignment(self._decode_states([state])[0])
