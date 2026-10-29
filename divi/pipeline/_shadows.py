# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Classical-shadow basis selection over an observable's Pauli terms.

Each snapshot measures every qubit in a Pauli basis, either drawn per qubit
from that qubit's basis probabilities or chosen deterministically. A basis
*covers* a term when it matches the term's Pauli on every qubit the term acts
on, which for random bases happens with probability ``Π_q β_q(P_q)`` —
``3**-|P|`` for uniform bases. Snapshots that cover the same terms run the same
measurement circuit, so they are merged into one group.

Basis probabilities are a ``(n_qubits, 4)`` array indexed by Pauli code
(``0`` identity, ``1`` X, ``2`` Z, ``3`` Y), with column ``0`` fixed at ``1``.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import PauliList, SparsePauliOp
from scipy.special import logsumexp

# Fitting stops once no probability moves by more than this in a sweep.
_FIT_TOLERANCE = 1e-10
_MAX_FIT_SWEEPS = 500
# Bound decay per covering basis, for the largest-weight term.
_DERANDOMISATION_RATE = 0.05

_RANDOM_SHADOW_STRATEGIES = frozenset({"shadows", "biased_shadows"})
_FIXED_SHADOW_STRATEGIES = frozenset({"derandomised_shadows"})
_SHADOW_STRATEGIES = _RANDOM_SHADOW_STRATEGIES | _FIXED_SHADOW_STRATEGIES

_Covers = tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], int]


@dataclass(frozen=True)
class _ShadowMethod:
    """How one shadow strategy selects bases and weighs the terms they cover.

    ``draw_covers(n)`` returns ``(covers, counts, empty)`` for ``n`` snapshots,
    and ``weigh_terms(hits, budget)`` each term's estimator weight from how many
    of a selection's snapshots cover it.
    """

    draw_covers: Callable[[int], _Covers]
    weigh_terms: Callable[[npt.NDArray[np.int64], int], npt.NDArray[np.float64]]
    is_random: bool


def _encode_paulis(paulis: PauliList) -> npt.NDArray[np.uint8]:
    """Per-term, per-qubit Pauli codes ``x + 2z``: 0 I, 1 X, 2 Z, 3 Y."""
    return paulis.x.astype(np.uint8) + 2 * paulis.z.astype(np.uint8)


def _make_uniform_bases(n_qubits: int) -> npt.NDArray[np.float64]:
    """Basis probabilities drawing X, Y and Z equally on every qubit."""
    return np.tile([1.0, 1 / 3, 1 / 3, 1 / 3], (n_qubits, 1))


def _compute_cover_probabilities(
    codes: npt.NDArray[np.uint8], bases: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Probability ``Π_q β_q(P_q)`` that one random basis covers each term."""
    return bases[np.arange(codes.shape[1]), codes].prod(axis=1)


def _sum_squared_coefficients(
    observables: Sequence[SparsePauliOp],
    per_obs_term_indices: Sequence[list[int]],
    n_terms: int,
) -> npt.NDArray[np.float64]:
    """``Σ_obs c_{obs,P}²`` per union term, merging repeats within an observable."""
    weights = np.zeros(n_terms)
    for observable, indices in zip(observables, per_obs_term_indices):
        coefficients = np.zeros(n_terms)
        np.add.at(coefficients, indices, np.real(observable.coeffs))
        weights += coefficients**2
    return weights


def _fit_biased_bases(
    codes: npt.NDArray[np.uint8], weights: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Basis probabilities minimising ``Σ_P w_P / π_P``, the locally-biased
    shadow variance bound of Hadfield et al. (arXiv:2006.15788).

    ``π_P`` is the cover probability. The bound is convex, and with every other
    qubit fixed it is minimised by ``β_q(a) ∝ sqrt(Σ_{P: P_q = a} w_P β_q(a) /
    π_P)``, so qubits are swept in turn until the probabilities settle. Letters
    no weighted term uses get zero. Cover probabilities are tracked as logs,
    since on wide registers they underflow.
    """
    bases = _make_uniform_bases(codes.shape[1])
    is_weighted = weights > 0
    codes, log_weights = codes[is_weighted], np.log(weights[is_weighted])
    log_covers = np.log(bases[np.arange(codes.shape[1]), codes]).sum(axis=1)
    active_qubits = np.flatnonzero(codes.any(axis=0))

    for _ in range(_MAX_FIT_SWEEPS):
        largest_change = 0.0
        for qubit in active_qubits:
            letters = codes[:, qubit]
            log_factors = np.log(bases[qubit, letters])
            log_shares = log_weights + log_factors - log_covers
            # Half the log of each letter's summed share: its unnormalised β.
            log_roots = np.array(
                [
                    0.5 * logsumexp(log_shares[letters == letter])
                    if (letters == letter).any()
                    else -np.inf
                    for letter in (1, 2, 3)
                ]
            )
            updated = np.exp(log_roots - log_roots.max())
            updated /= updated.sum()
            largest_change = max(
                largest_change, float(np.abs(updated - bases[qubit, 1:]).max())
            )
            bases[qubit, 1:] = updated
            log_covers += np.log(bases[qubit, letters]) - log_factors
        if largest_change < _FIT_TOLERANCE:
            break
    return bases


def _count_unique_rows(
    rows: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """The distinct rows of a boolean matrix and how often each occurs."""
    packed = np.ascontiguousarray(np.packbits(rows, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1])))[:, 0]
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return rows[first], counts


def _sample_shadow_covers(
    codes: npt.NDArray[np.uint8],
    bases: npt.NDArray[np.float64],
    n_draws: int,
    rng: np.random.Generator,
) -> _Covers:
    """Draw ``n_draws`` random Pauli bases and record which terms each covers.

    Returns ``(covers, counts, empty_draws)``: the distinct non-empty cover
    patterns (one row per pattern, one column per term), how many draws
    produced each, and how many draws covered no term.
    """
    active_qubits = np.flatnonzero(codes.any(axis=0))
    thresholds = np.cumsum(bases[active_qubits, 1:3], axis=1)
    # Rounding can leave the X+Z threshold below 1, so pin it when Y is never drawn.
    thresholds[bases[active_qubits, 3] == 0, 1] = np.inf
    draws = rng.random((n_draws, len(active_qubits)))
    letters = 1 + (draws[:, :, None] >= thresholds[None]).sum(axis=2)

    covers = np.ones((n_draws, len(codes)), dtype=bool)
    for column, qubit in enumerate(active_qubits):
        term_codes = codes[:, qubit]
        covers &= (term_codes == 0) | (letters[:, column, None] == term_codes)
    return _merge_covers(covers)


def _derandomise_shadow_covers(
    codes: npt.NDArray[np.uint8],
    weights: npt.NDArray[np.float64],
    n_bases: int,
) -> _Covers:
    """Choose ``n_bases`` Pauli bases deterministically and record which terms
    each covers.

    The derandomisation of Huang, Kueng and Preskill (arXiv:2103.07510): each
    basis is fixed qubit by qubit, taking the letter that most lowers a
    pessimistic bound on the terms' miss probability, with the undecided qubits
    taken as uniform. Every covering basis shrinks a term's bound by
    ``exp(-rate / w_P)`` for its weight ``w_P`` relative to the largest, so
    heavier terms are covered more often; zero-weight terms exert no pull.
    Returns ``(covers, counts, empty_bases)`` as :func:`_sample_shadow_covers`
    does.
    """
    active_qubits = np.flatnonzero(codes.any(axis=0))
    support = codes[:, active_qubits] != 0
    peak = weights.max()
    relative = weights / peak if peak > 0 else np.zeros_like(weights)
    rates = np.divide(
        _DERANDOMISATION_RATE,
        relative,
        out=np.zeros_like(relative),
        where=relative > 0,
    )
    shrinkage = -np.expm1(-rates)
    log_bounds = np.zeros(len(codes))

    covers = np.empty((n_bases, len(codes)), dtype=bool)
    for basis in range(n_bases):
        pull = shrinkage * np.exp(log_bounds - log_bounds.max())
        is_consistent = np.ones(len(codes), dtype=bool)
        # Chance that uniform letters on the undecided qubits complete each term.
        completion = 3.0 ** -support.sum(axis=1)
        for column, qubit in enumerate(active_qubits):
            letters, does_act = codes[:, qubit], support[:, column]
            completion = np.where(does_act, 3 * completion, completion)
            gains = pull * completion
            scores = [
                gains[is_consistent & (~does_act | (letters == letter))].sum()
                for letter in (1, 2, 3)
            ]
            is_consistent &= ~does_act | (letters == 1 + int(np.argmax(scores)))
        covers[basis] = is_consistent
        log_bounds -= rates * is_consistent
    return _merge_covers(covers)


def _merge_covers(
    covers: npt.NDArray[np.bool_],
) -> _Covers:
    """Distinct non-empty cover patterns, their counts, and the empty count."""
    patterns, counts = _count_unique_rows(covers)
    is_non_empty = patterns.any(axis=1)
    return (
        patterns[is_non_empty],
        counts[is_non_empty],
        int(counts[~is_non_empty].sum()),
    )


def _invert_or_zero(values: npt.NDArray) -> npt.NDArray[np.float64]:
    """``1 / values``, with ``0`` where a value is zero."""
    return np.divide(
        1.0, values, out=np.zeros(values.shape), where=values > 0, dtype=np.float64
    )


def _average_over_covers(
    hits: npt.NDArray[np.int64], budget: int
) -> npt.NDArray[np.float64]:
    """Weights averaging each term over the snapshots that cover it."""
    return budget * _invert_or_zero(hits)


def _make_shadow_method(
    strategy: str,
    codes: npt.NDArray[np.uint8],
    squared_coefficients: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> _ShadowMethod:
    """Build ``strategy``'s basis selection over terms with these coefficients.

    Fixed bases average each term over the snapshots that cover it; random
    bases weigh it by its inverse cover probability.
    """
    if strategy in _FIXED_SHADOW_STRATEGIES:
        magnitudes = np.sqrt(squared_coefficients)
        return _ShadowMethod(
            draw_covers=lambda n: _derandomise_shadow_covers(codes, magnitudes, n),
            weigh_terms=_average_over_covers,
            is_random=False,
        )
    bases = (
        _fit_biased_bases(codes, squared_coefficients)
        if strategy == "biased_shadows"
        else _make_uniform_bases(codes.shape[1])
    )
    inverse_cover = _invert_or_zero(_compute_cover_probabilities(codes, bases))
    return _ShadowMethod(
        draw_covers=lambda n: _sample_shadow_covers(codes, bases, n, rng),
        weigh_terms=lambda hits, budget: inverse_cover,
        is_random=True,
    )


def _warn_uncovered_terms(
    spec_key: tuple,
    cover_indices: Sequence[list[int]],
    squared_coefficients: npt.NDArray[np.float64],
) -> None:
    """Warn when fixed shadow bases leave a weighted term unmeasured."""
    is_covered = squared_coefficients == 0
    for terms in cover_indices:
        is_covered[terms] = True
    if is_covered.all():
        return
    warnings.warn(
        f"Derandomised shadows measured {(~is_covered).sum()}/{len(is_covered)} "
        f"term(s) of spec {spec_key!r} in no basis; those terms contribute zero "
        "to the estimate. Raise the shot count to cover them.",
        UserWarning,
        stacklevel=2,
    )
