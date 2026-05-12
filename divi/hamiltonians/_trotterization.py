# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Trotterization strategies for Hamiltonian simulation.

Strategies consume and return :class:`qiskit.quantum_info.SparsePauliOp`.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, Protocol
from warnings import warn

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _empty_spo,
    _n_qubits,
    _sort_hamiltonian_terms_spo,
)

# Maximum number of distinct input SPOs cached per strategy instance.
_STRATEGY_CACHE_MAXSIZE = 8


class _LRUCache(OrderedDict):
    """``OrderedDict`` with a maxsize cap. Most-recent entry is at the end."""

    def __init__(self, maxsize: int = _STRATEGY_CACHE_MAXSIZE) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def __getitem__(self, key):  # type: ignore[override]
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            return self[key]
        return default


def _warn_truncation_no_op(
    keep_fraction: float | None, keep_top_n: int | None, n_terms: int
) -> bool:
    """Emit the truncation-is-a-no-op warning if applicable. Returns True when warned."""
    if keep_fraction is not None and keep_fraction == 1.0:
        warn(
            "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian.",
            UserWarning,
        )
        return True
    if keep_top_n is not None and keep_top_n >= n_terms:
        warn(
            "keep_top_n is greater than or equal to the number of terms; "
            "returning the full Hamiltonian.",
            UserWarning,
        )
        return True
    return False


class TrotterizationStrategy(Protocol):
    """Trotterization strategy protocol."""

    @property
    def stateful(self) -> bool:
        """True if the strategy retains state across ``process_hamiltonian`` calls.
        This should be true for strategies that might re-process the Hamiltonian during execution.
        """
        ...

    def process_hamiltonian(self, hamiltonian: SparsePauliOp) -> SparsePauliOp:
        """Trotterize the Hamiltonian (SPO in, SPO out)."""
        ...


@dataclass(frozen=True)
class ExactTrotterization(TrotterizationStrategy):
    """Exact Trotterization strategy."""

    keep_fraction: float | None = None
    """Fraction of terms to keep by coefficient magnitude (largest first). Must be in (0, 1]. If None, keep all terms."""
    keep_top_n: int | None = None
    """Number of top terms to keep by coefficient magnitude. Must be >= 1. If None, keep all terms. Mutually exclusive with keep_fraction."""

    # Bounded LRU cache of ``process_hamiltonian`` results keyed on ``id(spo)``.
    _cache: _LRUCache = field(default_factory=_LRUCache, compare=False, hash=False)

    def __post_init__(self):
        if self.keep_fraction is not None and self.keep_top_n is not None:
            raise ValueError(
                "At most one of keep_fraction or keep_top_n may be provided."
            )

        if self.keep_fraction is not None and (
            self.keep_fraction <= 0 or self.keep_fraction > 1
        ):
            raise ValueError(
                f"keep_fraction must be in (0, 1], got {self.keep_fraction}"
            )

        if self.keep_top_n is not None and (
            not isinstance(self.keep_top_n, int) or self.keep_top_n <= 0
        ):
            raise ValueError(
                f"keep_top_n must be a positive integer (>= 1), got {self.keep_top_n}"
            )

    @property
    def stateful(self) -> bool:
        # Despite having a _cache, this strategy is stateless because it only
        # uses the cache as memoization, not as state.
        return False

    def process_hamiltonian(self, hamiltonian: SparsePauliOp) -> SparsePauliOp:
        """Truncate the Hamiltonian to its top-magnitude terms."""
        if self.keep_fraction is None and self.keep_top_n is None:
            return hamiltonian.simplify()

        # Cache keyed on ``id(spo)``. The stored tuple holds a strong
        # reference to the SPO so its id cannot be reused while cached.
        cache_key = id(hamiltonian)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached[1]

        if _warn_truncation_no_op(
            self.keep_fraction, self.keep_top_n, hamiltonian.size
        ):
            return hamiltonian.simplify()

        result_spo = self._truncate_spo(hamiltonian)
        self._cache[cache_key] = (hamiltonian, result_spo)
        return result_spo

    def _truncate_spo(self, spo: SparsePauliOp) -> SparsePauliOp:
        """Sort by |coeff|, slice to the kept tail, and re-attach the constant."""
        non_id_spo, constant = _clean_hamiltonian_spo(spo)
        if non_id_spo.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")
        sorted_spo = _sort_hamiltonian_terms_spo(non_id_spo, order="magnitude")

        if self.keep_fraction is not None:
            absolute_coeffs = np.abs(sorted_spo.coeffs.real)
            target = absolute_coeffs.sum() * self.keep_fraction
            cumsum_from_end = np.cumsum(absolute_coeffs[::-1])
            n_keep = np.searchsorted(cumsum_from_end, target, side="left") + 1
            slice_idx = -min(n_keep, len(absolute_coeffs))
        elif self.keep_top_n is not None:
            slice_idx = -self.keep_top_n
        else:
            raise RuntimeError(
                "keep_fraction and keep_top_n are both None; at least one must be set."
            )

        kept_spo = SparsePauliOp(
            sorted_spo.paulis[slice_idx:], sorted_spo.coeffs[slice_idx:]
        )
        if constant != 0:
            const_spo = SparsePauliOp.from_sparse_list(
                [("", [], constant)], num_qubits=_n_qubits(spo)
            )
            return (kept_spo + const_spo).simplify()
        return kept_spo.simplify()


@dataclass(frozen=True)
class QDrift(TrotterizationStrategy):
    """``QDrift`` Trotterization strategy."""

    keep_fraction: float | None = None
    """Fraction of terms to keep deterministically by coefficient magnitude (largest first). Must be in (0, 1]. If None, all terms go to the sampling pool. Mutually exclusive with keep_top_n."""
    keep_top_n: int | None = None
    """Number of top terms to keep deterministically by coefficient magnitude. Must be >= 1. If None, all terms go to the sampling pool. Mutually exclusive with keep_fraction."""
    sampling_budget: int | None = None
    """Number of terms to sample from the remaining Hamiltonian per cost evaluation. If None, only kept terms are applied (equivalent to ``ExactTrotterization``)."""
    sampling_strategy: Literal["uniform", "weighted"] = "uniform"
    """How to sample terms — ``"uniform"`` (equal probability) or ``"weighted"`` (by coefficient magnitude)."""
    seed: int | None = None
    """Random seed for reproducible sampling. If None, sampling is non-deterministic."""
    n_hamiltonians_per_iteration: int = 10
    """Number of Hamiltonian samples per cost evaluation; losses are averaged over them."""

    # Caches the ``(keep_spo, to_sample_spo)`` split keyed on ``id(spo)``.
    # ``to_sample_spo is None`` means all terms were kept (no sampling).
    _cache: _LRUCache = field(default_factory=_LRUCache, compare=False, hash=False)
    _rng: np.random.Generator = field(init=False, compare=False, hash=False)
    # Sampled SPO from the most recent call, preserving duplicates from
    # sampling-with-replacement; concatenated with deterministically-kept terms.
    _last_sampled_spo: SparsePauliOp | None = field(
        default=None, init=False, compare=False, hash=False
    )

    def __post_init__(self):
        if (
            self.keep_fraction is None
            and self.keep_top_n is None
            and self.sampling_budget is None
        ):
            warn(
                "Neither keep_fraction, keep_top_n, nor sampling_budget is set; "
                "the Hamiltonian will be returned unchanged.",
                UserWarning,
            )
        elif self.sampling_budget is None:
            warn(
                "sampling_budget is not set; only the kept terms will be applied, "
                "equivalent to ExactTrotterization.",
                UserWarning,
            )

        if self.sampling_strategy not in {"uniform", "weighted"}:
            raise ValueError(
                f"Invalid sampling_strategy: {self.sampling_strategy}. Must be 'uniform' or 'weighted'."
            )

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed}")

        if self.n_hamiltonians_per_iteration < 1:
            raise ValueError(
                f"n_hamiltonians_per_iteration must be >= 1, got {self.n_hamiltonians_per_iteration}"
            )

        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def stateful(self) -> bool:
        return True

    def process_hamiltonian(self, hamiltonian: SparsePauliOp) -> SparsePauliOp:
        r"""Apply the ``QDrift`` randomized channel to a Hamiltonian.

        Implements the ``QDrift`` protocol (Campbell 2019): for H = Σ c_i P_i,
        randomly sample L terms and rescale their coefficients so that
        E[H_sampled] = H.

        Rescaling rules (L = sampling_budget, λ = Σ\|c_i\|, N = #terms):
          - Weighted: term_i → (λ / (L · \|c_i\|)) · c_i · P_i
          - Uniform:  term_i → (N / L) · c_i · P_i
        """
        if (
            self.keep_fraction is None
            and self.keep_top_n is None
            and self.sampling_budget is None
        ):
            return hamiltonian.simplify()

        if hamiltonian.size == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            return _empty_spo(_n_qubits(hamiltonian))

        triggered_exact_trotterization = (
            self.keep_fraction is not None or self.keep_top_n is not None
        )

        # Cache keyed on ``id(spo)``; the cached tuple's first slot holds a
        # strong reference to the SPO so its id cannot be reused while cached.
        cache_key = id(hamiltonian)
        cached = self._cache.get(cache_key)
        if cached is not None:
            _, keep_spo, to_sample_spo = cached
        else:
            keep_spo = None

            if triggered_exact_trotterization:
                all_kept = (
                    self.keep_fraction is not None and self.keep_fraction == 1.0
                ) or (
                    self.keep_top_n is not None and self.keep_top_n >= hamiltonian.size
                )
                if all_kept:
                    # Two warnings: the truncation no-op + "nothing left to sample".
                    _warn_truncation_no_op(
                        self.keep_fraction, self.keep_top_n, hamiltonian.size
                    )
                    warn(
                        "All terms were kept; there are no terms left to sample. "
                        "Returning the full Hamiltonian.",
                        UserWarning,
                    )
                    self._cache[cache_key] = (hamiltonian, None, None)
                    return hamiltonian

                keep_spo = ExactTrotterization(
                    keep_fraction=self.keep_fraction, keep_top_n=self.keep_top_n
                )._truncate_spo(hamiltonian)
                # ``atol=0`` to keep small but genuine residual terms in the
                # sampling pool; default ``atol`` would silently drop them
                # when the input has wide coefficient magnitudes.
                to_sample_spo = (hamiltonian - keep_spo).simplify(atol=0)
            else:
                to_sample_spo = hamiltonian.simplify()

            self._cache[cache_key] = (hamiltonian, keep_spo, to_sample_spo)

        # All-kept branch (re-entered via cache).
        if to_sample_spo is None:
            return hamiltonian

        if self.sampling_budget is None:
            return hamiltonian.simplify() if keep_spo is None else keep_spo

        if to_sample_spo.size == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            if keep_spo is None:
                return _empty_spo(_n_qubits(hamiltonian))
            return keep_spo

        if to_sample_spo.size == 1:
            sampled_spo = to_sample_spo
        else:
            absolute_coeffs = np.abs(to_sample_spo.coeffs)
            coeff_sum = absolute_coeffs.sum()
            if coeff_sum == 0:
                warn(
                    "All term coefficients are zero; returning the kept Hamiltonian.",
                    UserWarning,
                )
                if keep_spo is None:
                    return _empty_spo(_n_qubits(hamiltonian))
                return keep_spo
            if self.sampling_strategy == "weighted":
                probs = absolute_coeffs / coeff_sum
                # Guard against ``probs.sum() == 1 + ε`` for very large term
                # counts; ``np.random.choice`` rejects probabilities that
                # don't sum to exactly 1.
                probs /= probs.sum()
            else:
                probs = None
            indices = self._rng.choice(
                to_sample_spo.size,
                size=self.sampling_budget,
                replace=True,
                p=probs,
            )

            # Rescale so that E[H_sampled] = H.
            if self.sampling_strategy == "weighted":
                # p_i = |c_i|/λ → scale by λ / (L · |c_i|).
                scale = coeff_sum / (self.sampling_budget * absolute_coeffs[indices])
            else:
                # p_i = 1/N → scale by N/L.
                scale = np.full(
                    self.sampling_budget,
                    to_sample_spo.size / self.sampling_budget,
                )
            sampled_coeffs = scale * to_sample_spo.coeffs[indices]
            sampled_spo = SparsePauliOp(to_sample_spo.paulis[indices], sampled_coeffs)

        if keep_spo is not None:
            faithful_spo = sampled_spo + keep_spo
        else:
            faithful_spo = sampled_spo
        object.__setattr__(self, "_last_sampled_spo", faithful_spo)

        if keep_spo is not None:
            return (sampled_spo + keep_spo).simplify()
        return sampled_spo.simplify()
