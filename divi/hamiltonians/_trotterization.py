# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Trotterization strategies for Hamiltonian simulation."""

from dataclasses import dataclass, field
from typing import Literal, Protocol
from warnings import warn

import numpy as np
import pennylane as qml

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian,
    _get_terms_iterable,
    _hamiltonian_term_count,
    _is_empty_hamiltonian,
    _is_multi_term_sum,
    _sort_hamiltonian_terms,
)


class TrotterizationStrategy(Protocol):
    """Trotterization strategy protocol."""

    @property
    def stateful(self) -> bool:
        """True if the strategy retains state across ``process_hamiltonian`` calls.
        This should be true for strategies that might re-process the Hamiltonian during execution.
        """
        ...

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """Trotterize the Hamiltonian."""
        ...


@dataclass(frozen=True)
class ExactTrotterization(TrotterizationStrategy):
    """Exact Trotterization strategy."""

    keep_fraction: float | None = None
    """Fraction of terms to keep by coefficient magnitude (largest first). Must be in (0, 1]. If None, keep all terms."""
    keep_top_n: int | None = None
    """Number of top terms to keep by coefficient magnitude. Must be >= 1. If None, keep all terms. Mutually exclusive with keep_fraction."""

    # Caches processed Hamiltonian to avoid re-sorting and re-slicing when the
    # same Hamiltonian is passed repeatedly (e.g. across optimizer evaluations).
    _cache: dict = field(default_factory=dict, compare=False, hash=False)

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
            self.keep_top_n <= 0 or not isinstance(self.keep_top_n, int)
        ):
            raise ValueError(
                f"keep_top_n must be a positive integer (>= 1), got {self.keep_top_n}"
            )

    @property
    def stateful(self) -> bool:
        # Despite having a _cache, this strategy is stateless because it only
        # uses the cache as memoization, not as state.
        return False

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """Exact Trotterize the Hamiltonian."""
        if self.keep_fraction is None and self.keep_top_n is None:
            return hamiltonian.simplify()

        if hamiltonian in self._cache:
            return self._cache[hamiltonian]

        if self.keep_fraction is not None and self.keep_fraction == 1.0:
            warn(
                "keep_fraction is 1.0 (no truncation); returning the full Hamiltonian.",
                UserWarning,
            )
            return hamiltonian.simplify()

        if self.keep_top_n is not None and self.keep_top_n >= _hamiltonian_term_count(
            hamiltonian
        ):
            warn(
                "keep_top_n is greater than or equal to the number of terms; "
                "returning the full Hamiltonian.",
                UserWarning,
            )
            return hamiltonian.simplify()

        non_id_terms, constant = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(non_id_terms):
            raise ValueError("Hamiltonian contains only constant terms.")
        sorted_non_id_terms = _sort_hamiltonian_terms(non_id_terms, order="magnitude")

        if not _is_multi_term_sum(sorted_non_id_terms):
            return (sorted_non_id_terms + constant * qml.Identity()).simplify()

        if self.keep_top_n is not None:
            slice_idx = -self.keep_top_n

        if self.keep_fraction is not None:
            absolute_coeffs = np.abs(sorted_non_id_terms.terms()[0])
            target = absolute_coeffs.sum() * self.keep_fraction
            cumsum_from_end = np.cumsum(absolute_coeffs[::-1])
            n_keep = np.searchsorted(cumsum_from_end, target, side="left") + 1
            slice_idx = -min(n_keep, len(absolute_coeffs))

        coeffs, terms = sorted_non_id_terms.terms()
        sliced_operands = [
            c * t for c, t in zip(list(coeffs)[slice_idx:], list(terms)[slice_idx:])
        ]
        if constant != 0:
            sliced_operands.append(constant * qml.Identity())
        result = qml.sum(*sliced_operands).simplify()

        self._cache[hamiltonian] = result

        return result


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

    # Caches the (keep_hamiltonian, to_sample_hamiltonian) split so we avoid
    # recomputing the deterministic part when the same Hamiltonian is passed
    # repeatedly; only the sampling step changes each call.
    _cache: dict = field(default_factory=dict, compare=False, hash=False)
    _rng: np.random.Generator = field(init=False, compare=False, hash=False)
    _last_sampled_terms: list | None = field(
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

        if self.sampling_strategy not in ["uniform", "weighted"]:
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

    def process_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
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

        triggered_exact_trotterization = (
            True
            if self.keep_fraction is not None or self.keep_top_n is not None
            else False
        )

        if hamiltonian in self._cache:
            keep_hamiltonian, to_sample_hamiltonian = self._cache[hamiltonian]
        else:
            if triggered_exact_trotterization:
                keep_hamiltonian = ExactTrotterization(
                    keep_fraction=self.keep_fraction, keep_top_n=self.keep_top_n
                ).process_hamiltonian(hamiltonian)
                to_sample_hamiltonian = (hamiltonian - keep_hamiltonian).simplify()
            else:
                keep_hamiltonian = None
                to_sample_hamiltonian = hamiltonian.simplify()

            self._cache[hamiltonian] = (keep_hamiltonian, to_sample_hamiltonian)

            if triggered_exact_trotterization and qml.equal(
                keep_hamiltonian, hamiltonian
            ):
                warn(
                    "All terms were kept; there are no terms left to sample. "
                    "Returning the full Hamiltonian.",
                    UserWarning,
                )
                return hamiltonian

        if self.sampling_budget is None:
            if keep_hamiltonian is None:
                return hamiltonian.simplify()
            return keep_hamiltonian

        if triggered_exact_trotterization and qml.equal(keep_hamiltonian, hamiltonian):
            return hamiltonian

        # to_sample_hamiltonian already set above (from cache or computation)
        terms_list = list(_get_terms_iterable(to_sample_hamiltonian))
        if len(terms_list) == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            if keep_hamiltonian is None:
                return qml.Hamiltonian([], [])
            return keep_hamiltonian

        if not _is_multi_term_sum(to_sample_hamiltonian):
            # Single term: no sampling needed, return as-is.
            sampled_terms = [to_sample_hamiltonian]
        else:
            absolute_coeffs = np.abs(to_sample_hamiltonian.terms()[0])
            coeff_sum = absolute_coeffs.sum()
            if coeff_sum == 0:
                warn(
                    "All term coefficients are zero; returning the kept Hamiltonian.",
                    UserWarning,
                )
                return keep_hamiltonian
            probs = (
                (absolute_coeffs / coeff_sum).tolist()
                if self.sampling_strategy == "weighted"
                else None
            )
            indices = self._rng.choice(
                len(terms_list),
                size=self.sampling_budget,
                replace=True,
                p=probs,
            )

            # --- QDrift coefficient rescaling ---
            # Each sampled term must be rescaled so that E[H_sampled] = H.
            if self.sampling_strategy == "weighted":
                # Weighted (p_i = |c_i|/λ): scale by λ / (L · |c_i|)
                sampled_terms = [
                    (coeff_sum / (self.sampling_budget * absolute_coeffs[i]))
                    * terms_list[i]
                    for i in indices
                ]
            else:
                # Uniform (p_i = 1/N): scale by N/L
                n_terms = len(terms_list)
                sampled_terms = [
                    (n_terms / self.sampling_budget) * terms_list[i] for i in indices
                ]

        # Store individual sampled terms for Campbell's faithful protocol.
        # Each term retains its rescaled coefficient for individual evolution gates.
        all_individual_terms = list(sampled_terms)
        if keep_hamiltonian is not None:
            if _is_multi_term_sum(keep_hamiltonian):
                all_individual_terms.extend(_get_terms_iterable(keep_hamiltonian))
            else:
                all_individual_terms.append(keep_hamiltonian)
        object.__setattr__(self, "_last_sampled_terms", all_individual_terms)

        sampled_sum = qml.ops.Sum(*sampled_terms)
        if keep_hamiltonian is not None:
            sampled_sum = sampled_sum + keep_hamiltonian
        return sampled_sum.simplify()
