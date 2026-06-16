# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless trotterization strategies for Hamiltonian simulation."""

from dataclasses import dataclass
from typing import Literal, Protocol
from warnings import warn

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    _sort_hamiltonian_terms_spo,
    generate_empty_spo,
)


@dataclass(frozen=True)
class TrotterizationResult:
    """Explicit output of a trotterization decision."""

    effective_hamiltonian: SparsePauliOp
    """Simplified Hamiltonian used for observable construction."""

    sampled_terms: SparsePauliOp | None = None
    """Faithful sampled sequence with replacement multiplicities preserved."""


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

    def process_hamiltonian(
        self,
        hamiltonian: SparsePauliOp,
        *,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Return a trotterization result without retaining call state."""
        ...


@dataclass(frozen=True)
class ExactTrotterization(TrotterizationStrategy):
    """Exact Trotterization strategy."""

    keep_fraction: float | None = None
    """Fraction of terms to keep by coefficient magnitude (largest first). Must be in (0, 1]. If None, keep all terms."""
    keep_top_n: int | None = None
    """Number of top terms to keep by coefficient magnitude. Must be >= 1. If None, keep all terms. Mutually exclusive with keep_fraction."""

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

    def process_hamiltonian(
        self,
        hamiltonian: SparsePauliOp,
        *,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Truncate the Hamiltonian to its top-magnitude terms."""
        if self.keep_fraction is None and self.keep_top_n is None:
            return TrotterizationResult(hamiltonian.simplify())

        if _warn_truncation_no_op(
            self.keep_fraction, self.keep_top_n, hamiltonian.size
        ):
            return TrotterizationResult(hamiltonian.simplify())

        return TrotterizationResult(self._truncate_spo(hamiltonian))

    def _truncate_spo(self, spo: SparsePauliOp) -> SparsePauliOp:
        """Sort by |coeff|, slice to the kept tail, and re-attach the constant."""
        non_id_spo, constant = _clean_hamiltonian_spo(spo, raise_on_constant=True)
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
                [("", [], constant)],
                num_qubits=_require_qiskit_num_qubits(spo.num_qubits),
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

        if self.sampling_budget is not None and (
            not isinstance(self.sampling_budget, int) or self.sampling_budget < 1
        ):
            raise ValueError(
                f"sampling_budget must be a positive integer (>= 1), "
                f"got {self.sampling_budget}"
            )

        if self.n_hamiltonians_per_iteration < 1:
            raise ValueError(
                f"n_hamiltonians_per_iteration must be >= 1, got {self.n_hamiltonians_per_iteration}"
            )

    def process_hamiltonian(
        self,
        hamiltonian: SparsePauliOp,
        *,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
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
            return TrotterizationResult(hamiltonian.simplify())

        if hamiltonian.size == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            return TrotterizationResult(
                generate_empty_spo(_require_qiskit_num_qubits(hamiltonian.num_qubits))
            )

        triggered_exact_trotterization = (
            self.keep_fraction is not None or self.keep_top_n is not None
        )

        keep_spo = None
        if triggered_exact_trotterization:
            all_kept = (
                self.keep_fraction is not None and self.keep_fraction == 1.0
            ) or (self.keep_top_n is not None and self.keep_top_n >= hamiltonian.size)
            if all_kept:
                _warn_truncation_no_op(
                    self.keep_fraction, self.keep_top_n, hamiltonian.size
                )
                warn(
                    "All terms were kept; there are no terms left to sample. "
                    "Returning the full Hamiltonian.",
                    UserWarning,
                )
                return TrotterizationResult(hamiltonian)

            keep_spo = ExactTrotterization(
                keep_fraction=self.keep_fraction, keep_top_n=self.keep_top_n
            )._truncate_spo(hamiltonian)
            to_sample_spo = (hamiltonian - keep_spo).simplify(atol=0)
        else:
            to_sample_spo = hamiltonian.simplify()

        if self.sampling_budget is None:
            effective = hamiltonian.simplify() if keep_spo is None else keep_spo
            return TrotterizationResult(effective)

        if to_sample_spo.size == 0:
            warn(
                "No terms to sample; returning the kept Hamiltonian.",
                UserWarning,
            )
            if keep_spo is None:
                return TrotterizationResult(
                    generate_empty_spo(
                        _require_qiskit_num_qubits(hamiltonian.num_qubits)
                    )
                )
            return TrotterizationResult(keep_spo)

        absolute_coeffs = np.abs(to_sample_spo.coeffs)
        coeff_sum = absolute_coeffs.sum()
        if coeff_sum == 0:
            warn(
                "All term coefficients are zero; returning the kept Hamiltonian.",
                UserWarning,
            )
            if keep_spo is None:
                return TrotterizationResult(
                    generate_empty_spo(
                        _require_qiskit_num_qubits(hamiltonian.num_qubits)
                    )
                )
            return TrotterizationResult(keep_spo)
        if self.sampling_strategy == "weighted":
            probs = absolute_coeffs / coeff_sum
            # Guard against ``probs.sum() == 1 + ε`` for very large term
            # counts; ``np.random.choice`` rejects probabilities that
            # don't sum to exactly 1.
            probs /= probs.sum()
        else:
            probs = None
        if rng is None:
            rng = np.random.default_rng(self.seed)
        indices = rng.choice(
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
            effective = (sampled_spo + keep_spo).simplify()
        else:
            faithful_spo = sampled_spo
            effective = sampled_spo.simplify()
        return TrotterizationResult(
            effective_hamiltonian=effective,
            sampled_terms=faithful_spo,
        )
