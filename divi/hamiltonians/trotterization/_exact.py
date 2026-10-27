# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic trotterization with optional magnitude truncation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    _sort_hamiltonian_terms_spo,
)
from divi.hamiltonians.trotterization._base import (
    TrotterizationResult,
    TrotterizationStrategy,
    TrotterSpec,
    _warn_truncation_no_op,
)


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

        if self.keep_fraction is not None:
            if not 0 < self.keep_fraction <= 1:
                raise ValueError(
                    f"keep_fraction must be in (0, 1], got {self.keep_fraction}"
                )
            object.__setattr__(self, "keep_fraction", float(self.keep_fraction))

        if self.keep_top_n is not None:
            if (
                not isinstance(self.keep_top_n, Integral)
                or isinstance(self.keep_top_n, bool)
                or self.keep_top_n <= 0
            ):
                raise ValueError(
                    f"keep_top_n must be a positive integer (>= 1), got {self.keep_top_n}"
                )
            object.__setattr__(self, "keep_top_n", int(self.keep_top_n))

    def process_hamiltonian(
        self,
        spec: TrotterSpec,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Truncate the Hamiltonian to its top-magnitude terms."""
        hamiltonian = spec.hamiltonian
        if self.keep_fraction is None and self.keep_top_n is None:
            return TrotterizationResult(hamiltonian.simplify())

        _clean_hamiltonian_spo(hamiltonian, raise_on_constant=True)
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
