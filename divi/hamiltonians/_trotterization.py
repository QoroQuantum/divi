# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless trotterization strategies for Hamiltonian simulation."""

from dataclasses import dataclass
from typing import Literal, Protocol
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    _sort_hamiltonian_terms_spo,
    _spo_to_qiskit_basis_gates,
    generate_empty_spo,
)


@dataclass(frozen=True)
class TrotterizationResult:
    """Output of one trotterization step: the Hamiltonian to build the circuit
    from, plus (for sampling strategies) the exact sequence of sampled terms."""

    effective_hamiltonian: SparsePauliOp
    """Simplified Hamiltonian the observable/circuit is built from."""

    sampled_terms: SparsePauliOp | None = None
    """For sampling strategies such as QDrift, the drawn terms in order with
    repeats kept (one entry per draw). ``None`` for deterministic strategies."""

    def synthesize_evolution(
        self,
        qc: QuantumCircuit,
        *,
        time: float,
        n_steps: int,
        order: int,
        qubits: list[int],
        basis_gates: list[str],
    ) -> QuantumCircuit:
        """Append this result's time-evolution gates to ``qc``.

        A sampling result (``sampled_terms`` set) applies one evolution gate
        per sampled term — preserving sampling-with-replacement multiplicities
        — repeated ``n_steps`` times at ``time / n_steps`` per step. A
        deterministic result synthesizes ``exp(-i t H)`` from
        ``effective_hamiltonian`` via
        :class:`~qiskit.circuit.library.PauliEvolutionGate`
        (:class:`~qiskit.synthesis.LieTrotter` for ``order == 1``, else
        :class:`~qiskit.synthesis.SuzukiTrotter`), then lowers the circuit to
        ``basis_gates``.

        Adjoint evolution is realized via negative time; single-term
        Hamiltonians use positive time to preserve the ``exp(-i t H)`` sign
        convention even when ``H`` carries its own coefficient sign. Returns the
        resulting circuit (a new object when synthesis required transpilation,
        otherwise ``qc``).
        """
        if self.sampled_terms is not None:
            step_time = -time / n_steps
            for _ in range(n_steps):
                _spo_to_qiskit_basis_gates(qc, self.sampled_terms, step_time, qubits)
            return qc

        if self.effective_hamiltonian.size >= 2:
            synthesis = (
                LieTrotter(reps=n_steps, preserve_order=True)
                if order == 1
                else SuzukiTrotter(order=order, reps=n_steps, preserve_order=True)
            )
            qc.append(
                PauliEvolutionGate(
                    self.effective_hamiltonian, time=-time, synthesis=synthesis
                ),
                qubits,
            )
            # Lower to the gate set the QASM body emitter accepts. Trotter
            # synthesis can emit ``rxx``/``ryy``/``rzz``-style compound rotations
            # the QASM2 emitter raises on; ``optimization_level=0`` keeps it to a
            # cheap gate-by-gate substitution.
            try:
                return transpile(qc, basis_gates=basis_gates, optimization_level=0)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to lower the Trotter-synthesised circuit to the "
                    "requested basis-gate set. This usually means "
                    "PauliEvolutionGate synthesis emitted a gate the QASM2 "
                    f"emitter does not handle. Supported gates: {sorted(basis_gates)}."
                ) from exc

        # Single-term Hamiltonian — positive-time convention.
        _spo_to_qiskit_basis_gates(qc, self.effective_hamiltonian, time, qubits)
        return qc


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

    def process_hamiltonian_batch(
        self,
        hamiltonian: SparsePauliOp,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Return ``n_samples`` results from one Hamiltonian.

        The default calls :meth:`process_hamiltonian` ``n_samples`` times;
        strategies with expensive deterministic preprocessing (e.g. QDrift's
        keep/sample split) override this to share that work across the batch.
        """
        return [
            self.process_hamiltonian(hamiltonian, rng=rng) for _ in range(n_samples)
        ]


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
class _QDriftSamplingPlan:
    """Deterministic QDrift keep/sample split, reused across one batch's draws."""

    to_sample_spo: SparsePauliOp
    keep_spo: SparsePauliOp | None
    absolute_coeffs: npt.NDArray[np.floating]
    coeff_sum: float
    probs: npt.NDArray[np.floating] | None


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
        prepared = self._plan(hamiltonian)
        if isinstance(prepared, TrotterizationResult):
            return prepared
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return self._sample(prepared, rng)

    def process_hamiltonian_batch(
        self,
        hamiltonian: SparsePauliOp,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Sample ``n_samples`` Hamiltonians, computing the deterministic
        keep/sample split once and drawing only the random terms per sample."""
        prepared = self._plan(hamiltonian)
        if isinstance(prepared, TrotterizationResult):
            return [prepared] * n_samples
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return [self._sample(prepared, rng) for _ in range(n_samples)]

    def _plan(
        self, hamiltonian: SparsePauliOp
    ) -> "TrotterizationResult | _QDriftSamplingPlan":
        """Deterministic part shared by every sample of one batch.

        Returns a finished :class:`TrotterizationResult` for the cases needing
        no random draw, otherwise a :class:`_QDriftSamplingPlan` for :meth:`_draw`.
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
        return _QDriftSamplingPlan(
            to_sample_spo=to_sample_spo,
            keep_spo=keep_spo,
            absolute_coeffs=absolute_coeffs,
            coeff_sum=coeff_sum,
            probs=probs,
        )

    def _sample(
        self, plan: "_QDriftSamplingPlan", rng: np.random.Generator
    ) -> TrotterizationResult:
        """Draw one sampled Hamiltonian from a prepared plan (the random part)."""
        # _plan returns a finished result (never a plan) when sampling_budget
        # is None, so reaching _sample guarantees it is set.
        if self.sampling_budget is None:
            raise RuntimeError("_sample requires sampling_budget; call _plan first.")
        to_sample_spo = plan.to_sample_spo
        indices = rng.choice(
            to_sample_spo.size,
            size=self.sampling_budget,
            replace=True,
            p=plan.probs,
        )

        # Rescale so that E[H_sampled] = H.
        if self.sampling_strategy == "weighted":
            # p_i = |c_i|/λ → scale by λ / (L · |c_i|).
            scale = plan.coeff_sum / (
                self.sampling_budget * plan.absolute_coeffs[indices]
            )
        else:
            # p_i = 1/N → scale by N/L.
            scale = np.full(
                self.sampling_budget,
                to_sample_spo.size / self.sampling_budget,
            )
        sampled_coeffs = scale * to_sample_spo.coeffs[indices]
        sampled_spo = SparsePauliOp(to_sample_spo.paulis[indices], sampled_coeffs)

        if plan.keep_spo is not None:
            faithful_spo = sampled_spo + plan.keep_spo
            effective = (sampled_spo + plan.keep_spo).simplify()
        else:
            faithful_spo = sampled_spo
            effective = sampled_spo.simplify()
        return TrotterizationResult(
            effective_hamiltonian=effective,
            sampled_terms=faithful_spo,
        )
