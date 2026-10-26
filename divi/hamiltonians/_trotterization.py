# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless trotterization strategies for Hamiltonian simulation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
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
    """Simplified Hamiltonian the observable/circuit is built from. For a
    sampling result, the first step's draw."""

    sampled_terms: tuple[SparsePauliOp, ...] | None = None
    """For sampling strategies such as QDrift, one independent draw per Trotter
    step, each holding the drawn terms in order with repeats kept (one entry
    per draw). ``None`` for deterministic strategies."""

    def synthesize_evolution(
        self,
        qc: QuantumCircuit,
        *,
        time: float,
        n_steps: int,
        order: int,
        qubits: Sequence[int],
        basis_gates: Sequence[str],
    ) -> QuantumCircuit:
        """Append this result's time-evolution gates to ``qc``.

        Every path approximates ``exp(-i t H)``. A sampling result
        (``sampled_terms`` set) is first order and carries its own step count:
        each draw is one step at ``time / len(sampled_terms)``, applying one
        evolution gate per drawn term (sampling-with-replacement multiplicities
        kept). A deterministic result synthesises ``effective_hamiltonian`` via
        :class:`~qiskit.circuit.library.PauliEvolutionGate` over ``n_steps``
        steps (:class:`~qiskit.synthesis.LieTrotter` for ``order == 1``, else
        :class:`~qiskit.synthesis.SuzukiTrotter`). Every path is then lowered
        to ``basis_gates``; the returned circuit is a new object.

        Raises:
            ValueError: If a sampling result is synthesised with ``order != 1``.
            RuntimeError: If the circuit cannot be lowered to ``basis_gates``.
        """
        if self.sampled_terms is not None:
            if order != 1:
                raise ValueError(
                    f"Sampled Trotterization supports only order=1, got order={order}."
                )
            # Concatenating keeps every step's terms in order, one gate per draw.
            _spo_to_qiskit_basis_gates(
                qc,
                SparsePauliOp.sum(list(self.sampled_terms)),
                time / len(self.sampled_terms),
                qubits,
            )
        elif self.effective_hamiltonian.size >= 2:
            synthesis = (
                LieTrotter(reps=n_steps, preserve_order=True)
                if order == 1
                else SuzukiTrotter(order=order, reps=n_steps, preserve_order=True)
            )
            qc.append(
                PauliEvolutionGate(
                    self.effective_hamiltonian, time=time, synthesis=synthesis
                ),
                qubits,
            )
        else:
            _spo_to_qiskit_basis_gates(qc, self.effective_hamiltonian, time, qubits)

        # ``optimization_level=0`` keeps lowering to a gate-by-gate substitution.
        try:
            return transpile(qc, basis_gates=list(basis_gates), optimization_level=0)
        except Exception as exc:
            raise RuntimeError(
                "Failed to lower the Trotter-synthesised circuit to the "
                "requested basis-gate set. This usually means "
                "PauliEvolutionGate synthesis emitted a gate the QASM2 "
                f"emitter does not handle. Supported gates: {sorted(basis_gates)}."
            ) from exc


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
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Return a trotterization result for an ``n_steps`` evolution without
        retaining call state; sampling strategies draw independently per step."""
        ...

    def process_hamiltonian_batch(
        self,
        hamiltonian: SparsePauliOp,
        n_samples: int,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Return ``n_samples`` results from one Hamiltonian.

        The default calls :meth:`process_hamiltonian` ``n_samples`` times;
        strategies with expensive deterministic preprocessing (e.g. QDrift's
        keep/sample split) override this to share that work across the batch.
        """
        return [
            self.process_hamiltonian(hamiltonian, n_steps=n_steps, rng=rng)
            for _ in range(n_samples)
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
        hamiltonian: SparsePauliOp,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        """Truncate the Hamiltonian to its top-magnitude terms."""
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
    """Number of terms to sample from the remaining Hamiltonian per Trotter step; each step draws its own terms independently. If None, only kept terms are applied (equivalent to ``ExactTrotterization``)."""
    sampling_strategy: Literal["uniform", "weighted"] | Mapping[str, float] = field(
        default="uniform", hash=False
    )
    """How to sample terms — ``"uniform"`` (equal probability), ``"weighted"`` (by coefficient magnitude), or a mapping from Pauli label to sampling probability. A mapping must cover exactly the terms left to sample (after ``simplify()`` and any ``keep_*`` split), with every probability finite and > 0 and the probabilities summing to 1."""
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

        if isinstance(self.sampling_strategy, Mapping):
            labels = list(self.sampling_strategy)
            if not all(isinstance(label, str) for label in labels):
                raise ValueError(
                    f"sampling_strategy keys must be Pauli label strings, got {labels}"
                )
            try:
                probs = np.array(
                    [float(p) for p in self.sampling_strategy.values()], dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sampling_strategy probabilities must be real numbers, "
                    f"got {list(self.sampling_strategy.values())}"
                ) from exc
            if not np.all(np.isfinite(probs) & (probs > 0)):
                raise ValueError(
                    f"sampling_strategy probabilities must be finite and > 0, "
                    f"got {dict(zip(labels, probs.tolist()))}"
                )
            if not np.isclose(probs.sum(), 1.0, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"sampling_strategy probabilities must sum to 1, got {probs.sum()}"
                )
            object.__setattr__(
                self, "sampling_strategy", dict(zip(labels, probs.tolist()))
            )
        elif self.sampling_strategy not in {"uniform", "weighted"}:
            raise ValueError(
                f"Invalid sampling_strategy: {self.sampling_strategy}. Must be 'uniform' or 'weighted'."
            )

        if self.seed is not None:
            if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
                raise ValueError(f"seed must be an integer, got {self.seed}")
            if self.seed < 0:
                raise ValueError(f"seed must be non-negative, got {self.seed}")
            object.__setattr__(self, "seed", int(self.seed))

        if self.sampling_budget is not None:
            if (
                not isinstance(self.sampling_budget, Integral)
                or isinstance(self.sampling_budget, bool)
                or self.sampling_budget < 1
            ):
                raise ValueError(
                    f"sampling_budget must be a positive integer (>= 1), "
                    f"got {self.sampling_budget}"
                )
            object.__setattr__(self, "sampling_budget", int(self.sampling_budget))

        if not isinstance(self.n_hamiltonians_per_iteration, Integral) or isinstance(
            self.n_hamiltonians_per_iteration, bool
        ):
            raise ValueError(
                f"n_hamiltonians_per_iteration must be an integer, got "
                f"{self.n_hamiltonians_per_iteration}"
            )
        if self.n_hamiltonians_per_iteration < 1:
            raise ValueError(
                f"n_hamiltonians_per_iteration must be >= 1, got {self.n_hamiltonians_per_iteration}"
            )
        object.__setattr__(
            self,
            "n_hamiltonians_per_iteration",
            int(self.n_hamiltonians_per_iteration),
        )

    def process_hamiltonian(
        self,
        hamiltonian: SparsePauliOp,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        r"""Apply the ``QDrift`` randomised channel to a Hamiltonian.

        Implements the ``QDrift`` protocol (Campbell 2019): for H = Σ c_i P_i,
        randomly sample L terms for each of the ``n_steps`` Trotter steps,
        independently per step, and rescale their coefficients so that
        E[H_sampled] = H for every step.

        Rescaling rules (L = sampling_budget, λ = Σ\|c_i\|, N = #terms):
          - Weighted: term_i → (λ / (L · \|c_i\|)) · c_i · P_i
          - Uniform:  term_i → (N / L) · c_i · P_i
          - Mapping q: term_i → (1 / (L · q_i)) · c_i · P_i

        Raises:
            ValueError: If a mapping ``sampling_strategy`` does not cover
                exactly the terms left to sample.
        """
        prepared = self._plan(hamiltonian)
        if isinstance(prepared, TrotterizationResult):
            return prepared
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return self._sample(prepared, rng, n_steps)

    def process_hamiltonian_batch(
        self,
        hamiltonian: SparsePauliOp,
        n_samples: int,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Sample ``n_samples`` Hamiltonians, computing the deterministic
        keep/sample split once and drawing only the random terms per sample."""
        prepared = self._plan(hamiltonian)
        if isinstance(prepared, TrotterizationResult):
            return [prepared] * n_samples
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return [self._sample(prepared, rng, n_steps) for _ in range(n_samples)]

    def _plan(
        self, hamiltonian: SparsePauliOp
    ) -> "TrotterizationResult | _QDriftSamplingPlan":
        """Deterministic part shared by every sample of one batch.

        Returns a finished :class:`TrotterizationResult` for the cases needing
        no random draw, otherwise a :class:`_QDriftSamplingPlan` for
        :meth:`_sample`.
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
            _clean_hamiltonian_spo(hamiltonian, raise_on_constant=True)
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
            if self.sampling_budget is None:
                return TrotterizationResult(keep_spo)
            to_sample_spo = (hamiltonian - keep_spo).simplify(atol=0)
        else:
            to_sample_spo = hamiltonian.simplify()

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
        if isinstance(self.sampling_strategy, Mapping):
            labels = to_sample_spo.paulis.to_labels()
            missing = sorted(set(labels) - self.sampling_strategy.keys())
            if missing:
                raise ValueError(
                    f"sampling_strategy has no probability for the sampled terms {missing}"
                )
            unknown = sorted(self.sampling_strategy.keys() - set(labels))
            if unknown:
                raise ValueError(
                    f"sampling_strategy has probabilities for {unknown}, "
                    f"which are not among the sampled terms {sorted(labels)}"
                )
            probs = np.array([self.sampling_strategy[label] for label in labels])
        elif self.sampling_strategy == "weighted":
            probs = absolute_coeffs / coeff_sum
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
        self, plan: "_QDriftSamplingPlan", rng: np.random.Generator, n_steps: int
    ) -> TrotterizationResult:
        """Draw one independent set of terms per Trotter step from a prepared
        plan (the random part); ``effective_hamiltonian`` is the first step's."""
        # _plan returns a finished result (never a plan) when sampling_budget is None.
        budget = self.sampling_budget
        if budget is None:
            raise RuntimeError("_sample requires sampling_budget; call _plan first.")
        steps = tuple(self._draw_step(plan, rng, budget) for _ in range(n_steps))
        return TrotterizationResult(
            effective_hamiltonian=steps[0].simplify(),
            sampled_terms=steps,
        )

    def _draw_step(
        self, plan: "_QDriftSamplingPlan", rng: np.random.Generator, budget: int
    ) -> SparsePauliOp:
        """One step's ``budget`` rescaled draws followed by the kept terms."""
        to_sample_spo = plan.to_sample_spo
        indices = rng.choice(
            to_sample_spo.size, size=budget, replace=True, p=plan.probs
        )

        # Rescale so that E[H_sampled] = H.
        if self.sampling_strategy == "weighted":
            # p_i = |c_i|/λ → scale by λ / (L · |c_i|).
            scale = plan.coeff_sum / (budget * plan.absolute_coeffs[indices])
        elif self.sampling_strategy == "uniform":
            # p_i = 1/N → scale by N/L.
            scale = np.full(budget, to_sample_spo.size / budget)
        elif plan.probs is not None:
            scale = 1.0 / (budget * plan.probs[indices])
        else:
            raise RuntimeError("A mapping sampling_strategy requires plan.probs.")
        sampled_coeffs = scale * to_sample_spo.coeffs[indices]
        sampled_spo = SparsePauliOp(to_sample_spo.paulis[indices], sampled_coeffs)
        if plan.keep_spo is not None:
            return sampled_spo + plan.keep_spo
        return sampled_spo
