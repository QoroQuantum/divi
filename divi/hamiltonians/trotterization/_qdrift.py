# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QDrift: randomised trotterization by sampling Hamiltonian terms."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    generate_empty_spo,
)
from divi.hamiltonians.trotterization._base import (
    TrotterizationResult,
    TrotterizationStrategy,
    TrotterSpec,
    _warn_truncation_no_op,
)
from divi.hamiltonians.trotterization._exact import ExactTrotterization
from divi.hamiltonians.trotterization._state_drift import (
    StateDriftCache,
    state_drift_distribution,
)


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
    sampling_strategy: (
        Literal["uniform", "weighted", "state_drift"] | Mapping[str, float]
    ) = field(default="uniform", hash=False)
    """How to sample terms — ``"uniform"`` (equal probability), ``"weighted"`` (by coefficient magnitude), ``"state_drift"``, or a mapping from Pauli label to sampling probability. A mapping must cover exactly the terms left to sample (after ``simplify()`` and any ``keep_*`` split), with every probability finite and > 0 and the probabilities summing to 1.

    ``"state_drift"`` chooses the probabilities from the initial state, observable and time of the evolution (StateDRIFT, `arXiv:2610.31515 <https://arxiv.org/abs/2610.31515>`_), so it works only in :class:`~divi.qprog.algorithms.TimeEvolution` with a single observable; it requires ``sampling_budget`` and excludes ``keep_fraction`` and ``keep_top_n``. Programs that supply no evolution raise when they first build their circuits."""
    seed: int | None = None
    """Random seed for reproducible sampling. If None, sampling is non-deterministic."""
    n_hamiltonians_per_iteration: int = 10
    """Number of Hamiltonian samples per cost evaluation; losses are averaged over them."""
    _state_drift_cache: StateDriftCache = field(
        default_factory=StateDriftCache, repr=False, compare=False, hash=False
    )
    """Densities of each evolution seen; ``replace`` and copies share it."""

    def __post_init__(self):
        if self.sampling_strategy == "state_drift":
            if self.keep_fraction is not None or self.keep_top_n is not None:
                raise ValueError(
                    "sampling_strategy='state_drift' samples every term; "
                    "keep_fraction and keep_top_n are not supported."
                )
            if self.sampling_budget is None:
                raise ValueError(
                    "sampling_strategy='state_drift' requires sampling_budget."
                )

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
        elif self.sampling_strategy not in {"uniform", "weighted", "state_drift"}:
            raise ValueError(
                f"Invalid sampling_strategy: {self.sampling_strategy}. Must be "
                "'uniform', 'weighted', 'state_drift' or a mapping."
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
        spec: TrotterSpec,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> TrotterizationResult:
        r"""Apply the ``QDrift`` randomised channel to ``spec.hamiltonian``.

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
                exactly the terms left to sample, or a ``"state_drift"``
                strategy gets a spec without an initial state and time, or
                with no observable or several.
            TypeError: If a ``"state_drift"`` strategy gets an observable that
                is not a ``SparsePauliOp`` or a non-numeric time.
        """
        prepared = self._plan(spec, n_steps)
        if isinstance(prepared, TrotterizationResult):
            return prepared
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return self._sample(prepared, rng, n_steps)

    def process_hamiltonian_batch(
        self,
        spec: TrotterSpec,
        n_samples: int,
        *,
        n_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[TrotterizationResult]:
        """Sample ``n_samples`` Hamiltonians, computing the deterministic
        keep/sample split once and drawing only the random terms per sample."""
        prepared = self._plan(spec, n_steps)
        if isinstance(prepared, TrotterizationResult):
            return [prepared] * n_samples
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return [self._sample(prepared, rng, n_steps) for _ in range(n_samples)]

    def _state_drift_probabilities(
        self, spec: TrotterSpec, n_steps: int
    ) -> dict[str, float]:
        """StateDRIFT probabilities for the evolution ``spec`` describes."""
        if spec.initial_state is None or spec.time is None:
            raise ValueError(
                "sampling_strategy='state_drift' needs the initial state and "
                "time of a time evolution; only TimeEvolution supplies them."
            )
        observable = spec.observable
        if isinstance(observable, Sequence) and not isinstance(observable, str):
            if len(observable) != 1:
                raise ValueError(
                    "sampling_strategy='state_drift' needs exactly one "
                    f"observable, got {len(observable)}."
                )
            (observable,) = observable
        if observable is None:
            raise ValueError(
                "sampling_strategy='state_drift' needs exactly one observable, "
                "got none."
            )
        if not isinstance(observable, SparsePauliOp):
            raise TypeError(
                "sampling_strategy='state_drift' needs a SparsePauliOp "
                f"observable, got {type(observable).__name__}."
            )
        time = spec.time
        if isinstance(time, bool) or not isinstance(time, Real):
            raise TypeError(
                "sampling_strategy='state_drift' needs a numeric time, got "
                f"{type(time).__name__}."
            )
        budget = self.sampling_budget
        if budget is None:
            raise RuntimeError("sampling_budget is not set.")
        return state_drift_distribution(
            self._state_drift_cache.densities(
                spec.hamiltonian, observable, spec.initial_state
            ),
            time=float(time),
            n_samples=n_steps * budget,
            n_trajectories=self.n_hamiltonians_per_iteration,
        )

    def _plan(
        self,
        spec: TrotterSpec,
        n_steps: int,
    ) -> TrotterizationResult | _QDriftSamplingPlan:
        """Deterministic part shared by every sample of one batch.

        Returns a finished :class:`TrotterizationResult` for the cases needing
        no random draw, otherwise a :class:`_QDriftSamplingPlan` for
        :meth:`_sample`.
        """
        hamiltonian = spec.hamiltonian
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
        distribution = self.sampling_strategy
        if distribution == "state_drift":
            distribution = self._state_drift_probabilities(spec, n_steps)
        if isinstance(distribution, Mapping):
            labels = to_sample_spo.paulis.to_labels()
            missing = sorted(set(labels) - distribution.keys())
            if missing:
                raise ValueError(
                    f"sampling_strategy has no probability for the sampled terms {missing}"
                )
            unknown = sorted(distribution.keys() - set(labels))
            if unknown:
                raise ValueError(
                    f"sampling_strategy has probabilities for {unknown}, "
                    f"which are not among the sampled terms {sorted(labels)}"
                )
            probs = np.array([distribution[label] for label in labels])
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
        self, plan: _QDriftSamplingPlan, rng: np.random.Generator, n_steps: int
    ) -> TrotterizationResult:
        """Draw one independent set of terms per Trotter step from a prepared
        plan (the random part); ``effective_hamiltonian`` is the first step's."""
        budget = self.sampling_budget
        if budget is None:
            raise RuntimeError("sampling_budget is not set.")
        steps = tuple(self._draw_step(plan, rng, budget) for _ in range(n_steps))
        return TrotterizationResult(
            effective_hamiltonian=steps[0].simplify(),
            sampled_terms=steps,
        )

    def _draw_step(
        self, plan: _QDriftSamplingPlan, rng: np.random.Generator, budget: int
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
