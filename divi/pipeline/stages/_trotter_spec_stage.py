# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._core import _assert_hermitian_spo
from divi.hamiltonians import (
    ExactTrotterization,
    QDrift,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import _clean_hamiltonian_spo
from divi.pipeline.abc import (
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    SpecStage,
    StageOutput,
    StageToken,
)
from divi.pipeline.transformations import (
    group_by_base_key,
    reduce_mean,
    reduce_merge_histograms,
)


class TrotterSpecStage(SpecStage[SparsePauliOp]):
    """SpecStage that turns a Hamiltonian into a batch of MetaCircuits via a TrotterizationStrategy.

    Accepts a :class:`~qiskit.quantum_info.SparsePauliOp`, runs the strategy
    to obtain one or more SPO samples, and invokes
    ``meta_circuit_factory(processed_spo, ham_id)`` for each.
    """

    @property
    def axis_name(self) -> str:
        return "ham"

    def cache_key_extras(self, env):
        """Invalidate the forward-pass cache per evaluation for QDrift.

        QDrift re-samples a fresh batch each optimizer evaluation, seeded
        deterministically from ``env.evaluation_counter``; folding the counter
        into the cache key reuses one sample across the cost and gradient
        passes of a single evaluation, then resamples on the next. Deterministic
        strategies (e.g. ``ExactTrotterization``) declare no extras and stay
        cached for the pipeline's lifetime.
        """
        if isinstance(self._trotterization_strategy, QDrift):
            return (env.evaluation_counter,)
        return ()

    def __init__(
        self,
        trotterization_strategy: TrotterizationStrategy,
        meta_circuit_factory: Callable[..., MetaCircuit],
    ) -> None:
        """
        Args:
            trotterization_strategy: Strategy for term selection/sampling (e.g. ``ExactTrotterization``, ``QDrift``).
            meta_circuit_factory: Factory callable
                ``(TrotterizationResult, ham_id) -> MetaCircuit``.
        """
        super().__init__(name=type(self).__name__)

        self._trotterization_strategy = (
            trotterization_strategy
            if trotterization_strategy is not None
            else ExactTrotterization()
        )
        self._meta_circuit_factory = meta_circuit_factory

    def _prepare(
        self, items: SparsePauliOp
    ) -> tuple[SparsePauliOp, TrotterizationStrategy, int, dict]:
        """Validate input and compute the shared ``(spo, strategy, n_samples, token)`` tuple.

        Shared between :meth:`expand` and :meth:`dry_expand`.
        """
        hamiltonian = items

        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError(
                f"TrotterSpecStage expects a SparsePauliOp, got {type(hamiltonian).__name__}"
            )

        _assert_hermitian_spo(hamiltonian)
        spo_clean, _ = _clean_hamiltonian_spo(hamiltonian)

        if spo_clean.size == 0:
            raise ValueError("Hamiltonian contains only constant terms.")

        strategy = self._trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)

        token = {
            "strategy": type(strategy).__name__,
            "n_terms": spo_clean.size,
            "n_qubits": spo_clean.num_qubits,
            "n_samples": n_samples,
        }
        return spo_clean, strategy, n_samples, token

    def expand(
        self, batch: SparsePauliOp, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Transform Hamiltonian into a keyed batch of MetaCircuits (one per strategy output)."""
        spo_clean, strategy, n_samples, token = self._prepare(batch)

        rng = self._rng_for_evaluation(strategy, env)
        results = strategy.process_hamiltonian_batch(spo_clean, n_samples, rng=rng)
        metas: MetaCircuitBatch = {
            (("ham", ham_id),): self._meta_circuit_factory(result, ham_id)
            for ham_id, result in enumerate(results)
        }

        return StageOutput(batch=metas, token=token)

    def dry_expand(
        self, batch: SparsePauliOp, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Analytic path: build one prototype MetaCircuit, fan it out ``n_samples`` times.

        For stochastic strategies (e.g. QDrift) each sample would in
        principle produce a slightly different DAG. Dry runs only count
        circuits, so a single prototype from ham_id=0 is reused — saving
        (n_samples - 1) expensive factory invocations. For the dominant
        deterministic case (``ExactTrotterization`` with ``n_samples=1``)
        this reduces to the same single factory call as :meth:`expand`.
        """
        spo_clean, strategy, n_samples, token = self._prepare(batch)

        prototype = self._meta_circuit_factory(
            strategy.process_hamiltonian(
                spo_clean, rng=self._rng_for_evaluation(strategy, env)
            ),
            0,
        )
        metas: MetaCircuitBatch = {
            (("ham", ham_id),): prototype for ham_id in range(n_samples)
        }
        return StageOutput(batch=metas, token=token)

    @staticmethod
    def _rng_for_evaluation(
        strategy: TrotterizationStrategy,
        env: PipelineEnv,
    ):
        if not isinstance(strategy, QDrift):
            return None
        if strategy.seed is None:
            if env.rng is None:
                return None
            seed = int(env.rng.integers(0, 2**63))
            return np.random.default_rng(seed)
        if env.evaluation_counter == 0:
            return np.random.default_rng(strategy.seed)
        return np.random.default_rng(
            np.random.SeedSequence([strategy.seed, env.evaluation_counter])
        )

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        if not isinstance(token, dict):
            return {}
        info = dict(token)
        # ExactTrotterization is deterministic — its n_samples is structurally
        # always 1 and adds no information for the user. For stochastic
        # strategies (QDrift et al.) the count is a load-bearing knob.
        if info.get("strategy") == "ExactTrotterization":
            info.pop("n_samples", None)
        return info

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        # Auto-detect: scalar results → average, dict results → merge histograms
        sample = next((v for vals in grouped.values() for v in vals), None)
        if isinstance(sample, dict):
            return reduce_merge_histograms(grouped)
        return reduce_mean(grouped)
