# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import pennylane as qp

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
    _hamiltonian_term_count,
    _is_empty_hamiltonian,
)
from divi.pipeline.abc import (
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    SpecStage,
    StageToken,
)
from divi.pipeline.transformations import (
    group_by_base_key,
    reduce_mean,
    reduce_merge_histograms,
)


class TrotterSpecStage(SpecStage[qp.operation.Operator]):
    """SpecStage that turns a Hamiltonian into a batch of MetaCircuits via a TrotterizationStrategy.

    Takes the initial_spec (a Hamiltonian), runs it through the strategy to obtain
    one or more Hamiltonian samples, and calls ``meta_circuit_factory(processed_hamiltonian, ham_id)``
    for each.
    """

    @property
    def axis_name(self) -> str | None:
        return "ham"

    @property
    def stateful(self) -> bool:
        return self._trotterization_strategy.stateful

    def __init__(
        self,
        trotterization_strategy: TrotterizationStrategy,
        meta_circuit_factory: Callable[..., MetaCircuit],
    ) -> None:
        """
        Args:
            trotterization_strategy: Strategy for term selection/sampling (e.g. ``ExactTrotterization``, ``QDrift``).
            meta_circuit_factory: Factory callable ``(hamiltonian, ham_id) -> MetaCircuit``.
        """
        super().__init__(name=type(self).__name__)

        self._trotterization_strategy = (
            trotterization_strategy
            if trotterization_strategy is not None
            else ExactTrotterization()
        )
        self._meta_circuit_factory = meta_circuit_factory

    def _prepare(
        self, items: qp.operation.Operator
    ) -> tuple[qp.operation.Operator, TrotterizationStrategy, int, dict]:
        """Validate input and compute the shared (hamiltonian, strategy, n_samples, token) tuple.

        Reused by :meth:`expand` and :meth:`dry_expand` so the Hamiltonian
        cleaning and token construction don't drift between paths.
        """
        hamiltonian = items

        if not isinstance(hamiltonian, qp.operation.Operator):
            raise TypeError(
                f"TrotterSpecStage expects a PennyLane Operator (Hamiltonian), got {type(hamiltonian).__name__}"
            )

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)

        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        strategy = self._trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)

        token = {
            "strategy": type(strategy).__name__,
            "n_terms": _hamiltonian_term_count(hamiltonian_clean),
            "n_qubits": len(hamiltonian_clean.wires),
            "n_samples": n_samples,
        }
        return hamiltonian_clean, strategy, n_samples, token

    def expand(
        self, items: qp.operation.Operator, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Transform Hamiltonian into a keyed batch of MetaCircuits (one per strategy output)."""
        hamiltonian_clean, strategy, n_samples, token = self._prepare(items)

        metas: dict[object, MetaCircuit] = {}
        for ham_id in range(n_samples):
            processed = strategy.process_hamiltonian(hamiltonian_clean)
            meta = self._meta_circuit_factory(processed, ham_id)
            metas[(("ham", ham_id),)] = meta

        return metas, token

    def dry_expand(
        self, items: qp.operation.Operator, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Analytic path: build one prototype MetaCircuit, fan it out ``n_samples`` times.

        For stochastic strategies (e.g. QDrift) each sample would in
        principle produce a slightly different DAG. Dry runs only count
        circuits, so a single prototype from ham_id=0 is reused — saving
        (n_samples - 1) expensive factory invocations. For the dominant
        deterministic case (``ExactTrotterization`` with ``n_samples=1``)
        this reduces to the same single factory call as :meth:`expand`.
        """
        hamiltonian_clean, strategy, n_samples, token = self._prepare(items)

        prototype = self._meta_circuit_factory(
            strategy.process_hamiltonian(hamiltonian_clean), 0
        )
        metas = {(("ham", ham_id),): prototype for ham_id in range(n_samples)}
        return metas, token

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        if not isinstance(token, dict):
            return {}
        return dict(token)

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        # Auto-detect: scalar results → average, dict results → merge histograms
        sample = next((v for vals in grouped.values() for v in vals), None)
        if isinstance(sample, dict):
            return reduce_merge_histograms(grouped)
        return reduce_mean(grouped)
