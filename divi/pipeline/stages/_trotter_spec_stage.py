# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pennylane as qml

from divi.circuits import MetaCircuit
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
    _clean_hamiltonian,
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


class TrotterSpecStage(SpecStage[qml.operation.Operator]):
    """SpecStage that turns a Hamiltonian into a batch of MetaCircuits via a TrotterizationStrategy.

    Takes the initial_spec (a Hamiltonian), runs it through the strategy to obtain
    one or more Hamiltonian samples, and calls meta_circuit_factory(processed_hamiltonian, ham_id)
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
            trotterization_strategy: Strategy for term selection/sampling (e.g. ExactTrotterization, QDrift).
            meta_circuit_factory: Factory callable ``(hamiltonian, ham_id) -> MetaCircuit``.
        """
        super().__init__(name=type(self).__name__)

        self._trotterization_strategy = (
            trotterization_strategy
            if trotterization_strategy is not None
            else ExactTrotterization()
        )
        self._meta_circuit_factory = meta_circuit_factory

    def expand(
        self, items: qml.operation.Operator, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Transform Hamiltonian into a keyed batch of MetaCircuits (one per strategy output)."""
        hamiltonian = items

        if not isinstance(hamiltonian, qml.operation.Operator):
            raise TypeError(
                f"TrotterSpecStage expects a PennyLane Operator (Hamiltonian), got {type(hamiltonian).__name__}"
            )

        hamiltonian_clean, _ = _clean_hamiltonian(hamiltonian)

        if _is_empty_hamiltonian(hamiltonian_clean):
            raise ValueError("Hamiltonian contains only constant terms.")

        strategy = self._trotterization_strategy
        n_samples = getattr(strategy, "n_hamiltonians_per_iteration", 1)

        metas: dict[object, MetaCircuit] = {}

        for ham_id in range(n_samples):
            processed = strategy.process_hamiltonian(hamiltonian_clean)
            meta = self._meta_circuit_factory(processed, ham_id)
            metas[(("ham", ham_id),)] = meta

        return metas, None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        # Auto-detect: scalar results → average, dict results → merge histograms
        sample = next((v for vals in grouped.values() for v in vals), None)
        if isinstance(sample, dict):
            return reduce_merge_histograms(grouped)
        return reduce_mean(grouped)
