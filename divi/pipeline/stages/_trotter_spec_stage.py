# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._core import _assert_hermitian_spo
from divi.hamiltonians import (
    ExactTrotterization,
    TrotterizationStrategy,
)
from divi.hamiltonians._term_ops import _clean_hamiltonian_spo
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


class TrotterSpecStage(SpecStage[SparsePauliOp]):
    """SpecStage that turns a Hamiltonian into a batch of MetaCircuits via a TrotterizationStrategy.

    Accepts a :class:`~qiskit.quantum_info.SparsePauliOp`, runs the strategy
    to obtain one or more SPO samples, and invokes
    ``meta_circuit_factory(processed_spo, ham_id)`` for each.
    """

    @property
    def axis_name(self) -> str:
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
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Transform Hamiltonian into a keyed batch of MetaCircuits (one per strategy output)."""
        spo_clean, strategy, n_samples, token = self._prepare(batch)

        metas: MetaCircuitBatch = {}
        for ham_id in range(n_samples):
            processed = strategy.process_hamiltonian(spo_clean)
            meta = self._meta_circuit_factory(processed, ham_id)
            metas[(("ham", ham_id),)] = meta

        return metas, token

    def dry_expand(
        self, batch: SparsePauliOp, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
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
            strategy.process_hamiltonian(spo_clean), 0
        )
        metas: MetaCircuitBatch = {
            (("ham", ham_id),): prototype for ham_id in range(n_samples)
        }
        return metas, token

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
