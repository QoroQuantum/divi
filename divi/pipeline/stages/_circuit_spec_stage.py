# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that wraps pre-built MetaCircuit(s) into a pipeline batch."""

from collections.abc import Mapping, Sequence
from typing import Any

from divi.circuits import MetaCircuit
from divi.pipeline.abc import (
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    SpecStage,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key

#: Accepted input types for ``CircuitSpecStage.expand``.
CircuitSpec = MetaCircuit | Sequence[MetaCircuit] | Mapping[str, MetaCircuit]


class CircuitSpecStage(SpecStage[CircuitSpec]):
    """SpecStage that wraps one or more pre-built MetaCircuits into a batch.

    Accepts three input shapes:

    - A single ``MetaCircuit`` → ``{(("circuit", 0),): meta}``
    - A ``Sequence[MetaCircuit]`` → indexed by position
    - A ``Mapping[str, MetaCircuit]`` → indexed by key name

    Use this when the MetaCircuit(s) are already constructed and don't
    need Hamiltonian decomposition.
    """

    @property
    def axis_name(self) -> str:
        return "circuit"

    def __init__(self) -> None:
        super().__init__(name=type(self).__name__)

    def expand(
        self, batch: CircuitSpec, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Wrap input MetaCircuit(s) into a keyed batch."""
        if isinstance(batch, MetaCircuit):
            out: MetaCircuitBatch = {(("circuit", 0),): batch}
            fmt = "single"
        elif isinstance(batch, Mapping):
            out = {(("circuit", name),): meta for name, meta in batch.items()}
            fmt = "mapping"
        elif isinstance(batch, Sequence):
            out = {(("circuit", i),): meta for i, meta in enumerate(batch)}
            fmt = "sequence"
        else:
            raise TypeError(
                f"CircuitSpecStage expects a MetaCircuit, sequence, or mapping, "
                f"got {type(batch).__name__}"
            )
        return out, fmt

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        meta = next(iter(batch.values()), None)
        if meta is None or not meta.circuit_bodies:
            return {}
        # Use the first body variant — QEM/twirl expansions share qubit
        # layout and gate count with the original.
        _, dag = meta.circuit_bodies[0]
        n_1q = sum(1 for node in dag.op_nodes() if len(node.qargs) == 1)
        n_2q = sum(1 for node in dag.op_nodes() if len(node.qargs) == 2)
        return {
            "n_qubits": meta.n_qubits,
            "n_gates": dag.size(),
            "n_1q_gates": n_1q,
            "n_2q_gates": n_2q,
        }

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Strip the ``'circuit'`` axis from result keys."""
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped.items()
        }
