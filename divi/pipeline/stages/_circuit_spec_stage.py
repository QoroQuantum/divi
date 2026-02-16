# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that wraps pre-built MetaCircuit(s) into a pipeline batch."""

from collections.abc import Mapping, Sequence

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
        self, items: CircuitSpec, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Wrap input MetaCircuit(s) into a keyed batch."""
        if isinstance(items, MetaCircuit):
            batch: MetaCircuitBatch = {(("circuit", 0),): items}
            fmt = "single"
        elif isinstance(items, Mapping):
            batch = {(("circuit", name),): meta for name, meta in items.items()}
            fmt = "mapping"
        elif isinstance(items, Sequence):
            batch = {(("circuit", i),): meta for i, meta in enumerate(items)}
            fmt = "sequence"
        else:
            raise TypeError(
                f"CircuitSpecStage expects a MetaCircuit, sequence, or mapping, "
                f"got {type(items).__name__}"
            )
        return batch, fmt

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Strip the ``'circuit'`` axis from result keys."""
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped.items()
        }
