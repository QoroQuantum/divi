# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for pipeline tests (stages, execute fn, meta circuit factory)."""

from typing import cast

import numpy as np
import pennylane as qml

from divi.circuits import MetaCircuit
from divi.pipeline import PipelineEnv, PipelineTrace
from divi.pipeline._compilation import _compile_batch
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    SpecStage,
    Stage,
    StageToken,
)
from divi.pipeline.stages import MeasurementStage


class DummySpecStage(SpecStage[str]):
    """Simple spec stage that emits a single logical circuit."""

    def __init__(self, meta: MetaCircuit | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._meta = meta or cast(MetaCircuit, object())

    def expand(
        self, items: str, env: PipelineEnv
    ) -> tuple[MetaCircuitBatch, StageToken]:
        return {(("spec", "circ"),): self._meta}, None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results


class FanoutAndSumStage(BundleStage):
    """Fan-out stage used to validate expansion lineage and reduce fan-in."""

    def __init__(self, branch_prefix: str, n_children: int) -> None:
        super().__init__(name=f"{type(self).__name__}:{branch_prefix}")
        self._branch_prefix = branch_prefix
        self._n_children = n_children

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: MetaCircuitBatch = {}
        for parent_key, meta in batch.items():
            for idx in range(self._n_children):
                child_key = parent_key + ((self._branch_prefix, idx),)
                out[child_key] = meta
        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        rolled_up: ChildResults = {}
        for child_key, value in results.items():
            # Strip this stage's axis (last axis we added in expand)
            parent_key = tuple(
                e
                for e in child_key
                if not (isinstance(e, tuple) and e[0] == self._branch_prefix)
            )
            rolled_up[parent_key] = rolled_up.get(parent_key, 0) + value
        return rolled_up


class StatefulFanoutStage(FanoutAndSumStage):
    """Like FanoutAndSumStage but stateful=True to exercise run_forward_pass cache/partial rerun."""

    @property
    def stateful(self) -> bool:
        return True


def two_group_meta() -> MetaCircuit:
    """MetaCircuit with 0.9*Z + 0.4*X for MeasurementStage to produce 2 groups."""
    qscript = qml.tape.QuantumScript(
        ops=[qml.Hadamard(0)],
        measurements=[qml.expval(0.9 * qml.Z(0) + 0.4 * qml.X(0))],
    )
    return MetaCircuit(source_circuit=qscript, symbols=np.array([], dtype=object))


def two_group_pipeline_stages(
    meta: MetaCircuit | None = None,
    fanout: tuple[str, int] | None = None,
) -> list[Stage]:
    """Stages: DummySpecStage(two_group_meta) -> MeasurementStage -> optional FanoutAndSum."""
    stages: list[Stage] = [
        DummySpecStage(meta=meta if meta is not None else two_group_meta()),
        MeasurementStage(),
    ]
    if fanout is not None:
        prefix, n = fanout
        stages.append(FanoutAndSumStage(prefix, n))
    return stages


def ones_execute_fn(
    trace: PipelineTrace,
    env: PipelineEnv,
) -> ChildResults:
    """Return 1 for each branch key so reduce stages get correct key structure (BranchKeys)."""
    try:
        _, lineage_by_label = _compile_batch(trace.final_batch)
        return {branch_key: 1 for branch_key in lineage_by_label.values()}
    except (ValueError, AttributeError):
        return {key: 1 for key in trace.final_batch}
