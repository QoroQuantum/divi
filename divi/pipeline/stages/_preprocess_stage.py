# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stage that applies a circuit preprocessor's transform to the batch."""

from typing import TYPE_CHECKING

from divi.pipeline.abc import (
    BundleStage,
    MetaCircuitBatch,
    PipelineEnv,
    StageOutput,
)

if TYPE_CHECKING:
    from divi.pipeline._preprocessor import CircuitPreprocessor


class PreprocessStage(BundleStage):
    """Apply a :class:`~divi.pipeline.CircuitPreprocessor`'s transform to the
    post-spec batch.

    Sits immediately after the spec stage. Transforms every ``MetaCircuit`` via
    ``preprocessor.preprocess`` before mitigation and the terminal measurement, so
    a single shared pipeline serves the cost, sampling, and metric routines.
    """

    def __init__(self, preprocessor: "CircuitPreprocessor") -> None:
        super().__init__(name="PreprocessStage")
        self._preprocessor = preprocessor

    @property
    def consumes_dag_bodies(self) -> bool:
        return self._preprocessor.consumes_dag_bodies

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        return StageOutput(
            batch={
                key: self._preprocessor.preprocess(meta) for key, meta in batch.items()
            }
        )
