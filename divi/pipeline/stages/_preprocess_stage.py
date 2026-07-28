# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Stage that applies a circuit preprocessor's transform to the batch."""

from copy import deepcopy
from typing import TYPE_CHECKING

from qiskit.dagcircuit import DAGCircuit

from divi.pipeline.abc import (
    BundleStage,
    MetaCircuitBatch,
    PipelineEnv,
    StageOutput,
)

if TYPE_CHECKING:
    from divi.circuits import MetaCircuit
    from divi.pipeline._preprocessor import CircuitPreprocessor


def _with_private_bodies(meta: "MetaCircuit") -> "MetaCircuit":
    """A copy of ``meta`` whose body DAGs are shared with nothing outside it.

    An analytic upstream stage hands the same DAG object to several batch entries,
    so this gives each entry its own. Bodies within one entry that were already the
    same object stay shared, as the upstream arranged them.
    """
    if not meta.circuit_bodies:
        return meta
    copies: dict[int, DAGCircuit] = {}
    for _, dag in meta.circuit_bodies:
        if id(dag) not in copies:
            copies[id(dag)] = deepcopy(dag)
    bodies = tuple((tag, copies[id(dag)]) for tag, dag in meta.circuit_bodies)
    return meta.set_circuit_bodies(bodies)


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

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Run the transform against private copies of the incoming DAG bodies.

        The transform is the routine's identity, so it cannot be skipped. Copying
        first upholds the shared-reference contract even if ``preprocess`` mutates
        in place, which lets the upstream stage keep its analytic path.
        """
        return StageOutput(
            batch={
                key: self._preprocessor.preprocess(_with_private_bodies(meta))
                for key, meta in batch.items()
            }
        )
