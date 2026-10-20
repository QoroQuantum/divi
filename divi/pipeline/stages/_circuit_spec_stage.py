# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that wraps pre-built MetaCircuit(s) into a pipeline batch."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from divi.circuits import MetaCircuit
from divi.pipeline._dry_run import _two_qubit_depth
from divi.pipeline._result_keys_operations import group_by_base_key
from divi.pipeline.abc import (
    ChildResults,
    MetaCircuitBatch,
    PipelineEnv,
    SpecStage,
    StageOutput,
    StageToken,
)

#: Accepted input types for ``CircuitSpecStage.expand``.
CircuitSpec = (
    MetaCircuit
    | Sequence[MetaCircuit]
    | npt.NDArray[np.object_]
    | Mapping[str, MetaCircuit]
)


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

    @staticmethod
    def _convert_by_shape(items, convert, *, single_type, expected: str):
        """Apply *convert* across a single item, a sequence, or a mapping.

        ``str`` is rejected before the ``Sequence`` branch: it satisfies that
        protocol and would otherwise be converted character by character.
        """
        if isinstance(items, single_type):
            return convert(items)
        if isinstance(items, str):
            raise TypeError(f"{expected}, got str")
        if isinstance(items, Mapping):
            return {key: convert(value) for key, value in items.items()}
        if isinstance(items, Sequence) or (
            isinstance(items, np.ndarray) and items.ndim == 1
        ):
            return [convert(value) for value in items]
        raise TypeError(f"{expected}, got {type(items).__name__}")

    def expand(
        self, batch: CircuitSpec, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Wrap input MetaCircuit(s) into a keyed batch."""
        if isinstance(batch, MetaCircuit):
            out: MetaCircuitBatch = {(("circuit", 0),): batch}
            fmt = "single"
        elif isinstance(batch, Mapping):
            out = {(("circuit", name),): meta for name, meta in batch.items()}
            fmt = "mapping"
        elif isinstance(batch, Sequence) or (
            isinstance(batch, np.ndarray) and batch.ndim == 1
        ):
            out = {(("circuit", i),): meta for i, meta in enumerate(batch)}
            fmt = "sequence"
        else:
            raise TypeError(
                f"CircuitSpecStage expects a MetaCircuit, sequence, or mapping, "
                f"got {type(batch).__name__}"
            )
        return StageOutput(batch=out, token=fmt)

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        meta = next(iter(batch.values()), None)
        if meta is None or not meta.circuit_bodies:
            return {}
        # Use the first body variant — QEM/twirl expansions share qubit
        # layout and gate count with the original.
        _, dag = meta.circuit_bodies[0]
        arity = Counter(len(node.qargs) for node in dag.op_nodes())
        return {
            "n_qubits": meta.n_qubits,
            "n_gates": dag.size(),
            "n_1q_gates": arity[1],
            "n_2q_gates": arity[2],
            "depth": dag.depth(),
            "depth_2q": _two_qubit_depth(dag),
        }

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Strip the ``'circuit'`` axis from result keys."""
        if self.axis_name in env.axes_to_preserve:
            return results
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped.items()
        }
