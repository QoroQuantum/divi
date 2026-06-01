# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that converts PennyLane QuantumScript(s) or QNode(s) into a pipeline batch."""

from collections.abc import Mapping, Sequence

import numpy as np
import pennylane as qp
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP
from pennylane.workflow.qnode import QNode

from divi.circuits import (
    MetaCircuit,
    qscript_to_meta,
)
from divi.circuits._pennylane_utils import (
    _qnode_to_symbolic_qscript,
    _validate_single_measurement,
)
from divi.pipeline.abc import MetaCircuitBatch, PipelineEnv, StageToken
from divi.pipeline.stages import CircuitSpecStage

#: Input types accepted by :class:`PennyLaneSpecStage`.
PennyLaneInput = qp.tape.QuantumScript | QNode

_SUPPORTED_MEASUREMENTS = (ProbabilityMP, ExpectationMP, CountsMP)


class PennyLaneSpecStage(CircuitSpecStage):
    """SpecStage that converts PennyLane circuits into MetaCircuit(s).

    Accepts ``QuantumScript`` or ``QNode`` objects (single, sequence, or
    mapping). QNodes are traced into a symbolic ``QuantumScript`` — the
    trainable arguments are seeded with ``sympy`` symbols — before conversion.

    **QNode parameter handling:**

    - **Scalar parameters** (``def circuit(x, y)``) become sympy symbols.
    - **Single 1-D array parameter** (``def circuit(params)``), including
      PennyLane templates such as ``AngleEmbedding`` and ``IQPEmbedding``, is
      auto-sized to the device wire count and traced symbolically.
    - **Structured-shape or multiple array parameters** (e.g.
      ``StronglyEntanglingLayers``, or ``circuit(inputs, weights)``) can't be
      inferred here. Pass a ``QuantumScript`` with explicit sympy symbols, or
      use :class:`~divi.qprog.algorithms.CustomVQA`'s ``arg_shapes`` /
      ``data_arg`` to declare the shapes.

    **Supported measurements:** ``probs()``, ``expval()``, and ``counts()``.
    Other PennyLane measurement types (``sample``, ``state``, ``var``, etc.)
    are not supported by the pipeline execution backend.
    """

    # pyrefly: ignore[bad-override]
    def expand(
        self,
        batch: PennyLaneInput | Sequence[PennyLaneInput] | Mapping[str, PennyLaneInput],
        env: PipelineEnv,
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Convert PennyLane circuit(s) to MetaCircuit(s) and build a keyed batch."""
        converted = self._convert(batch)
        self._reject_unbindable_param_sets(converted, env)
        return super().expand(converted, env)

    @staticmethod
    def _reject_unbindable_param_sets(
        converted: MetaCircuit | list[MetaCircuit] | dict[str, MetaCircuit],
        env: PipelineEnv,
    ) -> None:
        """Raise when ``env.param_sets`` carries columns but a converted circuit
        has no bindable parameters.

        Without this guard a concrete-valued tape (no symbolic gate parameters,
        no explicit ``trainable_params`` subset) silently drops the supplied
        parameter sets instead of binding them — producing a parameter-
        independent result. Mark the slots trainable, or pass parametric gates.
        """
        param_sets = np.asarray(env.param_sets, dtype=float)
        n_columns = param_sets.shape[1] if param_sets.ndim == 2 else 0
        if n_columns == 0:
            return
        if isinstance(converted, MetaCircuit):
            metas = [converted]
        elif isinstance(converted, Mapping):
            metas = list(converted.values())
        else:
            metas = list(converted)
        if any(len(m.parameters) == 0 for m in metas):
            raise ValueError(
                f"env.param_sets has {n_columns} parameter column(s) but a "
                "converted circuit exposes no bindable parameters. A concrete-"
                "valued QuantumScript binds nothing — set qscript.trainable_params "
                "to the slots to bind, or use parametric gates."
            )

    @staticmethod
    def _pennylane_to_meta(item: PennyLaneInput) -> MetaCircuit:
        """Convert a single QuantumScript or QNode into a MetaCircuit."""
        if isinstance(item, QNode):
            item = _qnode_to_symbolic_qscript(item)
        _validate_single_measurement(
            item,
            allowed=_SUPPORTED_MEASUREMENTS,
            caller="PennyLaneSpecStage",
            description="probs(), expval(), or counts()",
        )
        return qscript_to_meta(item)

    @staticmethod
    def _convert(
        items: PennyLaneInput | Sequence[PennyLaneInput] | Mapping[str, PennyLaneInput],
    ) -> MetaCircuit | list[MetaCircuit] | dict[str, MetaCircuit]:
        """Dispatch input shape and convert each input to MetaCircuit."""
        if isinstance(items, (qp.tape.QuantumScript, QNode)):
            return PennyLaneSpecStage._pennylane_to_meta(items)
        if isinstance(items, str):
            raise TypeError(
                f"PennyLaneSpecStage expects a QuantumScript, QNode, sequence, or "
                f"mapping, got str"
            )
        if isinstance(items, Mapping):
            return {
                k: PennyLaneSpecStage._pennylane_to_meta(v) for k, v in items.items()
            }
        if isinstance(items, Sequence):
            return [PennyLaneSpecStage._pennylane_to_meta(v) for v in items]
        raise TypeError(
            f"PennyLaneSpecStage expects a QuantumScript, QNode, sequence, or "
            f"mapping, got {type(items).__name__}"
        )
