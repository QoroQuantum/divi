# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that converts PennyLane QuantumScript(s) or QNode(s) into a pipeline batch."""

import inspect
from collections.abc import Mapping, Sequence

import numpy as np
import pennylane as qml
import sympy as sp
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP
from pennylane.workflow.qnode import QNode

from divi.circuits import MetaCircuit
from divi.pipeline.abc import MetaCircuitBatch, PipelineEnv, StageToken
from divi.pipeline.stages._circuit_spec_stage import CircuitSpecStage

#: Input types accepted by :class:`PennyLaneSpecStage`.
PennyLaneInput = qml.tape.QuantumScript | QNode

_SUPPORTED_MEASUREMENTS = (ProbabilityMP, ExpectationMP, CountsMP)

_PROBE_SIZE = 100


class PennyLaneSpecStage(CircuitSpecStage):
    """SpecStage that converts PennyLane circuits into MetaCircuit(s).

    Accepts ``QuantumScript`` or ``QNode`` objects (single, sequence, or
    mapping).  QNodes are converted to ``QuantumScript`` via
    :func:`~pennylane.tape.make_qscript`; parametric QNodes automatically
    receive ``sympy`` symbols for each function parameter.

    **QNode parameter handling:**

    - **Scalar parameters** (``def circuit(x, y)``) are converted directly
      to sympy symbols.
    - **Single array parameter** (``def circuit(params)``) is auto-detected:
      the stage probes the function to discover the array size and creates
      a matching sympy array.
    - **Multiple array parameters** are not supported — pass a
      ``QuantumScript`` with explicit sympy symbols instead.

    **Supported measurements:** ``probs()``, ``expval()``, and ``counts()``.
    Other PennyLane measurement types (``sample``, ``state``, ``var``, etc.)
    are not supported by the pipeline execution backend.
    """

    def expand(
        self,
        items: PennyLaneInput | Sequence[PennyLaneInput] | Mapping[str, PennyLaneInput],
        env: PipelineEnv,
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Convert PennyLane circuit(s) to MetaCircuit(s) and build a keyed batch."""
        return super().expand(self._convert(items), env)

    @staticmethod
    def _qnode_to_qscript(qnode: QNode) -> qml.tape.QuantumScript:
        """Convert a QNode into a QuantumScript, creating sympy symbols for parameters.

        Tries scalar symbols first.  If that fails (e.g. the function
        expects an array parameter), falls back to probing the function
        with a dummy numpy array to discover the parameter count and
        creates a sympy array of matching size.
        """
        sig = inspect.signature(qnode.func)
        n_params = len(sig.parameters)
        symbols = sp.symbols(f"p0:{n_params}")  # always a tuple (empty when n=0)

        # Phase 1: try scalar symbols
        try:
            return qml.tape.make_qscript(qnode.func)(*symbols)
        except TypeError:
            pass

        # Phase 2: array parameter — probe to discover size
        if n_params != 1:
            raise TypeError(
                "Failed to convert QNode — the function appears to use "
                "array parameters or numpy operations on its arguments. "
                "QNodes with multiple array parameters are not supported. "
                "Pass a QuantumScript with explicit sympy symbols instead."
            )

        try:
            probe_qs = qml.tape.make_qscript(qnode.func)(np.zeros(_PROBE_SIZE))
        except Exception as e:
            raise TypeError(
                "Failed to convert QNode — could not probe the function "
                "to discover its parameter structure. Pass a QuantumScript "
                "with explicit sympy symbols instead."
            ) from e

        n_gate_params = len(probe_qs.get_parameters())
        if n_gate_params >= _PROBE_SIZE:
            raise TypeError(
                f"QNode appears to use {n_gate_params}+ array elements "
                f"(probe saturated at {_PROBE_SIZE}). Pass a QuantumScript "
                f"with explicit sympy symbols instead."
            )
        sym_array = sp.symarray("p", (n_gate_params,))

        try:
            return qml.tape.make_qscript(qnode.func)(sym_array)
        except (TypeError, IndexError) as e:
            raise TypeError(
                "Failed to convert QNode with array parameter. "
                "Pass a QuantumScript with explicit sympy symbols instead."
            ) from e

    @staticmethod
    def _validate_measurements(qscript: qml.tape.QuantumScript) -> None:
        """Validate that the QuantumScript has exactly one supported measurement."""
        measurements = qscript.measurements
        if len(measurements) != 1 or not isinstance(
            measurements[0], _SUPPORTED_MEASUREMENTS
        ):
            names = [type(m).__name__ for m in measurements]
            raise ValueError(
                f"PennyLaneSpecStage requires exactly one measurement of type "
                f"probs(), expval(), or counts(). Got: {names}"
            )

    @staticmethod
    def _pennylane_to_meta(item: PennyLaneInput) -> MetaCircuit:
        """Convert a single QuantumScript or QNode into a MetaCircuit."""
        if isinstance(item, QNode):
            item = PennyLaneSpecStage._qnode_to_qscript(item)
        PennyLaneSpecStage._validate_measurements(item)
        params = item.get_parameters()
        symbols = np.array([p for p in params if isinstance(p, sp.Basic)], dtype=object)
        return MetaCircuit(source_circuit=item, symbols=symbols)

    @staticmethod
    def _convert(
        items: PennyLaneInput | Sequence[PennyLaneInput] | Mapping[str, PennyLaneInput],
    ) -> MetaCircuit | list[MetaCircuit] | dict[str, MetaCircuit]:
        """Dispatch input shape and convert each input to MetaCircuit."""
        if isinstance(items, (qml.tape.QuantumScript, QNode)):
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
